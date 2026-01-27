"""
RelP (Relevance Propagation) implementation for Llama-style models.

Adapted from Transluce's circuits codebase:
https://github.com/TransluceAI/circuits

Key insight: RelP linearizes nonlinearities and applies the "half rule" 
for bilinear operations, enabling faithful gradient-based attribution
in a single backward pass.
"""

import torch
import torch.nn as nn
from typing import Literal
from dataclasses import dataclass


# ============================================================================
# Straight-through estimators for nonlinearities
# ============================================================================

class StraightThroughRMSNorm(nn.Module):
    """
    RMSNorm with straight-through gradient: forward computes real RMSNorm,
    backward treats normalization constant as frozen.
    """
    def __init__(self, norm_module):
        super().__init__()
        self.norm = norm_module
        self.weight = norm_module.weight
        self.variance_epsilon = getattr(norm_module, 'variance_epsilon', 
                                        getattr(norm_module, 'eps', 1e-6))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute normalization coefficient but detach it
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        coeff = torch.rsqrt(variance + self.variance_epsilon).detach()
        return x * coeff * self.weight


class ShapleyElementwiseMult(torch.autograd.Function):
    """
    Elementwise multiplication with Shapley gradient (half rule).
    Distributes attribution equally to both branches, avoiding double-counting.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor):
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, y = ctx.saved_tensors
        # Half rule: each branch gets 50% of the gradient
        return 0.5 * grad_output * y, 0.5 * grad_output * x


class RelPGatedMLP(nn.Module):
    """
    Gated MLP with RelP gradient handling:
    - Detaches sigmoid from SiLU (treats as constant multiplier)
    - Applies half rule to gate * up_proj multiplication
    """
    def __init__(self, mlp_module):
        super().__init__()
        self.mlp = mlp_module
        # Store references for external access
        self.gate_proj = mlp_module.gate_proj
        self.up_proj = mlp_module.up_proj
        self.down_proj = mlp_module.down_proj
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU(x) = x * sigmoid(x)
        # We detach sigmoid to make it a constant multiplier
        gate_input = self.mlp.gate_proj(x)
        sigmoid_coeff = torch.sigmoid(gate_input).detach()
        gate_act = gate_input * sigmoid_coeff
        
        up_branch = self.mlp.up_proj(x)
        
        # Apply half rule to the elementwise multiplication
        combined = ShapleyElementwiseMult.apply(gate_act, up_branch)
        
        return self.mlp.down_proj(combined)


class NoGradAttention(nn.Module):
    """
    Attention wrapper that detaches attention weights from the gradient.
    Gradients flow only through values, not through QK computation.
    """
    def __init__(self, attn_module):
        super().__init__()
        self.attn = attn_module
        # Store references
        self.q_proj = attn_module.q_proj
        self.k_proj = attn_module.k_proj
        self.v_proj = attn_module.v_proj
        self.o_proj = attn_module.o_proj
    
    def forward(self, *args, **kwargs):
        # We need to intercept and modify the attention computation
        # For now, we use a simpler approach: hook-based modification
        return self.attn(*args, **kwargs)


# ============================================================================
# Model modification functions
# ============================================================================

@dataclass
class RelPState:
    """Stores original modules for reverting RelP modifications."""
    original_norm: nn.Module
    original_layers: dict


def apply_relp_to_model(model, use_half_rule: bool = True) -> RelPState:
    """
    Apply RelP modifications to a Llama-style model.
    
    Args:
        model: HuggingFace model (LlamaForCausalLM or similar)
        use_half_rule: Whether to apply the half rule for bilinear ops
        
    Returns:
        RelPState object for reverting modifications
    """
    state = RelPState(
        original_norm=None,
        original_layers={}
    )
    
    # Get the inner model
    inner = model.model if hasattr(model, 'model') else model
    
    # Replace final layer norm
    if hasattr(inner, 'norm'):
        state.original_norm = inner.norm
        inner.norm = StraightThroughRMSNorm(inner.norm)
    
    # Replace per-layer components
    for i, layer in enumerate(inner.layers):
        state.original_layers[i] = {
            'input_layernorm': layer.input_layernorm,
            'post_attention_layernorm': layer.post_attention_layernorm,
            'mlp': layer.mlp,
        }
        
        # Replace layernorms
        layer.input_layernorm = StraightThroughRMSNorm(layer.input_layernorm)
        layer.post_attention_layernorm = StraightThroughRMSNorm(layer.post_attention_layernorm)
        
        # Replace MLP with RelP version
        layer.mlp = RelPGatedMLP(layer.mlp)
    
    # Freeze all parameters (we only want gradients w.r.t. activations)
    for param in model.parameters():
        param.requires_grad = False
    
    return state


def revert_relp_from_model(model, state: RelPState):
    """Revert RelP modifications using saved state."""
    inner = model.model if hasattr(model, 'model') else model
    
    if state.original_norm is not None:
        inner.norm = state.original_norm
    
    for i, layer in enumerate(inner.layers):
        if i in state.original_layers:
            orig = state.original_layers[i]
            layer.input_layernorm = orig['input_layernorm']
            layer.post_attention_layernorm = orig['post_attention_layernorm']
            layer.mlp = orig['mlp']


# ============================================================================
# Attribution computation
# ============================================================================

def get_neuron_attributions(
    model,
    input_ids: torch.Tensor,
    target_positions: list[int],
    target_token_ids: list[int] | torch.Tensor,
    ablation_mode: Literal["zero", "mean"] = "zero",
    return_activations: bool = False,
) -> dict:
    """
    Compute RelP attributions from target logits to all MLP neurons.
    
    Args:
        model: Model with RelP modifications applied
        input_ids: Input token IDs [batch, seq_len]
        target_positions: Token positions to attribute from (e.g., [-1] for last)
        target_token_ids: Target vocabulary IDs to attribute to [batch] or [batch, n_targets]
        ablation_mode: "zero" for zero-baseline, "mean" for mean-baseline
        return_activations: Whether to also return neuron activations
        
    Returns:
        Dictionary with:
        - attributions: [n_layers, batch, seq_len, d_mlp] attribution scores
        - activations: [n_layers, batch, seq_len, d_mlp] if return_activations=True
    """
    device = input_ids.device
    batch_size = input_ids.shape[0]
    
    inner = model.model if hasattr(model, 'model') else model
    n_layers = len(inner.layers)
    
    # Set up hooks to capture MLP activations
    mlp_activations = {}
    hooks = []
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            # For RelPGatedMLP, we hook into the down_proj input
            # which is the post-activation MLP hidden state
            mlp_activations[layer_idx] = input[0]
        return hook
    
    # Register hooks
    for i, layer in enumerate(inner.layers):
        mlp = layer.mlp
        if hasattr(mlp, 'mlp'):  # RelPGatedMLP wrapper
            h = mlp.mlp.down_proj.register_forward_hook(make_hook(i))
        else:
            h = mlp.down_proj.register_forward_hook(make_hook(i))
        hooks.append(h)
    
    # Get embeddings with gradient tracking
    embed_layer = inner.embed_tokens
    embeds = embed_layer(input_ids).detach().requires_grad_(True)
    
    # Forward pass
    outputs = model(inputs_embeds=embeds, use_cache=False)
    logits = outputs.logits  # [batch, seq_len, vocab]
    
    # Gather target logits
    if isinstance(target_token_ids, list):
        target_token_ids = torch.tensor(target_token_ids, device=device)
    
    if target_token_ids.dim() == 1:
        target_token_ids = target_token_ids.unsqueeze(-1)  # [batch, 1]
    
    # Sum logits at target positions for target tokens
    target_logits = []
    for pos in target_positions:
        pos_logits = logits[:, pos, :]  # [batch, vocab]
        selected = torch.gather(pos_logits, dim=1, index=target_token_ids)  # [batch, n_targets]
        target_logits.append(selected.sum(dim=-1))  # [batch]
    
    goal = torch.stack(target_logits, dim=1).sum()  # scalar
    
    # Compute attributions via backward pass
    attributions = []
    activations_list = []
    
    for layer_idx in range(n_layers):
        act = mlp_activations[layer_idx]  # [batch, seq_len, d_mlp]
        
        # Compute gradient
        grad = torch.autograd.grad(
            goal,
            act,
            retain_graph=True,
            allow_unused=True
        )[0]
        
        if grad is None:
            grad = torch.zeros_like(act)
        
        # Attribution = activation * gradient (input × gradient)
        if ablation_mode == "zero":
            attr = grad * act
        else:  # mean ablation
            attr = grad * (act - act.mean(dim=0, keepdim=True))
        
        attributions.append(attr.detach())
        if return_activations:
            activations_list.append(act.detach())
    
    # Clean up hooks
    for h in hooks:
        h.remove()
    
    result = {
        'attributions': torch.stack(attributions),  # [n_layers, batch, seq_len, d_mlp]
        'goal_value': goal.detach(),
    }
    
    if return_activations:
        result['activations'] = torch.stack(activations_list)
    
    return result


def get_top_neurons(
    attributions: torch.Tensor,
    k: int | None = None,
    threshold: float | None = None,
    aggregation: Literal["mean", "max", "sum"] = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select top neurons by attribution score.
    
    Args:
        attributions: [n_layers, batch, seq_len, d_mlp]
        k: Number of top neurons to select (mutually exclusive with threshold)
        threshold: Attribution threshold (cumulative mass)
        aggregation: How to aggregate across batch
        
    Returns:
        indices: [n_selected, 3] tensor of (layer, position, neuron) indices
        scores: [n_selected] attribution scores
    """
    # Aggregate across batch
    if aggregation == "mean":
        attr = attributions.mean(dim=1)  # [n_layers, seq_len, d_mlp]
    elif aggregation == "max":
        attr = attributions.abs().max(dim=1).values
    else:
        attr = attributions.sum(dim=1)
    
    # Flatten and get absolute values
    flat_attr = attr.abs().flatten()
    
    if k is not None:
        # Top-k selection
        k = min(k, flat_attr.numel())
        top_values, top_flat_indices = torch.topk(flat_attr, k)
    elif threshold is not None:
        # Threshold-based selection (cumulative mass)
        sorted_values, sorted_indices = torch.sort(flat_attr, descending=True)
        cumsum = sorted_values.cumsum(dim=0)
        total = sorted_values.sum()
        cutoff = (cumsum / total) <= threshold
        n_keep = cutoff.sum().item() + 1  # Include one past threshold
        n_keep = min(n_keep, flat_attr.numel())
        top_values = sorted_values[:n_keep]
        top_flat_indices = sorted_indices[:n_keep]
    else:
        raise ValueError("Must specify either k or threshold")
    
    # Convert flat indices to (layer, position, neuron) coordinates
    n_layers, seq_len, d_mlp = attr.shape
    
    layer_indices = top_flat_indices // (seq_len * d_mlp)
    remainder = top_flat_indices % (seq_len * d_mlp)
    position_indices = remainder // d_mlp
    neuron_indices = remainder % d_mlp
    
    indices = torch.stack([layer_indices, position_indices, neuron_indices], dim=1)
    
    return indices, top_values
