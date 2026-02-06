"""
RelP (Relevance Propagation) implementation for Llama-style models.

Uses Transluce's actual circuits codebase directly rather than reimplementing.
https://github.com/TransluceAI/circuits
"""

import torch
from typing import Literal
from dataclasses import dataclass

# Import Transluce's actual gradient handling code
# Copied from https://github.com/TransluceAI/circuits (circuits/core/grad.py)
from neurons_bench.transluce.grad import (
    stop_nonlinear_grad_for_llama,
    revert_stop_nonlinear_grad_for_llama,
)


# ============================================================================
# Model modification functions (thin wrappers around Transluce's code)
# ============================================================================

@dataclass
class RelPState:
    """Stores state for reverting RelP modifications."""
    applied: bool = False


def apply_relp_to_model(model, use_half_rule: bool = True) -> RelPState:
    """
    Apply RelP modifications to a Llama-style model.
    
    Uses Transluce's stop_nonlinear_grad_for_llama directly, which:
    1. Replaces all RMSNorm with StraightThroughLlamaRMSNorm
    2. Replaces all attention with NoQKGradAttention (blocks QK gradients)
    3. Replaces all MLP with RelPGradMLP (detach sigmoid, half rule)
    
    Args:
        model: HuggingFace model (LlamaForCausalLM or similar)
        use_half_rule: Whether to apply the half rule for bilinear ops
        
    Returns:
        RelPState object for reverting modifications
    """
    stop_nonlinear_grad_for_llama(
        model,
        use_relp_grad=True,
        use_half_rule=use_half_rule,
    )
    
    # Freeze all parameters (we only want gradients w.r.t. activations)
    for param in model.parameters():
        param.requires_grad = False
    
    return RelPState(applied=True)


def revert_relp_from_model(model, state: RelPState):
    """Revert RelP modifications using Transluce's revert function."""
    if state.applied:
        revert_stop_nonlinear_grad_for_llama(model)
        state.applied = False


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
    
    inner = model.model if hasattr(model, 'model') else model
    n_layers = len(inner.layers)
    
    # Set up hooks to capture MLP activations
    mlp_activations = {}
    hooks = []
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            # Capture input to down_proj = post-activation MLP hidden state
            mlp_activations[layer_idx] = input[0]
        return hook
    
    # Register hooks on down_proj
    for i, layer in enumerate(inner.layers):
        mlp = layer.mlp
        if hasattr(mlp, 'mlp'):  # Wrapped MLP (RelPGradMLP etc)
            h = mlp.mlp.down_proj.register_forward_hook(make_hook(i))
        elif hasattr(mlp, 'down_proj'):
            h = mlp.down_proj.register_forward_hook(make_hook(i))
        else:
            continue
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
        if layer_idx not in mlp_activations:
            continue
            
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
        
        # Attribution = activation * gradient (input x gradient)
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
