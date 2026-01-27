"""
SAEBench adapter for neuron-based evaluation.

This module creates an SAE-like interface for MLP neurons, allowing
SAEBench evaluations to run on raw neurons instead of learned features.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NeuronSAEConfig:
    """Configuration mimicking SAE config for compatibility."""
    model_name: str
    d_in: int  # Residual stream dimension
    d_sae: int  # MLP dimension (number of neurons)
    hook_name: str  # e.g., "blocks.12.mlp.hook_post"
    hook_layer: int
    dtype: str = "bfloat16"
    
    # Additional fields SAEBench may expect
    normalize_activations: str = "none"
    device: str = "cuda"
    
    def __post_init__(self):
        # Ensure dtype is a string
        if isinstance(self.dtype, torch.dtype):
            self.dtype = str(self.dtype).split('.')[-1]


class NeuronBasedSAE(nn.Module):
    """
    Wraps MLP neurons to look like an SAE for SAEBench compatibility.
    
    Key insight: MLP activations are already a "privileged basis" due to
    the element-wise nonlinearity. We treat each neuron as a "feature".
    
    The encode/decode operations project between residual stream and 
    MLP activation space using the model's actual MLP weights.
    """
    
    def __init__(
        self,
        model,
        layer: int,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        
        self.model = model
        self.layer = layer
        self.device = device
        self.dtype = dtype
        
        # Get the MLP module
        inner = model.model if hasattr(model, 'model') else model
        mlp = inner.layers[layer].mlp
        
        # Handle RelP wrapper if present
        if hasattr(mlp, 'mlp'):
            mlp = mlp.mlp
        
        # Get MLP weights
        self.gate_proj = mlp.gate_proj
        self.up_proj = mlp.up_proj
        self.down_proj = mlp.down_proj
        
        # Dimensions
        self.d_in = mlp.gate_proj.in_features  # residual stream dim
        self.d_sae = mlp.gate_proj.out_features  # MLP intermediate dim
        
        # Create config
        self.cfg = NeuronSAEConfig(
            model_name=model_name,
            d_in=self.d_in,
            d_sae=self.d_sae,
            hook_name=f"blocks.{layer}.mlp.hook_post",
            hook_layer=layer,
            dtype=str(dtype).split('.')[-1],
            device=device,
        )
        
        # Create W_enc and W_dec for compatibility
        # W_enc: [d_in, d_sae] - projects residual stream to MLP activations
        # W_dec: [d_sae, d_in] - projects MLP activations back to residual stream
        
        # For encoding, we'd normally need gate_proj and up_proj combined
        # But for feature analysis, we primarily care about the decoder
        
        # W_dec is just the down_proj weight
        self.W_dec = nn.Parameter(
            mlp.down_proj.weight.data.clone(),  # [d_in, d_sae]
            requires_grad=False
        )
        
        # W_enc approximation: transpose of W_dec (not exact but reasonable)
        # A better approximation would use the actual encoding path
        self.W_enc = nn.Parameter(
            self.W_dec.data.T.clone(),  # [d_sae, d_in]
            requires_grad=False
        )
        
        # Biases (set to zero - neurons don't have separate biases in this formulation)
        self.b_enc = nn.Parameter(torch.zeros(self.d_sae, device=device, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(self.d_in, device=device, dtype=dtype))
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode residual stream activations to MLP neuron activations.
        
        This simulates what happens in the actual MLP forward pass:
        the gated activation of (gate_proj * up_proj).
        
        Args:
            x: [batch, seq_len, d_in] residual stream activations
            
        Returns:
            [batch, seq_len, d_sae] MLP neuron activations
        """
        # Compute gated activation like the real MLP
        gate = self.gate_proj(x)
        gate_act = torch.nn.functional.silu(gate)
        up = self.up_proj(x)
        
        return gate_act * up
    
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """
        Decode MLP neuron activations back to residual stream.
        
        Args:
            feature_acts: [batch, seq_len, d_sae] MLP neuron activations
            
        Returns:
            [batch, seq_len, d_in] residual stream contribution
        """
        return self.down_proj(feature_acts)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode then decode."""
        encoded = self.encode(x)
        return self.decode(encoded)
    
    @property
    def dict_size(self) -> int:
        """Number of features (neurons) in this SAE."""
        return self.d_sae


class IdentityNeuronSAE(nn.Module):
    """
    Even simpler: treat activations as-is without any transformation.
    
    This is useful when activations are already captured at the right
    hook point (e.g., hook_post for MLP outputs).
    """
    
    def __init__(
        self,
        d_in: int,
        model_name: str,
        layer: int,
        hook_name: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        
        self.d_in = d_in
        self.d_sae = d_in  # Identity: same dimensionality
        
        hook_name = hook_name or f"blocks.{layer}.hook_resid_post"
        
        self.cfg = NeuronSAEConfig(
            model_name=model_name,
            d_in=d_in,
            d_sae=d_in,
            hook_name=hook_name,
            hook_layer=layer,
            dtype=str(dtype).split('.')[-1],
            device=device,
        )
        
        # Identity weights
        self.W_enc = nn.Parameter(
            torch.eye(d_in, device=device, dtype=dtype),
            requires_grad=False
        )
        self.W_dec = nn.Parameter(
            torch.eye(d_in, device=device, dtype=dtype),
            requires_grad=False
        )
        self.b_enc = nn.Parameter(torch.zeros(d_in, device=device, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_in, device=device, dtype=dtype))
        
        self.device = device
        self.dtype = dtype
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        return feature_acts
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    @property
    def dict_size(self) -> int:
        return self.d_sae


def create_neuron_sae_for_layer(
    model,
    layer: int,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> NeuronBasedSAE:
    """
    Create a NeuronBasedSAE for a specific layer.
    
    Args:
        model: The language model
        layer: Layer index
        model_name: Model name for config
        device: Device
        dtype: Data type
        
    Returns:
        NeuronBasedSAE instance
    """
    return NeuronBasedSAE(
        model=model,
        layer=layer,
        model_name=model_name,
        device=device,
        dtype=dtype,
    )


def get_mlp_hook_name(layer: int, hook_type: str = "post") -> str:
    """
    Get the hook name for MLP activations.
    
    Args:
        layer: Layer index
        hook_type: "post" for post-activation, "pre" for pre-activation
        
    Returns:
        Hook name string
    """
    if hook_type == "post":
        return f"blocks.{layer}.mlp.hook_post"
    else:
        return f"blocks.{layer}.mlp.hook_pre"
