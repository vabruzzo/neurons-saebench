"""
NeuronSelector: Select important neurons for a concept using RelP attribution.

This module bridges RelP attribution with SAEBench's concept-based evaluation.
Instead of using SAE latent activations, we use RelP to identify which MLP
neurons are causally important for a concept.
"""

import torch
import torch.nn as nn
from typing import Literal
from dataclasses import dataclass
from tqdm import tqdm

from neurons_bench.relp import (
    apply_relp_to_model,
    revert_relp_from_model,
    get_neuron_attributions,
    get_top_neurons,
)


@dataclass
class NeuronSelection:
    """Result of neuron selection for a concept."""
    # Indices of selected neurons: [n_selected, 3] as (layer, position, neuron)
    indices: torch.Tensor
    # Attribution scores for selected neurons
    scores: torch.Tensor
    # Mean attribution across all neurons (for normalization)
    mean_attribution: float
    # Total number of neurons considered
    total_neurons: int


class NeuronSelector:
    """
    Select neurons important for a concept using RelP attribution.
    
    Usage:
        selector = NeuronSelector(model, tokenizer, device)
        selection = selector.select_for_concept(
            positive_examples=["The cat sat", "A dog ran"],
            negative_examples=["The 123 is", "Numbers 456"],
            k=50,
        )
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        
        # Apply RelP modifications
        self.relp_state = apply_relp_to_model(model)
        
        # Get model dimensions
        inner = model.model if hasattr(model, 'model') else model
        self.n_layers = len(inner.layers)
        self.d_mlp = inner.layers[0].mlp.gate_proj.out_features if hasattr(inner.layers[0].mlp, 'gate_proj') else inner.layers[0].mlp.mlp.gate_proj.out_features
        
    def __del__(self):
        """Revert RelP modifications on cleanup."""
        if hasattr(self, 'relp_state'):
            revert_relp_from_model(self.model, self.relp_state)
    
    def _tokenize_batch(self, texts: list[str], max_length: int = 128) -> dict:
        """Tokenize a batch of texts."""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
    
    def get_attributions_for_examples(
        self,
        examples: list[str],
        batch_size: int = 8,
        target_position: int = -1,
    ) -> torch.Tensor:
        """
        Get neuron attributions for a set of examples.
        
        Args:
            examples: List of text examples
            batch_size: Batch size for processing
            target_position: Which token position to attribute from (-1 = last)
            
        Returns:
            attributions: [n_layers, n_examples, max_seq_len, d_mlp]
        """
        all_attributions = []
        
        for i in range(0, len(examples), batch_size):
            batch_texts = examples[i:i + batch_size]
            inputs = self._tokenize_batch(batch_texts)
            
            # Get the next token prediction for each example
            with torch.no_grad():
                outputs = self.model(**inputs)
                next_token_logits = outputs.logits[:, target_position, :]
                predicted_tokens = next_token_logits.argmax(dim=-1)
            
            # Compute attributions targeting the predicted token
            result = get_neuron_attributions(
                self.model,
                inputs.input_ids,
                target_positions=[target_position],
                target_token_ids=predicted_tokens,
                ablation_mode="zero",
            )
            
            all_attributions.append(result['attributions'])
        
        # Concatenate along batch dimension
        return torch.cat(all_attributions, dim=1)
    
    def select_for_concept(
        self,
        positive_examples: list[str],
        negative_examples: list[str] | None = None,
        k: int | None = 50,
        threshold: float | None = None,
        batch_size: int = 8,
        method: Literal["diff", "positive_only", "contrast"] = "diff",
        aggregation: Literal["mean", "max"] = "mean",
    ) -> NeuronSelection:
        """
        Select neurons important for a concept.
        
        Args:
            positive_examples: Examples where concept is present
            negative_examples: Examples where concept is absent (optional)
            k: Number of neurons to select (mutually exclusive with threshold)
            threshold: Attribution mass threshold (0-1)
            batch_size: Batch size for processing
            method: How to combine positive/negative attributions
                - "diff": positive - negative (requires negative_examples)
                - "positive_only": only use positive examples
                - "contrast": |positive| - |negative| (requires negative_examples)
            aggregation: How to aggregate across examples
            
        Returns:
            NeuronSelection with selected neuron indices and scores
        """
        # Get attributions for positive examples
        pos_attr = self.get_attributions_for_examples(
            positive_examples, batch_size
        )  # [n_layers, n_pos, seq_len, d_mlp]
        
        if method == "positive_only" or negative_examples is None:
            # Just use positive attributions
            combined_attr = pos_attr
        else:
            # Get attributions for negative examples
            neg_attr = self.get_attributions_for_examples(
                negative_examples, batch_size
            )  # [n_layers, n_neg, seq_len, d_mlp]
            
            # Aggregate across examples first
            if aggregation == "mean":
                pos_mean = pos_attr.mean(dim=1)  # [n_layers, seq_len, d_mlp]
                neg_mean = neg_attr.mean(dim=1)
            else:
                pos_mean = pos_attr.abs().max(dim=1).values
                neg_mean = neg_attr.abs().max(dim=1).values
            
            if method == "diff":
                combined_attr = pos_mean - neg_mean
            else:  # contrast
                combined_attr = pos_mean.abs() - neg_mean.abs()
            
            # Add back a fake batch dimension for compatibility
            combined_attr = combined_attr.unsqueeze(1)
        
        # Select top neurons
        indices, scores = get_top_neurons(
            combined_attr,
            k=k,
            threshold=threshold,
            aggregation=aggregation,
        )
        
        return NeuronSelection(
            indices=indices,
            scores=scores,
            mean_attribution=combined_attr.abs().mean().item(),
            total_neurons=combined_attr.numel(),
        )
    
    def select_by_activation_difference(
        self,
        positive_examples: list[str],
        negative_examples: list[str],
        k: int = 50,
        batch_size: int = 8,
    ) -> NeuronSelection:
        """
        Select neurons by mean activation difference (SAEBench-style).
        
        This mimics how SAEBench selects features: by comparing mean
        activations on positive vs negative examples.
        
        Args:
            positive_examples: Examples where concept is present
            negative_examples: Examples where concept is absent
            k: Number of neurons to select
            batch_size: Batch size for processing
            
        Returns:
            NeuronSelection with selected neuron indices and scores
        """
        all_pos_acts = []
        all_neg_acts = []
        
        # Collect activations for positive examples
        for i in range(0, len(positive_examples), batch_size):
            batch_texts = positive_examples[i:i + batch_size]
            inputs = self._tokenize_batch(batch_texts)
            
            result = get_neuron_attributions(
                self.model,
                inputs.input_ids,
                target_positions=[-1],
                target_token_ids=inputs.input_ids[:, -1],  # dummy target
                return_activations=True,
            )
            all_pos_acts.append(result['activations'])
        
        # Collect activations for negative examples  
        for i in range(0, len(negative_examples), batch_size):
            batch_texts = negative_examples[i:i + batch_size]
            inputs = self._tokenize_batch(batch_texts)
            
            result = get_neuron_attributions(
                self.model,
                inputs.input_ids,
                target_positions=[-1],
                target_token_ids=inputs.input_ids[:, -1],
                return_activations=True,
            )
            all_neg_acts.append(result['activations'])
        
        # Concatenate and compute mean difference
        pos_acts = torch.cat(all_pos_acts, dim=1)  # [n_layers, n_pos, seq_len, d_mlp]
        neg_acts = torch.cat(all_neg_acts, dim=1)  # [n_layers, n_neg, seq_len, d_mlp]
        
        # Mean across examples, focusing on last position
        pos_mean = pos_acts[:, :, -1, :].mean(dim=1)  # [n_layers, d_mlp]
        neg_mean = neg_acts[:, :, -1, :].mean(dim=1)  # [n_layers, d_mlp]
        
        # Difference
        diff = pos_mean - neg_mean  # [n_layers, d_mlp]
        
        # Flatten and get top-k
        flat_diff = diff.abs().flatten()
        k = min(k, flat_diff.numel())
        top_scores, top_flat_indices = torch.topk(flat_diff, k)
        
        # Convert to (layer, neuron) indices (no position since we used last)
        layer_indices = top_flat_indices // self.d_mlp
        neuron_indices = top_flat_indices % self.d_mlp
        
        # Add position index (always last = -1, represented as 0 here)
        position_indices = torch.zeros_like(layer_indices)
        
        indices = torch.stack([layer_indices, position_indices, neuron_indices], dim=1)
        
        return NeuronSelection(
            indices=indices,
            scores=top_scores,
            mean_attribution=diff.abs().mean().item(),
            total_neurons=diff.numel(),
        )


def create_neuron_mask(
    selection: NeuronSelection,
    n_layers: int,
    seq_len: int,
    d_mlp: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Create a boolean mask from a neuron selection.
    
    Args:
        selection: NeuronSelection object
        n_layers: Number of layers
        seq_len: Sequence length
        d_mlp: MLP dimension
        device: Device for tensor
        
    Returns:
        mask: [n_layers, seq_len, d_mlp] boolean tensor
    """
    mask = torch.zeros(n_layers, seq_len, d_mlp, dtype=torch.bool, device=device)
    
    for idx in selection.indices:
        layer, pos, neuron = idx.tolist()
        if pos < seq_len:
            mask[layer, pos, neuron] = True
    
    return mask
