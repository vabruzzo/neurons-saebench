"""
neurons_bench: RelP attribution analysis for MLP neurons.

Uses Transluce's RelP method to trace model outputs to individual
MLP neurons in Llama 3.1 8B, validating against independent neuron
descriptions at neurons.transluce.org.

Reference: https://transluce.org/neuron-circuits
"""

__version__ = "0.1.0"

from neurons_bench.relp import (
    apply_relp_to_model,
    revert_relp_from_model,
    get_neuron_attributions,
    get_top_neurons,
)

__all__ = [
    "apply_relp_to_model",
    "revert_relp_from_model",
    "get_neuron_attributions",
    "get_top_neurons",
]
