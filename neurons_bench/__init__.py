"""
neurons_bench: Comparing neurons + RelP vs SAEs on SAEBench

This package implements neuron-based circuit tracing using RelP attribution
and benchmarks against SAE features on SAEBench metrics.
"""

__version__ = "0.1.0"

from neurons_bench.relp import (
    apply_relp_to_model,
    revert_relp_from_model,
    get_neuron_attributions,
)
from neurons_bench.neuron_selector import NeuronSelector
from neurons_bench.saebench_adapter import NeuronBasedSAE

__all__ = [
    "apply_relp_to_model",
    "revert_relp_from_model", 
    "get_neuron_attributions",
    "NeuronSelector",
    "NeuronBasedSAE",
]
