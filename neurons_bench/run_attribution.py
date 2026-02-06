"""
RelP Attribution Analysis on Llama 3.1 8B.

Uses Transluce's RelP method to attribute model outputs to individual MLP neurons.
Tests on novel factual recall prompts and validates against Transluce's independent
neuron descriptions at neurons.transluce.org.

Reference: https://transluce.org/neuron-circuits
"""

import argparse
import json
import os
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from neurons_bench.relp import (
    apply_relp_to_model,
    revert_relp_from_model,
    get_neuron_attributions,
    get_top_neurons,
)


# ============================================================================
# Known "bias" neurons in Llama 3.1 8B Instruct
# These are always-active neurons that don't encode task-specific information.
# From Transluce (https://transluce.org/neuron-circuits):
#   "We manually filtered out a few neurons which we found were present in
#    the circuit we traced, across every dataset and at many token positions.
#    These neurons are always activated and thus do not seem to provide useful
#    task-specific information."
# ============================================================================

BIAS_NEURONS = {
    # From Transluce's paper (explicitly listed)
    (23, 306), (20, 3972), (18, 7417), (16, 1241),
    (13, 4208), (11, 11321), (10, 11570), (9, 4255),
    (7, 6673), (6, 5866), (5, 7012), (2, 4786),
    # Additional always-on neurons we identified
    (0, 491), (1, 2427),
}


# ============================================================================
# Test cases: Novel factual recall prompts
# These are NOT from Transluce's evaluation - they are our own prompts,
# chosen to test whether RelP finds interpretable neurons on unseen tasks.
# ============================================================================

TEST_CASES = [
    ("The capital of France is", " Paris"),
    ("The largest planet in our solar system is", " Jupiter"),
    ("Water is composed of hydrogen and", " oxygen"),
    ("The author of Romeo and Juliet is", " Shakespeare"),
    ("2 + 2 =", " 4"),
    ("The opposite of hot is", " cold"),
    ("The color of grass is", " green"),
    ("Dogs say", " bark"),
]


def load_model(model_name: str, device: str = "cuda"):
    """Load model with eager attention (required for RelP)."""
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # MUST use eager attention for RelP.
    # Flash attention bypasses ALL_ATTENTION_FUNCTIONS dispatch,
    # preventing Transluce's NoQKGradAttention from intercepting.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    return model, tokenizer


def is_bias_neuron(layer: int, neuron: int) -> bool:
    """Check if a neuron is a known always-on bias neuron."""
    return (layer, neuron) in BIAS_NEURONS


def run_attribution(
    model,
    tokenizer,
    device: str,
    k: int = 20,
    filter_bias: bool = True,
) -> list[dict]:
    """
    Run RelP attribution on factual recall prompts.

    For each prompt, finds the top-k neurons contributing to the
    target token prediction.

    Args:
        model: Llama model
        tokenizer: Tokenizer
        device: Device
        k: Number of top neurons to report
        filter_bias: Whether to filter known bias neurons
    """
    results = []

    relp_state = apply_relp_to_model(model)

    try:
        for prompt, target in tqdm(TEST_CASES, desc="Attribution"):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            target_id = tokenizer.encode(target, add_special_tokens=False)[0]

            result = get_neuron_attributions(
                model,
                inputs.input_ids,
                target_positions=[-1],
                target_token_ids=[target_id],
                ablation_mode="zero",
            )

            attributions = result["attributions"]
            # Get more than k so we have enough after filtering
            fetch_k = k * 3 if filter_bias else k
            indices, scores = get_top_neurons(attributions, k=fetch_k)

            # Compute sparsity
            total_attr = attributions.abs().sum().item()
            top_k_attr = scores[:k].sum().item()
            sparsity = top_k_attr / total_attr if total_attr > 0 else 0

            # Collect top neurons, optionally filtering bias neurons
            top_neurons = []
            for idx, score in zip(indices, scores):
                layer_idx, pos_idx, neuron_idx = idx.tolist()

                if filter_bias and is_bias_neuron(layer_idx, neuron_idx):
                    continue

                top_neurons.append({
                    "layer": layer_idx,
                    "neuron": neuron_idx,
                    "score": score.item(),
                    "url": f"https://neurons.transluce.org/{layer_idx}/{neuron_idx}/+",
                })

                if len(top_neurons) >= k:
                    break

            results.append({
                "prompt": prompt,
                "target": target,
                "sparsity": sparsity,
                "top_neurons": top_neurons,
            })

    finally:
        revert_relp_from_model(model, relp_state)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="RelP Attribution Analysis on Llama 3.1 8B"
    )
    parser.add_argument(
        "--model", type=str, default="NousResearch/Meta-Llama-3.1-8B-Instruct",
        help="Model name (default: NousResearch mirror of Llama 3.1 8B Instruct)"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument(
        "--no-filter", action="store_true",
        help="Don't filter known bias neurons"
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="Number of top neurons to report per prompt"
    )

    args = parser.parse_args()
    filter_bias = not args.no_filter

    print("=" * 70)
    print("RelP ATTRIBUTION ANALYSIS")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Bias neuron filtering: {'ON' if filter_bias else 'OFF'}")
    print(f"Top-k neurons: {args.k}")
    print()

    # Load model
    model, tokenizer = load_model(args.model, args.device)

    # Run attribution
    results = run_attribution(
        model, tokenizer, args.device,
        k=args.k, filter_bias=filter_bias,
    )

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS: Which neurons drive specific outputs?")
    print("=" * 70)

    if filter_bias:
        print(f"\n(Filtering {len(BIAS_NEURONS)} known bias neurons)")

    for result in results:
        print(f"\n'{result['prompt']}' -> '{result['target']}'")
        print(f"  Top neurons:")
        for n in result["top_neurons"][:5]:
            print(f"    L{n['layer']}/N{n['neuron']}: {n['score']:.4f}")
            print(f"      {n['url']}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    output = {
        "model": args.model,
        "bias_filtering": filter_bias,
        "bias_neurons": [list(n) for n in BIAS_NEURONS],
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }

    output_path = os.path.join(args.output_dir, "attribution_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    print("Check top neurons at neurons.transluce.org")
    print("A match = neuron description relates to the target output")
    print("These prompts are NOT from Transluce's evaluation suite,")
    print("so matches constitute independent validation.")
    print("=" * 70)


if __name__ == "__main__":
    main()
