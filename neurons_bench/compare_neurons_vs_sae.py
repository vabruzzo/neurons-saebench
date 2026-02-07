"""
Side-by-side comparison: RelP attribution to neurons vs SAE latents.

For the same prompts, we run:
  1. RelP attribution → top MLP neurons → check descriptions at neurons.transluce.org
  2. SAE encoding → top SAE latents → check descriptions at neuronpedia.org

This directly answers Neel's question: "See if there seems to be anything where
the neurons are giving you non-trivial qualitative insight that is both consistent
with the SAE and not obvious by just reading the transcript."

SAE: EleutherAI/sae-llama-3.1-8b-64x (MultiTopK, 64x expansion)
     Loaded via the `sparsify` (eai-sparsify) library.
     Hookpoint: layers.{N}.mlp (post-MLP activations)

Neurons: MLP post-activation hidden states at the same layer.
     Attribution via RelP backward pass from target logit.
     Descriptions at neurons.transluce.org.

SAE latent descriptions: neuronpedia.org/llama-scope
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
# Known bias neurons to filter (same as run_attribution.py)
# ============================================================================

BIAS_NEURONS = {
    (23, 306), (20, 3972), (18, 7417), (16, 1241),
    (13, 4208), (11, 11321), (10, 11570), (9, 4255),
    (7, 6673), (6, 5866), (5, 7012), (2, 4786),
    (0, 491), (1, 2427),
}


# ============================================================================
# Test cases — same prompts used for both neurons and SAE
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


# ============================================================================
# Model + SAE loading
# ============================================================================

def load_model(model_name: str, device: str = "cuda"):
    """Load model with eager attention (required for RelP)."""
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def load_sae(sae_repo: str, layer: int, device: str = "cuda"):
    """
    Load an SAE from HuggingFace via the sparsify library.
    
    Install: pip install eai-sparsify
    """
    try:
        from sparsify import Sae
    except ImportError:
        try:
            from sae import Sae
        except ImportError:
            raise ImportError(
                "Install the sparsify library: pip install eai-sparsify\n"
                "Or: pip install sae"
            )

    hookpoint = f"layers.{layer}.mlp"
    print(f"Loading SAE: {sae_repo} @ {hookpoint}...")
    sae = Sae.load_from_hub(sae_repo, hookpoint=hookpoint)
    sae = sae.to(device)
    return sae


# ============================================================================
# SAE feature extraction
# ============================================================================

def get_mlp_output_at_layer(model, inputs, layer: int) -> torch.Tensor:
    """
    Extract MLP OUTPUT at a specific layer (d_model-sized, after down_proj).
    This is what the SAE is trained on (hookpoint: layers.N.mlp).
    """
    activations = {}

    def hook(module, inp, out):
        activations["mlp_out"] = out.detach()

    inner = model.model if hasattr(model, "model") else model
    mlp = inner.layers[layer].mlp
    handle = mlp.register_forward_hook(hook)

    try:
        with torch.no_grad():
            model(**inputs)
        return activations["mlp_out"]
    finally:
        handle.remove()


@torch.no_grad()
def get_top_sae_features(
    sae,
    mlp_output: torch.Tensor,
    layer: int,
    k: int = 10,
) -> list[dict]:
    """
    Encode MLP output through SAE, return top-k firing latents.
    
    Args:
        sae: Loaded SAE (from sparsify)
        mlp_output: [1, seq_len, d_model] MLP output (after down_proj)
        layer: Layer index (for Neuronpedia URLs)
        k: Number of top features to return
    """
    # Get last token activations
    last_token_acts = mlp_output[:, -1, :]  # [1, d_model]

    # Encode through SAE
    # sparsify's encode returns a TopK named tuple or similar
    encoded = sae.encode(last_token_acts)

    # Handle different return types from sparsify
    if hasattr(encoded, "top_indices") and hasattr(encoded, "top_acts"):
        # TopK sparse output
        indices = encoded.top_indices[0]  # [k_active]
        values = encoded.top_acts[0]  # [k_active]
    elif isinstance(encoded, torch.Tensor):
        # Dense output
        values, indices = torch.topk(encoded[0].abs(), k)
    else:
        # Try to handle as sparse tensor or similar
        dense = encoded.to_dense() if hasattr(encoded, "to_dense") else encoded
        if isinstance(dense, torch.Tensor):
            values, indices = torch.topk(dense[0].abs(), k)
        else:
            raise TypeError(f"Unknown SAE output type: {type(encoded)}")

    # Sort by magnitude and take top-k
    sorted_idx = torch.argsort(values.abs(), descending=True)[:k]
    top_indices = indices[sorted_idx]
    top_values = values[sorted_idx]

    features = []
    for idx, val in zip(top_indices, top_values):
        features.append({
            "feature_index": idx.item(),
            "activation": val.item(),
            "neuronpedia_url": f"https://www.neuronpedia.org/llama-scope/layers.{layer}.mlp/{idx.item()}",
        })

    return features


# ============================================================================
# RelP neuron attribution (reuses run_attribution.py logic)
# ============================================================================

def get_top_relp_neurons(
    model,
    tokenizer,
    prompt: str,
    target: str,
    device: str,
    k: int = 10,
    filter_bias: bool = True,
) -> list[dict]:
    """Run RelP and return top-k attributed neurons (filtering bias neurons)."""
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
    fetch_k = k * 3 if filter_bias else k
    indices, scores = get_top_neurons(attributions, k=fetch_k)

    neurons = []
    for idx, score in zip(indices, scores):
        layer_idx, pos_idx, neuron_idx = idx.tolist()
        if filter_bias and (layer_idx, neuron_idx) in BIAS_NEURONS:
            continue
        neurons.append({
            "layer": layer_idx,
            "neuron": neuron_idx,
            "score": score.item(),
            "url": f"https://neurons.transluce.org/{layer_idx}/{neuron_idx}/+",
        })
        if len(neurons) >= k:
            break

    return neurons


# ============================================================================
# Main comparison
# ============================================================================

def run_comparison(
    model,
    tokenizer,
    sae,
    sae_layer: int,
    device: str,
    k: int = 10,
) -> list[dict]:
    """
    For each test case, get:
      1. Top-k neurons via RelP attribution
      2. Top-k SAE latents via SAE encoding
    """
    results = []

    # Apply RelP for neuron attribution
    relp_state = apply_relp_to_model(model)

    try:
        for prompt, target in tqdm(TEST_CASES, desc="Comparing"):
            # --- Neurons via RelP ---
            top_neurons = get_top_relp_neurons(
                model, tokenizer, prompt, target, device, k=k,
            )

            # --- SAE features ---
            # Get MLP output (d_model-sized, after down_proj) — this is what the SAE encodes.
            # RelP only changes the backward pass, forward is normal.
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            mlp_output = get_mlp_output_at_layer(model, inputs, sae_layer)
            top_sae_features = get_top_sae_features(sae, mlp_output, sae_layer, k=k)

            results.append({
                "prompt": prompt,
                "target": target,
                "neurons_relp": top_neurons,
                "sae_latents": top_sae_features,
            })

    finally:
        revert_relp_from_model(model, relp_state)

    return results


def print_comparison(results: list[dict], sae_layer: int):
    """Pretty-print the side-by-side comparison."""
    for r in results:
        print(f"\n{'='*70}")
        print(f"Prompt: \"{r['prompt']}\" → \"{r['target']}\"")
        print(f"{'='*70}")

        print(f"\n  {'NEURONS (RelP attribution, all layers)'}")
        print(f"  {'─'*50}")
        for i, n in enumerate(r["neurons_relp"][:5]):
            print(f"  {i+1}. L{n['layer']}/N{n['neuron']}  score={n['score']:.4f}")
            print(f"     {n['url']}")

        print(f"\n  {'SAE LATENTS (encoding at layer ' + str(sae_layer) + ')'}")
        print(f"  {'─'*50}")
        for i, f in enumerate(r["sae_latents"][:5]):
            print(f"  {i+1}. Feature #{f['feature_index']}  activation={f['activation']:.4f}")
            print(f"     {f['neuronpedia_url']}")


def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side: RelP neurons vs SAE latents"
    )
    parser.add_argument(
        "--model", type=str, default="NousResearch/Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--sae-repo", type=str, default="EleutherAI/sae-llama-3.1-8b-64x",
        help="HuggingFace SAE repo (loaded via sparsify)",
    )
    parser.add_argument("--layer", type=int, default=23,
        help="Layer for SAE encoding (neurons use all layers via RelP)",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="results")

    args = parser.parse_args()

    print("=" * 70)
    print("SIDE-BY-SIDE: Neurons (RelP) vs SAE Latents")
    print("=" * 70)
    print(f"Model:     {args.model}")
    print(f"SAE:       {args.sae_repo}")
    print(f"SAE layer: {args.layer}")
    print(f"Top-k:     {args.k}")
    print()
    print("Neurons: RelP attribution from target logit → all MLP neurons (all layers)")
    print("SAE:     Encode MLP activations at one layer → top firing latents")
    print()

    # Load model
    model, tokenizer = load_model(args.model, args.device)

    # Load SAE
    sae = load_sae(args.sae_repo, args.layer, args.device)

    # Run comparison
    results = run_comparison(model, tokenizer, sae, args.layer, args.device, k=args.k)

    # Print results
    print_comparison(results, args.layer)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        "model": args.model,
        "sae_repo": args.sae_repo,
        "sae_layer": args.layer,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    output_path = os.path.join(args.output_dir, "neurons_vs_sae_comparison.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary
    print(f"\n{'='*70}")
    print("HOW TO INTERPRET")
    print(f"{'='*70}")
    print("1. Click neuron URLs → check if Transluce description matches the target")
    print("2. Click SAE feature URLs → check if Neuronpedia description matches")
    print("3. Compare: do they identify the same concept? Does one give more insight?")
    print("4. Key question: do neurons tell you something SAEs don't, or vice versa?")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
