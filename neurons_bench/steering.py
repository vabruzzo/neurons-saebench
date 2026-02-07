"""
Neuron steering: multiply a neuron's activation by α and measure output change.

Replicates Section 6.2 of the Transluce paper:
  "This single neuron can be steered to flip the top output from the capital
   to the state in a majority of examples."

We test: if you scale the Shakespeare neuron (L23/N13724) by 0, does the model
stop predicting "Shakespeare"? If you scale it by 2, does it predict Shakespeare
more confidently?

Formally, for a set of neurons V and scalar α:
  M_steer(V, α)(x) = M(x; do(v = α·v(x)) for v in V)

This is a causal intervention — it directly tests whether the neuron is
*causally responsible* for the output, not just correlated with it.
"""

import argparse
import json
import os
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================================
# Test cases with their expected top neurons (from our attribution results)
# ============================================================================

STEERING_CASES = [
    {
        "prompt": "The author of Romeo and Juliet is",
        "target": " Shakespeare",
        "neurons": [(23, 13724)],
        "label": "Shakespeare neuron",
    },
    {
        "prompt": "The largest planet in our solar system is",
        "target": " Jupiter",
        "neurons": [(21, 4954)],
        "label": "Planet neuron",
    },
    {
        "prompt": "The capital of France is",
        "target": " Paris",
        "neurons": [(31, 12763)],
        "label": "Paris neuron",
    },
    {
        "prompt": "2 + 2 =",
        "target": " 4",
        "neurons": [(31, 11514)],
        "label": "Math neuron",
    },
    {
        "prompt": "The color of grass is",
        "target": " green",
        "neurons": [(31, 6411)],
        "label": "Green neuron",
    },
    {
        "prompt": "Dogs say",
        "target": " bark",
        "neurons": [(28, 7776)],
        "label": "Bark neuron",
    },
]

# Alpha values to test (following the paper's range)
ALPHA_VALUES = [0, 0.25, 0.5, 1.0, 2.0, 4.0]


# ============================================================================
# Model loading
# ============================================================================

def load_model(model_name: str, device: str = "cuda"):
    """Load model."""
    print(f"Loading {model_name}...")
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


# ============================================================================
# Steering
# ============================================================================

def steer_and_measure(
    model,
    tokenizer,
    prompt: str,
    target_token: str,
    neurons: list[tuple[int, int]],
    alpha: float,
    device: str,
) -> dict:
    """
    Steer neurons by multiplying their activations by α, then measure output.

    Args:
        model: The LLM
        tokenizer: Tokenizer
        prompt: Input prompt
        target_token: Expected target token (e.g. " Shakespeare")
        neurons: List of (layer, neuron_index) to steer
        alpha: Scaling factor (0 = ablate, 1 = normal, 2 = amplify)
        device: Device

    Returns:
        Dict with target probability, top predictions, etc.
    """
    inner = model.model if hasattr(model, "model") else model
    hooks = []

    # Register hooks to scale neuron activations
    for layer_idx, neuron_idx in neurons:
        def make_hook(n_idx, a):
            def hook_fn(module, inp, out):
                # inp[0] is the input to down_proj = MLP hidden state [B, L, d_mlp]
                # We modify it in-place before down_proj processes it
                inp[0][:, :, n_idx] = inp[0][:, :, n_idx] * a
                return None  # don't modify output, let down_proj use modified input
            return hook_fn

        mlp = inner.layers[layer_idx].mlp
        target = mlp.mlp.down_proj if hasattr(mlp, "mlp") else mlp.down_proj

        # Use a pre-hook on down_proj to modify its input
        # Actually, we need a forward_pre_hook to modify input before down_proj runs
        h = target.register_forward_pre_hook(
            lambda module, args, n_idx=neuron_idx, a=alpha: (
                _scale_neuron(args, n_idx, a),
            )
        )
        hooks.append(h)

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        target_id = tokenizer.encode(target_token, add_special_tokens=False)[0]

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # [vocab]
            probs = torch.softmax(logits.float(), dim=-1)

            target_prob = probs[target_id].item()
            target_logit = logits[target_id].item()

            # Top 5 predictions
            top_probs, top_ids = torch.topk(probs, 5)
            top_tokens = [tokenizer.decode(tid) for tid in top_ids]

            return {
                "target_prob": target_prob,
                "target_logit": target_logit,
                "target_rank": (probs >= probs[target_id]).sum().item(),
                "top5": [
                    {"token": tok, "prob": p.item()}
                    for tok, p in zip(top_tokens, top_probs)
                ],
            }
    finally:
        for h in hooks:
            h.remove()


def _scale_neuron(args, neuron_idx: int, alpha: float):
    """Scale a specific neuron in the input tensor."""
    x = args[0]  # [B, L, d_mlp]
    x[:, :, neuron_idx] = x[:, :, neuron_idx] * alpha
    return x


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Neuron steering: scale neurons and measure output changes"
    )
    parser.add_argument(
        "--model", type=str, default="NousResearch/Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="results")

    args = parser.parse_args()

    print("=" * 70)
    print("NEURON STEERING")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Alpha values: {ALPHA_VALUES}")
    print(f"Method: M(x; do(v = α·v(x)))")
    print()

    model, tokenizer = load_model(args.model, args.device)

    all_results = []

    for case in STEERING_CASES:
        prompt = case["prompt"]
        target = case["target"]
        neurons = case["neurons"]
        label = case["label"]

        neuron_str = ", ".join(f"L{l}/N{n}" for l, n in neurons)
        print(f"\n{'='*60}")
        print(f"Prompt: \"{prompt}\" -> \"{target}\"")
        print(f"Steering: {neuron_str} ({label})")
        print(f"{'='*60}")
        print(f"  {'α':>6} | {'P(target)':>10} | {'Rank':>5} | Top prediction")
        print(f"  {'─'*55}")

        case_results = []
        for alpha in ALPHA_VALUES:
            result = steer_and_measure(
                model, tokenizer, prompt, target, neurons, alpha, args.device,
            )
            result["alpha"] = alpha

            top1 = result["top5"][0]
            marker = " <-- target" if top1["token"].strip() == target.strip() else ""
            print(
                f"  {alpha:>6.2f} | {result['target_prob']:>10.4f} | "
                f"{result['target_rank']:>5} | "
                f"\"{top1['token']}\" ({top1['prob']:.4f}){marker}"
            )

            case_results.append(result)

        # Compute effect: how much does ablation (α=0) reduce target probability?
        baseline = next(r for r in case_results if r["alpha"] == 1.0)
        ablated = next(r for r in case_results if r["alpha"] == 0.0)

        prob_drop = baseline["target_prob"] - ablated["target_prob"]
        print(f"\n  Effect of ablation (α=0):")
        print(f"    P(target) drops: {baseline['target_prob']:.4f} -> {ablated['target_prob']:.4f} (Δ = {prob_drop:+.4f})")
        print(f"    Rank changes: {baseline['target_rank']} -> {ablated['target_rank']}")

        if prob_drop > 0.01:
            print(f"    ** Causal: ablating this neuron reduces target probability **")
        else:
            print(f"    Not strongly causal for this specific neuron alone")

        all_results.append({
            "prompt": prompt,
            "target": target,
            "neurons": [{"layer": l, "neuron": n} for l, n in neurons],
            "label": label,
            "steering_results": case_results,
        })

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Neuron':<25} | {'P(α=1)':>8} | {'P(α=0)':>8} | {'Drop':>8} | Causal?")
    print(f"  {'─'*70}")
    for r in all_results:
        baseline_p = next(s["target_prob"] for s in r["steering_results"] if s["alpha"] == 1.0)
        ablated_p = next(s["target_prob"] for s in r["steering_results"] if s["alpha"] == 0.0)
        drop = baseline_p - ablated_p
        causal = "YES" if drop > 0.01 else "no"
        neurons_str = ", ".join(f"L{n['layer']}/N{n['neuron']}" for n in r["neurons"])
        print(f"  {neurons_str:<25} | {baseline_p:>8.4f} | {ablated_p:>8.4f} | {drop:>+8.4f} | {causal}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        "model": args.model,
        "alpha_values": ALPHA_VALUES,
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
    }
    output_path = os.path.join(args.output_dir, "steering_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
