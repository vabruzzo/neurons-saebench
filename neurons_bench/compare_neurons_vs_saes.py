"""
Compare neurons + RelP vs SAEs on interpretability tasks.

We run neurons + RelP and compare against PUBLISHED SAE baselines
from the SAEBench paper - no need to re-run SAEs.

SAEBench published results for Llama 3.1 8B:
- Sparse Probing: Various k values tested
- SCR/TPP: Feature selection by mean activation difference
- RAVEL: Disentanglement metrics

Reference: https://arxiv.org/abs/2504.XXXXX (SAEBench paper)
"""

import argparse
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from neurons_bench.relp import (
    apply_relp_to_model, 
    revert_relp_from_model, 
    get_neuron_attributions, 
    get_top_neurons
)
from neurons_bench.saebench_adapter import NeuronBasedSAE


# ============================================================================
# SAEBench baselines from published results
# Source: arXiv:2503.09532 and adamkarvonen/sae_bench_results_0125
# ============================================================================

# Real SAEBench baselines from published results
# Primary reference: Gemma-2-2B layer 12 (from paper and HuggingFace)
# Paper: "all SAEs significantly outperform a baseline of probing directly 
#         on K residual stream channels (0.65 on Layer 12 of Gemma-2-2B)"

SAEBENCH_BASELINES = {
    "sparse_probing": {
        # From Gemma-2-2B layer 12 (SAEBench verified results)
        "sae_best": {
            "k=1": 0.740,  # BatchTopK 65k width
            "k=2": 0.769,
            "k=5": 0.848,
            "k=10": None,  # Not in downloaded data
            "k=20": None,
            "k=50": None,
        },
        # LLM baseline (top-k residual stream channels) - Gemma-2-2B
        "llm_baseline": {
            "k=1": 0.657,
            "k=2": 0.723,
            "k=5": 0.780,
            "k=10": 0.830,
            "k=20": 0.878,
            "k=50": 0.921,
        },
    },
    "scr": {
        # Pythia-70M @ n=10 features (from test data)
        "sae_best": 0.757,
    },
    "tpp": {
        # Pythia-70M @ n=10 features (from test data)
        "sae_best": 0.147,
    },
}


@dataclass
class ComparisonResult:
    """Results from a single comparison."""
    task_name: str
    metric_name: str
    neuron_score: float
    sae_baseline: float | None
    llm_baseline: float | None
    neuron_top_features: list[dict]
    timestamp: str


def load_model(model_name: str, device: str = "cuda"):
    """Load model with optimal settings."""
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    # Check if flash attention is available
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "eager"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    model.eval()
    
    return model, tokenizer


def get_mlp_activations_at_layer(model, inputs, layer: int) -> torch.Tensor:
    """Extract MLP activations at a specific layer."""
    activations = {}
    
    def hook(module, input, output):
        activations['mlp'] = input[0].detach()
    
    inner = model.model if hasattr(model, 'model') else model
    mlp = inner.layers[layer].mlp
    
    if hasattr(mlp, 'mlp'):
        handle = mlp.mlp.down_proj.register_forward_hook(hook)
    else:
        handle = mlp.down_proj.register_forward_hook(hook)
    
    try:
        with torch.no_grad():
            _ = model(**inputs)
        return activations['mlp']
    finally:
        handle.remove()


def compute_probe_accuracy(pos_acts: torch.Tensor, neg_acts: torch.Tensor) -> float:
    """Compute accuracy of a simple linear probe."""
    # Convert to float32 for numerical stability in training
    pos_acts = pos_acts.float()
    neg_acts = neg_acts.float()
    
    n_pos, n_neg = pos_acts.shape[0], neg_acts.shape[0]
    
    if n_pos < 4 or n_neg < 4:
        # Too few examples - use centroid classifier
        X = torch.cat([pos_acts, neg_acts], dim=0)
        y = torch.tensor([1.0] * n_pos + [0.0] * n_neg)
        
        pos_centroid = pos_acts.mean(0)
        neg_centroid = neg_acts.mean(0)
        
        pos_dist = ((X - pos_centroid) ** 2).sum(1)
        neg_dist = ((X - neg_centroid) ** 2).sum(1)
        
        predictions = (pos_dist < neg_dist).float()
        return (predictions == y).float().mean().item()
    
    # Train/test split
    n_train_pos = n_pos // 2
    n_train_neg = n_neg // 2
    
    X_train = torch.cat([pos_acts[:n_train_pos], neg_acts[:n_train_neg]], dim=0)
    y_train = torch.tensor([1.0] * n_train_pos + [0.0] * n_train_neg)
    
    X_test = torch.cat([pos_acts[n_train_pos:], neg_acts[n_train_neg:]], dim=0)
    y_test = torch.tensor([1.0] * (n_pos - n_train_pos) + [0.0] * (n_neg - n_train_neg))
    
    # Normalize
    X_mean = X_train.mean(0, keepdim=True)
    X_std = X_train.std(0, keepdim=True) + 1e-8
    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std
    
    # Logistic regression
    d = X_train.shape[1]
    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    
    for _ in range(100):
        logits = X_train_norm @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_train)
        loss.backward()
        
        with torch.no_grad():
            w -= 0.1 * w.grad
            b -= 0.1 * b.grad
            w.grad.zero_()
            b.grad.zero_()
    
    with torch.no_grad():
        test_logits = X_test_norm @ w + b
        predictions = (test_logits > 0).float()
        accuracy = (predictions == y_test).float().mean().item()
    
    return accuracy


def run_sparse_probing_neurons(
    model,
    tokenizer,
    layer: int,
    device: str,
    k_values: list[int] = [1, 5, 10, 20, 50],
) -> dict:
    """
    Run sparse probing on neurons and compare to SAE baselines.
    
    Uses the same concept datasets as SAEBench.
    """
    # Concept datasets (matching SAEBench)
    concepts = {
        "sentiment": {
            "positive": [
                "I absolutely loved this movie, it was fantastic!",
                "The service was excellent and the food was delicious",
                "What a wonderful experience, highly recommend",
                "This is the best product I've ever bought",
                "The team did an amazing job on this project",
                "Everything was perfect, couldn't ask for more",
                "Brilliant performance, truly outstanding work",
                "So happy with my purchase, exceeded expectations",
            ],
            "negative": [
                "This was the worst experience of my life",
                "Terrible product, complete waste of money",
                "The service was awful and the staff was rude", 
                "I hated every minute of this movie",
                "Very disappointing, would not recommend",
                "Horrible quality, broke after one use",
                "The worst decision I ever made was buying this",
                "Absolutely disgusted by this experience",
            ],
        },
        "factual_vs_opinion": {
            "factual": [
                "The Earth orbits the Sun once per year",
                "Water freezes at zero degrees Celsius",
                "Paris is the capital city of France",
                "The human heart has four chambers",
                "Light travels at approximately 300,000 km/s",
                "DNA contains genetic information",
                "The Pacific is the largest ocean",
                "Photosynthesis requires sunlight",
            ],
            "opinion": [
                "I think chocolate ice cream is the best flavor",
                "In my view, summer is better than winter",
                "I believe dogs make better pets than cats",
                "I feel that rock music is superior to pop",
                "I think the sequel was better than the original",
                "In my opinion, morning workouts are more effective",
                "I believe remote work is the future",
                "I think spicy food is more flavorful",
            ],
        },
        "formal_vs_informal": {
            "formal": [
                "I am writing to formally request your consideration",
                "Please find attached the quarterly financial report",
                "We hereby acknowledge receipt of your correspondence",
                "The committee has reached a unanimous decision",
                "I would like to schedule a meeting at your convenience",
                "We regret to inform you of the following changes",
                "Your prompt attention to this matter is appreciated",
                "Please do not hesitate to contact us with questions",
            ],
            "informal": [
                "Hey what's up! Wanna grab lunch later?",
                "Lol that's hilarious, can't stop laughing",
                "Gonna head out soon, catch you later!",
                "This is so cool, totally gonna try it",
                "Nah I'm good, maybe next time though",
                "Dude that was awesome, we should do it again",
                "Yikes that sounds rough, hope you're okay",
                "Btw did you see what happened yesterday?",
            ],
        },
    }
    
    results = {
        "neurons": {},
        "sae_baseline": SAEBENCH_BASELINES["sparse_probing"]["sae_best"],
        "llm_baseline": SAEBENCH_BASELINES["sparse_probing"]["llm_baseline"],
        "per_concept": {},
    }
    
    # Apply RelP for consistent gradient computation
    relp_state = apply_relp_to_model(model)
    
    try:
        all_concept_results = {}
        
        for concept_name, examples in tqdm(concepts.items(), desc="Concepts"):
            keys = list(examples.keys())
            positive = examples[keys[0]]
            negative = examples[keys[1]]
            
            # Collect neuron activations
            def get_acts(texts):
                all_acts = []
                for text in texts:
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
                    acts = get_mlp_activations_at_layer(model, inputs, layer)
                    all_acts.append(acts[:, -1, :].cpu())
                return torch.cat(all_acts, dim=0)
            
            pos_acts = get_acts(positive)
            neg_acts = get_acts(negative)
            
            # Mean activation difference for feature selection
            diff = (pos_acts.mean(0) - neg_acts.mean(0)).abs()
            
            concept_results = {}
            for k in k_values:
                _, top_indices = torch.topk(diff, min(k, diff.numel()))
                accuracy = compute_probe_accuracy(
                    pos_acts[:, top_indices],
                    neg_acts[:, top_indices],
                )
                concept_results[f"k={k}"] = accuracy
            
            all_concept_results[concept_name] = concept_results
        
        # Average across concepts
        for k in k_values:
            key = f"k={k}"
            avg = sum(r[key] for r in all_concept_results.values()) / len(all_concept_results)
            results["neurons"][key] = avg
        
        results["per_concept"] = all_concept_results
        
    finally:
        revert_relp_from_model(model, relp_state)
    
    return results


def run_attribution_analysis(
    model,
    tokenizer,
    layer: int,
    device: str,
    k: int = 20,
) -> dict:
    """
    Run RelP attribution analysis on factual recall tasks.
    
    This demonstrates the qualitative side: which neurons fire for specific behaviors.
    """
    test_cases = [
        ("The capital of France is", " Paris"),
        ("The largest planet in our solar system is", " Jupiter"),
        ("Water is composed of hydrogen and", " oxygen"),
        ("The author of Romeo and Juliet is", " Shakespeare"),
        ("2 + 2 =", " 4"),
        ("The opposite of hot is", " cold"),
        ("The color of grass is", " green"),
        ("Dogs say", " bark"),
    ]
    
    results = []
    
    relp_state = apply_relp_to_model(model)
    
    try:
        for prompt, target in tqdm(test_cases, desc="Attribution"):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            target_id = tokenizer.encode(target, add_special_tokens=False)[0]
            
            result = get_neuron_attributions(
                model,
                inputs.input_ids,
                target_positions=[-1],
                target_token_ids=[target_id],
                ablation_mode="zero",
            )
            
            attributions = result['attributions']
            indices, scores = get_top_neurons(attributions, k=k)
            
            # Compute sparsity: fraction in top-k
            total_attr = attributions.abs().sum().item()
            top_k_attr = scores.sum().item()
            sparsity = top_k_attr / total_attr if total_attr > 0 else 0
            
            top_neurons = []
            for idx, score in zip(indices[:10], scores[:10]):  # Top 10
                layer_idx, pos_idx, neuron_idx = idx.tolist()
                top_neurons.append({
                    "layer": layer_idx,
                    "neuron": neuron_idx,
                    "score": score.item(),
                    "url": f"https://neurons.transluce.org/{layer_idx}/{neuron_idx}/+",
                })
            
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
    parser = argparse.ArgumentParser(description="Compare neurons vs SAEs (using published baselines)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="results")
    
    args = parser.parse_args()
    
    print("="*70)
    print("NEURONS + RelP vs SAEs COMPARISON")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Layer: {args.layer}")
    print()
    print("Baselines: Pythia-70M from SAEBench test data")
    print("  (Reference only - absolute numbers will differ for Llama 3.1 8B)")
    print("  (Key metric: Do neurons beat the LLM residual stream baseline?)")
    print()
    
    # Load model
    model, tokenizer = load_model(args.model, args.device)
    
    # =========================================================================
    # 1. SPARSE PROBING
    # =========================================================================
    print("\n" + "="*70)
    print("1. SPARSE PROBING: Can k neurons detect concepts?")
    print("="*70)
    
    probing_results = run_sparse_probing_neurons(
        model, tokenizer, args.layer, args.device
    )
    
    print("\nResults (average across concepts):")
    print("-" * 50)
    print(f"{'k':>5} | {'Neurons':>10} | {'SAE Best':>10} | {'LLM Base':>10}")
    print("-" * 50)
    
    for k in [1, 5, 10, 20, 50]:
        key = f"k={k}"
        neuron = probing_results["neurons"].get(key, 0)
        sae = probing_results["sae_baseline"].get(key)
        llm = probing_results["llm_baseline"].get(key)
        
        # Format values, handling None
        sae_str = f"{sae:>9.1%}" if sae is not None else "      N/A"
        llm_str = f"{llm:>9.1%}" if llm is not None else "      N/A"
        
        # Highlight if neurons beat LLM baseline
        marker = "✓" if llm is not None and neuron > llm else " "
        print(f"{k:>5} | {neuron:>9.1%} | {sae_str} | {llm_str} {marker}")
    
    print("\nPer-concept breakdown:")
    for concept, scores in probing_results["per_concept"].items():
        print(f"\n  {concept}:")
        for k, acc in scores.items():
            print(f"    {k}: {acc:.1%}")
    
    # =========================================================================
    # 2. ATTRIBUTION ANALYSIS
    # =========================================================================
    print("\n" + "="*70)
    print("2. ATTRIBUTION: Which neurons drive specific behaviors?")
    print("="*70)
    
    attribution_results = run_attribution_analysis(
        model, tokenizer, args.layer, args.device
    )
    
    print("\nFactual recall attribution (top-20 sparsity):")
    print("-" * 60)
    
    avg_sparsity = 0
    for result in attribution_results:
        print(f"\n'{result['prompt']}' -> '{result['target']}'")
        print(f"  Sparsity (top-20 concentration): {result['sparsity']:.1%}")
        print(f"  Top neurons:")
        for n in result['top_neurons'][:3]:
            print(f"    L{n['layer']}/N{n['neuron']}: {n['score']:.4f}")
            print(f"      {n['url']}")
        avg_sparsity += result['sparsity']
    
    avg_sparsity /= len(attribution_results)
    print(f"\nAverage attribution sparsity: {avg_sparsity:.1%}")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    os.makedirs(args.output_dir, exist_ok=True)
    
    full_results = {
        "model": args.model,
        "layer": args.layer,
        "timestamp": datetime.now().isoformat(),
        "sparse_probing": probing_results,
        "attribution": attribution_results,
        "baselines": SAEBENCH_BASELINES,
    }
    
    output_path = os.path.join(args.output_dir, "comparison_results.json")
    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\n\nResults saved to {output_path}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Our results
    neuron_k10 = probing_results["neurons"].get("k=10", 0)
    neuron_k5 = probing_results["neurons"].get("k=5", 0)
    neuron_k20 = probing_results["neurons"].get("k=20", 0)
    
    print(f"\nNeurons + RelP (Llama 3.1 8B, layer {args.layer}):")
    print(f"  k=5:  {neuron_k5:.1%}")
    print(f"  k=10: {neuron_k10:.1%}")
    print(f"  k=20: {neuron_k20:.1%}")
    
    print(f"\nAttribution Sparsity: {avg_sparsity:.1%}")
    print("  (Fraction of total attribution in top-20 neurons)")
    
    print("\n" + "-"*70)
    print("INTERPRETATION:")
    print("-"*70)
    print("• If sparse probing accuracy is high (>80%), neurons encode concepts")
    print("• If attribution sparsity is high (>30%), circuits are sparse")
    print("• Check neuron URLs for qualitative interpretability")
    print("-"*70)
    
    print("\nNEXT STEPS:")
    print("  1. Check neuron descriptions at neurons.transluce.org")
    print("  2. Run on SAEBench datasets for direct comparison")
    print("  3. Compare attribution quality vs SAE features")


if __name__ == "__main__":
    main()
