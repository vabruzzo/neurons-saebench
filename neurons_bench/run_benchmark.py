"""
Main benchmark runner: Compare neurons + RelP vs SAEs on SAEBench.

Usage:
    python -m neurons_bench.run_benchmark --model meta-llama/Llama-3.1-8B-Instruct --layer 16

For GPU cloud setup:
    - Recommend: A100 40GB or better
    - Use bfloat16 for memory efficiency
    - Flash Attention 2 will be auto-enabled if available
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_model(model_name: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """
    Load model optimized for GPU inference.
    
    Optimal settings for A100:
    - bfloat16 for memory efficiency
    - Flash Attention 2 (auto-enabled)
    - gradient_checkpointing disabled (not training)
    """
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # MUST use eager attention for RelP to work.
    # Flash attention bypasses ALL_ATTENTION_FUNCTIONS dispatch.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
        use_cache=False,  # Disable KV cache for activation hooks
    )
    model.eval()
    
    # Print model info
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Model loaded: {n_params:.1f}B parameters, dtype={dtype}")
    
    return model, tokenizer


def run_sparse_probing_on_neurons(
    model,
    tokenizer,
    layer: int,
    device: str = "cuda",
    k_values: list[int] = [1, 2, 5, 10, 20, 50],
    output_dir: str = "results",
):
    """
    Run sparse probing evaluation on MLP neurons.
    
    This replicates SAEBench's sparse probing but uses raw MLP neurons
    instead of SAE features.
    """
    from neurons_bench.saebench_adapter import NeuronBasedSAE
    
    # Create neuron-based SAE
    neuron_sae = NeuronBasedSAE(
        model=model,
        layer=layer,
        model_name=model.config._name_or_path,
        device=device,
    )
    
    print(f"Created NeuronBasedSAE for layer {layer}")
    print(f"  d_in (residual): {neuron_sae.d_in}")
    print(f"  d_sae (neurons): {neuron_sae.d_sae}")
    
    # Import SAEBench components
    try:
        from sae_bench.evals.sparse_probing.eval_config import SparseProbingEvalConfig
        from sae_bench.evals.sparse_probing import main as sparse_probing_main
        from transformer_lens import HookedTransformer
        
        print("SAEBench imported successfully")
        
        # Create config
        config = SparseProbingEvalConfig(
            random_seed=42,
            k_values=k_values,
        )
        
        # Run evaluation
        # Note: This requires the full SAEBench infrastructure
        # For now, we'll implement a simplified version
        
    except ImportError as e:
        print(f"SAEBench import failed: {e}")
        print("Running simplified sparse probing evaluation...")
    
    return run_simplified_sparse_probing(model, tokenizer, neuron_sae, layer, device, k_values, output_dir)


def run_simplified_sparse_probing(
    model,
    tokenizer,
    neuron_sae,
    layer: int,
    device: str,
    k_values: list[int],
    output_dir: str,
):
    """
    Simplified sparse probing that doesn't require full SAEBench.
    
    Tests: Can k neurons predict binary concepts?
    """
    from neurons_bench.relp import apply_relp_to_model, get_neuron_attributions
    
    # Simple concept datasets
    concepts = {
        "sentiment": {
            "positive": [
                "This movie was absolutely wonderful and I loved every moment",
                "The food was delicious and the service was excellent", 
                "I had an amazing experience, highly recommend",
                "Best purchase I've ever made, totally worth it",
                "The team did a fantastic job, exceeded expectations",
            ],
            "negative": [
                "This was terrible and I hated every second of it",
                "The worst experience of my life, completely awful",
                "Total waste of money, very disappointed",
                "Horrible quality, would not recommend to anyone",
                "The service was atrocious and the staff was rude",
            ],
        },
        "factual_vs_opinion": {
            "factual": [
                "The Earth orbits around the Sun",
                "Water boils at 100 degrees Celsius",
                "Paris is the capital of France",
                "The human body has 206 bones",
                "Light travels at 299,792 kilometers per second",
            ],
            "opinion": [
                "I think pizza is the best food ever",
                "In my opinion, summer is the best season",
                "I believe chocolate is better than vanilla",
                "I feel that dogs make better pets than cats",
                "I think the movie was really boring",
            ],
        },
    }
    
    results = {}
    
    # Apply RelP to model for attribution
    relp_state = apply_relp_to_model(model)
    
    try:
        for concept_name, examples in concepts.items():
            print(f"\nEvaluating concept: {concept_name}")
            
            pos_key = list(examples.keys())[0]
            neg_key = list(examples.keys())[1]
            positive = examples[pos_key]
            negative = examples[neg_key]
            
            # Collect neuron activations for positive examples
            pos_activations = []
            for text in positive:
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get MLP activations at the target layer
                with torch.no_grad():
                    acts = get_mlp_activations(model, inputs, layer)
                    # Average over sequence, take last position
                    pos_activations.append(acts[:, -1, :].cpu())
            
            pos_acts = torch.cat(pos_activations, dim=0)  # [n_pos, d_mlp]
            
            # Collect neuron activations for negative examples
            neg_activations = []
            for text in negative:
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    acts = get_mlp_activations(model, inputs, layer)
                    neg_activations.append(acts[:, -1, :].cpu())
            
            neg_acts = torch.cat(neg_activations, dim=0)  # [n_neg, d_mlp]
            
            # Compute mean activation difference
            pos_mean = pos_acts.mean(dim=0)
            neg_mean = neg_acts.mean(dim=0)
            diff = (pos_mean - neg_mean).abs()
            
            # Test classification with top-k neurons
            concept_results = {}
            for k in k_values:
                # Select top-k neurons by activation difference
                _, top_k_indices = torch.topk(diff, min(k, diff.numel()))
                
                # Simple linear probe on top-k neurons
                X_train = torch.cat([pos_acts[:3, top_k_indices], neg_acts[:3, top_k_indices]], dim=0)
                y_train = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float32)
                
                X_test = torch.cat([pos_acts[3:, top_k_indices], neg_acts[3:, top_k_indices]], dim=0)
                y_test = torch.tensor([1] * (len(positive) - 3) + [0] * (len(negative) - 3), dtype=torch.float32)
                
                # Simple logistic regression
                accuracy = train_simple_probe(X_train, y_train, X_test, y_test)
                concept_results[f"top_{k}_accuracy"] = accuracy
                print(f"  k={k}: accuracy={accuracy:.2%}")
            
            results[concept_name] = concept_results
            
    finally:
        # Revert RelP modifications
        from neurons_bench.relp import revert_relp_from_model
        revert_relp_from_model(model, relp_state)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, f"sparse_probing_neurons_layer{layer}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    return results


def get_mlp_activations(model, inputs, layer: int) -> torch.Tensor:
    """Extract MLP activations from a specific layer."""
    activations = {}
    
    def hook(module, input, output):
        # input[0] is the activation before down_proj
        activations['mlp'] = input[0].detach()
    
    inner = model.model if hasattr(model, 'model') else model
    mlp = inner.layers[layer].mlp
    
    # Handle RelP wrapper
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


def train_simple_probe(X_train, y_train, X_test, y_test, epochs=100, lr=0.1):
    """Train a simple logistic regression probe."""
    # Convert to float32 for numerical stability
    X_train = X_train.float()
    X_test = X_test.float()
    y_train = y_train.float()
    y_test = y_test.float()
    
    # Normalize
    X_mean = X_train.mean(dim=0, keepdim=True)
    X_std = X_train.std(dim=0, keepdim=True) + 1e-8
    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std
    
    # Simple logistic regression
    d = X_train.shape[1]
    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    
    for _ in range(epochs):
        logits = X_train_norm @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_train)
        loss.backward()
        
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()
    
    # Evaluate
    with torch.no_grad():
        test_logits = X_test_norm @ w + b
        predictions = (test_logits > 0).float()
        accuracy = (predictions == y_test).float().mean().item()
    
    return accuracy


def run_relp_attribution_demo(
    model,
    tokenizer,
    layer: int,
    device: str = "cuda",
):
    """
    Demo: Show RelP attribution identifying important neurons for a behavior.
    """
    from neurons_bench.relp import apply_relp_to_model, revert_relp_from_model, get_neuron_attributions, get_top_neurons
    
    print("\n" + "="*60)
    print("RelP Attribution Demo")
    print("="*60)
    
    # Test prompt
    prompt = "The capital of France is"
    expected_token = " Paris"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    expected_id = tokenizer.encode(expected_token, add_special_tokens=False)[0]
    
    print(f"Prompt: '{prompt}'")
    print(f"Expected completion: '{expected_token}' (token_id={expected_id})")
    
    # Apply RelP
    relp_state = apply_relp_to_model(model)
    
    try:
        # Get attributions
        result = get_neuron_attributions(
            model,
            inputs.input_ids,
            target_positions=[-1],
            target_token_ids=[expected_id],
            ablation_mode="zero",
        )
        
        attributions = result['attributions']  # [n_layers, 1, seq_len, d_mlp]
        print(f"\nAttribution shape: {attributions.shape}")
        
        # Get top neurons
        indices, scores = get_top_neurons(attributions, k=20)
        
        print(f"\nTop 20 neurons for predicting '{expected_token}':")
        print("-" * 50)
        for i, (idx, score) in enumerate(zip(indices, scores)):
            layer_idx, pos_idx, neuron_idx = idx.tolist()
            print(f"{i+1:2d}. Layer {layer_idx:2d}, Pos {pos_idx:2d}, Neuron {neuron_idx:5d}: score={score:.4f}")
        
    finally:
        revert_relp_from_model(model, relp_state)
    
    return indices, scores


def main():
    parser = argparse.ArgumentParser(description="Run neurons vs SAEs benchmark")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model name or path")
    parser.add_argument("--layer", type=int, default=16,
                        help="Layer to evaluate (default: 16, middle of Llama 3.1 8B)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--demo-only", action="store_true",
                        help="Only run the RelP attribution demo")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Neurons + RelP vs SAEs Benchmark")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Layer: {args.layer}")
    print(f"Device: {args.device}")
    print()
    
    # Setup
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model, tokenizer = setup_model(args.model, args.device, dtype)
    
    if args.demo_only:
        # Just run the demo
        run_relp_attribution_demo(model, tokenizer, args.layer, args.device)
    else:
        # Run full benchmark
        print("\n" + "="*60)
        print("Running Sparse Probing Evaluation")
        print("="*60)
        
        results = run_sparse_probing_on_neurons(
            model, tokenizer, args.layer, args.device,
            k_values=[1, 2, 5, 10, 20, 50],
            output_dir=args.output_dir,
        )
        
        # Run demo as well
        run_relp_attribution_demo(model, tokenizer, args.layer, args.device)
        
        print("\n" + "="*60)
        print("Benchmark Complete!")
        print("="*60)


if __name__ == "__main__":
    main()
