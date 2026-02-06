"""
SAEBench Sparse Probing Evaluation for Neurons + RelP.

This implements the actual SAEBench sparse probing methodology:
- Real datasets (4000 train, 1000 test)
- Feature selection on training data only
- Sklearn LogisticRegression (SAEBench standard)
- Comparison: MLP neurons vs residual stream baseline
"""

import os
import json
import argparse
from datetime import datetime
from typing import Literal

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

from neurons_bench.relp import (
    apply_relp_to_model,
    revert_relp_from_model,
    get_neuron_attributions,
)


def load_model(model_name: str, device: str = "cuda"):
    """Load model with optimal settings."""
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    # MUST use eager attention for RelP to work.
    # Flash attention bypasses ALL_ATTENTION_FUNCTIONS dispatch.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    
    return model, tokenizer


def get_mlp_activations(model, inputs, layer: int) -> torch.Tensor:
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


def get_residual_activations(model, inputs, layer: int) -> torch.Tensor:
    """Extract residual stream activations after a specific layer."""
    activations = {}
    
    def hook(module, input, output):
        # output is the residual after this layer
        if isinstance(output, tuple):
            activations['residual'] = output[0].detach()
        else:
            activations['residual'] = output.detach()
    
    inner = model.model if hasattr(model, 'model') else model
    handle = inner.layers[layer].register_forward_hook(hook)
    
    try:
        with torch.no_grad():
            _ = model(**inputs)
        return activations['residual']
    finally:
        handle.remove()


def extract_activations_batch(
    model,
    tokenizer,
    texts: list[str],
    layer: int,
    feature_space: Literal["mlp_neurons", "residual_stream"],
    device: str,
    batch_size: int = 8,
    max_length: int = 128,
) -> np.ndarray:
    """
    Extract activations for a batch of texts.
    
    Args:
        model: The language model
        tokenizer: Tokenizer
        texts: List of text strings
        layer: Layer index
        feature_space: "mlp_neurons" or "residual_stream"
        device: Device
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        
    Returns:
        activations: [n_texts, d_features] numpy array
    """
    all_acts = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Extracting L{layer} {feature_space}", leave=False):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        
        # Get activations
        if feature_space == "mlp_neurons":
            acts = get_mlp_activations(model, inputs, layer)
        else:  # residual_stream
            acts = get_residual_activations(model, inputs, layer)
        
        # Take last non-padding token for each example
        attention_mask = inputs.attention_mask
        seq_lengths = attention_mask.sum(dim=1) - 1  # Last token index
        
        batch_acts = []
        for j, seq_len in enumerate(seq_lengths):
            batch_acts.append(acts[j, seq_len, :].cpu().float().numpy())
        
        all_acts.extend(batch_acts)
    
    return np.array(all_acts)


def select_features_by_activation_diff(
    activations: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Select top-k features by mean activation difference between classes.
    
    For multi-class: use one-vs-rest and take union of top features.
    """
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    if n_classes == 2:
        # Binary: simple difference
        pos_mask = labels == unique_labels[1]
        neg_mask = labels == unique_labels[0]
        
        pos_mean = activations[pos_mask].mean(axis=0)
        neg_mean = activations[neg_mask].mean(axis=0)
        
        diff = np.abs(pos_mean - neg_mean)
        top_indices = np.argsort(diff)[-k:]
        
    else:
        # Multi-class: aggregate differences
        all_diffs = np.zeros(activations.shape[1])
        
        for label in unique_labels:
            this_class = activations[labels == label].mean(axis=0)
            other_classes = activations[labels != label].mean(axis=0)
            diff = np.abs(this_class - other_classes)
            all_diffs += diff
        
        top_indices = np.argsort(all_diffs)[-k:]
    
    return top_indices


def select_features_by_relp(
    model,
    tokenizer,
    texts: list[str],
    labels: np.ndarray,
    layer: int,
    k: int,
    device: str,
    max_examples: int = 200,  # Limit for speed
) -> np.ndarray:
    """
    Select top-k features using RelP attribution.
    
    For each example, compute attribution to predicted token,
    then aggregate by class.
    """
    unique_labels = np.unique(labels)
    
    # Subsample if too many examples
    if len(texts) > max_examples:
        indices = np.random.choice(len(texts), max_examples, replace=False)
        texts = [texts[i] for i in indices]
        labels = labels[indices]
    
    # Get attributions per class
    class_attributions = {label: [] for label in unique_labels}
    
    for text, label in tqdm(zip(texts, labels), total=len(texts), desc="RelP attribution", leave=False):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        
        # Get predicted token
        with torch.no_grad():
            outputs = model(**inputs)
            pred_token = outputs.logits[0, -1, :].argmax().item()
        
        # Get attribution
        result = get_neuron_attributions(
            model,
            inputs.input_ids,
            target_positions=[-1],
            target_token_ids=[pred_token],
            ablation_mode="zero",
        )
        
        # Get attribution at target layer, last position
        attr = result['attributions'][layer, 0, -1, :].cpu().numpy()
        class_attributions[label].append(attr)
    
    # Aggregate and select
    if len(unique_labels) == 2:
        pos_attr = np.mean(class_attributions[unique_labels[1]], axis=0)
        neg_attr = np.mean(class_attributions[unique_labels[0]], axis=0)
        diff = np.abs(pos_attr - neg_attr)
    else:
        all_diffs = np.zeros(len(class_attributions[unique_labels[0]][0]))
        for label in unique_labels:
            this_class = np.mean(class_attributions[label], axis=0)
            other_attrs = []
            for other_label in unique_labels:
                if other_label != label:
                    other_attrs.extend(class_attributions[other_label])
            other_mean = np.mean(other_attrs, axis=0)
            diff = np.abs(this_class - other_mean)
            all_diffs += diff
        diff = all_diffs
    
    top_indices = np.argsort(np.abs(diff))[-k:]
    return top_indices


def run_sparse_probing(
    model,
    tokenizer,
    dataset_name: str,
    layer: int,
    feature_space: Literal["mlp_neurons", "residual_stream"],
    feature_selection: Literal["activation_diff", "relp"],
    k_values: list[int],
    device: str,
    data_dir: str = "data/saebench",
    relp_state=None,
) -> dict:
    """
    Run SAEBench sparse probing evaluation.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        dataset_name: Name of SAEBench dataset
        layer: Layer to extract from
        feature_space: "mlp_neurons" or "residual_stream"
        feature_selection: "activation_diff" or "relp"
        k_values: List of k values for top-k feature selection
        device: Device
        data_dir: Directory containing datasets
        relp_state: Pre-applied RelP state (for RelP feature selection)
        
    Returns:
        Dictionary with results for each k value
    """
    # Load dataset
    train_ds = load_from_disk(os.path.join(data_dir, dataset_name, "train"))
    test_ds = load_from_disk(os.path.join(data_dir, dataset_name, "test"))
    
    train_texts = train_ds["text"]
    test_texts = test_ds["text"]
    
    # Encode labels
    le = LabelEncoder()
    train_labels = le.fit_transform(train_ds["label"])
    test_labels = le.transform(test_ds["label"])
    
    n_classes = len(le.classes_)
    print(f"  Classes: {n_classes}, Train: {len(train_texts)}, Test: {len(test_texts)}")
    
    # Extract activations
    train_acts = extract_activations_batch(
        model, tokenizer, train_texts, layer, feature_space, device
    )
    test_acts = extract_activations_batch(
        model, tokenizer, test_texts, layer, feature_space, device
    )
    
    print(f"  Train activations shape: {train_acts.shape}")
    print(f"  Test activations shape: {test_acts.shape}")
    
    results = {}
    
    for k in k_values:
        # Feature selection on training data ONLY
        if feature_selection == "activation_diff":
            top_indices = select_features_by_activation_diff(train_acts, train_labels, k)
        else:  # relp
            top_indices = select_features_by_relp(
                model, tokenizer, train_texts, train_labels, layer, k, device
            )
        
        # Train logistic regression (SAEBench standard)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(train_acts[:, top_indices], train_labels)
        
        # Evaluate
        accuracy = clf.score(test_acts[:, top_indices], test_labels)
        results[f"k={k}"] = accuracy
        
        print(f"    k={k}: {accuracy:.1%}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="SAEBench Sparse Probing for Neurons")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--datasets", type=str, default="bias_in_bios,amazon_reviews,ag_news",
                        help="Comma-separated list of datasets")
    parser.add_argument("--layers", type=str, default="24,28",
                        help="Comma-separated list of layers")
    parser.add_argument("--k-values", type=str, default="1,2,5,10,20,50",
                        help="Comma-separated list of k values")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data-dir", type=str, default="data/saebench")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--skip-relp", action="store_true",
                        help="Skip RelP feature selection (much faster)")
    
    args = parser.parse_args()
    
    datasets = args.datasets.split(",")
    layers = [int(l) for l in args.layers.split(",")]
    k_values = [int(k) for k in args.k_values.split(",")]
    
    print("="*70)
    print("SAEBench SPARSE PROBING - Neurons + RelP vs Residual Baseline")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Datasets: {datasets}")
    print(f"Layers: {layers}")
    print(f"K values: {k_values}")
    print()
    
    # Load model
    model, tokenizer = load_model(args.model, args.device)
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset}")
        print(f"{'='*70}")
        
        for layer in layers:
            print(f"\n--- Layer {layer} ---")
            
            key = f"{dataset}_L{layer}"
            all_results[key] = {}
            
            # 1. MLP Neurons + Activation Diff
            print(f"\n[1/3] MLP Neurons + Activation Diff")
            all_results[key]["neurons_actdiff"] = run_sparse_probing(
                model, tokenizer, dataset, layer,
                feature_space="mlp_neurons",
                feature_selection="activation_diff",
                k_values=k_values,
                device=args.device,
                data_dir=args.data_dir,
            )
            
            # 2. MLP Neurons + RelP (optional, slow)
            if not args.skip_relp:
                print(f"\n[2/3] MLP Neurons + RelP")
                relp_state = apply_relp_to_model(model)
                try:
                    all_results[key]["neurons_relp"] = run_sparse_probing(
                        model, tokenizer, dataset, layer,
                        feature_space="mlp_neurons",
                        feature_selection="relp",
                        k_values=k_values,
                        device=args.device,
                        data_dir=args.data_dir,
                        relp_state=relp_state,
                    )
                finally:
                    revert_relp_from_model(model, relp_state)
            else:
                print(f"\n[2/3] Skipping RelP (--skip-relp)")
                all_results[key]["neurons_relp"] = None
            
            # 3. Residual Stream Baseline
            print(f"\n[3/3] Residual Stream Baseline (LLM Baseline)")
            all_results[key]["residual_baseline"] = run_sparse_probing(
                model, tokenizer, dataset, layer,
                feature_space="residual_stream",
                feature_selection="activation_diff",
                k_values=k_values,
                device=args.device,
                data_dir=args.data_dir,
            )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"saebench_results_{timestamp}.json")
    
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nResults saved to {output_path}")
    
    # Print comparison table
    print("\n" + "="*70)
    print("SUMMARY: Neurons vs Residual Baseline")
    print("="*70)
    
    for dataset in datasets:
        print(f"\n{dataset}:")
        for layer in layers:
            key = f"{dataset}_L{layer}"
            neurons = all_results[key]["neurons_actdiff"]
            baseline = all_results[key]["residual_baseline"]
            
            print(f"\n  Layer {layer}:")
            print(f"  {'k':>5} | {'Neurons':>10} | {'Residual':>10} | {'Diff':>8}")
            print(f"  {'-'*45}")
            
            for k in k_values:
                k_key = f"k={k}"
                n_acc = neurons.get(k_key, 0)
                b_acc = baseline.get(k_key, 0)
                diff = n_acc - b_acc
                marker = "✓" if diff > 0 else ""
                print(f"  {k:>5} | {n_acc:>9.1%} | {b_acc:>9.1%} | {diff:>+7.1%} {marker}")


if __name__ == "__main__":
    main()
