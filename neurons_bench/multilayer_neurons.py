"""
Multi-layer neuron concatenation: treat neurons as a pseudo-SAE.

From Neel's suggestion:
  "What if you just took neurons from layers at 25%, 50%, 65%, and 85% of
   the way through the model, concatenated them, and tried to treat this
   like an SAE? How useful is this? Run this through SAE bench."

For Llama 3.1 8B (32 layers):
  25% = layer 8
  50% = layer 16
  65% = layer 21
  85% = layer 27
  → concatenated MLP activations: 4 × 14,336 = 57,344 dimensions

We run sparse probing on this concatenated representation, using the same
methodology as SAEBench:
  1. Collect MLP activations at each layer
  2. Concatenate across layers → one 57K-dim vector per example
  3. Select top-k dimensions by mean activation difference (feature selection)
  4. Train a linear probe on those k dimensions
  5. Compare accuracy against:
     - Single-layer neurons (each layer individually)
     - Residual stream baseline (SAEBench's "LLM baseline")

Datasets: Same as SAEBench sparse probing
  - LabHC/bias_in_bios_class_set1 (profession classification)
  - canrager/amazon_reviews_mcauley_1and5 (product category)
  - fancyzhx/ag_news (topic classification)

This is a self-contained implementation (no SAEBench dependencies).
"""

import argparse
import json
import os
import random
from datetime import datetime

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pandas as pd


# ============================================================================
# Config
# ============================================================================

DEFAULT_CONFIG = {
    "random_seed": 42,
    "train_set_size": 4000,
    "test_set_size": 1000,
    "context_length": 128,
    "llm_batch_size": 8,
    "k_values": [1, 2, 5, 10, 20, 50],
}

# Layer fractions for concatenation
LAYER_FRACTIONS = [0.25, 0.50, 0.65, 0.85]

# Datasets — same as SAEBench sparse probing
DATASETS = {
    "LabHC/bias_in_bios_class_set1": {
        "chosen_classes": ["0", "1", "2", "6", "9"],
        "loader": "bias_in_bios",
    },
    "canrager/amazon_reviews_mcauley_1and5": {
        "chosen_classes": ["1", "2", "3", "5", "6"],
        "loader": "amazon_reviews",
    },
    "fancyzhx/ag_news": {
        "chosen_classes": ["0", "1", "2", "3"],
        "loader": "ag_news",
    },
}


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


def get_layer_indices(n_layers: int) -> list[int]:
    """Compute layer indices from fractions."""
    layers = [int(f * n_layers) for f in LAYER_FRACTIONS]
    # Clamp to valid range
    layers = [min(l, n_layers - 1) for l in layers]
    return layers


# ============================================================================
# Activation collection (multi-layer)
# ============================================================================

def get_multilayer_mlp_activations(
    model,
    tokenizer,
    texts: list[str],
    layers: list[int],
    context_length: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """
    Collect MLP activations at multiple layers and concatenate.

    Returns: [num_texts, d_mlp * len(layers)] mean-pooled activations.
    """
    inner = model.model if hasattr(model, "model") else model
    all_per_layer = {l: [] for l in layers}

    for i in tqdm(range(0, len(texts), batch_size), desc="Collecting activations", leave=False):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=context_length,
            return_tensors="pt",
        ).to(device)

        layer_acts = {}
        hooks = []

        for layer_idx in layers:
            def make_hook(l):
                def hook_fn(module, inp, out):
                    layer_acts[l] = inp[0].detach()
                return hook_fn

            mlp = inner.layers[layer_idx].mlp
            target = mlp.mlp.down_proj if hasattr(mlp, "mlp") else mlp.down_proj
            h = target.register_forward_hook(make_hook(layer_idx))
            hooks.append(h)

        try:
            with torch.no_grad():
                model(**inputs)

            # Build attention mask (non-pad, non-bos, non-eos)
            mask = torch.ones_like(inputs.input_ids, dtype=torch.bool)
            if tokenizer.pad_token_id is not None:
                mask &= inputs.input_ids != tokenizer.pad_token_id
            if tokenizer.bos_token_id is not None:
                mask &= inputs.input_ids != tokenizer.bos_token_id
            if tokenizer.eos_token_id is not None:
                mask &= inputs.input_ids != tokenizer.eos_token_id

            mask_float = mask.float().unsqueeze(-1)  # [B, L, 1]
            count = mask_float.sum(dim=1).clamp(min=1.0)  # [B, 1]

            for layer_idx in layers:
                acts = layer_acts[layer_idx]  # [B, L, D]
                # Mean pool over non-masked positions
                pooled = (acts * mask_float).sum(dim=1) / count  # [B, D]
                all_per_layer[layer_idx].append(pooled.cpu())

        finally:
            for h in hooks:
                h.remove()

    # Concatenate batches per layer, then concatenate across layers
    per_layer_tensors = []
    for layer_idx in layers:
        per_layer_tensors.append(torch.cat(all_per_layer[layer_idx], dim=0))

    # Concatenated: [num_texts, d_mlp * n_layers]
    concatenated = torch.cat(per_layer_tensors, dim=1)

    return concatenated


def get_single_layer_mlp_activations(
    model,
    tokenizer,
    texts: list[str],
    layer: int,
    context_length: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Collect mean-pooled MLP activations at a single layer."""
    return get_multilayer_mlp_activations(
        model, tokenizer, texts, [layer],
        context_length, batch_size, device,
    )


def get_residual_stream_activations(
    model,
    tokenizer,
    texts: list[str],
    layer: int,
    context_length: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """
    Collect residual stream activations (SAEBench's "LLM baseline").
    This is the output of the full layer (after attention + MLP).
    """
    inner = model.model if hasattr(model, "model") else model
    all_acts = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Residual stream", leave=False):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=context_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # hidden_states[layer+1] = output of layer (0 is embeddings)
            acts = outputs.hidden_states[layer + 1].detach()  # [B, L, D]

        # Mean pool
        mask = torch.ones_like(inputs.input_ids, dtype=torch.bool)
        if tokenizer.pad_token_id is not None:
            mask &= inputs.input_ids != tokenizer.pad_token_id
        if tokenizer.bos_token_id is not None:
            mask &= inputs.input_ids != tokenizer.bos_token_id
        if tokenizer.eos_token_id is not None:
            mask &= inputs.input_ids != tokenizer.eos_token_id

        mask_float = mask.float().unsqueeze(-1)
        count = mask_float.sum(dim=1).clamp(min=1.0)
        pooled = (acts * mask_float).sum(dim=1) / count
        all_acts.append(pooled.cpu())

    return torch.cat(all_acts, dim=0)


# ============================================================================
# Dataset loading
# ============================================================================

def load_dataset_texts(
    dataset_name: str,
    chosen_classes: list[str],
    train_size: int,
    test_size: int,
    random_seed: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Load train/test text data for sparse probing."""
    rng = random.Random(random_seed)

    if "bias_in_bios" in dataset_name:
        return _load_bios(dataset_name, chosen_classes, train_size, test_size, random_seed)
    elif "amazon_reviews" in dataset_name:
        return _load_amazon(dataset_name, chosen_classes, train_size, test_size, random_seed)
    elif "ag_news" in dataset_name:
        return _load_ag_news(dataset_name, chosen_classes, train_size, test_size, random_seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _load_bios(name, classes, train_size, test_size, seed):
    base_name = name.split("_class_set")[0]
    dataset = load_dataset(base_name)
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    n_per_class_train = train_size // (2 * len(classes))
    n_per_class_test = test_size // (2 * len(classes))

    train_data, test_data = {}, {}
    for cls in classes:
        cls_int = int(cls)
        tr = train_df[train_df["profession"] == cls_int]
        te = test_df[test_df["profession"] == cls_int]

        # Balance by gender
        for split_df, split_data, n in [(tr, train_data, n_per_class_train), (te, test_data, n_per_class_test)]:
            texts = []
            for _, g in split_df.groupby("gender"):
                sampled = g.sample(n=min(n, len(g)), random_state=seed)
                texts.extend(sampled["hard_text"].tolist())
            random.Random(seed).shuffle(texts)
            split_data[cls] = texts

    return train_data, test_data


def _load_amazon(name, classes, train_size, test_size, seed):
    base_name = name.split("_class_set")[0]
    dataset = load_dataset(base_name)
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    n_per_class_train = train_size // (2 * len(classes))
    n_per_class_test = test_size // (2 * len(classes))

    train_data, test_data = {}, {}
    for cls in classes:
        cls_int = int(cls)
        for split_df, split_data, n in [(train_df, train_data, n_per_class_train), (test_df, test_data, n_per_class_test)]:
            cat_df = split_df[split_df["category"] == cls_int]
            texts = []
            for _, g in cat_df.groupby("rating"):
                sampled = g.sample(n=min(n, len(g)), random_state=seed)
                texts.extend(sampled["text"].tolist())
            random.Random(seed).shuffle(texts)
            split_data[cls] = texts

    return train_data, test_data


def _load_ag_news(name, classes, train_size, test_size, seed):
    dataset = load_dataset(name, streaming=False)
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    n_train = train_size // 2
    n_test = test_size // 2

    classes_int = [int(c) for c in classes]

    train_data, test_data = {}, {}
    for cls_int in classes_int:
        cls_str = str(cls_int)
        for split_df, split_data, n in [(train_df, train_data, n_train), (test_df, test_data, n_test)]:
            cls_df = split_df[split_df["label"] == cls_int]
            sampled = cls_df.sample(n=min(n, len(cls_df)), random_state=seed)
            split_data[cls_str] = sampled["text"].tolist()

    return train_data, test_data


# ============================================================================
# Sparse probing (mirrors SAEBench)
# ============================================================================

class Probe(nn.Module):
    def __init__(self, d_in: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.net = nn.Linear(d_in, 1, bias=True, dtype=dtype)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def sparse_probe_accuracy(
    train_acts: dict[str, torch.Tensor],
    test_acts: dict[str, torch.Tensor],
    class_name: str,
    k: int | None = None,
    device: str = "cuda",
) -> float:
    """
    Train a sparse linear probe for a one-vs-rest binary classification.
    Mirrors SAEBench's probe training pipeline.

    1. Prepare balanced binary data (class vs rest)
    2. If k is specified, select top-k features by mean activation difference
    3. Train logistic regression probe
    4. Return test accuracy
    """
    # Prepare data
    pos_train = train_acts[class_name]
    neg_parts = []
    neg_keys = [c for c in train_acts if c != class_name]
    samples_per = -(-len(pos_train) // len(neg_keys))  # ceil
    for nk in neg_keys:
        idx = torch.randperm(len(train_acts[nk]))[:samples_per]
        neg_parts.append(train_acts[nk][idx])
    neg_train = torch.cat(neg_parts)
    neg_train = neg_train[torch.randperm(len(neg_train))[:len(pos_train)]]

    X_train = torch.cat([pos_train, neg_train]).to(device)
    y_train = torch.cat([
        torch.ones(len(pos_train)),
        torch.zeros(len(neg_train)),
    ]).to(device)

    # Same for test
    pos_test = test_acts[class_name]
    neg_parts_test = []
    for nk in neg_keys:
        idx = torch.randperm(len(test_acts[nk]))[:samples_per]
        neg_parts_test.append(test_acts[nk][idx])
    neg_test = torch.cat(neg_parts_test)
    neg_test = neg_test[torch.randperm(len(neg_test))[:len(pos_test)]]

    X_test = torch.cat([pos_test, neg_test]).to(device)
    y_test = torch.cat([
        torch.ones(len(pos_test)),
        torch.zeros(len(neg_test)),
    ]).to(device)

    # Feature selection (top-k by mean diff)
    if k is not None:
        pos_mask = y_train == 1
        neg_mask = y_train == 0
        diff = (X_train[pos_mask].float().mean(0) - X_train[neg_mask].float().mean(0)).abs()
        _, top_idx = torch.topk(diff, min(k, diff.numel()))
        X_train = X_train[:, top_idx]
        X_test = X_test[:, top_idx]

    # Train probe
    dim = X_train.shape[1]
    dtype = X_train.dtype
    probe = Probe(dim, dtype).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    best_acc = 0.0
    patience = 0
    for epoch in range(100):
        probe.train()
        idx = torch.randperm(len(X_train))
        for b in range(0, len(X_train), 250):
            batch = idx[b : b + 250]
            logits = probe(X_train[batch])
            loss = criterion(logits, y_train[batch].to(dtype=dtype))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Test
        probe.eval()
        with torch.no_grad():
            preds = (probe(X_test) > 0).float()
            acc = (preds == y_test).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            patience = 0
        else:
            patience += 1
        if patience >= 10:
            break

    return best_acc


def run_sparse_probing(
    train_acts: dict[str, torch.Tensor],
    test_acts: dict[str, torch.Tensor],
    chosen_classes: list[str],
    k_values: list[int],
    label: str,
    device: str = "cuda",
) -> dict[str, float]:
    """Run sparse probing at all k values, averaged across classes."""
    results = {}

    for k in k_values:
        class_accs = []
        for cls in chosen_classes:
            acc = sparse_probe_accuracy(train_acts, test_acts, cls, k=k, device=device)
            class_accs.append(acc)
        avg = sum(class_accs) / len(class_accs)
        results[f"k={k}"] = avg
        print(f"  {label} k={k}: {avg:.3f}")

    # Also full (no feature selection)
    class_accs = []
    for cls in chosen_classes:
        acc = sparse_probe_accuracy(train_acts, test_acts, cls, k=None, device=device)
        class_accs.append(acc)
    results["full"] = sum(class_accs) / len(class_accs)
    print(f"  {label} full: {results['full']:.3f}")

    return results


# ============================================================================
# Main experiment
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-layer neuron concatenation sparse probing"
    )
    parser.add_argument(
        "--model", type=str, default="NousResearch/Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["all"] + list(DATASETS.keys()),
    )
    parser.add_argument(
        "--reference-layer", type=int, default=16,
        help="Single layer for the single-layer neuron baseline",
    )

    args = parser.parse_args()

    random.seed(DEFAULT_CONFIG["random_seed"])
    torch.manual_seed(DEFAULT_CONFIG["random_seed"])

    model, tokenizer = load_model(args.model, args.device)

    inner = model.model if hasattr(model, "model") else model
    n_layers = len(inner.layers)
    concat_layers = get_layer_indices(n_layers)
    d_mlp = inner.layers[0].mlp.down_proj.in_features if hasattr(inner.layers[0].mlp, "down_proj") else inner.layers[0].mlp.mlp.down_proj.in_features

    print("=" * 70)
    print("MULTI-LAYER NEURON CONCATENATION — Sparse Probing")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Total layers: {n_layers}")
    print(f"Concat layers: {concat_layers} (fractions: {LAYER_FRACTIONS})")
    print(f"MLP dim per layer: {d_mlp}")
    print(f"Concatenated dim: {d_mlp * len(concat_layers)}")
    print(f"Reference single layer: {args.reference_layer}")
    print(f"k values: {DEFAULT_CONFIG['k_values']}")
    print()

    datasets_to_run = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    all_results = {}

    for dataset_name in datasets_to_run:
        dataset_info = DATASETS[dataset_name]
        chosen_classes = dataset_info["chosen_classes"]

        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"Classes: {chosen_classes}")
        print(f"{'='*60}")

        # Load texts
        train_texts, test_texts = load_dataset_texts(
            dataset_name, chosen_classes,
            DEFAULT_CONFIG["train_set_size"],
            DEFAULT_CONFIG["test_set_size"],
            DEFAULT_CONFIG["random_seed"],
        )

        # Only use classes present in both
        shared = sorted(set(train_texts.keys()) & set(test_texts.keys()))
        if len(shared) < len(chosen_classes):
            print(f"  Warning: only {len(shared)} classes available (requested {len(chosen_classes)})")
            chosen_classes = shared

        # ============================================================
        # 1. Multi-layer concatenated neurons
        # ============================================================
        print(f"\n  Collecting multi-layer MLP activations (layers {concat_layers})...")
        concat_train = {}
        concat_test = {}
        for cls in tqdm(chosen_classes, desc="Classes"):
            concat_train[cls] = get_multilayer_mlp_activations(
                model, tokenizer, train_texts[cls], concat_layers,
                DEFAULT_CONFIG["context_length"], DEFAULT_CONFIG["llm_batch_size"],
                args.device,
            )
            concat_test[cls] = get_multilayer_mlp_activations(
                model, tokenizer, test_texts[cls], concat_layers,
                DEFAULT_CONFIG["context_length"], DEFAULT_CONFIG["llm_batch_size"],
                args.device,
            )

        print(f"\n  Multi-layer neurons ({len(concat_layers)} layers, {concat_train[chosen_classes[0]].shape[1]}d):")
        multilayer_results = run_sparse_probing(
            concat_train, concat_test, chosen_classes,
            DEFAULT_CONFIG["k_values"], "multilayer", args.device,
        )

        # ============================================================
        # 2. Single-layer neurons (baseline)
        # ============================================================
        print(f"\n  Collecting single-layer MLP activations (layer {args.reference_layer})...")
        single_train = {}
        single_test = {}
        for cls in tqdm(chosen_classes, desc="Classes"):
            single_train[cls] = get_single_layer_mlp_activations(
                model, tokenizer, train_texts[cls], args.reference_layer,
                DEFAULT_CONFIG["context_length"], DEFAULT_CONFIG["llm_batch_size"],
                args.device,
            )
            single_test[cls] = get_single_layer_mlp_activations(
                model, tokenizer, test_texts[cls], args.reference_layer,
                DEFAULT_CONFIG["context_length"], DEFAULT_CONFIG["llm_batch_size"],
                args.device,
            )

        print(f"\n  Single-layer neurons (layer {args.reference_layer}, {single_train[chosen_classes[0]].shape[1]}d):")
        single_results = run_sparse_probing(
            single_train, single_test, chosen_classes,
            DEFAULT_CONFIG["k_values"], "single_layer", args.device,
        )

        # ============================================================
        # 3. Residual stream baseline (SAEBench's "LLM baseline")
        # ============================================================
        print(f"\n  Collecting residual stream activations (layer {args.reference_layer})...")
        resid_train = {}
        resid_test = {}
        for cls in tqdm(chosen_classes, desc="Classes"):
            resid_train[cls] = get_residual_stream_activations(
                model, tokenizer, train_texts[cls], args.reference_layer,
                DEFAULT_CONFIG["context_length"], DEFAULT_CONFIG["llm_batch_size"],
                args.device,
            )
            resid_test[cls] = get_residual_stream_activations(
                model, tokenizer, test_texts[cls], args.reference_layer,
                DEFAULT_CONFIG["context_length"], DEFAULT_CONFIG["llm_batch_size"],
                args.device,
            )

        print(f"\n  Residual stream (layer {args.reference_layer}, {resid_train[chosen_classes[0]].shape[1]}d):")
        resid_results = run_sparse_probing(
            resid_train, resid_test, chosen_classes,
            DEFAULT_CONFIG["k_values"], "residual", args.device,
        )

        # ============================================================
        # Summary table
        # ============================================================
        print(f"\n  {'─'*60}")
        print(f"  {'k':>5} | {'Multi-layer':>12} | {'Single-layer':>12} | {'Residual':>12}")
        print(f"  {'─'*60}")
        for k in DEFAULT_CONFIG["k_values"]:
            key = f"k={k}"
            ml = multilayer_results.get(key, 0)
            sl = single_results.get(key, 0)
            rs = resid_results.get(key, 0)

            best = max(ml, sl, rs)
            ml_mark = " *" if ml == best else ""
            sl_mark = " *" if sl == best else ""
            rs_mark = " *" if rs == best else ""

            print(f"  {k:>5} | {ml:>10.1%}{ml_mark} | {sl:>10.1%}{sl_mark} | {rs:>10.1%}{rs_mark}")

        print(f"  {'full':>5} | {multilayer_results['full']:>10.1%}   | {single_results['full']:>10.1%}   | {resid_results['full']:>10.1%}")
        print(f"  {'─'*60}")
        print(f"  * = best at that k\n")

        all_results[dataset_name] = {
            "multilayer_neurons": multilayer_results,
            "single_layer_neurons": single_results,
            "residual_stream": resid_results,
            "concat_layers": concat_layers,
            "reference_layer": args.reference_layer,
        }

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        "model": args.model,
        "layer_fractions": LAYER_FRACTIONS,
        "concat_layers": concat_layers,
        "n_layers": n_layers,
        "d_mlp": d_mlp,
        "concat_dim": d_mlp * len(concat_layers),
        "timestamp": datetime.now().isoformat(),
        "config": DEFAULT_CONFIG,
        "results": all_results,
    }
    output_path = os.path.join(args.output_dir, "multilayer_neurons_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
