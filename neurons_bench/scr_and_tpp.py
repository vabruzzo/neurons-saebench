"""
SCR (Spurious Correlation Removal) and TPP (Targeted Probe Perturbation)
for MLP neurons, mirroring SAEBench's evaluation.

SAEBench reference: sae_bench/evals/scr_and_tpp/main.py

The SAEBench version:
  1. Collects residual stream activations
  2. Encodes through SAE → gets SAE latent activations
  3. Computes per-latent "effects" via (mean_diff) × (probe_weight · decoder_direction)
  4. Zero-ablates top-N latents in SAE space, decodes back (preserving error term)
  5. Measures probe accuracy on ablated activations

Our version (neurons):
  1. Collects MLP post-activation hidden states (14336-dim for Llama 3.1 8B)
  2. No encode/decode — neurons ARE the features
  3. Computes per-neuron effects via (mean_diff) × probe_weight
  4. Zero-ablates top-N neurons directly
  5. Measures probe accuracy on ablated activations

Datasets (same as SAEBench):
  - LabHC/bias_in_bios (profession × gender) — for SCR
  - canrager/amazon_reviews_mcauley_1and5 (category × rating) — for SCR
  Both are also used for TPP (without spurious correlation setup).

Metrics:
  TPP = intended_accuracy_drop - mean(unintended_accuracy_drops)
    Higher = better targeted ablation (hurts target concept, spares others)
  
  SCR = (ablated_cross_acc - original_cross_acc) / (clean_acc - original_cross_acc)
    Higher = better spurious correlation removal
"""

import argparse
import copy
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
# Config (mirrors SAEBench's ScrAndTppEvalConfig)
# ============================================================================

DEFAULT_CONFIG = {
    "random_seed": 42,
    "train_set_size": 4000,
    "test_set_size": 1000,
    "context_length": 128,
    "probe_train_batch_size": 16,
    "probe_test_batch_size": 500,
    "probe_epochs": 20,
    "probe_lr": 1e-3,
    "probe_l1_penalty": 1e-3,
    "early_stopping_patience": 20,
    "n_values": [2, 5, 10, 20, 50, 100],  # number of neurons to ablate
    "llm_batch_size": 8,
}

# SCR class pairs — same as SAEBench
SCR_COLUMN1_VALS = {
    "LabHC/bias_in_bios_class_set1": [
        ("professor", "nurse"),
        ("architect", "journalist"),
        ("surgeon", "psychologist"),
        ("attorney", "teacher"),
    ],
    "canrager/amazon_reviews_mcauley_1and5": [
        ("Books", "CDs_and_Vinyl"),
        ("Software", "Electronics"),
        ("Pet_Supplies", "Office_Products"),
        ("Industrial_and_Scientific", "Toys_and_Games"),
    ],
}

# TPP classes — same as SAEBench
TPP_CHOSEN_CLASSES = {
    "LabHC/bias_in_bios_class_set1": ["0", "1", "2", "6", "9"],
    "canrager/amazon_reviews_mcauley_1and5": ["1", "2", "3", "5", "6"],
}

# Dataset metadata — same as SAEBench
DATASET_METADATA = {
    "LabHC/bias_in_bios": {
        "text_column_name": "hard_text",
        "column1_name": "profession",
        "column2_name": "gender",
        "column1_mapping": {
            "accountant": 0, "architect": 1, "attorney": 2, "chiropractor": 3,
            "comedian": 4, "composer": 5, "dentist": 6, "dietitian": 7,
            "dj": 8, "filmmaker": 9, "interior_designer": 10, "journalist": 11,
            "model": 12, "nurse": 13, "painter": 14, "paralegal": 15,
            "pastor": 16, "personal_trainer": 17, "photographer": 18,
            "physician": 19, "poet": 20, "professor": 21, "psychologist": 22,
            "rapper": 23, "software_engineer": 24, "surgeon": 25, "teacher": 26,
            "yoga_teacher": 27,
        },
        "column2_mapping": {"male": 0, "female": 1},
    },
    "canrager/amazon_reviews_mcauley_1and5": {
        "text_column_name": "text",
        "column1_name": "category",
        "column2_name": "rating",
        "column1_mapping": {
            "All_Beauty": 0, "Toys_and_Games": 1, "Cell_Phones_and_Accessories": 2,
            "Industrial_and_Scientific": 3, "Gift_Cards": 4, "Musical_Instruments": 5,
            "Electronics": 6, "Handmade_Products": 7, "Arts_Crafts_and_Sewing": 8,
            "Baby_Products": 9, "Health_and_Household": 10, "Office_Products": 11,
            "Digital_Music": 12, "Grocery_and_Gourmet_Food": 13,
            "Sports_and_Outdoors": 14, "Home_and_Kitchen": 15,
            "Subscription_Boxes": 16, "Tools_and_Home_Improvement": 17,
            "Pet_Supplies": 18, "Video_Games": 19, "Kindle_Store": 20,
            "Clothing_Shoes_and_Jewelry": 21, "Patio_Lawn_and_Garden": 22,
            "Unknown": 23, "Books": 24, "Automotive": 25, "CDs_and_Vinyl": 26,
            "Beauty_and_Personal_Care": 27, "Amazon_Fashion": 28,
            "Magazine_Subscriptions": 29, "Software": 30,
            "Health_and_Personal_Care": 31, "Appliances": 32, "Movies_and_TV": 33,
        },
        "column2_mapping": {1.0: 1.0, 5.0: 5.0},
    },
}

COLUMN2_VALS = {
    "LabHC/bias_in_bios_class_set1": ("male", "female"),
    "canrager/amazon_reviews_mcauley_1and5": (1.0, 5.0),
}

# SCR paired class keys — same as SAEBench
PAIRED_CLASS_KEYS = {
    "male / female": "female_data_only",
    "professor / nurse": "nurse_data_only",
    "male_professor / female_nurse": "female_nurse_data_only",
}

POSITIVE_CLASS_LABEL = 1
NEGATIVE_CLASS_LABEL = 0


# ============================================================================
# Model loading
# ============================================================================

def load_model(model_name: str, device: str = "cuda"):
    """Load model with eager attention (required for RelP hooks if used later)."""
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
# Activation collection
# ============================================================================

def get_mlp_activations(
    model,
    tokenizer,
    texts: list[str],
    layer: int,
    context_length: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """
    Collect MLP post-activation hidden states for a list of texts.

    Returns: [num_texts, seq_len, d_mlp] tensor of activations.
    
    We keep the full sequence (like SAEBench) rather than just the last token,
    because probes are trained on mean-pooled activations across non-padding tokens.
    """
    all_activations = []
    inner = model.model if hasattr(model, "model") else model

    for i in tqdm(range(0, len(texts), batch_size), desc="Collecting activations", leave=False):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=context_length,
            return_tensors="pt",
        ).to(device)

        activations = {}

        def hook(module, inp, out):
            activations["mlp"] = inp[0].detach()

        mlp = inner.layers[layer].mlp
        target = mlp.mlp.down_proj if hasattr(mlp, "mlp") else mlp.down_proj
        handle = target.register_forward_hook(hook)

        try:
            with torch.no_grad():
                model(**inputs)

            # Mask padding tokens: zero out activations where input_ids == pad_token_id
            acts = activations["mlp"]  # [B, L, D]
            pad_mask = (inputs.input_ids == tokenizer.pad_token_id).unsqueeze(-1)  # [B, L, 1]
            acts = acts.masked_fill(pad_mask, 0.0)

            # Also mask BOS and EOS tokens (following SAEBench's mask_bos_pad_eos_tokens)
            if tokenizer.bos_token_id is not None:
                bos_mask = (inputs.input_ids == tokenizer.bos_token_id).unsqueeze(-1)
                acts = acts.masked_fill(bos_mask, 0.0)
            if tokenizer.eos_token_id is not None:
                eos_mask = (inputs.input_ids == tokenizer.eos_token_id).unsqueeze(-1)
                acts = acts.masked_fill(eos_mask, 0.0)

            all_activations.append(acts.cpu())
        finally:
            handle.remove()

    return torch.cat(all_activations, dim=0)


def mean_pool_activations(acts_BLD: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool activations across sequence length, ignoring zero-masked positions.
    
    SAEBench does this to get a single activation vector per example.
    Input: [B, L, D], Output: [B, D]
    """
    # Count non-zero positions per example
    nonzero_BL = (acts_BLD.sum(dim=-1) != 0.0).float()  # [B, L]
    nonzero_count_B = nonzero_BL.sum(dim=-1).clamp(min=1.0)  # [B]

    # Sum and divide by count
    summed_BD = acts_BLD.sum(dim=1)  # [B, D]
    meaned_BD = summed_BD / nonzero_count_B.unsqueeze(-1)

    return meaned_BD


# ============================================================================
# Probe training (mirrors SAEBench's probe_training.py)
# ============================================================================

class Probe(nn.Module):
    """Linear probe — same architecture as SAEBench."""

    def __init__(self, activation_dim: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.net = nn.Linear(activation_dim, 1, bias=True, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def prepare_probe_data(
    all_activations: dict[str, torch.Tensor],
    class_name: str,
    perform_scr: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare balanced binary classification data for a probe.
    Mirrors SAEBench's prepare_probe_data.
    """
    positive_acts = all_activations[class_name]
    device = positive_acts.device
    num_positive = len(positive_acts)

    if perform_scr:
        if class_name in PAIRED_CLASS_KEYS:
            neg_key = PAIRED_CLASS_KEYS[class_name]
        else:
            reversed_dict = {v: k for k, v in PAIRED_CLASS_KEYS.items()}
            neg_key = reversed_dict[class_name]
        negative_acts = all_activations[neg_key]
    else:
        negative_parts = []
        negative_keys = [k for k in all_activations if k != class_name]
        samples_per_class = -(-num_positive // len(negative_keys))  # ceil division
        for neg_key in negative_keys:
            indices = torch.randperm(len(all_activations[neg_key]))[:samples_per_class]
            negative_parts.append(all_activations[neg_key][indices])
        negative_acts = torch.cat(negative_parts)

    # Balance: take num_positive negatives
    indices = torch.randperm(len(negative_acts))[:num_positive]
    negative_acts = negative_acts[indices]

    combined = torch.cat([positive_acts, negative_acts])
    labels = torch.empty(len(combined), dtype=torch.int, device=device)
    labels[:num_positive] = POSITIVE_CLASS_LABEL
    labels[num_positive:] = NEGATIVE_CLASS_LABEL

    shuffle_idx = torch.randperm(len(combined))
    return combined[shuffle_idx], labels[shuffle_idx]


def train_probe(
    train_acts: torch.Tensor,
    train_labels: torch.Tensor,
    test_acts: torch.Tensor,
    test_labels: torch.Tensor,
    batch_size: int = 16,
    epochs: int = 20,
    lr: float = 1e-3,
    l1_penalty: float = 1e-3,
    early_stopping_patience: int = 20,
) -> tuple[Probe, float]:
    """Train a linear probe on GPU. Mirrors SAEBench's train_probe_gpu."""
    device = train_acts.device
    dtype = train_acts.dtype
    dim = train_acts.shape[1]

    probe = Probe(dim, dtype).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_acc = 0.0
    best_probe = None
    patience = 0

    for epoch in range(epochs):
        probe.train()
        indices = torch.randperm(len(train_acts))
        for i in range(0, len(train_acts), batch_size):
            batch_idx = indices[i : i + batch_size]
            logits = probe(train_acts[batch_idx])
            loss = criterion(logits, train_labels[batch_idx].to(dtype=dtype))
            if l1_penalty > 0:
                loss = loss + l1_penalty * torch.sum(torch.abs(probe.net.weight))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_acc = test_probe(probe, test_acts, test_labels, batch_size)

        if test_acc > best_acc:
            best_acc = test_acc
            best_probe = copy.deepcopy(probe)
            patience = 0
        else:
            patience += 1

        if patience >= early_stopping_patience:
            break

    assert best_probe is not None
    return best_probe, best_acc


@torch.no_grad()
def test_probe(
    probe: Probe,
    acts: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 500,
) -> float:
    """Test probe accuracy."""
    probe.eval()
    all_correct = []
    for i in range(0, len(labels), batch_size):
        logits = probe(acts[i : i + batch_size])
        preds = (logits > 0.0).long()
        correct = (preds == labels[i : i + batch_size]).float()
        all_correct.append(correct)
    return torch.cat(all_correct).mean().item()


def train_probes_for_all_classes(
    train_acts: dict[str, torch.Tensor],
    test_acts: dict[str, torch.Tensor],
    chosen_classes: list[str],
    perform_scr: bool,
    config: dict,
) -> tuple[dict[str, Probe], dict[str, float]]:
    """Train one probe per class. Returns probes and clean test accuracies."""
    probes = {}
    clean_accs = {}

    torch.set_grad_enabled(True)

    for class_name in chosen_classes:
        print(f"  Training probe for class: {class_name}")
        tr_data, tr_labels = prepare_probe_data(train_acts, class_name, perform_scr)
        te_data, te_labels = prepare_probe_data(test_acts, class_name, perform_scr)

        probe, acc = train_probe(
            tr_data, tr_labels, te_data, te_labels,
            batch_size=config["probe_train_batch_size"],
            epochs=config["probe_epochs"],
            lr=config["probe_lr"],
            l1_penalty=config["probe_l1_penalty"],
            early_stopping_patience=config["early_stopping_patience"],
        )
        probes[class_name] = probe
        clean_accs[class_name] = acc
        print(f"    Accuracy: {acc:.3f}")

    torch.set_grad_enabled(False)
    return probes, clean_accs


# ============================================================================
# Neuron effect computation and ablation
# ============================================================================

@torch.no_grad()
def compute_neuron_effects(
    probe: Probe,
    class_name: str,
    all_train_acts: dict[str, torch.Tensor],
    perform_scr: bool,
) -> torch.Tensor:
    """
    Compute per-neuron effects for a given class.
    
    SAEBench formula: effect_F = mean_activation_diff_F × (probe_weight @ decoder_weight)
    Our formula: effect_i = mean_activation_diff_i × probe_weight_i
    (Since neurons are the features — no decoder needed.)
    """
    train_data, train_labels = prepare_probe_data(all_train_acts, class_name, perform_scr)

    pos_mask = train_labels == POSITIVE_CLASS_LABEL
    neg_mask = train_labels == NEGATIVE_CLASS_LABEL

    mean_pos = train_data[pos_mask].float().mean(dim=0)
    mean_neg = train_data[neg_mask].float().mean(dim=0)
    mean_diff = mean_pos - mean_neg  # [D]

    probe_weight = probe.net.weight.squeeze().float()  # [D]

    effects = mean_diff * probe_weight  # [D]

    if perform_scr:
        effects = effects.abs()
    else:
        # For TPP: only consider positive activations (from positive class)
        mean_diff_clamped = mean_diff.clamp(min=0.0)
        effects = mean_diff_clamped * probe_weight

    return effects


def select_top_n_neurons(
    effects: torch.Tensor, n: int,
) -> tuple[torch.Tensor, list[int]]:
    """Select top-N neurons by effect magnitude. Returns (boolean mask, list of neuron indices)."""
    n = min(n, (effects != 0).sum().item())
    if n == 0:
        return torch.zeros_like(effects, dtype=torch.bool), []

    top_values, top_indices = torch.topk(effects.abs(), n)
    mask = torch.zeros_like(effects, dtype=torch.bool)
    mask[top_indices] = True
    return mask, top_indices.tolist()


def ablate_neurons(
    acts_BD: torch.Tensor,
    neuron_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Zero-ablate selected neurons.
    
    SAEBench does: encode → zero-ablate features → decode + error.
    For neurons: just zero out the selected dimensions directly.
    """
    ablated = acts_BD.clone()
    ablated[:, neuron_mask] = 0.0
    return ablated


# ============================================================================
# TPP evaluation
# ============================================================================

def run_tpp(
    model,
    tokenizer,
    dataset_name: str,
    layer: int,
    device: str,
    config: dict,
) -> dict:
    """
    Run Targeted Probe Perturbation.

    For each class:
      1. Find top-N neurons for that class (by activation difference × probe weight)
      2. Zero-ablate those neurons
      3. Measure: does probe accuracy drop for that class (intended)?
      4. Measure: does probe accuracy stay for other classes (unintended)?
      5. TPP = intended_drop - mean(unintended_drops)
    """
    print(f"\n{'='*60}")
    print(f"TPP: {dataset_name}")
    print(f"{'='*60}")

    chosen_classes = TPP_CHOSEN_CLASSES[dataset_name]

    # 1. Load dataset
    train_texts, test_texts = load_tpp_dataset(dataset_name, chosen_classes, config)

    # 2. Collect activations
    print("Collecting activations...")
    train_acts_BLD = {}
    test_acts_BLD = {}

    for class_name in tqdm(chosen_classes, desc="Classes"):
        train_acts_BLD[class_name] = get_mlp_activations(
            model, tokenizer, train_texts[class_name], layer,
            config["context_length"], config["llm_batch_size"], device,
        )
        test_acts_BLD[class_name] = get_mlp_activations(
            model, tokenizer, test_texts[class_name], layer,
            config["context_length"], config["llm_batch_size"], device,
        )

    # 3. Mean-pool
    train_acts = {k: mean_pool_activations(v).to(device) for k, v in train_acts_BLD.items()}
    test_acts = {k: mean_pool_activations(v).to(device) for k, v in test_acts_BLD.items()}

    # 4. Train probes
    print("Training probes...")
    probes, clean_accs = train_probes_for_all_classes(
        train_acts, test_acts, chosen_classes, perform_scr=False, config=config,
    )
    print(f"Clean accuracies: {clean_accs}")

    # 5. Compute effects and ablate
    print("Computing neuron effects and ablating...")
    results = {}
    ablated_neuron_indices = {}  # Track which neurons were ablated

    for ablated_class in chosen_classes:
        effects = compute_neuron_effects(
            probes[ablated_class], ablated_class, train_acts, perform_scr=False,
        )

        results[ablated_class] = {}
        ablated_neuron_indices[ablated_class] = {}

        for n in config["n_values"]:
            neuron_mask, neuron_ids = select_top_n_neurons(effects, n)
            ablated_neuron_indices[ablated_class][n] = neuron_ids

            # Report top neurons with transluce URLs
            if n <= 20:
                print(f"\n  Class '{ablated_class}', ablating top {n} neurons:")
                for nid in neuron_ids[:10]:  # Show first 10
                    print(f"    N{nid} (effect={effects[nid]:.4f})  "
                          f"https://neurons.transluce.org/{layer}/{nid}/+")

            # Ablate and test all classes
            class_accs = {}
            for eval_class in chosen_classes:
                te_data, te_labels = prepare_probe_data(test_acts, eval_class, perform_scr=False)
                ablated_data = ablate_neurons(te_data, neuron_mask)
                acc = test_probe(probes[eval_class], ablated_data, te_labels)
                class_accs[eval_class] = acc

            results[ablated_class][n] = class_accs

    # 6. Compute TPP metrics
    tpp_metrics = compute_tpp_metrics(results, clean_accs)

    return {
        "clean_accuracies": clean_accs,
        "ablation_results": {
            abl_class: {str(n): accs for n, accs in n_results.items()}
            for abl_class, n_results in results.items()
        },
        "ablated_neurons": {
            abl_class: {str(n): ids for n, ids in n_ids.items()}
            for abl_class, n_ids in ablated_neuron_indices.items()
        },
        "tpp_metrics": tpp_metrics,
    }


def compute_tpp_metrics(
    ablation_results: dict[str, dict[int, dict[str, float]]],
    clean_accs: dict[str, float],
) -> dict[str, float]:
    """
    Compute TPP metric: intended_drop - mean(unintended_drops).
    Mirrors SAEBench's create_tpp_plotting_dict.
    """
    classes = list(clean_accs.keys())
    all_metrics = {}

    for n in list(list(ablation_results.values())[0].keys()):
        intended_diffs = []
        unintended_diffs_all = []

        for target_class in classes:
            intended_clean = clean_accs[target_class]
            intended_ablated = ablation_results[target_class][n][target_class]
            intended_diff = intended_clean - intended_ablated
            intended_diffs.append(intended_diff)

            unintended_diffs = []
            for other_class in classes:
                if other_class == target_class:
                    continue
                other_clean = clean_accs[other_class]
                other_ablated = ablation_results[target_class][n][other_class]
                unintended_diffs.append(other_clean - other_ablated)

            unintended_diffs_all.append(
                sum(unintended_diffs) / len(unintended_diffs) if unintended_diffs else 0
            )

        # Average across classes
        avg_intended = sum(intended_diffs) / len(intended_diffs)
        avg_unintended = sum(unintended_diffs_all) / len(unintended_diffs_all)
        tpp = avg_intended - avg_unintended

        all_metrics[f"tpp_n={n}"] = tpp
        all_metrics[f"tpp_n={n}_intended"] = avg_intended
        all_metrics[f"tpp_n={n}_unintended"] = avg_unintended

    return all_metrics


# ============================================================================
# SCR evaluation
# ============================================================================

def run_scr(
    model,
    tokenizer,
    dataset_name: str,
    layer: int,
    device: str,
    config: dict,
) -> dict:
    """
    Run Spurious Correlation Removal.

    1. Create biased dataset (e.g., all professors are male, all nurses are female)
    2. Train probe on biased data — it learns the spurious correlation
    3. Find neurons encoding the spurious attribute (e.g., gender)
    4. Ablate those neurons
    5. Measure: does the probe now classify by real attribute instead of spurious one?
    """
    print(f"\n{'='*60}")
    print(f"SCR: {dataset_name}")
    print(f"{'='*60}")

    column1_vals_list = SCR_COLUMN1_VALS[dataset_name]
    column2_vals = COLUMN2_VALS[dataset_name]
    all_pair_results = {}

    for column1_vals in column1_vals_list:
        pair_name = f"{column1_vals[0]}_vs_{column1_vals[1]}"
        print(f"\n  Pair: {pair_name}")

        # 1. Load biased dataset
        train_texts, test_texts = load_scr_dataset(
            dataset_name, column1_vals, column2_vals, config,
        )

        chosen_classes = list(PAIRED_CLASS_KEYS.keys())

        # 2. Collect activations
        print("  Collecting activations...")
        train_acts_BLD = {}
        test_acts_BLD = {}

        for class_name in train_texts:
            train_acts_BLD[class_name] = get_mlp_activations(
                model, tokenizer, train_texts[class_name], layer,
                config["context_length"], config["llm_batch_size"], device,
            )
            test_acts_BLD[class_name] = get_mlp_activations(
                model, tokenizer, test_texts[class_name], layer,
                config["context_length"], config["llm_batch_size"], device,
            )

        train_acts = {k: mean_pool_activations(v).to(device) for k, v in train_acts_BLD.items()}
        test_acts = {k: mean_pool_activations(v).to(device) for k, v in test_acts_BLD.items()}

        # 3. Train probes on biased data
        print("  Training probes...")
        probes, clean_accs = train_probes_for_all_classes(
            train_acts, test_acts, chosen_classes, perform_scr=True, config=config,
        )

        # Also get cross-probe accuracies (e.g., gender probe on profession data)
        scr_clean_accs = dict(clean_accs)
        for class_name in chosen_classes:
            if class_name not in PAIRED_CLASS_KEYS:
                continue
            spurious_classes = [k for k in PAIRED_CLASS_KEYS if k != class_name]
            te_data, te_labels = prepare_probe_data(test_acts, class_name, perform_scr=True)
            for spurious_class in spurious_classes:
                cross_acc = test_probe(probes[spurious_class], te_data, te_labels)
                key = f"{spurious_class} probe on {class_name} data"
                scr_clean_accs[key] = cross_acc

        # 4. Compute effects and ablate
        print("  Computing effects and ablating...")
        ablation_results = {}
        scr_ablated_neurons = {}
        for ablated_class in chosen_classes:
            effects = compute_neuron_effects(
                probes[ablated_class], ablated_class, train_acts, perform_scr=True,
            )
            ablation_results[ablated_class] = {}
            scr_ablated_neurons[ablated_class] = {}
            for n in config["n_values"]:
                neuron_mask, neuron_ids = select_top_n_neurons(effects, n)
                scr_ablated_neurons[ablated_class][n] = neuron_ids

                # Report top neurons with transluce URLs
                if n <= 10:
                    print(f"\n    Class '{ablated_class}', ablating top {n} neurons:")
                    for nid in neuron_ids[:5]:
                        print(f"      N{nid} (effect={effects[nid]:.4f})  "
                              f"https://neurons.transluce.org/{layer}/{nid}/+")

                class_accs = {}
                for eval_class in chosen_classes:
                    te_data, te_labels = prepare_probe_data(
                        test_acts, eval_class, perform_scr=True,
                    )
                    ablated_data = ablate_neurons(te_data, neuron_mask)
                    acc = test_probe(probes[eval_class], ablated_data, te_labels)
                    class_accs[eval_class] = acc

                # Also test cross-probe after ablation
                for eval_class in chosen_classes:
                    if eval_class not in PAIRED_CLASS_KEYS:
                        continue
                    spurious_classes = [k for k in PAIRED_CLASS_KEYS if k != eval_class]
                    te_data, te_labels = prepare_probe_data(
                        test_acts, eval_class, perform_scr=True,
                    )
                    ablated_data = ablate_neurons(te_data, neuron_mask)
                    for spurious_class in spurious_classes:
                        cross_acc = test_probe(probes[spurious_class], ablated_data, te_labels)
                        key = f"{spurious_class} probe on {eval_class} data"
                        class_accs[key] = cross_acc

                ablation_results[ablated_class][n] = class_accs

        # 5. Compute SCR metrics
        scr_metrics = compute_scr_metrics(ablation_results, scr_clean_accs)

        all_pair_results[pair_name] = {
            "clean_accuracies": scr_clean_accs,
            "scr_metrics": scr_metrics,
            "ablated_neurons": {
                abl_class: {str(n): ids for n, ids in n_ids.items()}
                for abl_class, n_ids in scr_ablated_neurons.items()
            },
        }

    return all_pair_results


def compute_scr_metrics(
    ablation_results: dict[str, dict[int, dict[str, float]]],
    clean_accs: dict[str, float],
) -> dict[str, float]:
    """
    Compute SCR metric.
    Mirrors SAEBench's get_scr_plotting_dict.
    
    SCR = (ablated_cross_acc - original_cross_acc) / (clean_acc - original_cross_acc)
    """
    metrics = {}
    eval_probe_class = "male_professor / female_nurse"

    dir1_key = f"{eval_probe_class} probe on professor / nurse data"
    dir2_key = f"{eval_probe_class} probe on male / female data"

    dir1_acc = clean_accs.get(dir1_key, 0.5)
    dir2_acc = clean_accs.get(dir2_key, 0.5)

    for dir_idx, (ablated_class, eval_data_class) in enumerate([
        ("male / female", "professor / nurse"),
        ("professor / nurse", "male / female"),
    ], start=1):
        combined_key = f"{eval_probe_class} probe on {eval_data_class} data"
        clean_acc = clean_accs.get(eval_data_class, 0.5)
        original_acc = clean_accs.get(combined_key, 0.5)

        for n in ablation_results.get(ablated_class, {}).keys():
            changed_acc = ablation_results[ablated_class][n].get(combined_key, 0.5)

            if (clean_acc - original_acc) < 0.001:
                scr_score = 0.0
            else:
                scr_score = (changed_acc - original_acc) / (clean_acc - original_acc)

            metrics[f"scr_dir{dir_idx}_n={n}"] = scr_score

            # The "official" SCR metric uses the direction with the weaker baseline
            scr_metric_key = f"scr_n={n}"
            if dir1_acc < dir2_acc and dir_idx == 1:
                metrics[scr_metric_key] = scr_score
            elif dir1_acc > dir2_acc and dir_idx == 2:
                metrics[scr_metric_key] = scr_score

    return metrics


# ============================================================================
# Dataset loading (mirrors SAEBench's dataset_creation.py)
# ============================================================================

def load_tpp_dataset(
    dataset_name: str,
    chosen_classes: list[str],
    config: dict,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Load and split dataset for TPP."""
    base_name = dataset_name.split("_class_set")[0]
    meta = DATASET_METADATA[base_name]

    dataset = load_dataset(base_name)
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    text_col = meta["text_column_name"]
    col1_name = meta["column1_name"]
    col2_name = meta["column2_name"]

    # For TPP: balance by column2 (e.g., gender) within each column1 class (e.g., profession)
    samples_per_quadrant = config["train_set_size"] // 4
    test_samples_per_quadrant = config["test_set_size"] // 4

    train_data = _get_balanced_data(
        train_df, text_col, col1_name, col2_name,
        chosen_classes, meta["column1_mapping"],
        samples_per_quadrant, config["random_seed"],
    )
    test_data = _get_balanced_data(
        test_df, text_col, col1_name, col2_name,
        chosen_classes, meta["column1_mapping"],
        test_samples_per_quadrant, config["random_seed"],
    )

    # Keep only shared keys
    shared = set(train_data.keys()) & set(test_data.keys())
    train_data = {k: v for k, v in train_data.items() if k in shared}
    test_data = {k: v for k, v in test_data.items() if k in shared}

    return train_data, test_data


def _get_balanced_data(
    df: pd.DataFrame,
    text_col: str,
    col1_name: str,
    col2_name: str,
    chosen_classes: list[str],
    col1_mapping: dict,
    samples_per_quadrant: int,
    random_seed: int,
) -> dict[str, list[str]]:
    """Get balanced data for each class, balanced by the secondary column."""
    data = {}
    rng = random.Random(random_seed)

    for class_str in chosen_classes:
        class_idx = int(class_str)
        class_df = df[df[col1_name] == class_idx]

        groups = class_df.groupby(col2_name)
        min_count = min(len(g) for _, g in groups)
        n_per_group = min(min_count, samples_per_quadrant)

        if n_per_group < 10:
            continue

        texts = []
        for _, group_df in groups:
            sampled = group_df.sample(n=n_per_group, random_state=random_seed)
            texts.extend(sampled[text_col].tolist())

        rng.shuffle(texts)
        data[class_str] = texts

    return data


def load_scr_dataset(
    dataset_name: str,
    column1_vals: tuple[str, str],
    column2_vals: tuple[str, str],
    config: dict,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Load biased dataset for SCR.
    Mirrors SAEBench's get_spurious_corr_data.
    
    Creates perfectly biased data: e.g., all professors are male, all nurses are female.
    """
    import numpy as np

    base_name = dataset_name.split("_class_set")[0]
    meta = DATASET_METADATA[base_name]

    dataset = load_dataset(base_name)
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    text_col = meta["text_column_name"]
    col1_name = meta["column1_name"]
    col2_name = meta["column2_name"]
    col1_map = meta["column1_mapping"]
    col2_map = meta["column2_mapping"]

    train_samples = config["train_set_size"] // 4
    test_samples = config["test_set_size"] // 4

    def build_biased(df, n_per_quadrant):
        c1_pos_idx = col1_map[column1_vals[0]]
        c1_neg_idx = col1_map[column1_vals[1]]
        c2_pos_idx = col2_map[column2_vals[0]]
        c2_neg_idx = col2_map[column2_vals[1]]

        pos_pos = df[(df[col1_name] == c1_pos_idx) & (df[col2_name] == c2_pos_idx)][text_col].tolist()
        pos_neg = df[(df[col1_name] == c1_neg_idx) & (df[col2_name] == c2_pos_idx)][text_col].tolist()
        neg_pos = df[(df[col1_name] == c1_pos_idx) & (df[col2_name] == c2_neg_idx)][text_col].tolist()
        neg_neg = df[(df[col1_name] == c1_neg_idx) & (df[col2_name] == c2_neg_idx)][text_col].tolist()

        n = min(len(pos_pos) // 2, len(pos_neg), len(neg_pos), len(neg_neg), n_per_quadrant)

        rng = np.random.default_rng(config["random_seed"])

        # male/female: balanced by profession (column1)
        combined_male = pos_pos[:n] + pos_neg[:n]
        combined_female = neg_pos[:n] + neg_neg[:n]
        rng.shuffle(combined_male)
        rng.shuffle(combined_female)

        # professor/nurse: balanced by gender (column2)
        combined_prof = pos_pos[:n] + neg_pos[:n]
        combined_nurse = pos_neg[:n] + neg_neg[:n]
        rng.shuffle(combined_prof)
        rng.shuffle(combined_nurse)

        # biased: male_professor / female_nurse
        biased_pos = pos_pos[:n * 2]
        biased_neg = neg_neg[:n * 2]
        rng.shuffle(biased_pos)
        rng.shuffle(biased_neg)

        return {
            "male / female": combined_male[:n * 2],
            "female_data_only": combined_female[:n * 2],
            "professor / nurse": combined_prof[:n * 2],
            "nurse_data_only": combined_nurse[:n * 2],
            "male_professor / female_nurse": biased_pos[:n * 2],
            "female_nurse_data_only": biased_neg[:n * 2],
        }

    train_data = build_biased(train_df, train_samples)
    test_data = build_biased(test_df, test_samples)

    return train_data, test_data


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SCR/TPP evaluation for MLP neurons (mirrors SAEBench)"
    )
    parser.add_argument(
        "--model", type=str, default="NousResearch/Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--layer", type=int, default=28)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument(
        "--eval", type=str, choices=["tpp", "scr", "both"], default="both",
    )
    parser.add_argument(
        "--dataset", type=str, default="LabHC/bias_in_bios_class_set1",
        choices=["LabHC/bias_in_bios_class_set1", "canrager/amazon_reviews_mcauley_1and5"],
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SCR/TPP EVALUATION — MLP Neurons")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Layer: {args.layer}")
    print(f"Dataset: {args.dataset}")
    print(f"Eval: {args.eval}")
    print()

    random.seed(DEFAULT_CONFIG["random_seed"])
    torch.manual_seed(DEFAULT_CONFIG["random_seed"])

    model, tokenizer = load_model(args.model, args.device)

    results = {
        "model": args.model,
        "layer": args.layer,
        "dataset": args.dataset,
        "timestamp": datetime.now().isoformat(),
        "config": DEFAULT_CONFIG,
    }

    if args.eval in ("tpp", "both"):
        tpp_results = run_tpp(
            model, tokenizer, args.dataset, args.layer, args.device, DEFAULT_CONFIG,
        )
        results["tpp"] = tpp_results

        print("\n" + "=" * 60)
        print("TPP Results")
        print("=" * 60)
        for key, val in tpp_results["tpp_metrics"].items():
            if "intended" not in key and "unintended" not in key:
                print(f"  {key}: {val:.4f}")

    if args.eval in ("scr", "both"):
        scr_results = run_scr(
            model, tokenizer, args.dataset, args.layer, args.device, DEFAULT_CONFIG,
        )
        results["scr"] = scr_results

        print("\n" + "=" * 60)
        print("SCR Results")
        print("=" * 60)
        for pair, pair_results in scr_results.items():
            print(f"\n  {pair}:")
            for key, val in pair_results["scr_metrics"].items():
                if key.startswith("scr_n="):
                    print(f"    {key}: {val:.4f}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "scr_tpp_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
