"""
Robust evaluation of neurons + RelP for concept classification.

Improvements over compare_neurons_vs_saes.py:
1. More examples per concept (20+)
2. Multiple layers tested
3. Two feature selection methods: mean activation diff AND RelP attribution
4. Proper train/test split with no leakage
5. Multiple random seeds for variance estimation
"""

import argparse
import json
import os
from datetime import datetime
from dataclasses import dataclass
import random

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from neurons_bench.relp import (
    apply_relp_to_model,
    revert_relp_from_model,
    get_neuron_attributions,
    get_top_neurons,
)


# ============================================================================
# Expanded concept datasets (20+ examples each)
# ============================================================================

CONCEPTS = {
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
            "Amazing quality, worth every penny spent",
            "Incredible experience from start to finish",
            "The best decision I ever made was coming here",
            "Absolutely phenomenal, will definitely return",
            "Five stars, exceeded all my expectations",
            "Wonderful staff, exceptional service throughout",
            "I'm thrilled with the results, truly impressive",
            "Best in class, nothing else comes close",
            "Superb quality and fantastic customer service",
            "Couldn't be happier with this purchase",
            "Outstanding performance, highly recommended",
            "Perfect in every way, no complaints at all",
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
            "Complete disaster from start to finish",
            "Never coming back, totally unacceptable",
            "Regret this purchase, total waste of time",
            "Extremely poor quality, very disappointed",
            "Worst customer service I've ever experienced",
            "Absolutely terrible, avoid at all costs",
            "Nothing worked as advertised, complete scam",
            "Dreadful experience, would give zero stars",
            "Utterly disappointed with everything",
            "Horrible, horrible, horrible experience",
            "Stay away from this place, trust me",
            "The most frustrating experience ever",
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
            "Mammals are warm-blooded animals",
            "The Moon orbits the Earth",
            "Oxygen is essential for human respiration",
            "The Amazon is the longest river",
            "Humans have 206 bones in their body",
            "The Sun is a star at the center of our solar system",
            "Carbon dioxide is a greenhouse gas",
            "The Great Wall of China is over 13,000 miles long",
            "Diamonds are made of carbon atoms",
            "The speed of sound is about 343 meters per second",
            "Mount Everest is the highest peak on Earth",
            "Gold is a chemical element with symbol Au",
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
            "Coffee is definitely better than tea",
            "I feel that classical music is boring",
            "Mountains are more beautiful than beaches",
            "I think action movies are the best genre",
            "In my view, electric cars are overrated",
            "I believe breakfast is the most important meal",
            "I think video games are a waste of time",
            "In my opinion, city life is too stressful",
            "I feel that social media does more harm than good",
            "I believe reading is better than watching TV",
            "I think minimalism is the best lifestyle",
            "In my view, traveling is overrated",
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
            "We are pleased to announce the following appointment",
            "Kindly confirm your attendance at the earliest",
            "I trust this message finds you in good health",
            "The board has approved the proposed amendments",
            "We appreciate your continued support and cooperation",
            "Please be advised of the upcoming policy changes",
            "I respectfully submit this proposal for your review",
            "We acknowledge the validity of your concerns",
            "The organization maintains its commitment to excellence",
            "Your feedback has been duly noted and recorded",
            "We remain at your disposal for any clarification",
            "The terms and conditions are subject to revision",
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
            "OMG I can't believe that just happened",
            "Yo where you at? We're waiting for you",
            "That's sick bro, super jealous right now",
            "Haha no way, that's insane!",
            "Chill out man, it's not a big deal",
            "Sup dude, long time no see!",
            "Gonna bounce, got stuff to do",
            "My bad, totally forgot about that",
            "That's lit, count me in for sure",
            "Bruh that's wild, tell me more",
            "K cool, see ya tomorrow then",
            "Lmao you're so random sometimes",
        ],
    },
    "question_vs_statement": {
        "question": [
            "What time does the meeting start?",
            "How do I get to the train station?",
            "Where did you put my keys?",
            "Why is the sky blue?",
            "Can you help me with this problem?",
            "When will the package arrive?",
            "Who wrote this book?",
            "Which option is better?",
            "Is this the right way to do it?",
            "Have you finished your homework?",
            "Could you explain that again?",
            "What's the meaning of this word?",
            "How long have you been waiting?",
            "Did you remember to call them?",
            "Are we there yet?",
            "Would you like some coffee?",
            "What happened at the party?",
            "Why didn't you tell me earlier?",
            "How much does this cost?",
            "Do you know where she went?",
        ],
        "statement": [
            "The meeting starts at nine o'clock.",
            "I left your keys on the table.",
            "The train station is down the street.",
            "I finished all my homework yesterday.",
            "The package will arrive tomorrow morning.",
            "Shakespeare wrote this famous play.",
            "This is definitely the correct approach.",
            "I've been waiting for two hours.",
            "She went to the grocery store.",
            "The answer is forty-two.",
            "I called them this morning already.",
            "We're almost at our destination now.",
            "Coffee would be nice right now.",
            "The party was really fun last night.",
            "I forgot to mention it earlier.",
            "This item costs twenty dollars.",
            "My favorite color is blue.",
            "The weather is beautiful today.",
            "I agree with your assessment completely.",
            "The project deadline is next Friday.",
        ],
    },
}


def load_model(model_name: str, device: str = "cuda"):
    """Load model with optimal settings."""
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
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


def get_activations_batch(model, tokenizer, texts: list[str], layer: int, device: str) -> torch.Tensor:
    """Get MLP activations for a batch of texts."""
    all_acts = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        acts = get_mlp_activations_at_layer(model, inputs, layer)
        all_acts.append(acts[:, -1, :].cpu())  # Last token position
    return torch.cat(all_acts, dim=0)


def select_features_by_activation_diff(pos_acts: torch.Tensor, neg_acts: torch.Tensor, k: int) -> torch.Tensor:
    """Select top-k features by mean activation difference."""
    diff = (pos_acts.mean(0) - neg_acts.mean(0)).abs()
    _, top_indices = torch.topk(diff, min(k, diff.numel()))
    return top_indices


def select_features_by_relp(
    model, 
    tokenizer,
    pos_texts: list[str],
    neg_texts: list[str],
    layer: int,
    k: int,
    device: str,
) -> torch.Tensor:
    """
    Select top-k features using RelP attribution.
    
    For each positive example, we attribute to the model's predicted next token.
    We then look at which neurons have higher attribution for positive vs negative.
    """
    def get_attribution_for_texts(texts):
        """Get per-neuron attribution scores for a list of texts."""
        all_attr = []
        
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            
            # Get model's predicted next token
            with torch.no_grad():
                outputs = model(**inputs)
                pred_token = outputs.logits[0, -1, :].argmax().item()
            
            # Get attribution to that prediction
            result = get_neuron_attributions(
                model,
                inputs.input_ids,
                target_positions=[-1],
                target_token_ids=[pred_token],
                ablation_mode="zero",
            )
            
            # Get attribution at target layer, last position
            attr = result['attributions'][layer, 0, -1, :]  # [d_mlp]
            all_attr.append(attr.cpu())
        
        return torch.stack(all_attr)  # [n_texts, d_mlp]
    
    # Get attributions for positive and negative examples
    pos_attr = get_attribution_for_texts(pos_texts)
    neg_attr = get_attribution_for_texts(neg_texts)
    
    # Select features with largest difference in attribution
    diff = (pos_attr.mean(0) - neg_attr.mean(0)).abs()
    _, top_indices = torch.topk(diff, min(k, diff.numel()))
    
    return top_indices


def train_and_eval_probe(
    pos_train: torch.Tensor,
    neg_train: torch.Tensor,
    pos_test: torch.Tensor,
    neg_test: torch.Tensor,
    epochs: int = 200,
    lr: float = 0.1,
) -> float:
    """Train logistic regression probe and return test accuracy."""
    # Convert to float32
    pos_train = pos_train.float()
    neg_train = neg_train.float()
    pos_test = pos_test.float()
    neg_test = neg_test.float()
    
    n_train_pos, n_train_neg = pos_train.shape[0], neg_train.shape[0]
    n_test_pos, n_test_neg = pos_test.shape[0], neg_test.shape[0]
    
    # Build datasets
    X_train = torch.cat([pos_train, neg_train], dim=0)
    y_train = torch.tensor([1.0] * n_train_pos + [0.0] * n_train_neg)
    
    X_test = torch.cat([pos_test, neg_test], dim=0)
    y_test = torch.tensor([1.0] * n_test_pos + [0.0] * n_test_neg)
    
    # Normalize using training stats
    X_mean = X_train.mean(0, keepdim=True)
    X_std = X_train.std(0, keepdim=True) + 1e-8
    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std
    
    # Logistic regression
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


def run_robust_eval(
    model,
    tokenizer,
    layers: list[int],
    k_values: list[int],
    device: str,
    n_seeds: int = 3,
    train_ratio: float = 0.5,
    feature_selection: str = "activation_diff",  # or "relp"
) -> dict:
    """
    Run robust evaluation with proper methodology.
    
    Args:
        model: The language model
        tokenizer: Tokenizer
        layers: List of layers to evaluate
        k_values: List of k values for top-k feature selection
        device: Device
        n_seeds: Number of random seeds for variance estimation
        train_ratio: Fraction of data for training
        feature_selection: "activation_diff" or "relp"
    """
    results = {
        "config": {
            "layers": layers,
            "k_values": k_values,
            "n_seeds": n_seeds,
            "train_ratio": train_ratio,
            "feature_selection": feature_selection,
        },
        "by_layer": {},
        "by_concept": {},
    }
    
    # Apply RelP if using RelP feature selection
    relp_state = None
    if feature_selection == "relp":
        relp_state = apply_relp_to_model(model)
    
    try:
        for layer in layers:
            print(f"\n{'='*60}")
            print(f"Layer {layer}")
            print(f"{'='*60}")
            
            layer_results = {}
            
            for concept_name, examples in tqdm(CONCEPTS.items(), desc=f"L{layer} concepts"):
                keys = list(examples.keys())
                positive = examples[keys[0]]
                negative = examples[keys[1]]
                
                concept_results = {k: [] for k in k_values}
                
                for seed in range(n_seeds):
                    # Shuffle data with seed
                    rng = random.Random(seed)
                    pos_shuffled = positive.copy()
                    neg_shuffled = negative.copy()
                    rng.shuffle(pos_shuffled)
                    rng.shuffle(neg_shuffled)
                    
                    # Split
                    n_train_pos = int(len(pos_shuffled) * train_ratio)
                    n_train_neg = int(len(neg_shuffled) * train_ratio)
                    
                    pos_train = pos_shuffled[:n_train_pos]
                    pos_test = pos_shuffled[n_train_pos:]
                    neg_train = neg_shuffled[:n_train_neg]
                    neg_test = neg_shuffled[n_train_neg:]
                    
                    # Get activations
                    pos_train_acts = get_activations_batch(model, tokenizer, pos_train, layer, device)
                    neg_train_acts = get_activations_batch(model, tokenizer, neg_train, layer, device)
                    pos_test_acts = get_activations_batch(model, tokenizer, pos_test, layer, device)
                    neg_test_acts = get_activations_batch(model, tokenizer, neg_test, layer, device)
                    
                    for k in k_values:
                        # Feature selection on TRAINING data only
                        if feature_selection == "activation_diff":
                            top_indices = select_features_by_activation_diff(
                                pos_train_acts, neg_train_acts, k
                            )
                        else:  # relp
                            top_indices = select_features_by_relp(
                                model, tokenizer, pos_train, neg_train, layer, k, device
                            )
                        
                        # Train and evaluate
                        accuracy = train_and_eval_probe(
                            pos_train_acts[:, top_indices],
                            neg_train_acts[:, top_indices],
                            pos_test_acts[:, top_indices],
                            neg_test_acts[:, top_indices],
                        )
                        
                        concept_results[k].append(accuracy)
                
                # Compute mean and std for this concept
                for k in k_values:
                    accs = concept_results[k]
                    mean_acc = np.mean(accs)
                    std_acc = np.std(accs)
                    concept_results[k] = {"mean": mean_acc, "std": std_acc, "raw": accs}
                
                layer_results[concept_name] = concept_results
                
                # Print concept results
                print(f"\n  {concept_name}:")
                for k in k_values:
                    r = concept_results[k]
                    print(f"    k={k}: {r['mean']:.1%} ± {r['std']:.1%}")
            
            results["by_layer"][layer] = layer_results
            
            # Compute layer averages
            print(f"\n  Layer {layer} averages:")
            for k in k_values:
                means = [layer_results[c][k]["mean"] for c in CONCEPTS.keys()]
                avg = np.mean(means)
                print(f"    k={k}: {avg:.1%}")
    
    finally:
        if relp_state is not None:
            revert_relp_from_model(model, relp_state)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Robust evaluation of neurons + RelP")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--layers", type=str, default="16,20,24,28",
                        help="Comma-separated list of layers to evaluate")
    parser.add_argument("--k-values", type=str, default="1,5,10,20",
                        help="Comma-separated list of k values")
    parser.add_argument("--feature-selection", type=str, default="activation_diff",
                        choices=["activation_diff", "relp"],
                        help="Feature selection method")
    parser.add_argument("--n-seeds", type=int, default=3,
                        help="Number of random seeds for variance estimation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="results")
    
    args = parser.parse_args()
    
    layers = [int(l) for l in args.layers.split(",")]
    k_values = [int(k) for k in args.k_values.split(",")]
    
    print("="*70)
    print("ROBUST NEURONS + RelP EVALUATION")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Layers: {layers}")
    print(f"K values: {k_values}")
    print(f"Feature selection: {args.feature_selection}")
    print(f"Random seeds: {args.n_seeds}")
    print(f"Concepts: {list(CONCEPTS.keys())}")
    print(f"Examples per concept: {len(list(CONCEPTS.values())[0][list(list(CONCEPTS.values())[0].keys())[0]])}")
    print()
    
    # Load model
    model, tokenizer = load_model(args.model, args.device)
    
    # Run evaluation
    results = run_robust_eval(
        model=model,
        tokenizer=tokenizer,
        layers=layers,
        k_values=k_values,
        device=args.device,
        n_seeds=args.n_seeds,
        feature_selection=args.feature_selection,
    )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        args.output_dir, 
        f"robust_eval_{args.feature_selection}_{timestamp}.json"
    )
    
    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj
    
    with open(output_path, "w") as f:
        json.dump(convert_for_json(results), f, indent=2)
    
    print(f"\n\nResults saved to {output_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\nAverage accuracy by layer and k:")
    print("-" * 50)
    header = "Layer |" + " | ".join([f"k={k:>2}" for k in k_values])
    print(header)
    print("-" * 50)
    
    for layer in layers:
        row = f"  {layer:>2}  |"
        for k in k_values:
            means = [results["by_layer"][layer][c][k]["mean"] for c in CONCEPTS.keys()]
            avg = np.mean(means)
            row += f" {avg:>5.1%} |"
        print(row)
    
    print("-" * 50)
    print(f"\nRandom baseline: 50%")
    print(f"Feature selection: {args.feature_selection}")


if __name__ == "__main__":
    main()
