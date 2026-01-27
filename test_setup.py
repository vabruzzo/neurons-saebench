#!/usr/bin/env python
"""
Quick test to verify the setup works before running on GPU.

Run with: python test_setup.py

For CPU-only testing (no Llama 3.1 8B needed):
    python test_setup.py --cpu-only

For full GPU test:
    python test_setup.py --full
"""

import argparse
import sys


def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"  ✓ torch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ torch: {e}")
        return False
    
    try:
        import transformers
        print(f"  ✓ transformers {transformers.__version__}")
    except ImportError as e:
        print(f"  ✗ transformers: {e}")
        return False
    
    try:
        from neurons_bench import relp
        print("  ✓ neurons_bench.relp")
    except ImportError as e:
        print(f"  ✗ neurons_bench.relp: {e}")
        return False
    
    try:
        from neurons_bench import neuron_selector
        print("  ✓ neurons_bench.neuron_selector")
    except ImportError as e:
        print(f"  ✗ neurons_bench.neuron_selector: {e}")
        return False
    
    try:
        from neurons_bench import saebench_adapter
        print("  ✓ neurons_bench.saebench_adapter")
    except ImportError as e:
        print(f"  ✗ neurons_bench.saebench_adapter: {e}")
        return False
    
    return True


def test_relp_module():
    """Test the RelP module with a simple example."""
    print("\nTesting RelP module...")
    
    import torch
    from neurons_bench.relp import (
        StraightThroughRMSNorm,
        ShapleyElementwiseMult,
        RelPGatedMLP,
    )
    
    # Test ShapleyElementwiseMult
    x = torch.randn(2, 4, requires_grad=True)
    y = torch.randn(2, 4, requires_grad=True)
    
    z = ShapleyElementwiseMult.apply(x, y)
    loss = z.sum()
    loss.backward()
    
    # Check that gradients exist and follow half rule
    expected_grad_x = 0.5 * y.detach()
    expected_grad_y = 0.5 * x.detach()
    
    assert torch.allclose(x.grad, expected_grad_x, atol=1e-5), "Half rule not applied correctly to x"
    assert torch.allclose(y.grad, expected_grad_y, atol=1e-5), "Half rule not applied correctly to y"
    
    print("  ✓ ShapleyElementwiseMult (half rule verified)")
    
    return True


def test_gpu_availability():
    """Check GPU availability and settings."""
    print("\nChecking GPU...")
    
    import torch
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✓ CUDA available: {device_name}")
        print(f"    Memory: {memory_gb:.1f} GB")
        
        # Check for bfloat16 support
        if torch.cuda.is_bf16_supported():
            print("  ✓ bfloat16 supported")
        else:
            print("  ⚠ bfloat16 NOT supported (will use float16)")
        
        # Check for Flash Attention
        try:
            import flash_attn
            print(f"  ✓ Flash Attention available")
        except ImportError:
            print("  ⚠ Flash Attention not installed (will use eager attention)")
        
        return True
    else:
        print("  ⚠ CUDA not available - will use CPU")
        return False


def test_model_loading(model_name: str = "meta-llama/Llama-3.1-8B-Instruct", device: str = "cuda"):
    """Test model loading (requires GPU and ~16GB VRAM)."""
    print(f"\nTesting model loading: {model_name}")
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    
    print(f"  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("  ✓ Tokenizer loaded")
    
    print(f"  Loading model (dtype={dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        attn_implementation="flash_attention_2" if device == "cuda" else "eager",
    )
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  ✓ Model loaded: {n_params:.1f}B parameters")
    
    return model, tokenizer


def test_relp_on_model(model, tokenizer, device: str = "cuda"):
    """Test RelP attribution on the loaded model."""
    print("\nTesting RelP attribution...")
    
    import torch
    from neurons_bench.relp import apply_relp_to_model, revert_relp_from_model, get_neuron_attributions
    
    prompt = "The capital of France is"
    target_token = " Paris"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    target_id = tokenizer.encode(target_token, add_special_tokens=False)[0]
    
    print(f"  Prompt: '{prompt}'")
    print(f"  Target: '{target_token}' (id={target_id})")
    
    # Apply RelP
    print("  Applying RelP modifications...")
    relp_state = apply_relp_to_model(model)
    
    try:
        print("  Computing attributions...")
        result = get_neuron_attributions(
            model,
            inputs.input_ids,
            target_positions=[-1],
            target_token_ids=[target_id],
        )
        
        attributions = result['attributions']
        print(f"  ✓ Attribution shape: {attributions.shape}")
        
        # Find top neuron
        flat = attributions.abs().flatten()
        max_idx = flat.argmax().item()
        max_val = flat[max_idx].item()
        
        n_layers = attributions.shape[0]
        seq_len = attributions.shape[2]
        d_mlp = attributions.shape[3]
        
        layer = max_idx // (seq_len * d_mlp)
        remainder = max_idx % (seq_len * d_mlp)
        pos = remainder // d_mlp
        neuron = remainder % d_mlp
        
        print(f"  Top neuron: Layer {layer}, Position {pos}, Neuron {neuron}")
        print(f"  Attribution score: {max_val:.4f}")
        print(f"  URL: https://neurons.transluce.org/{layer}/{neuron}/+")
        
    finally:
        print("  Reverting RelP modifications...")
        revert_relp_from_model(model, relp_state)
    
    print("  ✓ RelP attribution test passed!")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true", help="Skip GPU tests")
    parser.add_argument("--full", action="store_true", help="Run full test with model loading")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    args = parser.parse_args()
    
    print("="*60)
    print("NEURONS_BENCH SETUP TEST")
    print("="*60)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed!")
        sys.exit(1)
    
    # Test RelP module
    if not test_relp_module():
        print("\n❌ RelP module test failed!")
        sys.exit(1)
    
    # Test GPU availability
    if not args.cpu_only:
        has_gpu = test_gpu_availability()
    else:
        has_gpu = False
        print("\nSkipping GPU tests (--cpu-only)")
    
    # Full test with model
    if args.full and has_gpu:
        device = "cuda"
        model, tokenizer = test_model_loading(args.model, device)
        test_relp_on_model(model, tokenizer, device)
    elif args.full and not has_gpu:
        print("\n⚠ Skipping model test (no GPU)")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    
    if has_gpu and not args.full:
        print("\nRun with --full to test model loading and RelP attribution")
    
    print("\nNext steps:")
    print("  1. Run demo: python -m neurons_bench.run_benchmark --demo-only")
    print("  2. Run comparison: python -m neurons_bench.compare_neurons_vs_saes")


if __name__ == "__main__":
    main()
