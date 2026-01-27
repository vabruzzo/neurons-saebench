# Neurons + RelP vs SAEs on SAEBench

This project tests whether neuron-based circuit tracing with RelP attribution can match or exceed sparse autoencoders (SAEs) on interpretability benchmarks.

## Key Hypothesis

Recent work from [Transluce](https://transluce.org/neuron-circuits) shows that MLP neurons + RelP achieve comparable faithfulness to learned features for circuit discovery. This project evaluates whether this translates to better performance on SAEBench's downstream metrics.

## Quick Start

### 0. Download SAEBench Baselines (Optional)

```bash
# Download published SAE results for comparison
python scripts/download_saebench_baselines.py

# This populates saebench_baselines.json with actual numbers from:
# - Pythia-70M, Pythia-160M
# - Gemma-2-2B
# - And other models if available
```

### 1. Cloud GPU Setup (Recommended: A100 40GB+)

```bash
# Clone and enter the repo
git clone <repo-url>
cd sae-bench-neurons

# Create environment with uv (fastest)
pip install uv
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .

# Or with pip
pip install -e .

# For SAEBench integration (optional)
pip install sae-bench
```

### 2. Run the Comparison

```bash
# Quick demo: See RelP attribution in action
python -m neurons_bench.run_benchmark --demo-only --model meta-llama/Llama-3.1-8B-Instruct

# Full comparison: Neurons vs SAEs
python -m neurons_bench.compare_neurons_vs_saes --model meta-llama/Llama-3.1-8B-Instruct --layer 16

# Run sparse probing evaluation
python -m neurons_bench.run_benchmark --model meta-llama/Llama-3.1-8B-Instruct --layer 16
```

### 3. Optimal GPU Settings

For best performance on cloud GPUs:

| Setting | Value | Why |
|---------|-------|-----|
| dtype | bfloat16 | Memory efficient, native support |
| Attention | flash_attention_2 | 2-4x faster, less memory |
| Batch size | 8-16 | Maximize throughput |
| Device map | auto | Efficient GPU utilization |

The code automatically uses these settings when CUDA is available.

## Project Structure

```
neurons_bench/
├── __init__.py           # Package exports
├── relp.py               # RelP implementation (from Transluce)
├── neuron_selector.py    # Select neurons using RelP attribution
├── saebench_adapter.py   # Make neurons work like SAEs
├── neuron_descriptions.py # Fetch from neurons.transluce.org
├── run_benchmark.py      # Main benchmark runner
└── compare_neurons_vs_saes.py  # Head-to-head comparison
```

## What This Measures

### 1. Attribution Sparsity
Does RelP produce sparse attributions? We measure what fraction of total attribution is concentrated in the top-k neurons.

### 2. Concept Detection (Sparse Probing)
Can k neurons separate positive/negative examples for a concept?
- Sentiment (positive vs negative)
- Formality (formal vs informal)
- Factual vs opinion

### 3. Qualitative Interpretability
Do the top neurons have meaningful descriptions? We link to [neurons.transluce.org](https://neurons.transluce.org) for human-readable explanations.

## Key Files

### `relp.py`
Core RelP implementation:
- `apply_relp_to_model()` - Modify model for RelP gradients
- `get_neuron_attributions()` - Single-pass attribution to all neurons
- `get_top_neurons()` - Select most important neurons

### `compare_neurons_vs_saes.py`
Head-to-head comparison:
- Same tasks evaluated on neurons and SAEs
- Outputs JSON with detailed results
- Prints summary statistics

## Results

Results are saved to `results/` directory:
- `comparison_results.json` - Full comparison data
- `sparse_probing_neurons_layer{N}.json` - Per-layer probing results

## Background

### Why This Matters

SAEs are expensive to train and maintain. If raw neurons + better attribution methods work comparably:
1. No need for auxiliary learned features
2. Direct interpretability of the model as-is
3. Applicable to any model immediately

### The RelP Method

RelP (Relevance Propagation) modifies gradient computation:
1. **Linearize nonlinearities**: Treat sigmoid as constant, RMSNorm as frozen
2. **Half rule**: Split credit equally at bilinear operations (gate × up_proj)
3. **Single backward pass**: Much faster than Integrated Gradients

This gives us faithful attribution without the noise/cost of IG.

## Citation

If you use this code, please cite:

```bibtex
@misc{arora2025language,
  author = {Arora, Aryaman and Wu, Zhengxuan and Steinhardt, Jacob and Schwettmann, Sarah},
  title = {Language Model Circuits are Sparse in the Neuron Basis},
  year = {2025},
  howpublished = {\url{https://transluce.org/neuron-circuits}}
}
```

## License

MIT
