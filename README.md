# Neurons + RelP Attribution Analysis

Can raw MLP neurons provide non-trivial interpretability insights via gradient-based attribution? This project uses Transluce's RelP method to attribute Llama 3.1 8B outputs to individual neurons, then validates the results against independent neuron descriptions at [neurons.transluce.org](https://neurons.transluce.org).

## Background

[Transluce](https://transluce.org/neuron-circuits) showed that MLP neurons + RelP achieve comparable faithfulness to learned SAE features for circuit discovery. This project tests whether RelP attribution surfaces interpretable, task-specific neurons on novel prompts not present in Transluce's evaluation.

## Key Finding

RelP attribution on novel factual recall prompts independently identifies neurons whose automated descriptions (from neurons.transluce.org) match the expected output. For example:

| Prompt | Top Neuron | Transluce Description |
|--------|-----------|----------------------|
| "The author of Romeo and Juliet is" | L23/N13724 | "Shakespeare or elements related to Shakespeare" |
| "The largest planet in our solar system is" | L21/N4954 | "planets or astrological elements... Saturn" |
| "The color of grass is" | L31/N6411 | "Green Realty Corp" (green-related) |
| "2 + 2 =" | L31/N11514 | "numerical tokens... statistics, percentages, or counts" |

These prompts were designed by us, not from Transluce's paper, making this an independent validation.

## Known Bias Neurons

RelP attribution in Llama 3.1 8B is dominated by "always-on" neurons (e.g., L0/N491, L1/N2427) that appear in every attribution regardless of task. This is expected behavior documented by Transluce, who manually filter these out. Our code filters them automatically.

## Setup

```bash
pip install uv
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

Requires a GPU (A100 40GB+ recommended). Uses eager attention (required for RelP hooks).

## Usage

```bash
# Run attribution analysis (filters bias neurons by default)
python -m neurons_bench.run_attribution \
  --model NousResearch/Meta-Llama-3.1-8B-Instruct

# Show all neurons including bias neurons
python -m neurons_bench.run_attribution --no-filter

# Adjust number of top neurons reported
python -m neurons_bench.run_attribution --k 20
```

## Project Structure

```
neurons_bench/
  __init__.py              # Package exports
  relp.py                  # RelP wrapper (thin layer over Transluce's code)
  run_attribution.py       # Main script: attribution on factual recall prompts
  transluce/
    __init__.py
    grad.py                # Transluce's gradient handling (from github.com/TransluceAI/circuits)
```

## How RelP Works

RelP modifies gradient computation to produce faithful attributions in a single backward pass:

1. **Linearize nonlinearities**: RMSNorm uses straight-through, sigmoid is detached
2. **Half rule**: Credit is split equally at bilinear operations (gate x up_proj)
3. **Attention**: QK gradients blocked, flow goes through OV circuit only
4. **Attribution**: activation x gradient gives per-neuron contribution to target logit

## Citation

```bibtex
@misc{arora2025language,
  author = {Arora, Aryaman and Wu, Zhengxuan and Steinhardt, Jacob and Schwettmann, Sarah},
  title = {Language Model Circuits are Sparse in the Neuron Basis},
  year = {2025},
  howpublished = {\url{https://transluce.org/neuron-circuits}}
}
```
