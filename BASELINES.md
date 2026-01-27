# SAEBench Baselines Reference

**Source**: SAEBench paper (arXiv:2503.09532) and `adamkarvonen/sae_bench_results_0125`

## Sparse Probing (Concept Detection)

**Task**: Can k features/neurons classify concepts (e.g., profession, sentiment)?

### Gemma-2-2B, Layer 12 (Primary Reference)

| k | LLM Baseline | SAE Best | SAE Improvement |
|---|--------------|----------|-----------------|
| 1 | 65.7% | 74.0% | **+8.3%** |
| 2 | 72.3% | 76.9% | +4.6% |
| 5 | 78.0% | 84.8% | +6.8% |
| 10 | 83.0% | — | — |
| 20 | 87.8% | — | — |
| 50 | 92.1% | — | — |

*Paper quote: "all SAEs significantly outperform a baseline of probing directly on K residual stream channels (0.65 on Layer 12 of Gemma-2-2B)"*

### Pythia-70M, Layer 4 (Smaller Model Reference)

| k | LLM Baseline | SAE Best | Gap |
|---|--------------|----------|-----|
| 1 | 76.0% | 77.4% | +1.4% |
| 5 | 85.3% | 87.9% | +2.6% |
| 10 | 89.0% | 89.9% | +0.9% |
| 20 | 91.1% | 91.3% | +0.2% |

**Key insight**: SAE improvement is larger for bigger models and at lower k.

---

## SCR (Spurious Correlation Removal)

**Task**: Can ablating k features remove unwanted correlations (e.g., gender→profession)?

| Model | SAE Metric (n=10) |
|-------|-------------------|
| Pythia-70M | 0.757 |

**Higher is better**: Measures spurious correlation removed.

---

## TPP (Targeted Probe Perturbation)

**Task**: Are concepts encoded by distinct feature groups?

| Model | Total Metric | Intended Δ | Unintended Δ |
|-------|--------------|------------|--------------|
| Pythia-70M | 0.147 | 0.160 | 0.013 |

**Goal**: High intended diff, low unintended diff = good disentanglement.

---

## Success Criteria for Neurons + RelP

From Neel: *"Just seeing if it can do anything non-trivial at all would be a victory."*

### Quantitative Targets (Sparse Probing)

| Level | k=1 Target | k=5 Target | Interpretation |
|-------|------------|------------|----------------|
| **Baseline** | >65.7% | >78.0% | Matches LLM residual stream |
| **Good** | >70% | >82% | Competitive with SAEs |
| **Great** | >74% | >85% | Matches best SAE |

### Qualitative Targets

1. ✅ **Sparse attribution**: Top-20 neurons account for >30% of total attribution
2. ✅ **Interpretable neurons**: Top neurons have meaningful descriptions at neurons.transluce.org
3. ✅ **Consistent patterns**: Same neurons appear across similar examples

---

## Data Sources

- **Paper**: [SAEBench: A Comprehensive Benchmark](https://arxiv.org/abs/2503.09532)
- **Results**: `huggingface.co/datasets/adamkarvonen/sae_bench_results_0125`
- **Test data**: `SAEBench/tests/acceptance/test_data/`
- **Neuronpedia**: [neuronpedia.org/sae-bench](https://neuronpedia.org/sae-bench)
