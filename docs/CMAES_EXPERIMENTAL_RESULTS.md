# CMA-ES Experimental Results: E88 vs Mamba2 vs Transformers

**Date**: 2026-01-31
**Source**: ~/elman CMA-ES hyperparameter evolution experiments

## Executive Summary

Comprehensive CMA-ES hyperparameter search was conducted across multiple architectures at 480M parameters. The experiments provide empirical validation of the expressivity hierarchy, with Mamba2 (linear SSM) outperforming E88 (nonlinear Elman), which outperforms Transformers on this language modeling task.

**Key Finding**: The empirical results *contradict* the theoretical expressivity hierarchy (E88 ⊃ Mamba2). This suggests that while E88 has greater theoretical computational power, practical factors (optimization landscape, training dynamics, parallel efficiency) favor Mamba2 at current scales.

---

## 1. CMA-ES Search Configuration

All experiments used:
- **Target parameters**: 480M ± 50M
- **Training time per config**: 10 minutes
- **Learning rate**: Fixed at 3e-4 (fair comparison)
- **Optimizer**: ScheduleFree AdamW
- **Data**: The Pile (byte-level tokenization)
- **Hardware**: Multi-GPU (4x RTX 6000 Ada)
- **Evaluations per model**: ~120 configurations (15 generations × 8 population)

---

## 2. Main Results: Best Loss by Architecture

| Architecture | Best Loss | Best Configuration | Params | Status |
|--------------|-----------|-------------------|--------|--------|
| **Mamba2** | **1.271** | d_state=96, expand=2, depth=25 | 494M | ✅ Best |
| **FLA-GDN** | 1.273 | expansion=2, depth=17, n_heads=24 | ~480M | ✅ |
| **E88** | 1.407 | n_heads=68, n_state=16, depth=23 | 488M | ✅ |
| Transformer | 1.505 | n_heads=8, expansion=4, depth=13 | 491M | ✅ |
| MinGRU | 1.528 | expansion=1, depth=14 | ~480M | ✅ |
| MinLSTM | 1.561 | expansion=1, depth=31 | ~480M | ✅ |
| MoM-E88 | 1.762 | n_heads=40, top_k=8, n_state=64, depth=12 | ~480M | ✅ |
| E90 (Dual-Rate) | 1.791 | n_heads=114, config=(8,16), depth=13 | ~500M | ✅ |
| GRU (CUDA) | 10.0 | - | ~480M | ❌ Diverged |
| LSTM (CUDA) | - | - | ~480M | ❌ Not tested |

---

## 3. Architecture-Specific Findings

### 3.1 E88: Nonlinear Multi-Head Elman

**Optimal Configuration**:
```python
E88(
    n_heads=68,      # Many small heads
    n_state=16,      # Smallest stable state size
    depth=23,        # Moderate depth
    dim=3840,        # Large model dimension
    expansion=1.0,   # Square state matrices
    use_gate=True,
    gate_activation='silu'
)
```

**Search Space Insights**:
- n_state=16 consistently outperformed 32, 48, 64 (smaller states, more heads)
- Optimal depth: 20-25 layers
- CUDA fused kernel only supports n_state ∈ {16, 32, 48, 64}

**Ablation Findings** (from E88_ABLATION_NOTES.md):
- Removing output RMSNorm: -0.10 nats improvement
- Removing convolutions: -0.03 improvement
- Removing output gating: -0.01 improvement
- Linear state ≈ tanh state (no difference!)
- SiLU and L2 normalization are critical for stability

### 3.2 Mamba2: Linear State Space Model

**Optimal Configuration**:
```python
Mamba2(
    d_state=96,      # Moderate state dimension
    expand=2,        # Standard expansion
    depth=25         # Deep
)
```

**Why Mamba2 Wins Despite Linear State**:
1. **Parallel scan efficiency**: O(log n) recurrence computation
2. **Input-dependent selectivity**: B, C, Δ functions of input
3. **Numerical stability**: Log-space updates prevent overflow
4. **Higher throughput**: More tokens processed per training minute

### 3.3 Transformer (LLaMA-style)

**Optimal Configuration**:
```python
Transformer(
    n_heads=8,       # Fewer heads with larger head_dim
    expansion=4,     # Standard FFN expansion
    depth=13         # Shallow but wide
)
```

**Performance Gap**: Transformers underperformed RNN/SSM variants at 480M scale. This may be due to:
- Quadratic attention complexity limiting context length
- Shorter training time favoring recurrent inductive biases
- Byte-level tokenization (256 vocab) favoring local patterns

---

## 4. Expressivity Hierarchy: Theory vs Practice

### Theoretical Hierarchy (from Lean proofs)
```
E88 (nonlinear Elman) ⊃ Mamba2 (linear SSM) ⊃ Linear attention
```

**Key Theorems**:
- Linear RNNs cannot compute XOR (LinearLimitations.lean)
- Linear RNNs cannot compute Threshold (LinearLimitations.lean)
- E88's tanh enables nonlinear computation (MultiLayerLimitations.lean)

### Empirical Results (This Benchmark)
```
Mamba2 (1.271) > FLA-GDN (1.273) > E88 (1.407) > Transformer (1.505)
```

### Reconciling the Gap

The discrepancy between theoretical expressivity and practical performance suggests:

1. **Optimization matters**: Mamba2's parallel scan may create a smoother optimization landscape
2. **Training efficiency**: Mamba2 processes more tokens per second (parallel vs sequential)
3. **Inductive bias**: Linear dynamics may be sufficient for language modeling at current scales
4. **Hyperparameter sensitivity**: E88 may require more careful tuning

---

## 5. Running Parity Experiments

**Status**: No dedicated parity task experiments found in ~/elman.

The codebase contains theoretical analysis of parity:
- `ElmanProofs/Expressivity/LinearLimitations.lean`: Proves linear RNNs cannot compute XOR
- Multi-layer extension shows XOR requires nonlinearity

**Recommended Follow-up**: Create synthetic parity benchmark to validate:
1. E88 can learn parity(x_1, ..., x_t) exactly
2. Mamba2 cannot (linear dynamics)
3. Performance gap increases with sequence length

---

## 6. E75/E87 Benchmark Comparison

From E75_E87_MULTIHEAD_BENCHMARK.md (100M scale, 10 min training):

| Architecture | Best Loss | Notes |
|--------------|-----------|-------|
| Mamba2 | 1.21 | Best overall |
| E75 (Multi-Head Delta) | 1.42 | 4 heads, n_state=32 |
| FLA-GDN | 1.57 | ICLR 2025 baseline |
| E87 (Sparse Block) | 1.67 | 16 blocks, top-4 |

**Key Finding**: Multi-head variants (E75, E88) significantly outperform sparse routing (E87, MoM-E88).

---

## 7. Key Experimental Artifacts

### CMA-ES Result Files
```
~/elman/benchmark_results/cmaes_e88_10min/results.json
~/elman/benchmark_results/cmaes_mamba2_10min/results.json
~/elman/benchmark_results/cmaes_transformer_10min/results.json
~/elman/benchmark_results/cmaes_fla-gdn_10min/results.json
~/elman/benchmark_results/cmaes_mingru_10min/results.json
~/elman/benchmark_results/cmaes_minlstm_10min/results.json
~/elman/benchmark_results/cmaes_search/mom-e88_480M_*/results.json
~/elman/benchmark_results/cmaes_e90/results.json
```

### Benchmark Documentation
```
~/elman-proofs/E75_E87_MULTIHEAD_BENCHMARK.md
~/elman/E88_ABLATION_NOTES.md
~/elman/E5_EXPERIMENT_REPORT.md
```

### CMA-ES Search Script
```
~/elman/cmaes_search.py  # Full CMA-ES hyperparameter evolution
```

---

## 8. Conclusions for Expressivity Research

1. **Mamba2 beats E88 empirically** despite E88's theoretical advantage
2. **Parallel efficiency dominates** at current training scales
3. **Sparse routing underperforms** dense multi-head variants
4. **Parity experiments needed** to validate theoretical expressivity claims
5. **n_state=16 optimal for E88** - many small heads > few large heads

### Implications for Lean Proofs

The theoretical expressivity hierarchy (E88 ⊃ Mamba2) is mathematically valid but:
- Does not account for training dynamics
- Does not account for optimization landscape
- Does not account for throughput differences

A complete theory should explain why theoretical advantages don't manifest in practice.

---

## Appendix: Search Space Definitions

```python
SEARCH_SPACES = {
    'e88': {
        'n_heads': (32, 160, 'int'),
        'n_state': (16, 64, 'e88_n_state'),  # Only {16,32,48,64} supported
        'depth': (12, 40, 'int'),
    },
    'mamba2': {
        'd_state': (64, 256, 'int_mult16'),
        'expand': (1, 3, 'int'),
        'depth': (16, 40, 'int'),
    },
    'transformer': {
        'n_heads': (8, 32, 'int'),
        'expansion': (2, 6, 'int'),
        'depth': (12, 36, 'int'),
    },
    # ... (see cmaes_search.py for full definitions)
}
```
