# Research Note: Batch Size and Learning Dynamics in Sequential RNNs

## Context for the Formal Theory Agent

This note comes from empirical observations in the Elman RNN project (~/elman). We've discovered something potentially important about how batch size affects learning quality in recurrent neural networks, and we'd like formal/theoretical investigation.

## Empirical Findings

We ran a large-scale CMA-ES architecture search (400+ evaluations per model) at ~480M parameters, training on sequential text data (commapile.txt, byte-level, vocab_size=256). Each evaluation trains for 20 minutes on a 48GB GPU. The memory probing system automatically finds the maximum batch size that fits in GPU memory for each config.

### The batch size effect

The CMA-ES fitness metric is **average loss over ALL training steps** in the 20-minute phase. Results, stratified by batch size:

**E88 (n_state=32, matrix state RNN):**
| Batch Size | n configs | Mean Loss | Best Loss | Median Loss |
|-----------|-----------|-----------|-----------|-------------|
| bs=1 | 63 | 1.019 | 0.940 | 0.998 |
| bs=2-4 | 95 | 1.079 | 0.974 | 1.050 |
| bs=5-16 | 118 | 1.192 | 1.054 | 1.165 |
| bs=17+ | 28 | 1.314 | 1.174 | 1.295 |

**E88 (n_state=16, matrix state RNN):**
| Batch Size | n configs | Mean Loss | Best Loss | Median Loss |
|-----------|-----------|-----------|-----------|-------------|
| bs=1 | 59 | 0.988 | 0.929 | 0.982 |
| bs=2-4 | 88 | 1.034 | 0.961 | 1.029 |
| bs=5-16 | 115 | 1.111 | 1.022 | 1.107 |
| bs=17+ | 42 | 1.257 | 1.156 | 1.238 |

**Critical: average parameter count is ~480M across ALL batch size buckets.** The effect is NOT due to larger models fitting at bs=1.

### Why bs=1 wins

In a fixed 20-minute window:
- bs=1 at chunk_size=512: ~3000 gradient steps, 1.5M tokens seen
- bs=16 at chunk_size=512: ~500 gradient steps, 4M tokens seen

bs=1 sees **fewer tokens** but does **6x more gradient updates**. And it wins decisively.

### The coherency hypothesis

With bs=1 and memory-mapped sequential data, the model reads consecutive 512-byte windows from the dataset. It's essentially **online learning on a stream** — seeing coherent, locally related content across consecutive gradient updates.

With bs=16, each step processes 16 random positions from the dataset simultaneously. The gradient is an average of 16 unrelated signals. While each individual gradient is "smoother," the learning trajectory is less coherent.

This raises the question: **does gradient coherency (consecutive related updates) matter more than gradient smoothness (averaging over diverse samples)?**

## Questions for Formal Investigation

### 1. Gradient update interaction in batched SGD

When we batch B samples and average their gradients, what exactly happens to the learning dynamics? For a recurrent network with state update `h_t = f(W_h, h_{t-1}, x_t)`:

- Each batch element produces gradients through its own 512-step unroll
- These gradients are averaged before the weight update
- The averaged gradient may cancel out specialized directions that individual samples would push toward

**Question:** Can we formally characterize when averaging gradients helps vs. hurts, specifically for recurrent architectures?

### 2. Online learning on coherent streams vs. shuffled data

Classical SGD theory assumes i.i.d. samples. But our bs=1 sequential setup violates this — consecutive windows come from the same document/topic. This means:

- Consecutive gradient updates push in similar directions (coherent signal)
- The model can "focus" on a topic for multiple steps before context-switching
- This is closer to **online learning on a non-stationary stream**

**Question:** Is there a formal framework for understanding why correlated sequential updates might outperform i.i.d. shuffled updates in certain regimes? Curriculum learning theory? Information-theoretic arguments?

### 3. The "dream consolidation" analogy (and its limits)

One could argue that large-batch training is like "experiencing a whole day at once" — diverse inputs consolidated into one update. But our results suggest the opposite: **frequent small coherent updates** beat **infrequent large diverse updates**.

**Question:** Does this connect to any known results in online learning theory, particularly for sequence models?

### 4. Implications for the expressivity results

Our Lean proofs establish expressivity properties of Elman RNNs (UTM simulation, etc.). But expressivity is about what the model CAN represent, not how efficiently it learns.

**Question:** Can we say anything formal about how batch size affects the **rate of convergence** for nonlinear recurrent models specifically? The sequential structure (BPTT through time) means each sample already provides a rich gradient signal — does this reduce the benefit of batching?

### 5. Fixed compute budget implications

In a fixed wall-time budget:
- More steps with smaller batches = more frequent weight updates
- Fewer steps with larger batches = smoother but rarer updates

The optimal tradeoff depends on:
- Noise in the gradient (signal-to-noise ratio)
- Curvature of the loss landscape
- Whether the model is in early training (far from optimum) vs. fine-tuning

**Question:** For the early-training regime (20 minutes, far from convergence), is there a theoretical prediction for the optimal batch size? Our empirical finding says bs=1 — is this expected?

## Raw Data Location

- E88 n32 results: `~/elman/benchmark_results/cmaes_multicontext/e88_n32_512/`
- E88 n16 results: `~/elman/benchmark_results/cmaes_multicontext/e88_n16_512/`
- Each eval has a `.done` JSON file with full config and loss metrics
- CMA-ES search script: `~/elman/cmaes_search_v2.py`

## Summary

The conventional wisdom that "larger batches smooth gradients and improve training" does not hold in our setting. For sequential RNNs trained for a fixed wall-time budget, bs=1 with coherent sequential data consistently outperforms larger batch sizes. We want to understand WHY — both empirically and theoretically. Is this a deep property of recurrent learning, or an artifact of short training runs?
