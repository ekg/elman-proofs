# Residual Recurrence in RNNs: A Comprehensive Research Report

## Executive Summary

Residual recurrence refers to the application of residual (skip) connections within recurrent neural networks, particularly in the temporal dimension. This comprehensive report surveys the academic literature, identifies key researchers and papers, analyzes theoretical findings, examines empirical results, and positions the E59 work within the broader research landscape.

**Key finding**: Residual recurrence has been extensively studied since the mid-2010s. The approach demonstrably improves gradient flow and training stability while preserving or enhancing expressivity. Modern architectures like RWKV, xLSTM, Mamba, and Griffin all leverage residual principles in their designs.

---

## Table of Contents

1. [History and Key Papers](#1-history-and-key-papers)
2. [Theoretical Foundations](#2-theoretical-foundations)
3. [Expressivity and Computational Power](#3-expressivity-and-computational-power)
4. [Tradeoffs: Memory and Forgetting](#4-tradeoffs-memory-and-forgetting)
5. [Language Modeling Applications](#5-language-modeling-applications)
6. [Scaling Studies](#6-scaling-studies)
7. [Combinations with Attention and SSMs](#7-combinations-with-attention-and-ssms)
8. [Gaps in Existing Research](#8-gaps-in-existing-research)
9. [How E59 Relates to Prior Work](#9-how-e59-relates-to-prior-work)
10. [Recommendations](#10-recommendations)
11. [Sources](#11-sources)

---

## 1. History and Key Papers

### 1.1 Early Foundations (1990s-2010s)

The conceptual roots of residual recurrence trace back to several key developments:

**Elman Networks (1990)**: Jeffrey Elman introduced the simple recurrent network (SRN), establishing the basic recurrence `h_t = f(W_h h_{t-1} + W_x x_t)`. While lacking explicit residual connections, this architecture highlighted the fundamental challenge of gradient flow through time.

**LSTM (1997)**: Hochreiter and Schmidhuber's Long Short-Term Memory introduced the cell state mechanism, which functions as a residual connection. The update `c_t = f_t * c_{t-1} + i_t * candidate` creates a path where gradients can flow unchanged when `f_t = 1`. This insight is crucial: **LSTM's cell state IS a form of residual connection**.

**Forget Gate Addition (1999)**: Gers, Schmidhuber, and Cummins introduced the forget gate, enabling LSTMs to reset their state. The formulation `c_{t+1} = f_t * c_t + i_t * g_t` explicitly contains a skip connection when `f_t` approaches 1.

### 1.2 The ResNet Revolution (2015-2016)

**Highway Networks (2015)**: Srivastava, Greff, and Schmidhuber applied LSTM principles to feedforward networks, creating Highway Networks. This demonstrated that residual/gated connections could enable training of networks with hundreds of layers.

**Deep Residual Learning (2015)**: He et al. introduced ResNet with the simple residual connection `y = F(x) + x`. This architecture won ImageNet and proved that identity mappings dramatically improve gradient flow.

**Identity Mappings in Deep Residual Networks (2016)**: He et al. showed that pure identity shortcuts outperform all variants. Key insight: the forward signal `x_L = x_l + sum_{i=l}^{L-1} F(x_i)` and backward gradient both propagate directly through identity connections.

**Bridging Residual Learning and RNNs (2016)**: [arXiv:1604.03640](https://arxiv.org/pdf/1604.03640) demonstrated formal equivalence between deep residual networks and specific shallow RNNs, establishing a theoretical bridge between the two paradigms.

### 1.3 Recurrent Highway Networks (2016-2017)

**Recurrent Highway Networks (ICML 2017)**: [Zilly et al.](https://arxiv.org/abs/1607.03474) extended highway networks to RNNs, enabling step-to-step transition depths larger than one. Key result: on Penn Treebank, increasing transition depth from 1 to 10 improved word-level perplexity from 90.6 to 65.4 using the same number of parameters.

**Highway-LSTM for Speech Recognition**: [Google Research](https://research.google/pubs/pub46171/) showed that 10-layer Highway-LSTM models outperform 5-layer standard LSTM models with the same parameters by 2% relative WER.

**Highway State Gating**: [Springer](https://link.springer.com/chapter/10.1007/978-3-319-94147-9_10) introduced a gating mechanism for state that allows the network to "choose" whether to pass information directly through time.

### 1.4 Independently Recurrent Neural Networks (2018)

**IndRNN**: [Li et al., CVPR 2018](https://arxiv.org/abs/1803.04831) proposed neurons that are independent within a layer but connected across layers. Key achievements:
- Process sequences over 5000 timesteps
- Train networks up to 21 layers deep
- Over 10x faster than LSTM
- Works with non-saturating activations (ReLU)

This architecture represents a form of residual thinking: by making within-layer neurons independent, gradient flow becomes simpler and more controllable.

### 1.5 Residual RNN Variants (2018-2025)

**Residual Recurrent Neural Networks for Learning Sequential Representations (2018)**: [MDPI](https://www.mdpi.com/2078-2489/9/3/56) directly reformulated the RNN unit to learn residual functions with reference to the hidden state:
- Solves gradient vanishing/exploding for large time scales
- Promotes backward update optimization

**Residual Stacked RNNs for Action Recognition (ECCV 2018)**: Introduced `x^l_t = m^l_t + x^{l-1}_t`, combining LSTM outputs with residual connections for spatiotemporal feature extraction.

**Residual LSTM for Distant Speech Recognition (2017)**: [arXiv:1701.03360](https://arxiv.org/pdf/1701.03360) designed deep recurrent architectures with residual connections for challenging acoustic conditions.

**Residual Echo State Networks (2024)**: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0925231224007379) extended Echo State Networks with temporal residual connections, achieving stable dynamics and fast learning in the reservoir computing framework.

**Deep Residual Echo State Networks (2025)**: [arXiv:2508.21172](https://arxiv.org/html/2508.21172) introduced hierarchical architectures with residual orthogonal connections for untrained RNNs.

**Residual-Time Gated Recurrent Unit (2025)**: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231225000682) introduced residual information into GRU's candidate hidden state in the backpropagation direction.

---

## 2. Theoretical Foundations

### 2.1 Gradient Flow Analysis

The fundamental theoretical insight underlying residual recurrence concerns gradient flow through time:

**Standard RNN Gradient**:
```
h_t = W @ f(h_{t-1}, x_t)
dh_T/dh_0 = prod_{t=1}^{T} dh_t/dh_{t-1}
```
For tanh activation, each `dh_t/dh_{t-1} = W * diag(1 - tanh^2(z_t))`, leading to:
- **Vanishing**: If spectral_radius(W) < 1, gradient decays exponentially
- **Exploding**: If spectral_radius(W) > 1, gradient explodes exponentially

**Pure Residual RNN Gradient**:
```
h_t = h_{t-1} + f(x_t)      # f independent of h
dh_t/dh_{t-1} = I           # Identity!
dh_T/dh_0 = I^T = I         # Perfectly preserved!
```

This is formalized in the project's `ResidualRecurrence.lean`:

```lean
theorem residual_gradient_preserved (n T : Nat) :
    (List.replicate T (1 : Matrix (Fin n) (Fin n) Real)).foldl (· * ·) 1 = 1
```

### 2.2 The Curse of Memory (NeurIPS 2024)

A crucial recent theoretical advance: [arXiv:2405.21064](https://arxiv.org/abs/2405.21064) discovered that as recurrent networks encode longer memories, they become increasingly sensitive to parameter changes, even without exploding gradients.

Key findings:
- The "curse of memory" affects all RNNs attempting to learn long-term dependencies
- This arises even when the recurrent Jacobian eigenvalues are constrained below 1
- Element-wise recurrence (as in SSMs and IndRNN) combined with careful parameterization mitigates these effects

### 2.3 Identity Mappings Theory

He et al.'s analysis of identity mappings provides the theoretical foundation:

**Forward propagation**: `x_L = x_l + sum_{i=l}^{L-1} F(x_i, W_i)`

The feature `x_L` of any deeper unit L can be represented as the feature `x_l` of any shallower unit l plus a residual function.

**Backward propagation**: `dL/dx_l = dL/dx_L * (1 + d/dx_l sum_{i=l}^{L-1} F)`

The gradient `dL/dx_l` decomposes into two additive terms:
1. `dL/dx_L` - directly propagated without concerning weight layers
2. `dL/dx_L * d/dx_l sum F` - propagated through weight layers

The "1" ensures gradients can flow directly, preventing complete vanishing.

### 2.4 Gersgorin Circle Theorem in RHN

Recurrent Highway Networks used Gersgorin's circle theorem for theoretical analysis of gradient flow. This provides bounds on eigenvalues of the recurrent Jacobian, enabling principled design of transition depths.

---

## 3. Expressivity and Computational Power

### 3.1 Universal Approximation

**Standard RNNs**: Siegelmann and Sontag (1992) proved that RNNs with rational weights are Turing complete with unbounded precision and computation time.

**Bounded-Precision Results**: [NeurIPS 2021](https://openreview.net/forum?id=IWJ9jvXAoVQ) showed that 54-neuron bounded-precision RNNs with growing memory can simulate Universal Turing Machines.

**Practical Hierarchy**: Weiss et al. (ACL 2018) established that with finite precision and linear computation time:
- LSTM and ReLU-RNN are strictly more powerful than squashing RNNs and GRU
- This is because LSTMs and ReLU-RNNs can implement counting behavior

### 3.2 Does Residual Recurrence Preserve Expressivity?

**Key insight from the literature**: Yes, residual recurrence preserves (and may enhance) expressivity.

**Theoretical argument**:
1. Pure residual RNN has same linear capacity as standard: both can represent weighted sums of inputs
2. Standard: `h_T = sum A^{T-1-t} B x_t` (position-dependent weights)
3. Residual: `h_T = h_0 + W @ sum x_t` (uniform weights)
4. Nonlinearity comes from separate mechanisms (gating), not recurrence itself

**From `ResidualRecurrence.lean`**:
```lean
theorem same_linear_capacity (n : Nat) :
    (standard_expressivity n).linear_capacity = (residual_expressivity n).linear_capacity
```

**Empirical validation**: IndRNN, Highway-LSTM, and RWKV all demonstrate that residual-style architectures match or exceed the performance of standard RNNs across diverse tasks.

### 3.3 Linear vs Nonlinear Recurrence

The project's formal proofs establish key separations:

**Linear recurrence** (Mamba2, S4D): `h_{t+1} = A h_t + B x_t`
- Forms a monoid under composition (associative)
- Enables parallel scan algorithms
- Limited to O(d) bits of information in d-dimensional state

**Nonlinear recurrence** (Elman, GRU): `h_{t+1} = sigma(W_h h_t + W_x x_t)`
- NOT associative (proven via counterexample)
- Requires sequential processing
- Can compute functions linear RNNs cannot (XOR, counting)

**Residual principle**: Use linear recurrence for gradient flow + separate nonlinearity for expressivity.

---

## 4. Tradeoffs: Memory and Forgetting

### 4.1 The Memory-Forgetting Tension

**Fading Memory as Inductive Bias**: [arXiv:2307.14823](https://arxiv.org/html/2307.14823) analyzed how residual connections influence RNN dynamics and fading memory properties:

- Networks close to the "edge of chaos" have long-term fading memory
- Residuals resulting in weakly subcritical dynamics allow networks to benefit from long memory timescales
- Heterogeneous residuals enable diversity of memory timescales

**Pure Residual Limitation**:
```
h_T = h_0 + sum_{t=0}^{T-1} W @ x_t
```
This is order-independent and cannot selectively forget, making it unsuitable for tasks requiring selective memory.

**Gated Residual Solution**:
```
h_t = g_t * h_{t-1} + (1-g_t) * f(x_t)
```
The gate `g_t in (0,1)` allows controlled forgetting while maintaining gradient flow through the `g * h` path.

### 4.2 Skip Connections and Memory Timescales

**Skip connections through time (SCTT)**: [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10592728/) showed that temporal skip connections support task-relevant dynamics while maintaining biological plausibility.

Key finding: RNNs trained with SCTT required fewer training steps than control networks on 7 of 9 tasks requiring long-term dependencies.

### 4.3 The Unreasonable Effectiveness of the Forget Gate

[van der Westhuizen & Lasenby, 2018](https://arxiv.org/pdf/1804.04849) analyzed LSTM variants and found that the forget gate is crucial for performance:

- When `f_t = 1, i_t = 0`: cell state preserved indefinitely (identity mapping)
- When `f_t = 0`: complete reset (maximum forgetting)
- The forget gate effectively implements a learnable interpolation between these extremes

---

## 5. Language Modeling Applications

### 5.1 Key Results on Standard Benchmarks

**Penn Treebank (PTB)**:
| Model | Perplexity | Year |
|-------|------------|------|
| Standard LSTM | ~90 | 2016 |
| Recurrent Highway Network (depth 10) | 65.4 | 2017 |
| AWD-LSTM | 57.3 | 2017 |
| AWD-LSTM + Neural Cache | 52.8 | 2017 |
| Mogrifier LSTM | ~54 | 2020 |
| xLSTM | competitive | 2024 |

**WikiText-2**:
| Model | Perplexity |
|-------|------------|
| AWD-LSTM | 65.8 |
| AWD-LSTM + Neural Cache | 52.0 |

**Character-level (Enwik8)**:
| Model | BPC | Training Cost |
|-------|-----|---------------|
| Transformer-XL | 1.17 | 4 GPUs, 4 days |
| SRU++ | 1.17 | 2 GPUs, less time |

### 5.2 Simple Recurrent Unit (SRU) and SRU++

[SRU](https://arxiv.org/abs/1709.02755) achieved significant efficiency gains by designing for parallelism:
- 10-16x speedup compared to LSTM
- As fast as word-level convolution

[SRU++](https://www.asapp.com/blog/reducing-the-high-cost-of-training-nlp-models-with-sru) combined attention with fast recurrence:
- 8.7x more efficient than Transformer-XL to match results
- 4-5x faster inference

### 5.3 RWKV: RNNs at Transformer Scale

[RWKV](https://arxiv.org/abs/2305.13048) demonstrated that residual-based RNNs can scale to transformer-competitive performance:

**Architecture**: Series of stacked residual blocks with:
- Time-mixing (attention-like with linear complexity)
- Channel-mixing (feed-forward)
- Token shift via linear interpolation

**Scale**:
- Trained up to 14 billion parameters
- Largest dense RNN ever trained
- Performs on par with similarly-sized Transformers

**Efficiency**:
- O(n) complexity vs O(n^2) for transformers
- 10-100x lower computational requirements at large context
- Supports infinite context length theoretically

### 5.4 xLSTM: Extended LSTM (NeurIPS 2024)

[xLSTM](https://arxiv.org/abs/2405.04517) from Sepp Hochreiter's group modernized LSTM with:
- Exponential gating with normalization
- Modified memory structures (sLSTM: scalar, mLSTM: matrix)
- **Integration into residual block backbones**

Key quote: "Integrating these LSTM extensions into residual block backbones yields xLSTM blocks that are then residually stacked into xLSTM architectures."

Results: Competitive with Transformers and State Space Models on language modeling.

---

## 6. Scaling Studies

### 6.1 RNN vs Transformer Scaling Laws

**Kaplan et al. (2020)** [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361) established that:
- Transformer performance depends weakly on architectural hyperparameters when total parameters fixed
- The precise architectural details are less important than overall scale

**Implications for RNNs**:
- Traditional RNNs don't scale as well due to sequential computation bottleneck
- Residual-style RNNs (RWKV, Mamba) can leverage parallel computation during training

### 6.2 RWKV Scaling

RWKV has been scaled systematically:
- 0.1B to 14B parameters
- Performance scales smoothly with size
- Matches transformer performance at each scale

### 6.3 Mamba Scaling

[Mamba](https://arxiv.org/abs/2312.00752) demonstrated:
- Faster than FlashAttention-2 beyond sequence length 2K
- 20-40x faster than standard scan implementation
- 4-5x higher inference throughput than similar-sized Transformers

[Mamba-2](https://pli.princeton.edu/blog/2024/mamba-2-algorithms-and-systems):
- State dimension: 16 -> 128 (8x expansion)
- Training speed: 2-8x faster than Mamba-1
- Linear memory while achieving transformer-competitive performance

### 6.4 Efficiency Comparison

| Architecture | Training | Inference | Memory | Scaling |
|--------------|----------|-----------|--------|---------|
| Transformer | Parallel | O(n^2) | O(n) KV-cache | Excellent |
| Standard LSTM | Sequential | O(n) | O(1) | Poor |
| Highway-LSTM | Semi-parallel | O(n) | O(1) | Good |
| RWKV | Parallel | O(n) | O(1) | Excellent |
| Mamba | Parallel | O(n) | O(1) | Excellent |

---

## 7. Combinations with Attention and SSMs

### 7.1 Attention as RNN

[Attention as an RNN (2024)](https://arxiv.org/abs/2405.13956) showed that attention can be viewed as a special RNN:
- Transformers can be viewed as RNN variants
- Unlike traditional RNNs, they cannot be efficiently updated with new tokens
- New methods based on parallel prefix scan address this

### 7.2 Hybrid Models

**Griffin (DeepMind, 2024)**: [arXiv:2402.19427](https://arxiv.org/html/2402.19427v1) combines gated linear recurrence with local attention:
- Uses RG-LRU (Real-Gated Linear Recurrent Unit)
- Hawk variant outperforms Mamba
- Griffin surpasses Llama-2 using 6x less training data

**Hybrid Transformer-RNN models**: Multiple works combine:
- RNN for efficient long-range processing
- Attention for precise short-range dependencies
- Residual connections throughout

### 7.3 State Space Models

**S4 to Mamba Evolution**:
1. S4 (2022): Structured State Space models with efficient convolution kernels
2. S4D: Diagonal variant for simplicity
3. H3: Incorporates multiplicative interactions
4. Mamba (2023): Selective SSMs with input-dependent state transitions
5. Mamba-2 (2024): Unified framework connecting SSMs and linear attention

**Key insight**: SSMs use residual structure inherently:
- Linear recurrence with identity-like dynamics
- Residual blocks in the overall architecture
- This enables the same gradient preservation benefits

### 7.4 Attention, State Space Models, and RNNs (NeurIPS 2024)

[Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/file/f271a36160097fbdb06a9adeb1605343-Paper-Conference.pdf) unified these architectures theoretically, showing they exist on a spectrum with shared principles around residual information flow.

---

## 8. Gaps in Existing Research

### 8.1 Theoretical Gaps

1. **Precise expressivity characterization**: While we know residual RNNs can match linear capacity, the exact characterization of what functions residual vs. standard RNNs can efficiently compute remains incomplete.

2. **Optimal gating mechanisms**: The tradeoff between gradient flow (favoring identity) and expressivity (favoring transformation) lacks a principled optimal solution.

3. **Memory-computation tradeoffs**: No complete theory explains when to use residual memory (sum) vs. selective memory (gating).

4. **Curse of memory in residual architectures**: How does the curse of memory (sensitivity to parameter changes for long memories) manifest in specifically residual architectures?

### 8.2 Empirical Gaps

1. **Systematic ablation of residual placement**: Few studies systematically compare:
   - Residual in time dimension only
   - Residual in depth dimension only
   - Both combined

2. **Very long sequence benchmarks**: Most benchmarks are under 10K tokens. Behavior at 100K+ tokens is less characterized for residual RNNs.

3. **Multimodal applications**: Residual RNN performance on vision, audio, and multimodal tasks is less explored than transformers.

### 8.3 Architectural Gaps

1. **Optimal combination with attention**: The right balance of recurrence, attention, and residual connections is not established.

2. **Self-gating variants**: Comparisons between different self-gating mechanisms (SiLU, GeLU, multiplicative) in residual RNNs are incomplete.

3. **Hardware-aware designs**: Few works optimize residual RNN architecture specifically for GPU/TPU characteristics.

---

## 9. How E59 Relates to Prior Work

Based on the project's `ResidualRecurrence.lean` and `RESEARCH_ROADMAP.md`, E59 (and related experiments E42, E33) relate to prior work as follows:

### 9.1 Confirmed Prior Findings

The project's formal proofs confirm theoretical results from the literature:

**Gradient preservation**:
```lean
theorem residual_gradient_preserved (n T : Nat) :
    (List.replicate T (1 : Matrix (Fin n) (Fin n) Real)).foldl (· * ·) 1 = 1
```
This formalizes what Highway Networks, ResNet, and RHN papers showed empirically.

**Residual beats standard for vanishing gradients**:
```lean
theorem residual_beats_standard (T : Nat) (rho : Real)
    (h_small : rho < 1) (h_pos : rho > 0) (h_T : T > 0) :
    (residual_gradient_mag T).gradient_bound > (standard_gradient_mag T rho).gradient_bound
```

### 9.2 Novel Contributions

The project appears to contribute:

1. **Formal mechanized proofs**: Using Lean 4/Mathlib to formally verify results that prior work established informally.

2. **Unified framework**: Comparing linear (Mamba2), nonlinear (Elman), and log-polynomial recurrence in a single theoretical framework.

3. **Self-gating emphasis**: Explicit formal treatment of how self-gating (`h * silu(h)`) provides nonlinearity while preserving residual gradient flow.

4. **Expressivity-gradient tradeoff**: Formal proofs that Mamba2's h-independent gradient differs from Elman's h-dependent gradient.

### 9.3 Position in Literature

The project's approach aligns with:
- **RWKV philosophy**: Linear recurrence + separate gating
- **xLSTM structure**: Residual block backbones with LSTM-like components
- **Mamba principles**: Selective (input-dependent) dynamics without h-dependence in gradients

The key insight formalized:
> "Don't put nonlinearity in the recurrence itself. Use linear recurrence + separate gating."

This matches the empirical findings from IndRNN, SRU, RWKV, and modern SSMs.

---

## 10. Recommendations

### 10.1 For E59 Development

1. **Benchmark against RWKV**: Since RWKV uses similar principles (residual stacking, linear-ish recurrence, token shift), direct comparison would position E59 clearly.

2. **Formalize the self-gating component**: The `h * silu(h)` self-gating is mentioned but less formally analyzed than the residual recurrence itself.

3. **Address the curse of memory**: Consider how the architecture handles the sensitivity issue from the NeurIPS 2024 paper.

4. **Scale experiments**: The 400M studies are good; consider how residual properties manifest at larger scales.

### 10.2 For Future Research

1. **Hybrid architectures**: Combine E59's residual recurrence with selective attention (like Griffin) for best of both worlds.

2. **Hardware optimization**: Design variants specifically for parallel scan on GPU (like Mamba-2's hardware-algorithm co-design).

3. **Long context evaluation**: Test on sequences beyond current benchmarks (100K+ tokens) where residual properties should shine.

4. **Biological plausibility**: The SCTT work suggests residual temporal connections may have biological analogues worth exploring.

### 10.3 Citation Recommendations

When publishing work on residual recurrence, cite:
- He et al. (2016) "Identity Mappings in Deep Residual Networks" for theoretical foundation
- Zilly et al. (2017) "Recurrent Highway Networks" for RNN-specific application
- Li et al. (2018) "IndRNN" for practical scaling demonstration
- Peng et al. (2023) "RWKV" for large-scale language modeling evidence
- Beck et al. (2024) "xLSTM" for modern LSTM evolution

---

## 11. Sources

### Primary Academic Papers

- [Residual Recurrent Neural Networks for Learning Sequential Representations](https://www.mdpi.com/2078-2489/9/3/56) - MDPI 2018
- [Recurrent Highway Networks](https://arxiv.org/abs/1607.03474) - ICML 2017
- [Independently Recurrent Neural Network (IndRNN)](https://arxiv.org/abs/1803.04831) - CVPR 2018
- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) - ECCV 2016
- [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048) - EMNLP 2023
- [xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517) - NeurIPS 2024
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) - 2023
- [RNNs: Vanishing and Exploding Gradients Are Not the End](https://arxiv.org/abs/2405.21064) - NeurIPS 2024
- [Griffin: Mixing Gated Linear Recurrences with Local Attention](https://arxiv.org/html/2402.19427v1) - 2024
- [Simple Recurrent Units for Highly Parallelizable Recurrence](https://arxiv.org/abs/1709.02755) - EMNLP 2018
- [Regularizing and Optimizing LSTM Language Models](https://arxiv.org/abs/1708.02182) - ICLR 2018
- [Highway-LSTM and Recurrent Highway Networks for Speech Recognition](https://research.google/pubs/pub46171/) - Google Research
- [Residual Echo State Networks](https://www.sciencedirect.com/science/article/pii/S0925231224007379) - Neurocomputing 2024
- [Fading Memory as Inductive Bias in Residual Recurrent Networks](https://arxiv.org/html/2307.14823) - Neural Networks 2024
- [Attention as an RNN](https://arxiv.org/abs/2405.13956) - 2024
- [On the Turing Completeness of Modern Neural Network Architectures](https://arxiv.org/abs/1901.03429) - ICLR 2019
- [Turing Completeness of Bounded-Precision RNNs](https://openreview.net/forum?id=IWJ9jvXAoVQ) - NeurIPS 2021
- [The Unreasonable Effectiveness of the Forget Gate](https://arxiv.org/pdf/1804.04849) - 2018
- [Bridging the Gaps Between Residual Learning, Recurrent Neural Networks and Visual Cortex](https://arxiv.org/pdf/1604.03640) - 2016

### GitHub Repositories

- [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) - Official RWKV implementation
- [Mamba](https://github.com/state-spaces/mamba) - Official Mamba implementation
- [SRU](https://github.com/asappresearch/sru) - Simple Recurrent Unit
- [IndRNN PyTorch](https://github.com/Sunnydreamrain/IndRNN_pytorch) - IndRNN implementation

### Tutorials and Surveys

- [Stanford CS224N Lecture 7: Fancy RNNs](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture07-fancy-rnn.pdf)
- [Understanding LSTMs - colah's blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [A Visual Guide to Mamba and State Space Models](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state)
- [From S4 to Mamba: A Comprehensive Survey](https://arxiv.org/abs/2503.18970)
- [Dive into Deep Learning - LSTM Chapter](https://d2l.ai/chapter_recurrent-modern/lstm.html)

---

*Report generated: 2026-01-13*
*For the elman-proofs project*
