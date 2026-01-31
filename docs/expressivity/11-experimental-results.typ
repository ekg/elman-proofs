// Section 11: Experimental Results
// CMA-ES Hyperparameter Search and Benchmark Validation

#let theorem(title, body) = block(
  fill: rgb("#f0f7ff"),
  stroke: rgb("#3366cc"),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
)[
  #strong[Theorem (#title)]#linebreak()
  #body
]

#let definition(title, body) = block(
  fill: rgb("#fff7f0"),
  stroke: rgb("#cc6633"),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
)[
  #strong[Definition (#title)]#linebreak()
  #body
]

#let observation(title, body) = block(
  fill: rgb("#f0fff7"),
  stroke: rgb("#33cc66"),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
)[
  #strong[Observation (#title)]#linebreak()
  #body
]

#let finding(body) = block(
  fill: rgb("#f7f7ff"),
  stroke: rgb("#6666cc"),
  inset: 12pt,
  radius: 4pt,
)[#body]

= Section 11: Experimental Results

_CMA-ES Hyperparameter Evolution, Benchmark Validation, and Theory-Practice Gap Analysis_

== 11.1 Overview

The preceding sections established theoretical expressivity bounds: E88 (nonlinear temporal) strictly contains Mamba2 (linear temporal) in computational power. This section presents comprehensive empirical results from CMA-ES hyperparameter evolution experiments, testing whether theoretical advantages manifest in practice.

#finding[
  *Key Finding*: Despite E88's provably greater expressivity, Mamba2 outperforms E88 on language modeling benchmarks. This reveals a theory-practice gap where optimization efficiency and training dynamics dominate over raw computational power.
]

== 11.2 CMA-ES Hyperparameter Search

=== 11.2.1 Methodology

#definition("CMA-ES Search Protocol")[
  Covariance Matrix Adaptation Evolution Strategy (CMA-ES) was used to optimize hyperparameters across all architectures:

  - *Target parameters*: 480M $plus.minus$ 50M
  - *Training time per configuration*: 10 minutes
  - *Learning rate*: Fixed at $3 times 10^(-4)$ (fair comparison)
  - *Optimizer*: ScheduleFree AdamW
  - *Data*: The Pile (byte-level tokenization, 256 vocab)
  - *Hardware*: 4$times$ RTX 6000 Ada (96GB VRAM total)
  - *Evaluations*: ~120 configurations per architecture (15 generations $times$ 8 population)
]

#definition("Search Spaces")[
  Each architecture was optimized over interpretable hyperparameter ranges:

  *E88*:
  - `n_heads`: 32--160 (integer)
  - `n_state`: $\{16, 32, 48, 64\}$ (CUDA kernel constraint)
  - `depth`: 12--40 (integer)

  *Mamba2*:
  - `d_state`: 64--256 (multiples of 16)
  - `expand`: 1--3 (integer)
  - `depth`: 16--40 (integer)

  *Transformer*:
  - `n_heads`: 8--32 (integer)
  - `expansion`: 2--6 (integer)
  - `depth`: 12--36 (integer)
]

=== 11.2.2 CMA-ES Evolution Dynamics

CMA-ES iteratively:
1. Samples candidate configurations from a multivariate Gaussian
2. Evaluates each candidate (10-minute training run)
3. Updates the covariance matrix toward high-performing regions
4. Repeats until convergence or budget exhaustion

This approach is well-suited for neural architecture search because it:
- Handles mixed continuous/discrete parameters
- Adapts search direction based on landscape curvature
- Avoids getting trapped in local optima

== 11.3 Main Results: Architecture Comparison at 480M Scale

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    align: (left, center, center, center, center),
    [*Architecture*], [*Best Loss*], [*Best Configuration*], [*Params*], [*Status*],
    [*Mamba2*], [*1.271*], [d_state=96, expand=2, depth=25], [494M], [Best],
    [FLA-GDN], [1.273], [expansion=2, depth=17, n_heads=24], [~480M], [--],
    [E88], [1.407], [n_heads=68, n_state=16, depth=23], [488M], [--],
    [Transformer], [1.505], [n_heads=8, expansion=4, depth=13], [491M], [--],
    [MinGRU], [1.528], [expansion=1, depth=14], [~480M], [--],
    [MinLSTM], [1.561], [expansion=1, depth=31], [~480M], [--],
    [MoM-E88], [1.762], [n_heads=40, top_k=8, depth=12], [~480M], [--],
    [E90 (Dual-Rate)], [1.791], [n_heads=114, depth=13], [~500M], [--],
    [GRU (CUDA)], [10.0], [-], [~480M], [Diverged],
  ),
  caption: [CMA-ES optimized benchmark results at 480M parameters. Loss is cross-entropy on held-out The Pile data.],
)

#observation("Result Summary")[
  The empirical ranking is:
  $ "Mamba2" (1.271) > "FLA-GDN" (1.273) > "E88" (1.407) > "Transformer" (1.505) $

  This *inverts* the theoretical expressivity hierarchy:
  $ "E88" supset "Mamba2" supset "FLA-GDN" supset "Linear Attention" $
]

== 11.4 E88-Specific Findings

=== 11.4.1 Optimal E88 Configuration

The CMA-ES search converged to:

#finding[
  *E88 Optimal*: 68 heads, n_state=16, depth=23, dim=3840

  - *Many small heads*: 68 heads $times$ 16-dim state $>$ 17 heads $times$ 64-dim state
  - *Moderate depth*: 20--25 layers optimal
  - *CUDA alignment*: n_state $in \{16, 32, 48, 64\}$ required for fused kernel
]

=== 11.4.2 E88 Ablation Studies

Controlled experiments revealed surprising findings about E88 components:

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,
    align: (left, center, center),
    [*Ablation*], [*Loss Change*], [*Interpretation*],
    [Remove output RMSNorm], [$-0.10$], [Improves (norm was harmful)],
    [Remove convolutions], [$-0.03$], [Slight improvement],
    [Remove output gating], [$-0.01$], [Negligible effect],
    [Linear state $approx$ tanh state], [$approx 0$], [*No difference!*],
    [Remove SiLU gating], [Divergence], [Critical for stability],
    [Remove L2 normalization], [Divergence], [Critical for stability],
  ),
  caption: [E88 ablation results. Negative loss change = improvement.],
)

#observation("Surprising Ablation Result")[
  The linear-state vs tanh-state ablation showed *no measurable difference* on language modeling loss.

  This is consistent with Section 3's analysis: for language modeling, the composition depth $D$ from layer stacking may be sufficient, and tanh's nonlinearity in the recurrence adds little practical value.

  *Interpretation*: The theoretical separation (E88 can compute parity, Mamba2 cannot) doesn't manifest in natural language distributions where such computations are rare.
]

== 11.5 Multi-Head Benchmark (E75/E87, 100M Scale)

Smaller-scale experiments at 100M parameters (10 min training, depth=20) tested multi-head variants:

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,
    align: (left, center, center, left),
    [*Architecture*], [*Best Variant*], [*Loss*], [*Notes*],
    [*Mamba2*], [d=896], [*1.21*], [SSM baseline -- Best],
    [*E75 Multi-Head*], [4 heads, n_state=32], [1.42], [Best Elman variant],
    [FLA-GDN], [d=768], [1.57], [ICLR 2025 baseline],
    [E87 Sparse Block], [16 blocks, top-4], [1.67], [MoE-style routing],
  ),
  caption: [E75/E87 benchmark results at 100M scale.],
)

=== 11.5.1 E75 Multi-Head Parameter Scan

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    align: (left, center, center, center, center),
    [*Model*], [*Heads*], [*n_state*], [*Loss*], [*Status*],
    [*E75h4n32*], [4], [32], [*1.42*], [Best],
    [E75h4n24], [4], [24], [1.56], [OK],
    [E75h5n24], [5], [24], [1.58], [OK],
    [E75h6n24], [6], [24], [1.62], [OK],
    [E75h4n20], [4], [20], [NaN], [Diverged],
    [E75h4n28], [4], [28], [NaN], [Diverged],
    [E75h5n20], [5], [20], [NaN], [Diverged],
  ),
  caption: [E75 multi-head parameter scan showing stability boundaries.],
)

#observation("Numerical Stability Boundaries")[
  *Critical finding*: `n_state` values not divisible by 8 (specifically 20, 28) cause NaN divergence for *all* head counts.

  This suggests a hardware/numerical alignment constraint beyond the theoretical architecture specification. The practical search space is constrained to `n_state` $in {16, 24, 32, 48, 64}$.
]

=== 11.5.2 Sparse Block Scaling (E87)

E87 uses MoE-style routing where only top-$k$ memory blocks are updated per timestep:

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,
    align: (center, center, center),
    [*Blocks*], [*Best top_k*], [*Loss*],
    [4], [2], [1.91],
    [8], [3], [1.76],
    [16], [4], [*1.67*],
    [32], [4--8], [3.8--4.2],
  ),
  caption: [E87 sparse block scaling. 32 blocks dilutes signal too much.],
)

#observation("Sparse Routing Limitation")[
  16 blocks with top-4 routing is optimal. Beyond 32 blocks, signal dilution causes severe performance degradation.

  *Dense multi-head (E75, E88) consistently outperforms sparse routing (E87, MoM-E88)*. This suggests that for current scales, the overhead of routing doesn't pay off.
]

== 11.6 Running Parity Validation

=== 11.6.1 Theoretical Prediction

From Section 6, we have the proven separation:

#theorem("Running Parity Separation")[
  For any $D$-layer linear-temporal model:
  $ forall "model" in cal(L)_D, quad not "CanComputeRunningParity"("model") $

  E88 with tanh saturation can compute running parity with appropriate weights.
]

=== 11.6.2 Experimental Setup

#definition("Running Parity Task")[
  - *Input*: Binary sequence $x_1, x_2, ..., x_T in \{0, 1\}^T$
  - *Target*: At each position $t$, output $y_t = x_1 xor x_2 xor ... xor x_t$
  - *Metric*: Per-position accuracy
]

=== 11.6.3 Predicted vs Observed Results

#finding[
  *Status*: Dedicated running parity experiments were not found in the ~/elman benchmark suite.

  *Theoretical Prediction*:
  - E88: ~99% accuracy (can represent parity with saturating dynamics)
  - Mamba2: ~50% accuracy (cannot compute XOR, reduces to random guessing)

  *Recommended Experiment*: Synthetic parity benchmark with sequences $T in \{32, 64, 128, 256, 512, 1024\}$ to validate separation.
]

The language modeling benchmarks do *not* test parity-like computations. The theory-practice gap suggests that natural language rarely requires the specific computational patterns that separate E88 from Mamba2.

== 11.7 Comparison with Theoretical Predictions

=== 11.7.1 Expressivity vs Performance

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,
    align: (left, center, center),
    [*Property*], [*Theory*], [*Empirical*],
    [XOR/Parity], [E88 > Mamba2], [Not tested],
    [Threshold counting], [E88 > Mamba2], [Not tested],
    [Binary fact retention], [E88 > Mamba2], [Not tested],
    [Language modeling], [E88 $>=$ Mamba2], [*Mamba2 > E88*],
    [Training efficiency], [Not specified], [Mamba2 >> E88],
  ),
  caption: [Theoretical predictions vs empirical observations.],
)

=== 11.7.2 Reconciling Theory and Practice

#observation("Theory-Practice Gap Analysis")[
  The theoretical expressivity hierarchy (E88 $supset$ Mamba2) is mathematically valid but does not account for:

  1. *Optimization landscape*: Mamba2's parallel scan creates smoother gradients
  2. *Training throughput*: Mamba2 processes more tokens per second (parallel vs sequential)
  3. *Inductive bias*: Linear dynamics may match language statistics at current scales
  4. *Hyperparameter sensitivity*: E88 requires more careful tuning

  The relevant question shifts from "What can this architecture compute?" to "What can this architecture *learn* within a training budget?"
]

=== 11.7.3 When Theory Predicts Practice

The theoretical predictions should manifest when:

1. *Task requires specific computations*: Parity, counting, state tracking
2. *Sequence length exceeds depth*: $T >> D$ where linear-temporal models have insufficient composition
3. *Evaluation is exact*: Tasks where approximate solutions don't suffice

For language modeling, none of these conditions are strongly met, explaining the reversed empirical ranking.

== 11.8 Why Mamba2 Wins on Language Modeling

=== 11.8.1 Parallel Scan Efficiency

#definition("Parallel Scan")[
  Mamba2 computes the recurrence:
  $ h_t = A_t h_(t-1) + B_t x_t $

  Using a parallel associative scan in $O(log T)$ parallel time, rather than $O(T)$ sequential time.

  For a 10-minute training budget, Mamba2 processes *more tokens* than E88 (which must run sequentially).
]

=== 11.8.2 Input-Dependent Selectivity

Unlike fixed-weight linear RNNs, Mamba2's selectivity mechanism makes $A, B, C, Delta$ all functions of the input. This provides:

- Adaptive forgetting ($Delta$ controls decay rate)
- Content-aware gating ($B, C$ project input relevance)
- Dynamic state transition ($A$ varies per position)

This input-dependence captures some of what E88 achieves through nonlinearity, but with parallel-friendly operations.

=== 11.8.3 Numerical Stability

Mamba2 uses log-space updates to prevent overflow in long sequences:

$ log(h_t) = log(A_t h_(t-1) + B_t x_t) $

E88's tanh saturation provides stability but also *information compression*---values are squeezed into $[-1, 1]$, potentially limiting precision.

== 11.9 Implications for Architecture Design

#finding[
  *Design Principles from Experiments*:

  1. *Many small heads > few large heads*: E88 optimal at 68$times$16, not 17$times$64
  2. *Dense > sparse*: Multi-head outperforms MoE-style routing at current scales
  3. *Alignment matters*: CUDA kernel constraints (`n_state mod 8 = 0`) affect practical performance
  4. *Theoretical power $!=$ empirical performance*: Optimization dynamics dominate expressivity
]

=== 11.9.1 Hybrid Architecture Opportunity

The experiments suggest a hybrid approach:

- Use *Mamba2* for efficient sequence processing (linear-temporal bulk)
- Add *E88-style heads* for tasks requiring nonlinear temporal computation
- Route based on input complexity

This combines Mamba2's training efficiency with E88's expressivity for targeted computations.

=== 11.9.2 Benchmark Recommendations

To properly validate the expressivity hierarchy, future benchmarks should include:

1. *Synthetic parity sequences*: Clean test of XOR computation
2. *Threshold counting tasks*: Test discontinuous decisions
3. *State machine simulation*: Test latching and multi-state tracking
4. *Long-range retrieval with interference*: Test binary fact retention under noise

These tasks are *designed* to separate architectures, unlike language modeling which conflates many factors.

== 11.10 Summary

#figure(
  table(
    columns: 2,
    stroke: 0.5pt,
    align: (left, left),
    [*Finding*], [*Implication*],
    [Mamba2 beats E88 empirically], [Theoretical expressivity $!=$ practical performance],
    [E88 linear $approx$ tanh on LM], [Tanh nonlinearity adds little for language],
    [n_state=16 optimal], [Many small heads > few large heads],
    [Sparse routing underperforms], [Dense computation preferred at current scale],
    [CUDA alignment critical], [Practical constraints shape search space],
    [Parity tasks not tested], [Expressivity separation needs targeted benchmarks],
  ),
  caption: [Summary of experimental findings.],
)

The experiments reveal a fundamental lesson: *theoretical computational power is necessary but not sufficient for practical performance*. Mamba2's linear temporal dynamics are strictly less expressive than E88's nonlinear dynamics, yet Mamba2 achieves better language modeling loss.

This does not invalidate the theoretical results---E88 genuinely can compute functions Mamba2 cannot. Rather, it demonstrates that:

1. Language modeling may not require those specific computations
2. Training dynamics matter as much as final expressivity
3. Parallel efficiency compounds over training time

The expressivity hierarchy remains valuable for understanding *what architectures can potentially compute*, while empirical benchmarks tell us *what architectures learn efficiently* for specific data distributions. Both perspectives are necessary for principled architecture design.
