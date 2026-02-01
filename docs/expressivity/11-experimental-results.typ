// Section 11: Experiments and Theory-Practice Gap

#import "traditional-math-style.typ": *

= The Theory-Practice Gap

E88 computes functions that linear-temporal models cannot. The hierarchy is strict. Yet when trained on language modeling, the empirical ranking inverts the theoretical one.

== The Empirical Results

CMA-ES hyperparameter optimization at 480M parameters:

#simpletable(
  columns: 8,
  align: (left, center, center, center, center, center, center, center),
  [*Architecture*], [*Loss*], [*Depth*], [*Hidden Dim*], [*Expansion*], [*Params*], [*LR*], [*Iters*],
  [Mamba2], [1.271], [25], [1792], [2], [494M], [3e-4], [120],
  [GDN], [1.273], [17], [1920], [2], [502M], [3e-4], [120],
  [E88], [1.407], [23], [3840], [1.0], [488M], [3e-4], [120],
  [Transformer], [1.505], [13], [1536], [4], [491M], [3e-4], [120],
  [MinGRU], [1.528], [14], [2944], [1], [486M], [3e-4], [120],
  [MinLSTM], [1.561], [31], [1792], [1], [498M], [3e-4], [120],
  [MoM-E88], [1.762], [12], [3840], [1.0], [480M], [3e-4], [240],
  [E90], [1.791], [13], [3072], [1.0], [497M], [3e-4], [80],
)

_Iterations_: CMA-ES evaluations (training runs). All models trained 10 minutes per config at 3e-4 learning rate with ScheduleFree AdamW optimizer. Iterations vary because some models converged faster (E90 needed only 80 evals to find optimal hyperparameters) while MoM-E88 required extended search (240 evals due to sparse routing complexity). Standard CMA-ES search: 15 generations × 8 population = 120 evaluations.

=== Key Hyperparameters by Architecture

#simpletable(
  columns: 3,
  align: (left, left, left),
  [*Architecture*], [*Best Configuration*], [*Expressivity-Critical Parameters*],
  [Mamba2], [d_state=96, expand=2, depth=25], [Linear SSM: no temporal nonlinearity],
  [GDN], [expansion=2, depth=17, n_heads=24], [Gated Delta: input-dependent gates],
  [E88], [n_heads=68, n_state=16, depth=23], [Nonlinear: tanh(Wh + Ux) enables XOR],
  [Transformer], [n_heads=8, expansion=4, depth=13], [Attention: quadratic, no recurrence],
  [MinGRU], [expansion=1, depth=14], [Minimal GRU: linear hidden state],
  [MinLSTM], [expansion=1, depth=31], [Minimal LSTM: gated but linear temporal],
  [MoM-E88], [n_heads=40, top_k=8, n_state=64, depth=12], [Sparse routing: activates 8 of 40 heads],
  [E90], [n_heads=114, config=(8,16), depth=13], [Dual-rate: 8-step fast, 16-step slow],
)

_Expansion factor_: E88 expansion=1.0 (square state matrices). Mamba2 expand=2 (hidden dim is 2× model dim). GDN expansion=2 (FFN factor). MinGRU/MinLSTM expansion=1 (minimal parameterization).

_State dimension_: E88 n_state=16 per head (68 heads × 16 = 1088 total state). Mamba2 d_state=96 (single state vector). E90 uses dual-rate with (8, 16) timestep config.

=== Complete E88 Architecture Configuration

#simpletable(
  columns: 2,
  align: (left, left),
  [*Parameter*], [*Value*],
  [n_heads], [68],
  [n_state], [16],
  [depth], [23],
  [dim], [3840],
  [expansion], [1.0],
  [use_gate], [True],
  [gate_activation], [silu],
  [Total state dimension], [1088 (68 × 16)],
)

=== Complete Mamba2 Architecture Configuration

#simpletable(
  columns: 2,
  align: (left, left),
  [*Parameter*], [*Value*],
  [d_state], [96],
  [expand], [2],
  [depth], [25],
  [Recurrence type], [Parallel scan (O(log n))],
  [Selectivity], [Input-dependent B, C, Δ],
  [Numerical updates], [Log-space (stability)],
)

=== Complete Transformer Architecture Configuration

#simpletable(
  columns: 2,
  align: (left, left),
  [*Parameter*], [*Value*],
  [n_heads], [8],
  [expansion], [4],
  [depth], [13],
  [Attention type], [LLaMA-style],
  [Context complexity], [Quadratic],
)

Mamba2 (cannot compute parity) achieved the best loss. E88 (provably more expressive) placed third.

=== CMA-ES Search Configuration

All experiments used:
- *Target parameters*: 480M ± 50M
- *Training time per config*: 10 minutes
- *Learning rate*: Fixed at 3e-4 (fair comparison)
- *Optimizer*: ScheduleFree AdamW
- *Data*: The Pile (byte-level tokenization, 256 vocab)
- *Hardware*: Multi-GPU (4× RTX 6000 Ada)
- *Evaluations per model*: ~120 configurations (15 generations × 8 population)

=== Search Space Definitions

#simpletable(
  columns: 4,
  align: (left, left, left, left),
  [*Architecture*], [*Parameter*], [*Range*], [*Type*],
  [E88], [n_heads], [32-160], [int],
  [], [n_state], [16, 32, 48, 64], [discrete (CUDA kernel constraint)],
  [], [depth], [12-40], [int],
  [Mamba2], [d_state], [64-256], [int (multiples of 16)],
  [], [expand], [1-3], [int],
  [], [depth], [16-40], [int],
  [Transformer], [n_heads], [8-32], [int],
  [], [expansion], [2-6], [int],
  [], [depth], [12-36], [int],
)

=== E75/E87 Benchmark Comparison (100M Scale)

From separate 100M parameter experiments (10 min training):

#simpletable(
  columns: 4,
  align: (left, center, left, left),
  [*Architecture*], [*Loss*], [*Best Config*], [*Notes*],
  [Mamba2], [1.21], [Standard config], [Best overall],
  [E75 (Multi-Head Delta)], [1.42], [4 heads, n_state=32], [Multi-head variant],
  [GDN], [1.57], [Standard config], [ICLR 2025 baseline],
  [E87 (Sparse Block)], [1.67], [16 blocks, top-4], [Sparse routing],
)

_Key Finding_: Multi-head variants (E75, E88) significantly outperform sparse routing (E87, MoM-E88) at current scales.

== The Ablation That Reveals Everything

We replaced E88's tanh with linear recurrence. If temporal nonlinearity matters for language modeling, this should degrade performance.

It did not. Loss was unchanged.

Running parity, exact threshold, state machine simulation---these do not manifest in the language modeling objective.

=== Complete E88 Ablation Results

#simpletable(
  columns: 3,
  align: (left, center, left),
  [*Component Removed*], [*Loss Change*], [*Interpretation*],
  [Output RMSNorm], [-0.10 nats], [Improvement: normalization hurts],
  [Convolutions], [-0.03 nats], [Small improvement: conv not needed],
  [Output gating], [-0.01 nats], [Minimal improvement],
  [Tanh → Linear state], [~0.00 nats], [No difference: nonlinearity unused!],
)

_Critical finding_: SiLU gate activation and L2 normalization are essential for stability. The tanh nonlinearity in the recurrence contributes nothing to language modeling performance, despite being theoretically necessary for computing parity.

== Two Types of Efficiency

#definition("Sample vs Wall-Clock Efficiency")[
  _Sample efficiency_: examples needed to learn a function class.

  _Wall-clock efficiency_: forward and backward passes per unit wall-clock time.
]

E88 processes one timestep at a time. Mamba2's parallel scan processes sequences simultaneously, achieving roughly 4× higher throughput.

In fixed wall-clock time, Mamba2 sees 4× as many examples. Training dynamics dominate expressivity.

== When Theory Predicts Practice

_When the task requires expressivity_: For running parity, Mamba2 cannot converge. Loss plateaus at random-chance (50%). E88 converges to near-perfect accuracy. The impossibility theorem manifests as an unbreakable floor.

_When training budget is unlimited_: Given infinite time, E88's expressivity advantage should manifest even for tasks both can approximate.

#simpletable(
  columns: 3,
  align: (left, center, center),
  [*Property*], [*Theory Predicts*], [*Empirical Observation*],
  [Running parity], [E88 > Mamba2], [Mamba2 stuck at 50%],
  [Language modeling], [E88 $>=$ Mamba2], [Mamba2 > E88],
)

== Interpreting the Gap

#keyinsight[
  Expressivity determines what can be computed with unlimited resources. Benchmark performance measures what is learned in fixed time.
]

Language modeling may not require the capabilities that separate E88 from Mamba2:

_Natural language may not require temporal nonlinearity._ Natural text may lie within what linear-temporal models can approximate.

_Benchmarks may not measure where it matters._ Perplexity averages over all predictions. Rare cases requiring temporal nonlinearity may be overwhelmed by common pattern matching.

_Theory is about expressivity, not learnability._ A function may be computable but unreachable by gradient descent. Expressivity is necessary but not sufficient.

== Lessons from Experiments

_Many small heads outperform few large heads._ For E88, 68 heads with 16-dimensional state outperformed configurations with fewer, larger heads.

_Dense architectures outperform sparse at current scales._ At 480M parameters, dense computation wins.

_Hardware alignment matters._ State dimensions that are multiples of 8 achieve efficient CUDA execution.

_Theoretical power does not equal empirical performance._ Expressivity is one factor among many.

#centerrule

Theory tells us what is possible with unlimited resources. Practice tells us what happens with finite resources on specific tasks.
