// Section 11: Experiments and Theory-Practice Gap

#import "traditional-math-style.typ": *

= The Theory-Practice Gap

The theorems in this document are airtight. E88 computes functions that linear-temporal models cannot. The hierarchy is strict. Yet when we train these models on language modeling, the empirical ranking inverts the theoretical one. This tension deserves careful examination.

== The Empirical Results

In experiments with CMA-ES hyperparameter optimization at 480M parameters, the results surprised us:

#simpletable(
  columns: 6,
  align: (left, center, center, center, center, center),
  [*Architecture*], [*Loss*], [*Depth*], [*Hidden Dim*], [*Params*], [*Iters*],
  [Mamba2], [1.271], [25], [1792], [494M], [120],
  [FLA-GDN], [1.273], [17], [1920], [502M], [120],
  [E88], [1.407], [23], [3840], [488M], [120],
  [Transformer], [1.505], [13], [1536], [491M], [120],
  [MinGRU], [1.528], [14], [2944], [486M], [120],
  [MinLSTM], [1.561], [31], [1792], [498M], [120],
  [MoM-E88], [1.762], [12], [3840], [480M], [240],
  [E90], [1.791], [13], [3072], [497M], [80],
)

_Iterations_: CMA-ES evaluations (training runs). All models trained 10 minutes per config at 3e-4 learning rate. Iterations vary because some models converged faster (E90 needed only 80 evals) while MoM-E88 required extended search (240 evals).

=== Key Hyperparameters by Architecture

#simpletable(
  columns: 3,
  align: (left, left, left),
  [*Architecture*], [*Best Configuration*], [*Expressivity-Critical Parameters*],
  [Mamba2], [d_state=96, expand=2, depth=25], [Linear SSM: no temporal nonlinearity],
  [FLA-GDN], [expansion=2, depth=17, n_heads=24], [Gated Delta: input-dependent gates],
  [E88], [n_heads=68, n_state=16, depth=23], [Nonlinear: tanh(Wh + Ux) enables XOR],
  [Transformer], [n_heads=8, expansion=4, depth=13], [Attention: quadratic, no recurrence],
  [MinGRU], [expansion=1, depth=14], [Minimal GRU: linear hidden state],
  [MinLSTM], [expansion=1, depth=31], [Minimal LSTM: gated but linear temporal],
  [MoM-E88], [n_heads=40, top_k=8, n_state=64, depth=12], [Sparse routing: activates 8 of 40 heads],
  [E90], [n_heads=114, config=(8,16), depth=13], [Dual-rate: 8-step fast, 16-step slow],
)

_Expansion factor_: E88 expansion=1.0 (square state matrices). Mamba2 expand=2 (hidden dim is 2× model dim). FLA-GDN expansion=2 (FFN factor). MinGRU/MinLSTM expansion=1 (minimal parameterization).

_State dimension_: E88 n_state=16 per head (68 heads × 16 = 1088 total state). Mamba2 d_state=96 (single state vector). E90 uses dual-rate with (8, 16) timestep config.

Mamba2---the architecture we proved cannot compute parity---achieved the best loss. E88---provably more expressive---placed third. The empirical ranking exactly inverts the theoretical hierarchy.

== The Ablation That Reveals Everything

We conducted an ablation study: replace E88's tanh with linear recurrence, converting it to a linear-temporal model. If temporal nonlinearity matters for language modeling, this should degrade performance.

It did not. The loss was unchanged within measurement noise.

The theoretical separation---running parity, exact threshold, state machine simulation---does not manifest in the language modeling objective. Whatever natural language requires, it is not the capabilities that distinguish E88 from Mamba2.

== Two Types of Efficiency

The resolution lies in distinguishing two notions of efficiency.

#definition("Sample vs Wall-Clock Efficiency")[
  _Sample efficiency_: examples needed to learn a function class.

  _Wall-clock efficiency_: forward and backward passes per unit wall-clock time.
]

E88's sequential recurrence processes one timestep at a time. Mamba2's parallel scan processes the entire sequence simultaneously. On modern hardware, Mamba2 achieves roughly 4× higher throughput---four times as many tokens processed per second.

In fixed wall-clock training time, Mamba2 sees 4× as many examples. If both architectures can learn the target function (language modeling), the one that sees more examples learns better. The training dynamics dominate the expressivity advantage.

== When Theory Predicts Practice

The theory-practice gap closes under specific conditions.

_When the task requires expressivity_: For running parity, Mamba2 cannot converge regardless of training budget. Any loss curve plateaus at random-chance accuracy (50%). E88, given sufficient training, converges to near-perfect accuracy. The impossibility theorem manifests as an unbreakable floor.

_When training budget is unlimited_: Given infinite time, E88's expressivity advantage should eventually manifest even for tasks that both can approximate. We rarely have infinite time.

#simpletable(
  columns: 3,
  align: (left, center, center),
  [*Property*], [*Theory Predicts*], [*Empirical Observation*],
  [Running parity], [E88 > Mamba2], [Mamba2 stuck at 50%],
  [Language modeling], [E88 $>=$ Mamba2], [Mamba2 > E88],
)

== Interpreting the Gap

#keyinsight[
  Expressivity determines what can be computed with unlimited resources. Benchmark performance measures what is learned in fixed time. The gap between them is not a flaw in the theory---it is information about the task.
]

The language modeling benchmark apparently does not require the capabilities that separate E88 from Mamba2. This could mean:

_Natural language does not require temporal nonlinearity._ The distribution of natural text may lie within what linear-temporal models can approximate, even though they cannot exactly compute running parity.

_The benchmarks do not measure where it matters._ Perplexity averages over all predictions. Rare cases requiring temporal nonlinearity---complex state tracking, deep nesting, multi-step reasoning---may be overwhelmed by common cases requiring only pattern matching.

_The theory is about expressivity, not learnability._ A function may be computable by an architecture but unreachable by gradient descent from random initialization. Expressivity is necessary but not sufficient for learning.

== Lessons from Experiments

Several empirical findings guide practical design.

_Many small heads outperform few large heads._ For E88, 68 heads with 16-dimensional state outperformed configurations with fewer, larger heads. The diversity of independent state machines matters more than the capacity of each one.

_Dense architectures outperform sparse at current scales._ Mixture-of-experts and sparse attention show promise at very large scales, but at 480M parameters, dense computation wins.

_Hardware alignment matters._ State dimensions that are multiples of 8 achieve efficient CUDA execution. A state dimension of 68 may theoretically allow more capacity than 64, but 64 runs faster.

_Theoretical power does not equal empirical performance._ This is perhaps the central lesson. Expressivity is one factor among many. Trainability, throughput, initialization, and optimization dynamics all contribute to final performance.

#centerrule

The theory-practice gap is not a contradiction. Theory tells us what is possible with unlimited resources. Practice tells us what happens with finite resources on specific tasks. E88's expressivity advantage is real and provable; Mamba2's training advantage is real and measurable. Which matters depends on the task.

The next section examines where the expressivity advantage is likely to matter: tasks whose composition depth exceeds what linear-temporal models can provide.
