// Section 13: The Theory-Practice Gap
// Sample Efficiency vs Wall-Clock Efficiency

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

= The Theory-Practice Gap

_Why Provably More Expressive Architectures Can Lose in Practice_

The preceding sections established a rigorous expressivity hierarchy: E88 (nonlinear temporal) strictly contains Mamba2 (linear temporal). Yet experiments show Mamba2 outperforms E88 on language modeling. This section analyzes why theoretical computational power does not translate directly to empirical performance.

== 13.1 The Central Paradox

#finding[
  *The Paradox*: E88 provably computes functions that Mamba2 cannot (running parity, threshold, XOR chains). Yet Mamba2 achieves lower loss on language modeling benchmarks.

  $ "Expressivity: E88" supset "Mamba2" $
  $ "Empirical loss: Mamba2 (1.27)" < "E88 (1.41)" $

  These facts are not contradictory---they measure different things.
]

The resolution lies in distinguishing two efficiency concepts:

#definition("Sample Efficiency")[
  The number of training examples required to learn a function.

  If architecture $A$ can represent function $f$ but requires $10^{12}$ samples to learn it, while architecture $B$ cannot represent $f$ at all, then $A$ is more expressive but possibly less sample-efficient for tasks where $f$ matters.
]

#definition("Wall-Clock Efficiency")[
  The number of functions evaluated per unit time during training.

  If architecture $A$ processes 100 tokens/second and $B$ processes 1000 tokens/second, then $B$ sees 10$times$ more data in the same wall-clock budget.
]

#observation("The Trade-off")[
  *Expressivity* determines _what can be learned_ given unlimited data.

  *Wall-clock efficiency* determines _how much data is seen_ in a fixed training budget.

  The winner depends on:
  1. Whether the target function requires the extra expressivity
  2. Whether the training budget is sufficient to exploit it
]

== 13.2 Sample Efficiency Analysis

=== 13.2.1 The Optimization Landscape

More expressive models typically have more complex loss landscapes:

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,
    align: (left, center, center),
    [*Property*], [*Linear-Temporal (Mamba2)*], [*Nonlinear-Temporal (E88)*],
    [Local minima], [Fewer], [More],
    [Saddle points], [Fewer], [More],
    [Gradient smoothness], [Higher (parallel scan)], [Lower (sequential tanh)],
    [Curvature variation], [Lower], [Higher],
  ),
  caption: [Optimization landscape characteristics by architecture type.],
)

#theorem("Expressivity-Smoothness Trade-off")[
  Let $cal(F)_"linear"$ and $cal(F)_"nonlinear"$ be the function classes representable by linear-temporal and nonlinear-temporal RNNs respectively.

  If $cal(F)_"nonlinear" supset.neq cal(F)_"linear"$, then the parameter space of nonlinear-temporal models has higher intrinsic dimension and more complex geometry.

  *Consequence*: Gradient descent is more likely to find good solutions quickly in the simpler space, even if the global optimum is worse.
]

=== 13.2.2 Gradient Flow Dynamics

Consider the gradient of loss with respect to parameters at time step $t$:

*Mamba2 (linear temporal)*:
$ (partial L) / (partial theta) = sum_(t=1)^T (partial L) / (partial h_t) dot underbrace(A^(T-t), "constant factor") dot (partial h_1) / (partial theta) $

The $A^(T-t)$ terms are _constant_ with respect to input---gradients flow uniformly.

*E88 (nonlinear temporal)*:
$ (partial L) / (partial theta) = sum_(t=1)^T (partial L) / (partial S_t) dot underbrace(product_(s=t)^T "tanh'"(dot.c), "input-dependent") dot (partial S_1) / (partial theta) $

The $"tanh'"(dot.c)$ terms depend on actual activations---gradients are _input-dependent_ and can vanish in saturated regions.

#observation("Gradient Flow Quality")[
  Mamba2's parallel scan provides:
  1. *Uniform gradient scaling*: All timesteps contribute equally (modulo $A^k$ decay)
  2. *Parallel computation*: Gradients computed in $O(log T)$ parallel time
  3. *Numerical stability*: Log-space computation prevents overflow

  E88's sequential tanh provides:
  1. *Adaptive gradient gating*: Saturated regions get low gradients (intentional for memory)
  2. *Sequential computation*: Gradients require $O(T)$ sequential steps
  3. *Information compression*: Values squeezed into $[-1, 1]$

  For training, uniformity often beats adaptivity.
]

== 13.3 Wall-Clock Efficiency Analysis

=== 13.3.1 Parallelism and Throughput

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,
    align: (left, center, center, center),
    [*Architecture*], [*Temporal Parallelism*], [*Tokens/Second (480M)*], [*Relative*],
    [Mamba2], [$O(log T)$ (scan)], [~8000], [1.0$times$],
    [E88], [$O(T)$ (sequential)], [~2000], [0.25$times$],
    [Transformer], [$O(1)$ (full parallel)], [~3000], [0.38$times$],
  ),
  caption: [Throughput comparison on 4$times$ RTX 6000 Ada GPUs.],
)

#definition("Training Tokens Seen")[
  In a fixed 10-minute training run:

  $ "Tokens"_"Mamba2" approx 8000 times 600 times 256 approx 1.2 times 10^9 $
  $ "Tokens"_"E88" approx 2000 times 600 times 256 approx 0.3 times 10^9 $

  Mamba2 sees 4$times$ more data in the same wall-clock budget.
]

=== 13.3.2 The 4$times$ Data Advantage

Even if E88 were perfectly sample-efficient (learning as much per token as Mamba2), the 4$times$ throughput difference means:

$ "Effective training" prop "Sample efficiency" times "Throughput" $

For E88 to match Mamba2's effective training, it would need to be 4$times$ more sample-efficient---extracting 4$times$ more information per token.

#observation("The Wall-Clock Handicap")[
  In practice, E88's sequential bottleneck means it starts every training comparison at a significant disadvantage. The theoretical expressivity gains must overcome this practical deficit.

  For language modeling---where the extra expressivity may not be exercised---the handicap dominates.
]

== 13.4 When Theory Predicts Practice

The theory-practice gap closes when both conditions hold:

#finding[
  *Condition 1: Task Requires Expressivity*

  The target function must genuinely require capabilities that linear-temporal models lack:
  - Running parity / XOR chains
  - Exact threshold counting
  - State machine simulation with absorbing states
  - Temporal decisions that cannot be approximated

  *Condition 2: Sufficient Training Budget*

  The training budget must be large enough to:
  - Overcome the wall-clock disadvantage
  - Navigate the more complex optimization landscape
  - Find the basin of attraction for the target function
]

=== 13.4.1 Task Analysis

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,
    align: (left, center, center, left),
    [*Task*], [*Requires Temporal Nonlinearity?*], [*Condition 1?*], [*Prediction*],
    [Running parity], [Yes (proven)], [$checkmark$], [E88 wins if trained long enough],
    [Threshold count], [Yes (proven)], [$checkmark$], [E88 wins if trained long enough],
    [Language modeling], [Unclear], [$times$?], [Mamba2 likely wins (faster)],
    [Code execution], [Likely yes], [$checkmark$?], [E88 may win for complex code],
    [Math reasoning], [Likely yes], [$checkmark$?], [E88 may win for multi-step],
  ),
  caption: [Task analysis for theory-practice alignment.],
)

=== 13.4.2 Training Budget Analysis

Consider training to convergence on running parity:

*Mamba2*: Cannot converge to correct solution (provably impossible). Any training budget yields ~50% accuracy (random guessing on each bit).

*E88*: Can converge to correct solution. Requires sufficient budget to:
1. Initialize near correct basin
2. Navigate optimization landscape
3. Fine-tune to high accuracy

#observation("Convergence vs Speed")[
  For tasks requiring temporal nonlinearity:
  - Mamba2 converges _quickly_ to _wrong_ answer
  - E88 converges _slowly_ to _right_ answer

  The question is whether "slowly" fits in the training budget.
]

== 13.5 The Language Modeling Case

=== 13.5.1 What Does Language Modeling Require?

Language modeling loss measures:
$ L = -1/(|cal(D)|) sum_((x,y) in cal(D)) log P(y | x) $

This aggregates over many sub-tasks:
- Syntactic prediction (grammar)
- Semantic coherence (meaning)
- Factual recall (knowledge)
- Reasoning chains (logic)

#observation("Decomposition Hypothesis")[
  Language modeling loss can be decomposed:
  $ L = L_"syntax" + L_"semantic" + L_"factual" + L_"reasoning" + L_"other" $

  The theoretical separation (E88 > Mamba2) primarily affects $L_"reasoning"$ and certain $L_"factual"$ cases.

  If $L_"reasoning"$ is a small fraction of total loss, the separation may be swamped by:
  1. Mamba2's advantage on faster training (lower $L_"syntax"$, $L_"semantic"$ from more data)
  2. Noise in benchmark design
]

=== 13.5.2 Empirical Decomposition

The ablation finding (Section 11) that removing tanh from E88 state has negligible effect on language modeling loss supports this decomposition:

#finding[
  *Ablation Result*: Linear-state E88 $approx$ tanh-state E88 on language modeling.

  *Interpretation*: The tanh's benefit (temporal nonlinearity) is not being exercised by the language modeling objective. Either:
  1. Language modeling doesn't require it, OR
  2. The benchmark doesn't measure it, OR
  3. The model hasn't learned to use it
]

== 13.6 Reconciling Theory and Practice

=== 13.6.1 The Role of Benchmarks

Standard language modeling benchmarks (perplexity on held-out data) measure:
- Average-case performance across many tokens
- Aggregate over diverse sub-tasks
- Reward smooth prediction, not discrete decisions

#observation("Benchmark Bias")[
  Language modeling benchmarks are _biased toward linear-temporal models_:

  1. They reward patterns learnable from local context (where Mamba2 excels)
  2. They don't specifically test temporal nonlinearity
  3. They don't penalize failure on rare but important tokens

  A model that predicts 99% of tokens well but fails on 1% requiring reasoning may have:
  - Excellent perplexity (99% dominates)
  - Terrible downstream performance (1% is critical)
]

=== 13.6.2 Better Benchmarks

To test the theoretical separation, benchmarks should:

1. *Isolate temporal nonlinearity*: Running parity, XOR chains, threshold counting
2. *Measure discrete accuracy*: Not smooth loss, but exact correctness
3. *Control for training budget*: Equal wall-clock time, or equal tokens seen

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,
    align: (left, left, left, left),
    [*Benchmark*], [*Tests*], [*Expected Gap*], [*Current Status*],
    [Running parity], [XOR computation], [E88 >> Mamba2], [Not run],
    [Threshold count], [Discontinuous decisions], [E88 > Mamba2], [Not run],
    [State machine], [Latching / absorbing], [E88 > Mamba2], [Not run],
    [Code execution], [Sequential reasoning], [E88 $>=$ Mamba2?], [Partial data],
    [Perplexity], [Average prediction], [Mamba2 > E88], [Confirmed],
  ),
  caption: [Benchmark coverage for expressivity separation.],
)

== 13.7 Practical Recommendations

=== 13.7.1 Architecture Selection

#finding[
  *For Language Modeling (Current Benchmarks)*:

  Use Mamba2 or similar linear-temporal architectures. The wall-clock efficiency advantage outweighs theoretical expressivity for current benchmarks.

  *For Reasoning-Heavy Tasks*:

  Consider E88 or hybrid architectures. Tasks requiring sequential decisions, exact counting, or state tracking may benefit from temporal nonlinearity.

  *For Maximum Capability*:

  Use hybrid architectures: linear-temporal bulk with nonlinear-temporal heads for reasoning. Route based on input complexity.
]

=== 13.7.2 Training Strategy

When using E88 or similar expressive architectures:

1. *Longer training*: Budget for the wall-clock disadvantage
2. *Targeted initialization*: Initialize near desired behavior if known
3. *Curriculum learning*: Start with simple temporal patterns, increase complexity
4. *Task-specific fine-tuning*: Pre-train with efficient architecture, fine-tune with expressive one

== 13.8 Summary

#figure(
  table(
    columns: 2,
    stroke: 0.5pt,
    align: (left, left),
    [*Aspect*], [*Finding*],
    [Expressivity hierarchy], [E88 $supset$ Mamba2 (proven)],
    [Language modeling loss], [Mamba2 < E88 (observed)],
    [Resolution], [Wall-clock efficiency + task mismatch],
    [Sample efficiency], [Similar when tasks don't require temporal nonlinearity],
    [Wall-clock efficiency], [Mamba2 ~4$times$ faster],
    [When theory wins], [Tasks requiring temporal nonlinearity + sufficient budget],
    [When practice wins], [Aggregate benchmarks + fixed training time],
  ),
  caption: [Summary of theory-practice gap analysis.],
)

The theory-practice gap is not a failure of the theory---it correctly identifies what architectures _can_ compute. The gap arises from:

1. *Benchmark design*: Current benchmarks don't isolate temporal nonlinearity
2. *Training budgets*: Wall-clock efficiency compounds over training
3. *Task distribution*: Natural language may not require the extra expressivity

#block(
  fill: rgb("#f0f7ff"),
  stroke: rgb("#3366cc"),
  inset: 12pt,
  radius: 4pt,
)[
  *The Core Lesson*: Theoretical expressivity is necessary but not sufficient for practical performance. The path from "can compute" to "does learn" requires:

  1. Training dynamics that find the solution
  2. Sufficient data to specialize the solution
  3. Tasks that exercise the extra capability

  When these conditions align, the theoretical hierarchy predicts the empirical ranking. When they don't, faster training wins.
]

