#set document(title: "E79: Coupled Memory-Modulation Matrix System", author: "Elman-Proofs")
#set page(margin: 1in)
#set text(size: 11pt)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

// Custom theorem-like environments
#let theorem(name, body) = block(
  width: 100%,
  inset: 8pt,
  stroke: (left: 2pt + blue),
  [*Theorem* (#name)*.* #body]
)

#let proposition(name, body) = block(
  width: 100%,
  inset: 8pt,
  stroke: (left: 2pt + green),
  [*Proposition* (#name)*.* #body]
)

#let definition(name, body) = block(
  width: 100%,
  inset: 8pt,
  stroke: (left: 2pt + orange),
  [*Definition* (#name)*.* #body]
)

#let proof(body) = block(
  width: 100%,
  inset: (left: 12pt),
  [_Proof._ #body #h(1fr) $square$]
)

#let corollary(body) = block(
  width: 100%,
  inset: 8pt,
  stroke: (left: 2pt + purple),
  [*Corollary.* #body]
)

#let conjecture(body) = block(
  width: 100%,
  inset: 8pt,
  stroke: (left: 2pt + red),
  [*Conjecture.* #body]
)

#let insight(body) = block(
  width: 100%,
  inset: 10pt,
  fill: rgb("#e7f3ff"),
  radius: 4pt,
  [*Insight:* #body]
)

#let warning(body) = block(
  width: 100%,
  inset: 10pt,
  fill: rgb("#fff3cd"),
  radius: 4pt,
  [*Key Point:* #body]
)

#align(center)[
  #text(size: 18pt, weight: "bold")[E79: Coupled Memory-Modulation Matrix System]

  #text(size: 12pt)[A Mathematical Analysis of Mutual Gating Control]

  #v(0.5em)
  #text(size: 10pt, style: "italic")[Formal verification in Lean 4 with Mathlib]
]

#v(1em)

= Introduction

E79 represents the culmination of 79 architectural experiments in recurrent neural network design. Its key innovation is *mutual gating control*: two $n times n$ matrix states where each controls the other's forgetting dynamics.

This document provides:
1. Complete mathematical specification of E79
2. Analysis of how M modulates S (and vice versa)
3. Jacobian and gradient flow analysis
4. Key insights from the Lean formalization
5. Testable predictions and open questions

= Mathematical Specification

== State Definition

E79 maintains two matrix states:

$ bold(S) in RR^(n times n) quad "Content Memory (primary associative storage)" $
$ bold(M) in RR^(n times n) quad "Modulation Memory (controls S's gating)" $

Total state: $2n^2$ real values. For $n = 32$, this is 2048 elements.

== Input Vectors

At each timestep, E79 receives:
- $bold(k) in RR^n$: Key vector for content addressing
- $bold(v) in RR^n$: Value to store
- $bold(q) in RR^n$: Query for output
- $bold(m) in RR^n$: Modulation key for M addressing

== The E79 Update Rule

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [E79 Forward Pass - Mutual Gating Control],
  align(left)[
    *Input:* State $(bold(S), bold(M))$, vectors $(bold(k), bold(v), bold(q), bold(m))$, biases $(bold(b)_S, bold(b)_M)$

    *Step 1: Normalize keys*
    $ hat(bold(k)) = bold(k) / norm(bold(k))_2, quad hat(bold(m)) = bold(m) / norm(bold(m))_2 $

    *Step 2: M controls S's decay gates* #text(fill: red)[(M → S coupling)]
    $ bold(g)^S_"row" &= sigma(bold(M) hat(bold(k)) + bold(b)_S) in (0,1)^n $ <eq:s_row_gate>
    $ bold(g)^S_"col" &= sigma(bold(M)^top hat(bold(k)) + bold(b)_S) in (0,1)^n $ <eq:s_col_gate>

    *Step 3: S delta rule update with M-controlled gating*
    $ bold(delta)_S &= bold(v) - bold(S) hat(bold(k)) $
    $ bold(S)' &= (bold(g)^S_"row" bold(g)^S_"col"{}^top) dot.circle bold(S) + bold(delta)_S hat(bold(k))^top $ <eq:S_update>

    *Step 4: S controls M's decay gates* #text(fill: blue)[(S → M coupling)]
    $ bold(g)^M_"row" &= sigma(bold(S) hat(bold(m)) + bold(b)_M) $
    $ bold(g)^M_"col" &= sigma(bold(S)^top hat(bold(m)) + bold(b)_M) $

    *Step 5: M delta rule update (M predicts S's changes)*
    $ bold(delta)_M &= bold(delta)_S - bold(M) hat(bold(m)) $
    $ bold(M)' &= (bold(g)^M_"row" bold(g)^M_"col"{}^top) dot.circle bold(M) + bold(delta)_M hat(bold(m))^top $ <eq:M_update>

    *Step 6: Output with self-gating*
    $ bold(o) = (bold(S)' bold(q)) dot.circle "silu"(bold(S)' bold(q)) $

    *Return:* New state $(bold(S)', bold(M)')$, output $bold(o)$
  ]
)

= Key Insight 1: Factorized Gating is Rank-Deficient Control

== The Factorized Gate Structure

The decay applied to S has the form:
$ "Gate"_(i j) = g^S_("row",i) times g^S_("col",j) $

This is a *rank-1 outer product*:
$ bold(G)^S = bold(g)^S_"row" (bold(g)^S_"col")^top in RR^(n times n) $

#theorem[Rank Deficiency][
  The factorized gate $bold(G)^S = bold(g)^S_"row" (bold(g)^S_"col")^top$ has rank at most 1.

  This means *$2n$ parameters control $n^2$ decay rates*.
]

#proof[
  Any outer product $bold(u) bold(v)^top$ has rank $<= 1$ since all columns are scalar multiples of $bold(u)$.
]

== Consequences of Rank Deficiency

#warning[
  You cannot independently control each element's decay. If row $i$ decays quickly ($g^S_("row",i)$ small), then *all elements in row $i$* decay quickly, regardless of column.
]

This constraint explains why E79 needs *two* coupled matrices:
- Single matrix with factorized gating has limited expressiveness
- The coupling between S and M compensates for each other's rank deficiency
- M can modulate S's gating to achieve richer decay patterns than either could alone

#proposition[Effective Degrees of Freedom][
  The factorized gate has $2n - 1$ effective degrees of freedom (not $2n$, due to the constraint that scaling $bold(g)_"row"$ by $c$ and $bold(g)_"col"$ by $1/c$ gives the same result).

  Compare to full gating: $n^2$ degrees of freedom.

  The ratio: $(2n-1) / n^2 approx 2/n$ for large $n$.
]

= Key Insight 2: Bidirectional Jacobian Coupling

== The Jacobian is NOT Lower-Triangular

#warning[
  Unlike the simplified description, the actual E79 Jacobian is *fully coupled* in both directions.
]

The full E79 state is $bold(z) = "vec"([bold(S); bold(M)]) in RR^(2n^2)$.

#theorem[Bidirectional Coupling][
  The Jacobian of the E79 update has the block structure:
  $ (diff bold(z)') / (diff bold(z)) = mat(bold(J)_(S S), bold(J)_(S M); bold(J)_(M S), bold(J)_(M M)) $
  where *both off-diagonal blocks are non-zero*:
  - $bold(J)_(S M) = (diff bold(S)') / (diff bold(M)) != bold(0)$: M affects S' through gating
  - $bold(J)_(M S) = (diff bold(M)') / (diff bold(S)) != bold(0)$: S affects M' through gating AND $bold(delta)_S$
]

#proof[
  *M → S coupling:* From @eq:S_update, the gates $bold(g)^S_"row", bold(g)^S_"col"$ depend on M:
  $ bold(g)^S_"row" = sigma(bold(M) hat(bold(k)) + bold(b)_S) $
  Therefore:
  $ (diff bold(S)') / (diff bold(M)) = (diff bold(S)') / (diff bold(g)^S) dot (diff bold(g)^S) / (diff bold(M)) != bold(0) $

  *S → M coupling:* From @eq:M_update, the gates $bold(g)^M_"row", bold(g)^M_"col"$ depend on S, and $bold(delta)_M$ depends on $bold(delta)_S$ which depends on S:
  $ (diff bold(M)') / (diff bold(S)) != bold(0) $
]

== Dynamical Systems Interpretation

#insight[
  E79 is a *fully coupled nonlinear dynamical system*, not a hierarchical cascade. The two matrices co-evolve and mutually regulate each other's dynamics.

  This is qualitatively similar to:
  - *Lotka-Volterra equations* (predator-prey dynamics)
  - *Coupled oscillators* in physics
  - *Mutual inhibition circuits* in neuroscience
]

= Key Insight 3: Gradient Flow Analysis

== How M Gets Gradients

#theorem[M Gradient Path][
  M influences the output through the gating path:
  $ "Loss" arrow bold(o) arrow bold(S)' arrow bold(g)^S_"row", bold(g)^S_"col" arrow bold(M) $

  The gradient:
  $ (partial cal(L)) / (partial bold(M)) = (partial cal(L)) / (partial bold(S)') dot (partial bold(S)') / (partial bold(g)^S) dot (partial bold(g)^S) / (partial bold(M)) $
]

#proof[
  From @eq:S_update: $S'_(i j) = g^S_("row",i) dot g^S_("col",j) dot S_(i j) + (delta_S)_i hat(k)_j$

  The gradient with respect to $g^S_("row",i)$:
  $ (partial S'_(i j)) / (partial g^S_("row",i)) = g^S_("col",j) dot S_(i j) $

  And $g^S_("row",i) = sigma(sum_l M_(i l) hat(k)_l + (b_S)_i)$, so:
  $ (partial g^S_("row",i)) / (partial M_(i l)) = sigma'(...) dot hat(k)_l $

  Composing via chain rule yields a non-zero path from Loss to M.
]

== Meta-Learning Interpretation

#insight[
  M receives gradients that encode: *"If you had gated S differently, the output would have been better."*

  This is *implicit meta-learning* --- M learns to control S's forgetting based on task loss, without explicit meta-supervision.
]

= Key Insight 4: Tied Keys Collapse the System

== The Tied Keys Theorem

#theorem[Tied Keys Reduction][
  If $bold(m) = bold(k)$ everywhere during training, then E79's expressive power collapses toward a single matrix.

  Specifically, M cannot organize independently from S when using the same addressing.
]

#proof[
  With $hat(bold(m)) = hat(bold(k))$, both matrices are updated and queried with the same key.

  The residual:
  $ bold(delta)_M = bold(v) - bold(S) hat(bold(k)) - bold(M) hat(bold(k)) = bold(v) - (bold(S) + bold(M)) hat(bold(k)) $

  The combined retrieval $(bold(S) + bold(M)) hat(bold(k))$ acts like a single matrix.
]

#warning[
  The separate modulation key $bold(m)$ is *essential* for E79 to be more than a single larger matrix.

  *Testable prediction:* If trained weights satisfy $bold(W)_m approx bold(W)_k$, then E79 is not utilizing its full capacity.
]

= Key Insight 5: State Efficiency vs Attention

== State Size Comparison

#table(
  columns: (auto, auto, auto, auto),
  align: (left, center, center, left),
  [*Model*], [*State Size*], [*Per-Step Cost*], [*Scaling*],
  [E79], [$2n^2$], [$O(n^2)$], [Fixed],
  [Attention], [$T times d$], [$O(T^2 d)$], [Grows with $T$],
  [E1 (vector)], [$n$], [$O(n d)$], [Fixed],
)

#theorem[Crossover Point][
  E79 uses less memory than attention when sequence length $T$ exceeds:
  $ T > (2n^2) / d $

  For $n = 32$, $d = 512$: crossover at $T > 4$.

  E79 compresses arbitrarily long sequences into fixed $2n^2$ state.
]

== The Compression Tradeoff

#insight[
  E79 trades *sequence-length scaling* for *fixed-size compression*.

  - Attention: Full context access, $O(T^2)$ cost
  - E79: Compressed context, $O(1)$ state but lossy

  E79's mutual gating helps determine *what to keep* in the limited state budget.
]

= Key Insight 6: K-Level Generalization

== The K-Level Hierarchy

#definition[K-Level Coupled Memory][
  For $K >= 1$, define matrices $bold(M)_0, bold(M)_1, ..., bold(M)_(K-1) in RR^(n times n)$:

  $ bold(r)_0 &= bold(v) - bold(M)_0 hat(bold(k))_0 quad "(Level 0 residual)" $
  $ bold(r)_i &= bold(r)_(i-1) - bold(M)_i hat(bold(k))_i quad "for" i = 1, ..., K-1 $

  Each level learns the residual of the previous level.
]

#table(
  columns: (auto, 1fr),
  align: (center, left),
  [$K$], [*Description*],
  [$1$], [Standard delta rule (E74). Single matrix.],
  [$2$], [E79. S + M with mutual gating.],
  [$3$], [Triple hierarchy. S + M + N.],
  [$K$], [Chain of $K$ mutually-gated residual predictors.],
)

== Diminishing Returns

#theorem[Residual Decay][
  If level $i$ converges (learns to predict $bold(r)_(i-1)$ well), then:
  $ norm(bold(r)_i) << norm(bold(r)_(i-1)) $

  Each additional level has diminishing marginal benefit.
]

#conjecture[
  There exists an optimal $K^*$ that depends on:
  1. *Task complexity*: Structure in residuals
  2. *Training time*: Deeper hierarchies converge slower
  3. *Compute budget*: Each level costs $n^2$ parameters and $O(n^2)$ compute

  The benchmark showing $n = 32$ optimal for 10-minute training suggests $K = 2$ is near-optimal for that regime.
]

= Testable Predictions

The formalization yields several experimentally testable predictions:

== Prediction 1: Key Divergence

*Measure:* $norm(bold(W)_m - bold(W)_k)_F / norm(bold(W)_k)_F$

*Expected:* This should be significantly positive (> 0.1) if E79 is utilizing both matrices effectively.

*If violated:* E79 has collapsed to approximately a single larger matrix.

== Prediction 2: Gate Utilization

*Measure:* Variance of $bold(g)^S_"row"$ and $bold(g)^S_"col"$ across inputs.

*Expected:* High variance indicates M is actively controlling S's forgetting.

*If violated:* Gates are near-constant, reducing to fixed decay.

== Prediction 3: Residual Decay Over Training

*Measure:* $norm(bold(delta)_M) / norm(bold(delta)_S)$ over training.

*Expected:* Should decrease if M learns to predict S's errors.

*If violated:* M is not learning useful residual structure.

== Prediction 4: Jacobian Spectral Radius

*Measure:* Largest eigenvalue magnitude of the coupled Jacobian.

*Expected:* Should be < 1 for stability.

*If violated:* Risk of gradient explosion or state divergence.

= Comparison to Related Architectures

#table(
  columns: (auto, auto, auto, auto),
  align: (left, left, left, left),
  [*Architecture*], [*Coupling*], [*Gating*], [*State*],
  [LSTM], [Hierarchical (cell/hidden)], [Input-dependent], [Vector],
  [Transformer], [None (parallel)], [Attention weights], [KV cache],
  [Mamba/SSM], [None], [Input-dependent], [Diagonal matrix],
  [E79], [*Mutual (bidirectional)*], [*Cross-matrix*], [*Full matrices*],
)

#insight[
  E79 is unique in having *bidirectional mutual control* between memory systems. This is more like biological neural circuits (e.g., cortical-thalamic loops, hippocampal-prefrontal interactions) where populations mutually regulate each other.
]

= Empirical Results Summary

From the benchmark (100M params, 10-minute training):

#table(
  columns: (auto, auto, auto, auto),
  align: (left, center, center, center),
  [*Model*], [*Loss*], [*tok/s*], [*State*],
  [Mamba2], [1.27], [78.7K], [SSM (parallel)],
  [*E79 n=32*], [*1.51*], [*31.5K*], [*2×32² = 2048*],
  [E1 (gated)], [1.53], [45.5K], [vector],
  [E42 (linear)], [1.59], [137K], [vector],
  [FLA-GDN], [1.99], [18.7K], [matrix],
)

Key observations:
- E79 beats E1 (1.51 vs 1.53): mutual gating helps
- n=32 optimal for 10-min training (larger n under-converged)
- 40% of Mamba2 throughput despite sequential scan

= Summary of Formalization Insights

#table(
  columns: (auto, 1fr),
  align: (center, left),
  [*\#*], [*Insight*],
  [1], [*Factorized gating is rank-deficient*: 2n params control n² decays. The coupling compensates.],
  [2], [*Jacobian is bidirectionally coupled*: Not hierarchical---true mutual control.],
  [3], [*Gradient flow enables meta-learning*: M learns "how to gate S" from task loss.],
  [4], [*Tied keys collapse the system*: Separate $bold(m) != bold(k)$ is essential.],
  [5], [*Fixed state beats attention for long sequences*: Crossover at $T > 2n^2/d$.],
  [6], [*K-level hierarchies have diminishing returns*: K=2 may be near-optimal.],
  [7], [*Mutual control resembles biological circuits*: Lotka-Volterra / coupled oscillator dynamics.],
)

= Open Questions

1. *Optimal K for K-level hierarchies*: Is K=2 optimal, or would K=3 help for harder tasks?

2. *Adaptive coupling*: Can we learn the coupling structure rather than hard-coding it?

3. *Parallel scan for coupled matrices*: Can we achieve Mamba2-like parallelism?

4. *Scaling laws*: How does E79 scale beyond 100M parameters?

5. *Biological analogs*: Are there neural circuits with similar mutual gating dynamics?

6. *Formal stability analysis*: Under what conditions is the coupled system guaranteed stable?

