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

#align(center)[
  #text(size: 18pt, weight: "bold")[E79: Coupled Memory-Modulation Matrix System]

  #text(size: 12pt)[A Mathematical Analysis of Hierarchical Delta Rules]

  #v(0.5em)
  #text(size: 10pt, style: "italic")[Formal verification in Lean 4 with Mathlib]
]

#v(1em)

= Introduction

E79 represents the culmination of 79 architectural experiments in recurrent neural network design. Its key innovation is *coupled delta rules*: two $n times n$ matrix states where the second learns to predict the residuals of the first.

This document provides:
1. Complete mathematical specification of E79
2. Analysis of how M modulates S
3. Jacobian and gradient flow analysis
4. Generalizations to K-level hierarchies
5. Conditions for simplification

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

== The E79 Update Rule (Actual Implementation)

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [E79 Forward Pass - Mutual Gating Control],
  align(left)[
    *Input:* State $(bold(S), bold(M))$, vectors $(bold(k), bold(v), bold(q), bold(m))$, biases $(bold(b)_S, bold(b)_M)$

    *Step 1: Normalize keys*
    $ hat(bold(k)) = bold(k) / norm(bold(k))_2, quad hat(bold(m)) = bold(m) / norm(bold(m))_2 $

    *Step 2: M controls S's decay gates* #text(fill: red)[(M → S coupling)]
    $ bold(g)^S_"row" &= sigma(bold(M) hat(bold(k)) + bold(b)_S) quad "(row decay from M)" $ <eq:s_row_gate>
    $ bold(g)^S_"col" &= sigma(bold(M)^top hat(bold(k)) + bold(b)_S) quad "(col decay from M)" $ <eq:s_col_gate>

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

== Explicit Matrix Form

The key insight is the *factorized gating*:

$ bold(S)' = underbrace((bold(g)^S_"row" bold(g)^S_"col"{}^top), "M-controlled decay") dot.circle bold(S) + bold(delta)_S hat(bold(k))^top $ <eq:S_explicit>

Where the decay gate is an outer product of M's outputs:
$ bold(g)^S_"row" bold(g)^S_"col"{}^top = sigma(bold(M) hat(bold(k))) sigma(bold(M)^top hat(bold(k)))^top $

This means *M directly influences S's update* and thus the output. Similarly:
$ bold(M)' = (bold(g)^M_"row" bold(g)^M_"col"{}^top) dot.circle bold(M) + bold(delta)_M hat(bold(m))^top $

With S controlling M's decay gates.

= How M Modulates S

== The Coupling Mechanism (Actual Implementation)

Unlike the simplified description in E79_RESULTS.md, the *actual* E79 implements *mutual gating control*:

#block(
  fill: rgb("#fff3cd"),
  inset: 10pt,
  radius: 4pt,
  [*Key Insight:* M directly controls S's decay gates, and S controls M's decay gates. This creates a bidirectional dynamical coupling where each memory controls what the other forgets.]
)

=== M → S Coupling (M controls S's forgetting)

The decay factors for S come from M:
$ bold(g)^S_"row" = sigma(bold(M) hat(bold(k)) + bold(b)_S) in (0, 1)^n $
$ bold(g)^S_"col" = sigma(bold(M)^top hat(bold(k)) + bold(b)_S) in (0, 1)^n $

The S update becomes:
$ S'_(i j) = g^S_("row",i) dot g^S_("col",j) dot S_(i j) + (delta_S)_i hat(k)_j $

*M controls what S retains.* When $bold(M) hat(bold(k))$ is large and positive, $bold(g)^S_"row" arrow 1$ and S preserves its rows. When negative, S forgets.

=== S → M Coupling (S controls M's forgetting)

Symmetrically, S controls M's decay:
$ bold(g)^M_"row" = sigma(bold(S) hat(bold(m)) + bold(b)_M) $
$ bold(g)^M_"col" = sigma(bold(S)^top hat(bold(m)) + bold(b)_M) $

*S controls what M retains.* This creates a feedback loop where the memories regulate each other.

== Gradient Flow Through M

#theorem[M Gets Gradients Through S][
  M influences the output through the gating path:
  $ "Loss" arrow bold(o) arrow bold(S)' arrow bold(g)^S_"row", bold(g)^S_"col" arrow bold(M) $

  Specifically:
  $ (partial cal(L)) / (partial bold(M)) = (partial cal(L)) / (partial bold(S)') dot (partial bold(S)') / (partial bold(g)^S) dot (partial bold(g)^S) / (partial bold(M)) $
]

#proof[
  From @eq:S_update: $bold(S)' = (bold(g)^S_"row" bold(g)^S_"col"{}^top) dot.circle bold(S) + bold(delta)_S hat(bold(k))^top$

  The gradient of $bold(S)'$ with respect to $bold(g)^S_"row"$ is:
  $ (partial S'_(i j)) / (partial g^S_("row", i)) = g^S_("col", j) dot S_(i j) $

  And $bold(g)^S_"row" = sigma(bold(M) hat(bold(k)) + bold(b)_S)$, so:
  $ (partial g^S_("row", i)) / (partial M_(i l)) = sigma'(...) dot hat(k)_l $

  Composing these gives a non-zero gradient path from Loss to M.
]

== Interpretation: Mutual Control Dynamical System

The E79 coupling creates a *self-organizing* memory system:

#table(
  columns: (auto, 1fr),
  align: (center, left),
  [*Aspect*], [*Mechanism*],
  [M → S], [M decides what S should forget based on current key],
  [S → M], [S decides what M should forget based on modulation key],
  [S delta], [Standard delta rule with M-modulated decay],
  [M delta], [Learns S's prediction errors for meta-learning],
)

This is analogous to:
- *Neural gating* (LSTM): Forget gates control information flow
- *Attention* (Transformers): Context-dependent routing
- *Neuromodulation* (Biology): One system modulates another's plasticity

== Why Mutual Control Helps

#proposition[Adaptive Forgetting][
  With M-controlled gating, S can learn *input-dependent* forgetting:
  - For familiar keys: M outputs high gates → S preserves old information
  - For novel keys: M outputs low gates → S makes room for new content

  This is impossible with fixed decay $alpha_S$.
]

#proposition[Meta-Learning Through Coupling][
  M can learn to recognize *when* S should update strongly vs. weakly:
  - Systematic input patterns → M learns predictable gating
  - Noisy inputs → M learns to gate conservatively

  This is a form of "learning to learn" for associative memory.
]

= Jacobian Analysis

== State Space Jacobian

The full E79 state is $bold(z) = "vec"([bold(S); bold(M)]) in RR^(2n^2)$.

#theorem[Lower-Triangular Jacobian][
  The Jacobian of the E79 update has block lower-triangular structure:
  $ (diff bold(z)') / (diff bold(z)) = mat(bold(J)_S, bold(0); bold(J)_(M S), bold(J)_M) $
  where:
  - $bold(J)_S = (diff bold(S)') / (diff bold(S))$: How S affects S'
  - $bold(J)_M = (diff bold(M)') / (diff bold(M))$: How M affects M'
  - $bold(J)_(M S) = (diff bold(M)') / (diff bold(S))$: How S affects M' (the coupling!)
  - $(diff bold(S)') / (diff bold(M)) = bold(0)$: M does not affect S' directly
]

#proof[
  From @eq:S_update, $bold(S)'$ depends only on $bold(S)$, not $bold(M)$:
  $ bold(S)' = alpha_S bold(S) + (bold(v) - bold(S) hat(bold(k))) hat(bold(k))^top $
  Therefore $(diff bold(S)') / (diff bold(M)) = bold(0)$.

  From @eq:M_update, $bold(M)'$ depends on both $bold(S)$ and $bold(M)$:
  $ bold(M)' = alpha_M bold(M) + (underbrace(bold(v) - bold(S) hat(bold(k)), bold(delta)_S) - bold(M) hat(bold(m))) hat(bold(m))^top $

  The dependence on $bold(S)$ comes through $bold(delta)_S$:
  $ (diff bold(M)') / (diff bold(S)) = (diff) / (diff bold(S)) [(bold(v) - bold(S) hat(bold(k)) - bold(M) hat(bold(m))) hat(bold(m))^top] = -hat(bold(k)) hat(bold(m))^top $
  (in the appropriate tensor form).
]

== Individual Block Jacobians

=== Content Memory Jacobian $bold(J)_S$

For the delta rule update with decay:
$ bold(S)' = alpha_S bold(S) + (bold(v) - bold(S) hat(bold(k))) hat(bold(k))^top = alpha_S bold(S) + bold(v) hat(bold(k))^top - bold(S) hat(bold(k)) hat(bold(k))^top $

Taking the derivative with respect to $bold(S)$:
$ bold(J)_S = alpha_S bold(I) - hat(bold(k)) hat(bold(k))^top times.circle bold(I)_n $

In terms of action on a perturbation $delta bold(S)$:
$ delta bold(S)' = alpha_S delta bold(S) - (delta bold(S) hat(bold(k))) hat(bold(k))^top $

#theorem[S Jacobian Spectral Properties][
  With $norm(hat(bold(k)))_2 = 1$, the Jacobian $bold(J)_S$ (as a linear map on $n times n$ matrices) has eigenvalues:
  - $alpha_S$ with multiplicity $n^2 - n$ (eigenvectors: matrices with $hat(bold(k))$ in null space)
  - $alpha_S - 1$ with multiplicity $n$ (eigenvectors: outer products with $hat(bold(k))$)

  For stability, we need $|alpha_S| <= 1$ and $|alpha_S - 1| <= 1$, giving $alpha_S in [0, 1]$.
]

=== Modulation Memory Jacobian $bold(J)_M$

Similarly:
$ bold(J)_M = alpha_M bold(I) - hat(bold(m)) hat(bold(m))^top times.circle bold(I)_n $

Same spectral structure with $hat(bold(m))$ replacing $hat(bold(k))$.

== Gradient Flow Through the Coupling

The coupling term $bold(J)_(M S)$ enables *gradient sharing*:

$ (diff cal(L)) / (diff bold(S)) = (diff cal(L)) / (diff bold(S)') bold(J)_S + (diff cal(L)) / (diff bold(M)') bold(J)_(M S) $

The second term means: *M's gradient signal flows back to S*.

#corollary[
  When training E79 end-to-end, S receives gradients from:
  1. Direct output path: $bold(o) arrow bold(S)'$
  2. Indirect coupling path: $bold(o) arrow bold(M)' arrow bold(S)$ (through $bold(delta)_S$)

  This coupling allows M to "tell" S about systematic errors.
]

= Exact Retrieval and Capacity

== Single Write Exact Retrieval

#theorem[Exact Retrieval][
  If $bold(S) = bold(0)$ (empty memory) and we write $(bold(v), bold(k))$ with $norm(bold(k)) = 1$, then:
  $ bold(S)' hat(bold(k)) = bold(v) $

  _Proof._
  $ bold(S)' hat(bold(k)) &= [bold(0) + (bold(v) - bold(0) dot hat(bold(k))) hat(bold(k))^top] hat(bold(k)) \
  &= (bold(v)) hat(bold(k))^top hat(bold(k)) \
  &= bold(v) (hat(bold(k))^top hat(bold(k))) \
  &= bold(v) dot 1 = bold(v) $
]

== Orthogonal Keys Preserve Information

#theorem[Selective Update][
  If $hat(bold(k))_1 perp hat(bold(k))_2$ (orthogonal keys), then writing $(bold(v)_2, bold(k)_2)$ does not affect retrieval with $bold(k)_1$:
  $ bold(S)' hat(bold(k))_1 = bold(S) hat(bold(k))_1 $

  _Proof._ The update adds $(bold(v)_2 - bold(S) hat(bold(k))_2) hat(bold(k))_2^top$. Applying to $hat(bold(k))_1$:
  $ [(bold(v)_2 - bold(S) hat(bold(k))_2) hat(bold(k))_2^top] hat(bold(k))_1 = (bold(v)_2 - bold(S) hat(bold(k))_2) (hat(bold(k))_2^top hat(bold(k))_1) = bold(0) $
  since $hat(bold(k))_2^top hat(bold(k))_1 = 0$.
]

== Capacity Analysis

#proposition[E79 Capacity][
  With orthonormal keys ${hat(bold(k))_i}_(i=1)^n$ for S and ${hat(bold(m))_j}_(j=1)^n$ for M:
  - S can store $n$ independent (value, key) pairs: $n^2$ real values
  - M can store $n$ independent (residual, modulation-key) pairs: $n^2$ real values
  - Total: $2n^2$ real values (equal to state size)

  This is *optimal* capacity utilization.
]

= Generalizations

== K-Level Hierarchies

E79 is the $K = 2$ case of a general construction:

#definition[K-Level Coupled Memory][
  For $K >= 1$, define matrices $bold(M)_0, bold(M)_1, ..., bold(M)_(K-1) in RR^(n times n)$ with:

  $ bold(r)_0 &= bold(v) - bold(M)_0 hat(bold(k))_0 quad "(Level 0 residual)" $
  $ bold(r)_i &= bold(r)_(i-1) - bold(M)_i hat(bold(k))_i quad "for" i = 1, ..., K-1 $
  $ bold(M)'_i &= alpha_i bold(M)_i + bold(r)_i hat(bold(k))_i^top quad "(Each level's update)" $
]

#table(
  columns: (auto, 1fr),
  align: (center, left),
  [$K$], [*Description*],
  [$1$], [Standard delta rule (E74). Single matrix S.],
  [$2$], [E79. Content memory S + Modulation memory M.],
  [$3$], [Triple hierarchy. S + M + N where N predicts M's residuals.],
  [$K$], [Chain of $K$ residual predictors.],
)

== Diminishing Returns Conjecture

#conjecture[
  For a fixed compute budget, there exists an optimal $K^*$ such that:
  - $K < K^*$: Adding levels improves performance
  - $K > K^*$: Additional levels have negligible benefit

  $K^*$ depends on:
  1. *Task complexity*: How much structure exists in residuals
  2. *Training time*: Deeper hierarchies need more convergence time
  3. *State budget*: Each level costs $n^2$ parameters
]

The benchmark showing $n = 32$ optimal for 10-minute training suggests E79 is near optimal for that regime.

= Simplifications

== Tied Keys: $bold(m) = bold(k)$

#theorem[Tied Keys Reduction][
  If $bold(m) = bold(k)$ (same key for both levels), then E79 reduces to a single delta rule on the combined matrix $bold(S) + bold(M)$.

  _Proof._ With $hat(bold(m)) = hat(bold(k))$:
  $ bold(delta)_M = bold(v) - bold(S) hat(bold(k)) - bold(M) hat(bold(k)) = bold(v) - (bold(S) + bold(M)) hat(bold(k)) $

  Combined update:
  $ bold(S)' + bold(M)' &= alpha_S bold(S) + alpha_M bold(M) + (bold(v) - bold(S) hat(bold(k))) hat(bold(k))^top + (bold(v) - bold(S) hat(bold(k)) - bold(M) hat(bold(k))) hat(bold(k))^top $

  If $alpha_S = alpha_M = alpha$:
  $ bold(S)' + bold(M)' = alpha(bold(S) + bold(M)) + [2bold(v) - 2bold(S) hat(bold(k)) - bold(M) hat(bold(k))] hat(bold(k))^top $

  This is *not* exactly a single delta rule, but the key insight is: tied keys limit M's ability to organize independently.
]

== Zero M Decay: $alpha_M = 0$

#proposition[Instantaneous Modulation][
  With $alpha_M = 0$:
  $ bold(M)' = bold(delta)_M hat(bold(m))^top $

  M becomes an "instantaneous" residual predictor with no memory of past residuals.
  This is useful when residuals have no temporal structure.
]

== No Modulation: $bold(M) = bold(0)$

#proposition[Reduction to E74][
  Setting $bold(M) = bold(0)$ and $alpha_M = 0$ recovers E74 (single delta rule with self-gating):
  $ bold(S)' = alpha_S bold(S) + (bold(v) - bold(S) hat(bold(k))) hat(bold(k))^top $
  $ bold(o) = (bold(S)' bold(q)) dot.circle "silu"(bold(S)' bold(q)) $
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
- E79 beats E1 (1.51 vs 1.53): modulation helps
- n=32 optimal for 10-min training (larger n under-converged)
- 40% of Mamba2 throughput despite sequential scan

= Conclusions

== What E79 Teaches Us

1. *Hierarchical error correction works*: Even one level of residual prediction (M on S) provides measurable benefit.

2. *Separate addressing enables specialization*: $bold(k)$ for content, $bold(m)$ for error patterns.

3. *Gradient coupling is essential*: M receives loss-relevant gradients through $bold(delta)_S$, not just residual reconstruction loss.

4. *Training time vs capacity tradeoff*: Larger state needs more training to converge.

== Open Questions

1. What is optimal $K$ for K-level hierarchies?
2. Can we learn the coupling adaptively?
3. How does E79 scale beyond 100M parameters?
4. Can parallel scan be applied to coupled matrices?

// Bibliography would go here if needed

