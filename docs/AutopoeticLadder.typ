#set document(title: "The Autopoietic Ladder: Self-Modulating Memory Architectures", author: "Elman-Proofs")
#set page(margin: 1in)
#set text(size: 11pt)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

#let definition(name, body) = block(
  width: 100%,
  inset: 8pt,
  stroke: (left: 2pt + orange),
  [*Definition* (#name)*.* #body]
)

#let insight(body) = block(
  width: 100%,
  inset: 10pt,
  fill: rgb("#e7f3ff"),
  radius: 4pt,
  [*Insight:* #body]
)

#let architecture(name, body) = block(
  width: 100%,
  inset: 10pt,
  fill: rgb("#f0f0f0"),
  radius: 4pt,
  stroke: 1pt + gray,
  [*#name* \ #body]
)

#align(center)[
  #text(size: 18pt, weight: "bold")[The Autopoietic Ladder]

  #text(size: 14pt)[Self-Modulating Memory Architectures]

  #v(0.5em)
  #text(size: 10pt, style: "italic")[From Fixed Decay to Pure Self-Reference]
]

#v(1em)

= Introduction

The term *autopoiesis* (from Greek: self-creation) describes systems that produce and maintain themselves. In the context of recurrent neural networks, we ask: *how can a memory system modulate its own dynamics?*

This document presents a hierarchy of increasingly autopoietic architectures, each building on the last. The central question: how far can we push the principle of self-modulation while maintaining trainability and efficiency?

= The Hierarchy of Automodulation

We present eight levels of automodulation, from fixed decay to continuous self-referential dynamics.

== Level 0: Fixed Gating (E74)

#architecture[E74: Fixed Decay Delta Rule][
  $ bold(S)' = alpha dot bold(S) + (bold(v) - bold(S) hat(bold(k))) hat(bold(k))^top $

  - $alpha in (0, 1)$: Fixed scalar decay
  - No automodulation
  - Decay is a hyperparameter, not learned
]

*Properties:*
- Simplest possible delta rule
- Decay cannot adapt to input or state
- Baseline for comparison

== Level 1: Vector Gating (E75)

#architecture[E75: Input-Dependent Vector Gate][
  $ bold(g) = sigma(bold(W)_beta bold(x) + bold(b)_beta) in (0,1)^n $
  $ bold(S)' = "diag"(bold(g)) dot bold(S) + (bold(v) - bold(S) hat(bold(k))) hat(bold(k))^top $

  - Per-row decay controlled by input
  - $n$ degrees of freedom in gating
  - External modulation (input → gate)
]

*Properties:*
- Input-dependent forgetting
- Still no state-dependence in gating
- Gate is "open loop" — doesn't see what S contains

== Level 2: Cross-Matrix Gating, Rank-1 (E79)

#architecture[E79: Mutual Rank-1 Gating][
  $ bold(g)^S_"row" = sigma(bold(M) hat(bold(k)) + bold(b)_S), quad bold(g)^S_"col" = sigma(bold(M)^top hat(bold(k)) + bold(b)_S) $
  $ bold(S)' = (bold(g)^S_"row" (bold(g)^S_"col")^top) dot.circle bold(S) + bold(delta)_S hat(bold(k))^top $

  Symmetrically, S gates M:
  $ bold(g)^M_"row" = sigma(bold(S) hat(bold(m)) + bold(b)_M), quad bold(g)^M_"col" = sigma(bold(S)^top hat(bold(m)) + bold(b)_M) $
  $ bold(M)' = (bold(g)^M_"row" (bold(g)^M_"col")^top) dot.circle bold(M) + bold(delta)_M hat(bold(m))^top $
]

*Properties:*
- State-dependent gating (M sees S, S sees M)
- Mutual modulation — bidirectional coupling
- *Constraint:* Gate is rank-1 (outer product of two vectors)
- $2n$ parameters control $n^2$ decay rates

#insight[
  The rank-1 constraint means rows and columns cannot be gated independently. If row $i$ decays, ALL of row $i$ decays regardless of column.
]

== Level 3: Cross-Matrix Gating, Full Rank (E80)

#architecture[E80: Full-Rank Mutual Gating][
  $ bold(G)^S = sigma(bold(M) + "outer"(bold(M) hat(bold(k)), hat(bold(k))) + bold(B)_S) in (0,1)^(n times n) $
  $ bold(S)' = bold(G)^S dot.circle bold(S) + bold(delta)_S hat(bold(k))^top $

  The gate $bold(G)^S$ is a full $n times n$ matrix, not rank-1.

  Symmetrically for M:
  $ bold(G)^M = sigma(bold(S) + "outer"(bold(S) hat(bold(m)), hat(bold(m))) + bold(B)_M) $
  $ bold(M)' = bold(G)^M dot.circle bold(M) + bold(delta)_M hat(bold(m))^top $
]

*Properties:*
- Full $n^2$ degrees of freedom in gating
- Each element $(i,j)$ can have independent decay
- The gate is computed FROM the other matrix but is not itself a hidden state

*Variation — Rank-r Gating:*
$ bold(G) = sigma(sum_(i=1)^r bold(u)_i bold(v)_i^top) $

This interpolates between rank-1 ($r=1$, E79) and full-rank ($r=n$, E80).

== Level 4: Gate Matrix as Hidden State (E81)

#architecture[E81: Evolving Gate Matrix][
  Two hidden states: $bold(S)$ (content) and $bold(G)$ (gate), both $n times n$.

  $ bold(S)' = sigma(bold(G)) dot.circle bold(S) + bold(delta)_S hat(bold(k))^top $
  $ bold(G)' = sigma(bold(S)) dot.circle bold(G) + bold(delta)_G hat(bold(m))^top $

  where $bold(delta)_G = bold(delta)_S - bold(G) hat(bold(m))$ (G learns to predict S's changes).
]

*Properties:*
- The gate itself is a hidden state that evolves over time
- G has memory — it accumulates information about good gating strategies
- Mutual modulation: S gates G, G gates S
- Both matrices use delta rule updates

#insight[
  In E81, the gate is not just computed — it is *learned online* as a hidden state. G develops a "theory" of when S should forget.
]

== Level 5: Self-Gating Matrix (E82)

#architecture[E82: Pure Self-Modulation][
  Single matrix $bold(S) in RR^(n times n)$ that gates itself:

  $ bold(G) = sigma(bold(S) hat(bold(m)) hat(bold(k))^top + alpha dot bold(S)) $
  $ bold(S)' = bold(G) dot.circle bold(S) + bold(delta)_S hat(bold(k))^top $

  The gate is computed from S itself — no separate modulation matrix.
]

*Properties:*
- Minimal architecture: single matrix
- Maximum autopoiesis: S determines its own forgetting
- Fixed point dynamics: S must find self-consistent evolution
- Risk: degenerate solutions (all-forget or all-remember)

*Stabilization strategies:*
- Use different key projections for gating vs. content
- Add skip connection: $bold(G) = sigma(...) + epsilon dot bold(I)$
- Regularize toward moderate gating

== Level 6: Circular K-Tower (E83)

#architecture[E83: Circular Mutual Gating][
  $K$ matrices $bold(M)_0, bold(M)_1, ..., bold(M)_(K-1)$, each $n times n$.

  Each matrix is gated by the next (modulo $K$):
  $ bold(G)_i = sigma(bold(M)_((i+1) mod K) hat(bold(k))_i hat(bold(k))_i^top + bold(B)_i) $
  $ bold(M)'_i = bold(G)_i dot.circle bold(M)_i + bold(delta)_i hat(bold(k))_i^top $

  For $K=3$: $bold(M)_0 arrow.l bold(M)_1 arrow.l bold(M)_2 arrow.l bold(M)_0$ (circular)
]

*Properties:*
- No "top" of the hierarchy — circular dependency
- Distributed autopoiesis across K matrices
- Each matrix is both controller and controlled
- Richer dynamics than pairwise coupling

*The K=2 case recovers E79/E80* (mutual pair).

== Level 7: Continuous Dynamics (E84)

#architecture[E84: Neural ODE Automodulation][
  Continuous-time evolution:
  $ (d bold(S)) / (d t) = -bold(S) + sigma(bold(G)) dot.circle bold(S) + "outer"(bold(v) - bold(S) hat(bold(k)), hat(bold(k))) $
  $ (d bold(G)) / (d t) = -bold(G) + sigma(bold(S)) dot.circle bold(G) + "outer"(bold(delta)_S - bold(G) hat(bold(m)), hat(bold(m))) $

  Integrate from $t=0$ to $t=T$ using ODE solver.
]

*Properties:*
- Adaptive computation: harder inputs → more integration steps
- Smooth dynamics, potentially better gradients
- The system finds its own "clock"
- Use adjoint method for memory-efficient gradients

= Comparison Table

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, center, center, center, left),
  [*Level*], [*Gate Rank*], [*State Size*], [*Gate DOF*], [*Key Property*],
  [E74], [0 (scalar)], [$n^2$], [1], [Fixed decay],
  [E75], [diag], [$n^2$], [$n$], [Input-dependent],
  [E79], [1], [$2n^2$], [$2n$], [Mutual, rank-1],
  [E80], [$n$], [$2n^2$], [$n^2$], [Mutual, full-rank],
  [E81], [$n$], [$2n^2$], [$n^2$ evolving], [Gate as state],
  [E82], [$n$], [$n^2$], [$n^2$ self], [Self-gating],
  [E83], [$n$ each], [$K n^2$], [$K n^2$], [Circular tower],
  [E84], [$n$], [$2n^2$], [continuous], [Neural ODE],
)

= The Information-Theoretic View

== Bits of Control

Each level provides different amounts of information for gating:

- *E74:* 0 bits (fixed)
- *E75:* $n log_2(1/epsilon)$ bits (n scalar gates at precision $epsilon$)
- *E79:* $2n log_2(1/epsilon)$ bits (2n values → rank-1 gate)
- *E80:* $n^2 log_2(1/epsilon)$ bits (full gate matrix)

== The Compression Principle

#insight[
  There's a tradeoff: more gating flexibility requires more parameters/computation, but may enable better compression of the input sequence into fixed-size state.

  The optimal level depends on:
  1. Sequence complexity (more structure → benefit from richer gating)
  2. Training budget (richer gating → harder to optimize)
  3. Inference budget (richer gating → more compute per step)
]

= Gradient Flow Analysis

== E79: Rank-1 Constraint

Gradient from loss to M:
$ (diff cal(L)) / (diff bold(M)) = (diff cal(L)) / (diff bold(S)') dot (diff bold(S)') / (diff bold(g)) dot (diff bold(g)) / (diff bold(M)) $

The bottleneck: $(diff bold(S}') / (diff bold(g))$ only has rank-1 structure.

== E80+: Full-Rank Gradient

With full-rank gating, every element of G receives independent gradient signal. This may enable:
- Faster learning of complex gating patterns
- Better credit assignment
- But also: risk of overfitting the gating

= Stability Considerations

== Fixed Points

Self-gating systems (E82) must avoid degenerate fixed points:
- $bold(G) = bold(0)$: Complete forgetting (S → 0)
- $bold(G) = bold(1)$: No forgetting (S accumulates without bound)

*Mitigation:*
- Initialize gate biases for moderate decay ($sigma^(-1)(0.9) approx 2.2$)
- Add regularization toward $bold(G) approx 0.5$
- Use spectral normalization on S

== Circular Dependencies (E83)

The circular gating $bold(M)_0 arrow.l bold(M)_1 arrow.l ... arrow.l bold(M)_0$ creates:
- No clear "ground truth" — all matrices bootstrap each other
- Potential for oscillation or divergence
- Need careful initialization and learning rate scheduling

= Implementation Considerations

== Computational Cost

#table(
  columns: (auto, auto, auto),
  align: (left, center, center),
  [*Level*], [*Forward Cost*], [*Backward Cost*],
  [E74], [$O(n^2)$], [$O(n^2)$],
  [E75], [$O(n^2 + n d)$], [$O(n^2 + n d)$],
  [E79], [$O(n^2)$ × 2], [$O(n^2)$ × 2],
  [E80], [$O(n^2)$ × 2], [$O(n^2)$ × 2],
  [E81], [$O(n^2)$ × 2], [$O(n^2)$ × 2],
  [E82], [$O(n^2)$], [$O(n^2)$],
  [E83], [$O(K n^2)$], [$O(K n^2)$],
  [E84], [$O(n^2 times "steps")$], [$O(n^2 times "steps")$],
)

== CUDA Kernel Strategy

For each level, the kernel structure is similar:
1. Load state matrices into shared memory
2. Compute gates (level-specific)
3. Apply gated decay + delta rule update
4. Store results

The main difference is how gates are computed:
- E79: Two matrix-vector products → outer product
- E80: Full matrix computation for gate
- E81: Same as E80, but gate persists across timesteps
- E82: Self-referential gate computation

= Open Questions

1. *Optimal rank for gating:* Is there a sweet spot between rank-1 (E79) and full-rank (E80)?

2. *Initialization for self-gating:* How to initialize E82 to avoid degenerate fixed points?

3. *Circular vs. hierarchical:* Does the circular structure (E83) outperform linear hierarchy?

4. *Continuous vs. discrete:* When does E84's adaptive computation help?

5. *Biological plausibility:* Do neural circuits implement any of these patterns?

= Conclusion

The autopoietic ladder reveals a spectrum of self-modulation strategies:

#table(
  columns: (auto, 1fr),
  align: (center, left),
  [*Level*], [*Key Insight*],
  [E74], [Baseline: no self-reference],
  [E75], [External modulation only],
  [E79], [Mutual modulation, rank-constrained],
  [E80], [Full-rank mutual modulation],
  [E81], [Gate itself evolves],
  [E82], [Pure self-reference],
  [E83], [Distributed circular control],
  [E84], [Continuous self-modulation],
)

Each step up the ladder increases the system's ability to control its own structure. The research question is: *where is the sweet spot* between expressiveness and trainability?

The E79 benchmark results (1.51 loss, beating E1's 1.53) suggest that even rank-1 mutual gating provides benefit. The higher levels remain to be empirically validated.

