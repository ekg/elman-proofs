// Section 3: The Linear-Temporal Limitation
// What Mamba2, FLA, and GDN Cannot Do

= The Linear-Temporal Limitation

This section establishes what models with linear temporal dynamics cannot compute. The results apply to Mamba2, Fast Linear Attention, Gated Delta Networks, and any architecture where within-layer state evolution is a linear function of time.

== The Core Limitation

#block(
  fill: rgb("#fff0f0"),
  stroke: rgb("#cc3333"),
  inset: 12pt,
  radius: 4pt,
)[
  *Main Theorem (MultiLayerLimitations.lean)*: A $D$-layer model with linear temporal dynamics at each layer cannot compute any function requiring more than $D$ levels of nonlinear composition---regardless of sequence length $T$.
]

The intuition: each layer contributes at most one level of nonlinear composition (the inter-layer activation). Time steps within a layer do not add composition depth because the temporal aggregation is linear.

== Linear Temporal Dynamics: The Definition

A layer has *linear temporal dynamics* if its state at time $T$ is:

$ h_T^L = sum_(t <= T) W(t, T) dot y_t^(L-1) $

where $W(t, T)$ are weight matrices and $y^(L-1)$ is the output of the previous layer. The key property: $h_T^L$ is a _linear function_ of the input sequence $y^(L-1)$.

Examples of linear temporal dynamics:
- *SSM*: $h_t = A h_(t-1) + B x_t$, giving $h_T = sum A^(T-t) B x_t$
- *Linear attention*: $"out" = sum (q dot k_i) v_i = q dot (sum k_i times.o v_i)$
- *Gated delta*: Despite "gating," the delta rule is linear in query

== The Multi-Layer Case

Consider a $D$-layer model. Let $phi$ be the inter-layer nonlinearity (e.g., SiLU, GeLU). The output of layer $L$ is:

$ y_t^L = phi(h_t^L) = phi(sum_(s <= t) W_s^L y_s^(L-1)) $

Even with nonlinear $phi$, the function computed has bounded complexity:

#block(
  fill: rgb("#f0f7ff"),
  stroke: rgb("#3366cc"),
  inset: 12pt,
  radius: 4pt,
)[
  *Theorem (Composition Depth Bound)*: The output $y_T^D$ of a $D$-layer linear-temporal model can be expressed as a composition of at most $D$ nonlinear functions applied to linear combinations of inputs.
]

This is in stark contrast to E88, where each timestep adds a nonlinear composition:

$ S_t = tanh(alpha S_(t-1) + delta k_t) = tanh(alpha tanh(alpha tanh(...) + ...) + ...) $

After $T$ steps, E88 has $T$ nested $tanh$ applications---composition depth $T$, not $1$.

== Concrete Impossibility Results

=== Running Threshold

*Task*: Output $1$ if $sum_(t <= T) x_t > tau$, else $0$.

#block(
  fill: rgb("#fff0f0"),
  stroke: rgb("#cc3333"),
  inset: 12pt,
  radius: 4pt,
)[
  *Theorem (ExactCounting.lean)*: No $D$-layer linear-temporal model can compute running threshold for any $D$.

  *Proof sketch*: Running threshold has a discontinuity when the sum crosses $tau$. But $D$-layer models with linear temporal dynamics output continuous functions (composition of continuous functions). Continuous functions cannot have discontinuities.
]

=== Running Parity

*Task*: Output the parity (XOR) of all inputs seen so far: $y_t = x_1 xor x_2 xor ... xor x_t$.

#block(
  fill: rgb("#fff0f0"),
  stroke: rgb("#cc3333"),
  inset: 12pt,
  radius: 4pt,
)[
  *Theorem (RunningParity.lean)*: No linear-temporal model can compute running parity.

  *Proof*: Parity violates the affine constraint. For any affine function $f$:
  $ f(0,0) + f(1,1) = f(0,1) + f(1,0) $

  But parity gives: $0 + 0 eq.not 1 + 1$. Since linear-temporal outputs are affine in inputs, parity is impossible.
]

=== XOR Chain

*Task*: Compute $y_t = x_1 xor x_2 xor ... xor x_t$ at each position.

This requires $T-1$ nonlinear decisions (each XOR is nonlinear). With composition depth $D$, a linear-temporal model can make at most $D$ decisions. For $T > D$, it fails.

== The Depth Compensation Fallacy

A common belief: "Just add more layers." But our proofs show this doesn't work:

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,
    align: (left, center, center, center),
    [*Task*], [*Required Depth*], [*D-layer Linear*], [*1-layer E88*],
    [Running threshold], [1 (but discontinuous)], [Impossible], [Possible],
    [Running parity], [$T$ (sequence length)], [Impossible], [Possible],
    [FSM simulation], [$|Q|$ (state count)], [Limited], [Full],
  ),
  caption: [Depth cannot compensate for linear temporal dynamics on these tasks.],
)

The key insight: *time does not create composition depth for linear systems*. The matrix $A^T$ is still just one linear operation, no matter how large $T$ is. But $tanh^T$ (E88's $T$ nested tanh applications) has true composition depth $T$.

== Mamba2, FLA, GDN: Where They Stand

All three architectures have linear temporal dynamics:

- *Mamba2*: $h_t = A_t h_(t-1) + B_t x_t$ with input-dependent $A_t, B_t$. Still linear in $h$.
- *FLA*: Linear attention is linear in query: $"out" = q dot M$ where $M = sum k_i times.o v_i$.
- *GDN*: Delta rule $S' = S + (v - S k) k^T$ is linear in $S$.

The input-dependent gating in Mamba2 doesn't help---it makes $A_t, B_t$ depend on $x_t$, but the recurrence remains linear in the state. Similarly, GDN's selective update is linear despite its "gated" name.

== When Linear Temporal Models Suffice

For language modeling with $D >= 32$ layers, the practical gap may be small:

- Typical language complexity: ~25 levels of nesting
- $D = 32$ provides sufficient composition depth
- Linear temporal models are often faster (simpler operations)

The limitation matters for:
- Algorithmic reasoning (counting, parity, state tracking)
- Tasks requiring temporal decisions
- Small-$D$ deployments where depth is constrained

The next section shows how E88's temporal nonlinearity overcomes these limitations.
