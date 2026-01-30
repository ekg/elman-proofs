// Section 1: Introduction - The Central Question
// Temporal Nonlinearity vs Depth in Sequence Models

= The Central Question

When designing sequence models, a fundamental architectural choice is _where to place nonlinearity_. Consider two approaches:

+ *Depth-wise nonlinearity*: Stack $D$ layers, each with linear temporal dynamics, with nonlinear activations between layers. This is the approach of Mamba2, Fast Linear Attention (FLA), and Gated Delta Networks (GDN).

+ *Time-wise nonlinearity*: Apply nonlinearity within the temporal recurrence itself. The E88 architecture does this: $S_t = tanh(alpha S_(t-1) + delta k_t)$.

The central question we address: *Does depth compensate for linear temporal dynamics?*

Our answer, backed by formal proofs in Lean 4: *No.* There exist functions computable by a single-layer E88 that no $D$-layer linear-temporal model can compute, regardless of how large $D$ is. The gap is not merely quantitative---it is a fundamental difference in computational class.

== The Architectural Dichotomy

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,
    align: (left, center, center),
    [*Architecture*], [*Temporal Dynamics*], [*Composition Depth*],
    [Mamba2, FLA, GDN], [Linear: $h_T = sum alpha^(T-t) dot f(x_t)$], [$D$ (layer count)],
    [E88], [Nonlinear: $S_t = tanh(alpha S_(t-1) + g(x_t))$], [$D times T$],
  ),
  caption: [The two paradigms differ in where nonlinearity enters the computation.],
)

The linear-temporal models aggregate information across time using weighted sums. No matter how many layers you stack, each layer performs linear temporal aggregation. The total composition depth equals the number of layers $D$.

E88 compounds nonlinearity at every timestep. Each application of $tanh$ adds one level of composition. Over $T$ timesteps and $D$ layers, the composition depth is $D times T$---linear in sequence length.

== Why This Matters

For typical language modeling at $D >= 32$ layers, the practical difference may be small. Language has approximately 25 levels of hierarchical structure, and $32 > 25$. Both approaches have "enough" depth.

But for algorithmic reasoning---counting, parity, state machine simulation---the difference is stark. These tasks require _temporal decisions_: decisions that depend on the sequence of past events, not just a summary. Linear temporal dynamics cannot make such decisions; nonlinear temporal dynamics can.

== Roadmap

The remainder of this document develops this argument formally:

- *Section 2*: Mathematical foundations---composition depth, linearity, affine constraints
- *Section 3*: The linear-temporal limitation---what Mamba2/FLA/GDN cannot do
- *Section 4*: E88's temporal nonlinearity---tanh saturation and fixed points
- *Section 5*: E23 vs E88---tape memory vs temporal saturation
- *Section 6*: Separation results---proven impossibilities
- *Section 7*: Practical implications---when to use what

All theorems are formalized in Lean 4 with Mathlib, available in the `ElmanProofs/` directory.
