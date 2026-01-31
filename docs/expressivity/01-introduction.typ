// Section 1: Introduction - Deep Context
// Temporal Nonlinearity vs Depth in Sequence Models

= Introduction: The Geometry of Computation in Sequence Models

== The Fundamental Question

Every sequence model must answer a deceptively simple question: *where should nonlinearity live?*

The possible answers define the design space:

+ *Between tokens* (Transformers): Nonlinearity flows through the attention-MLP stack at each position. Information combines across positions through linear attention, with nonlinear mixing happening depth-wise.

+ *Between layers* (Mamba2, FLA, GDN): Within each layer, information flows forward through time via purely linear operations. Nonlinearities (SiLU, gating, projections) operate within each timestep. Depth provides composition.

+ *Through time* (E88, classical RNNs): Nonlinearity applies to the temporal recurrence itself. Each timestep compounds nonlinear transformations, making the state a nonlinear function of the entire history.

The choice is not merely aesthetic. It determines what functions can be computed, how efficiently they can be learned, and what tasks will succeed or fail. This document establishes these relationships formally, with proofs mechanically verified in Lean 4.

== Historical Context: The Architecture Debate

The history of sequence modeling is a history of trading off expressivity and efficiency.

*The RNN Era (1990s-2017)*: Recurrent neural networks with nonlinear activations (tanh, LSTM gates) dominated. These had rich temporal dynamics---each timestep applied nonlinearity---but training was notoriously difficult. Vanishing gradients plagued deep temporal computation. The state $h_T = sigma(W h_(T-1) + V x_T)$ compounds nonlinearities, creating $O(T)$-deep computation graphs that gradients must traverse.

*The Transformer Revolution (2017-2022)*: Attention replaced recurrence. The key insight: remove sequential dependencies. Each position attends to all others in parallel, with nonlinearity flowing through depth (layers) rather than time. This solved training stability but introduced $O(T^2)$ complexity in sequence length.

*The SSM Resurgence (2022-present)*: Mamba and its successors (Mamba2, FLA, GDN) brought linear recurrence to scale. By making temporal dynamics linear---$h_t = A_t h_(t-1) + B_t x_t$---they achieved $O(T)$ complexity with training stability. The nonlinearity between layers, not within temporal updates, enables efficient parallelization.

*The E88 Alternative (2025)*: What if we kept temporal nonlinearity but made it hardware-friendly? E88's $S_t = tanh(alpha S_(t-1) + delta k_t^T)$ applies tanh to the accumulated state. This compounds nonlinearity through time while maintaining compute-dense matrix operations that modern accelerators handle efficiently.

This document resolves the debate formally: *temporal nonlinearity enables strictly more computation than depth alone*.

== The Key Insight: Two Kinds of Composition

The central result can be stated simply:

#block(
  fill: rgb("#f0f7ff"),
  stroke: rgb("#3366cc"),
  inset: 12pt,
  radius: 4pt,
)[
  *"Nonlinearity flows down (through layers), not forward (through time)."*

  For models with linear temporal dynamics, no matter how many layers $D$ you stack, each layer still aggregates information linearly across time. The total composition depth is $D$.

  For models with nonlinear temporal dynamics, each timestep adds one composition level. The total composition depth is $D times T$.

  The gap is a factor of $T$---the sequence length.
]

This explains why certain tasks that seem simple to humans---counting, parity, state tracking---defeat even very deep linear-temporal models. These tasks require $T$ sequential nonlinear decisions. Depth provides $D$ decisions. When $T > D$, the task becomes impossible.

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,
    align: (left, center, center),
    [*Architecture*], [*Temporal Dynamics*], [*Composition Depth*],
    [Mamba2, FLA, GDN], [Linear: $h_T = sum alpha^(T-t) dot f(x_t)$], [$D$ (layers only)],
    [E88], [Nonlinear: $S_t = tanh(alpha S_(t-1) + g(x_t))$], [$D times T$ (layers × time)],
    [Transformer], [Parallel attention + depth], [$D$ (but $O(T^2)$ cost)],
  ),
  caption: [Composition depth varies by architecture. The key is where nonlinearity enters.],
)

== Circuit Complexity: The TC${}^0$ Perspective

The distinction has a precise complexity-theoretic characterization.

*TC${}^0$* is the class of functions computable by constant-depth threshold circuits with polynomial size. It captures "shallow parallel computation"---lots of units, but bounded depth. Crucially, TC${}^0$ cannot compute certain functions that seem simple:

- *Parity* of $n$ bits requires depth $Omega(log n \/ log log n)$ in threshold circuits
- *Majority counting* is in TC${}^0$ but not in AC${}^0$ (circuits without threshold gates)
- *Iterated multiplication* requires unbounded depth in uniform TC${}^0$

Linear-temporal models are in TC${}^0$ when viewed as circuits. Each layer contributes constant depth, regardless of sequence length. The temporal aggregation---being linear---collapses to a single operation.

Nonlinear-temporal models escape TC${}^0$. Each timestep's tanh adds depth to the circuit representation. Over $T$ timesteps, the "circuit" has depth $O(T)$, not $O(1)$.

#block(
  fill: rgb("#fff7f0"),
  stroke: rgb("#cc6633"),
  inset: 12pt,
  radius: 4pt,
)[
  *The TC${}^0$ Bound Explains Everything*:

  - Why Mamba2 can't compute parity: parity $in.not$ TC${}^0$
  - Why E88 can: $O(T)$ depth escapes the TC${}^0$ limitation
  - Why depth doesn't help linear-temporal: more layers = more parallel TC${}^0$, not higher depth
  - Why the separation is fundamental: it's not about training---it's about computation
]

This connects our architectural analysis to the foundations of computational complexity. The proofs we develop are not ad hoc---they instantiate known separations in complexity theory.

== Emergent Computation: Output Feedback Creates Tape Memory

A second key insight: *output feedback transforms computational class*.

When a model can:
1. Write tokens to an output stream
2. Read those tokens back (via attention or recurrence)
3. Run for $T$ steps

...it creates an *emergent tape* of length $T$. This is the mechanism behind chain-of-thought (CoT) reasoning, scratchpad computation, and autoregressive self-conditioning.

#block(
  fill: rgb("#f0fff7"),
  stroke: rgb("#33cc66"),
  inset: 12pt,
  radius: 4pt,
)[
  *Theorem (OutputFeedback.lean)*: Output feedback elevates any architecture to bounded Turing machine power.

  Even a simple linear RNN with output feedback can simulate a bounded TM, because the feedback creates an emergent tape of length $T$.
]

This explains why chain-of-thought dramatically improves reasoning: it provides the working memory needed for multi-step computation. The "scratchpad" is not a prompting trick---it's a computational resource.

The hierarchy becomes:

$ "Fixed Mamba2" subset.eq.not "Fixed E88" subset.eq.not "E88+Feedback" tilde.equiv "Transformer+CoT" subset.eq.not "E23 (unbounded tape)" $

Each separation is witnessed by concrete problems:
- Mamba2 $subset.eq.not$ E88: Running parity (linear cannot threshold)
- E88 $subset.eq.not$ E88+Feedback: Palindrome recognition ($O(1)$ vs $O(T)$ memory)
- CoT $subset.eq.not$ E23: Halting problem (bounded vs unbounded tape)

== What This Document Proves

This document develops a complete theory of expressivity for sequence models, with all key results formally verified in Lean 4. The proofs are not informal arguments---they are machine-checked mathematical theorems.

=== Section 2: Mathematical Foundations

Establishes the core machinery:
- Linear RNN state is a weighted sum of inputs (LinearCapacity.lean:72)
- Linear outputs are additive and homogeneous (LinearLimitations.lean:62-78)
- Threshold functions are not linearly computable (LinearLimitations.lean:107)
- XOR is not affine (LinearLimitations.lean:218)
- Multi-layer models with linear temporal dynamics have composition depth $D$ (MultiLayerLimitations.lean)
- E88's tanh recurrence has composition depth $T$ per layer (RecurrenceLinearity.lean:215)

=== Section 3: The Linear-Temporal Limitation

Proves what Mamba2, FLA, and GDN *cannot* do:
- Running threshold is impossible for continuous models (ExactCounting.lean)
- Running parity requires nonlinearity at each step (RunningParity.lean)
- Depth does not compensate for linear temporal dynamics (MultiLayerLimitations.lean:231)

The key theorem: for any $D$-layer linear-temporal model, there exist functions computable by 1-layer E88 that the $D$-layer model cannot compute.

=== Section 4: E88 Temporal Nonlinearity

Proves what E88 *can* do and why:
- Tanh saturation creates stable fixed points (TanhSaturation.lean)
- For $alpha > 1$, nonzero fixed points exist (AttentionPersistence.lean:212)
- Latched states persist under perturbation (TanhSaturation.lean:204)
- Linear systems decay; E88 latches (BinaryFactRetention.lean)
- E88 heads are independent parallel state machines (MultiHeadTemporalIndependence.lean)
- E88 can compute running threshold and parity (ExactCounting.lean)

=== Section 5: E23 vs E88

Contrasts two paths to expressivity:
- E23 (tape-based): Turing-complete but memory-bandwidth-bound
- E88 (saturation-based): Sub-UTM but hardware-efficient
- Why E88 wins in practice: compute-dense operations, bounded gradients, natural batching

The deeper lesson: memory that emerges from dynamics (E88) aligns better with gradient-based learning and modern hardware than explicit tape storage (E23).

=== Section 6: Separation Results

Collects the proven impossibilities into a clean hierarchy:
- XOR is not affine (foundational)
- Running parity separates linear from nonlinear temporal
- Running threshold separates continuous from discontinuous
- Binary fact retention separates decaying from latching
- FSM simulation requires state persistence

Each result is a mathematical theorem, not an empirical observation.

=== Section 7: Practical Implications

Translates theory into practice:
- Architecture selection by task type
- Benchmark design for clean separation
- Experimental predictions derived from proofs
- Design principles for hybrid architectures

The key prediction: on running parity, E88 achieves ~99% accuracy while Mamba2 achieves ~50% (random)---regardless of depth.

=== Section 9: Output Feedback and Emergent Tape

Analyzes the computational effects of autoregressive feedback:
- Feedback creates emergent tape memory
- Chain-of-thought equals explicit tape in computational power
- Sequential vs random access: RNN feedback vs Transformer attention
- The hierarchy from fixed state to unbounded tape

This explains why CoT works and when output feedback matters.

== The Structure of the Argument

The document proceeds in three phases:

*Phase 1 (Sections 2-3): Establishing Limitations*

We prove that linear temporal dynamics impose fundamental constraints. These are not training failures---they are mathematical impossibilities. The core lemma: linear functions are continuous and additive, but threshold, XOR, and parity violate these properties.

*Phase 2 (Sections 4-6): Proving Separation*

We show that E88's temporal nonlinearity overcomes these limitations. The tanh recurrence creates:
- Fixed points for memory (unlike linear decay)
- Discontinuity approximation for threshold (unlike continuous linear output)
- Composition depth $T$ (unlike collapsed linear composition)

Each capability is proven formally.

*Phase 3 (Sections 7, 9): Practical Synthesis*

We connect theory to practice. Which architecture for which task? What benchmarks cleanly separate models? How does output feedback change the picture?

The analysis is complete: from foundational impossibility to practical recommendations.

== Reading Guide

*For practitioners*: Start with Section 3 (what linear-temporal models can't do) and Section 7 (practical implications). These give actionable guidance without deep formalism.

*For theorists*: Section 2 (foundations) and Section 6 (separation results) provide the formal core. The Lean code references allow verification of every claim.

*For the curious*: Section 5 (E23 vs E88) and Section 9 (output feedback) explore the deeper questions of what makes an architecture work.

All theorems are formalized in Lean 4 with Mathlib. The source files are in `ElmanProofs/Expressivity/`. When we say "proven," we mean mechanically verified---the proofs exist as checked Lean code.

== Summary: The Central Claims

#block(
  fill: rgb("#f7f0ff"),
  stroke: rgb("#9933cc"),
  inset: 12pt,
  radius: 4pt,
)[
  *Claim 1*: Linear temporal dynamics (Mamba2, FLA, GDN) cannot compute running parity, running threshold, or exact counting, regardless of depth $D$.

  *Claim 2*: Nonlinear temporal dynamics (E88) can compute these functions with a single layer.

  *Claim 3*: The separation is not about training---it's about computation. These are mathematical theorems, not empirical observations.

  *Claim 4*: Output feedback creates emergent tape memory, elevating any architecture to bounded TM power.

  *Claim 5*: The hierarchy is complete:
  $ "Linear-Temporal" subset.eq.not "E88" subset.eq.not "E88+Feedback" tilde.equiv "Transformer+CoT" subset.eq.not "E23 (UTM)" $
]

This is not a conjecture. Every claim is proven in Lean 4 and referenced to specific files and line numbers. The remainder of this document develops the proofs.

