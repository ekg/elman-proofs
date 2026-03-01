// Section 1: Introduction

#import "traditional-math-style.typ": *

= Introduction

== The Method: Formal Proofs as Architecture Design Tool

How should we design sequence models? The standard approach is empirical: propose an architecture, train it on benchmarks, report metrics. This paper demonstrates a different methodology: use formal proofs to explore the architecture space systematically.

The central principle is simple. Prove what a class of architectures _cannot_ compute. These impossibility results reveal what architectural features are _necessary_ for specific computational tasks. Each proof becomes a design constraint, narrowing the space of viable architectures.

We apply this method to the fundamental question: where should nonlinearity live in a sequence model? Through time, or between layers? The answer determines computational capacity in ways that training data alone cannot reveal.

== The Investigation: Three Architecture Classes

Our proof-guided exploration considers three dominant approaches:

*Transformers* @vaswani2017attention place nonlinearity between layers. Attention aggregates the sequence, then a feedforward network applies nonlinearity, then attention again. Time is processed all at once; depth accumulates vertically.

*State-space models* like Mamba2 @dao2024mamba2 also place nonlinearity between layers, but process time differently. Within each layer, temporal dynamics are linear: $h_t = A_t h_(t-1) + B_t x_t$ where $A_t, B_t$ depend on $x_t$ but not nonlinearly on $h_(t-1)$. This enables parallel computation through associative scans.

*Nonlinear-temporal models* like the classical Elman architecture @elman1990finding place nonlinearity in time itself. The E88 variant uses $S_t = tanh(alpha S_(t-1) + delta v_t k_t^top)$ where tanh compounds across timesteps. Every timestep adds composition depth.

These are not arbitrary designs. They represent fundamentally different compositional structures. Our proofs make this precise.

== The First Result: Impossibility of Linear-Temporal Computation

We begin by formalizing linear recurrence: $h_t = A_t h_(t-1) + b_t$ where $A_t, b_t$ may depend on inputs but not nonlinearly on $h_(t-1)$. The state after $T$ steps is a weighted sum:

$ h_T = sum_(t=0)^(T-1) product_(s=t+1)^(T-1) A_s dot.c b_t $

This representation immediately implies two consequences, both proven in `LinearCapacity.lean` and `LinearLimitations.lean`:

*Theorem 1 (Continuity).* Linear RNN output is a continuous function of inputs.

*Theorem 2 (Superposition).* Linear RNN output satisfies $f(x + y) = f(x) + f(y)$ and $f(c x) = c f(x)$.

Any function violating these properties cannot be computed by linear-temporal recurrence. Threshold functions are discontinuous. XOR violates superposition. Parity requires $T$ sequential XOR operations. All are impossible.

*Theorem 3 (Impossibility of Parity).* No linear-temporal RNN can compute running parity $x_1 xor x_2 xor dots xor x_t$ at each position $t$.

The proof is mechanical, verified in Lean 4. No depth can overcome this limitation: stacking $D$ layers provides $D$ nonlinear compositions _between_ layers, but within each layer, time still flows linearly. For sequence length $T > 2^D$, running parity requires more compositions than the model provides.

== The Design Implication: Composition Depth Gap

The impossibility proofs reveal the architectural constraint:

#theorem("Composition Depth Gap")[
  For a $D$-layer model processing sequences of length $T$:
  #h(1em) Linear temporal dynamics yields composition depth $D$.
  #h(1em) Nonlinear temporal dynamics yields composition depth $D times T$.
]

This is not a benchmark result. It is a mathematical fact about compositional structure. When $T > 2^D$, certain functions become impossible for linear-temporal models regardless of training.

The E88 architecture escapes this limitation through temporal nonlinearity. The tanh compounds at every timestep, providing $T$ levels of composition per layer. A single-layer E88 has more compositional depth than a 32-layer Mamba2 when $T > 2^(32)$.

== The Hierarchy: Formal Computational Classes

The proof-guided exploration establishes a strict hierarchy:

$ "Linear SSM" subset.neq "TC"^0 "(Transformer)" subset.neq "E88" subset.neq "E23 (UTM)" $

Each separation is witnessed by a concrete computable function:
- *Linear SSM → TC⁰:* Running parity (impossible for linear-temporal, computable in constant depth)
- *TC⁰ → E88:* Functions requiring depth proportional to $T$ (impossible for constant depth)
- *E88 → E23:* Functions requiring unbounded memory (E23 has explicit tape)

The containments are proven in `LinearLimitations.lean`, `TC0Bounds.lean`, `TC0VsUnboundedRNN.lean`, and `E23vsE88Comparison.lean`. The strictness follows from construction.

== The Empirical Puzzle: Theory vs Practice

Here the methodology encounters a puzzle. Despite E88's provably greater computational capacity, Mamba2 achieves better perplexity on language modeling. Theory predicts E88 should dominate. Experiments show otherwise.

This gap is not a failure of the method. It reveals the difference between what an architecture _can_ compute with unlimited resources and what it _learns efficiently_ with gradient descent on finite data. The impossibility proofs tell us about fundamental limits. Training dynamics tell us what happens within those limits.

The resolution matters for architecture design. If a task's composition depth is bounded by $D = 32$, linear-temporal models suffice and train faster. If a task requires temporal decisions (state tracking, algorithmic reasoning), the linear-temporal limitation becomes an empirical bottleneck.

== The Extension: Emergent Tape Memory

A final proof-guided insight: output feedback creates working memory. When a model writes output and reads it back, the output stream becomes an _emergent tape_. With output feedback, even limited architectures achieve bounded Turing machine power:

$ "E88+Feedback" equiv "Transformer+CoT" equiv "DTIME"(T) $

This explains chain-of-thought reasoning: it provides external storage that overcomes fixed state limitations. The theorems are in `OutputFeedback.lean` and `MultiPass.lean`.

== Summary of Methodology

This paper demonstrates a design method:

1. Formalize architecture classes precisely (linear vs nonlinear temporal dynamics)
2. Prove impossibility results (what cannot be computed)
3. Extract design constraints (what architectural features are necessary)
4. Construct witness functions showing separations are sharp
5. Verify all proofs mechanically in Lean 4

The result is not just a collection of theorems, but a systematic approach to architecture exploration. The proofs guide the search through design space, revealing which architectural choices matter and why.

All theorems are mechanically verified in Lean 4 @moura2021lean4 with Mathlib @mathlib2020. Clone the repository, run `lake build`, and verify that every proof compiles. The formalization is the method.
