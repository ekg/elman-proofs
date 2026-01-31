// Section 12: The Formal Verification System

#import "traditional-math-style.typ": *

= Formal Verification

The theorems in this document are not arguments. They are proofs---mechanically verified in Lean 4, checked by computer, with no gaps.

This distinction matters. Mathematical claims in machine learning often rest on intuition, approximation, or empirical validation. Our claims rest on formal proof. If the Lean type checker accepts the proof, the theorem is true. There is no wiggle room for subtle errors or unstated assumptions.

== The Verification Approach

Each theorem corresponds to a Lean definition and proof. The proof must satisfy Lean's type checker, which verifies that every step follows from the axioms and previously proven results. Mathlib provides the mathematical foundations: real analysis, linear algebra, topology.

The proofs are constructive where possible. When we say E88 computes running parity, we provide the parameter values and prove they work. When we say linear-temporal models cannot compute threshold, we provide the mathematical obstruction.

== Verification Status

#simpletable(
  columns: 2,
  align: (left, left),
  [*Result*], [*Source File*],
  [Linear state as weighted sum], [`LinearCapacity.lean`],
  [Linear cannot threshold/XOR], [`LinearLimitations.lean`],
  [Running parity impossibility], [`RunningParity.lean`],
  [Multi-layer limitation], [`MultiLayerLimitations.lean`],
  [Tanh saturation and latching], [`TanhSaturation.lean`],
  [E88 computes parity], [`TanhSaturation.lean`],
  [Exact counting separation], [`ExactCounting.lean`],
  [TC⁰ circuit bounds], [`TC0VsUnboundedRNN.lean`],
  [Output feedback / emergent tape], [`OutputFeedback.lean`],
  [Multi-pass RNN hierarchy], [`MultiPass.lean`],
  [DFA simulation bounds], [`ComputationalClasses.lean`],
)

All core expressivity theorems compile without `sorry` statements. The proofs are complete.

== What Verification Guarantees

Formal verification provides certainty about what it checks:

_Logical validity_: Every proof step is a valid inference.

_Type correctness_: All mathematical objects have the stated types.

_Explicit hypotheses_: The assumptions of each theorem are stated precisely.

_Completeness_: There are no gaps---every lemma invoked is itself proven.

Formal verification does _not_ guarantee:

_Relevance_: The theorems might not matter for practice.

_Applicability_: Real systems might not match the formalized abstractions.

_Optimality_: A proven bound might not be tight.

_Efficiency_: An existence proof does not provide an algorithm.

Our theorems concern idealized mathematical models. Real neural networks have finite precision, training noise, and optimization dynamics. The formalization captures expressivity---what functions the architecture _can_ compute---not what it will learn.

== How to Verify

Clone the repository: `github.com/ekg/elman-proofs`.

Run `lake build`.

If the build completes without error, every theorem in this document has been checked. The source code is the proof. The compiler is the verifier.

#centerrule

The formal foundation is solid. The theorems are true. The remaining sections explore their implications for practical deployment and the composition depth required by various tasks.
