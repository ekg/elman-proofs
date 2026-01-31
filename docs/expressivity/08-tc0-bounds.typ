// Section 8: TC0 Circuit Complexity Bounds

#import "traditional-math-style.typ": *

= Circuit Complexity and the Inverted Hierarchy

The hierarchy we have established among neural architectures has a classical counterpart in circuit complexity. The correspondence is illuminating---and the conclusion is surprising. The popular ranking of architectures by "power" inverts when we measure computational expressivity.

== The Boolean Circuit Hierarchy

Circuit complexity measures computational power by the resources a circuit needs. The classes form a nested hierarchy:

$ "NC"^0 subset.neq "AC"^0 subset.neq "TC"^0 subset.eq "NC"^1 subset.eq "P" $

NC⁰ consists of constant-depth circuits with bounded fan-in gates. AC⁰ allows unbounded fan-in AND and OR. TC⁰ adds threshold gates (MAJORITY). NC¹ allows logarithmic depth with bounded fan-in. P is polynomial-time computation.

The famous separation AC⁰ $subset.neq$ TC⁰ comes from parity: PARITY requires super-polynomial size in AC⁰ but has polynomial-size TC⁰ circuits. Adding threshold gates makes parity easy. This is exactly the separation we have been studying---now placed in classical context.

== Where the Architectures Fall

The question is: where do Transformers, SSMs, and recurrent networks sit in this hierarchy?

#theorem("Transformers Are TC⁰-Bounded")[
  A Transformer with $D$ layers and saturated (hard) attention can be simulated by a TC⁰ circuit of depth $O(D)$.
]#leanfile("TC0Bounds.lean:152")

The proof observes that attention computes weighted averages, which are threshold operations, and feedforward layers compute bounded-depth compositions of activations. The depth is constant (in the number of layers, not in the sequence length), so the circuit is TC⁰.

This means Transformers live at TC⁰---exactly at the boundary where threshold gates give power over AC⁰. They can compute parity. But their depth is constant.

#theorem("Linear SSMs Are Below TC⁰")[
  State-space models with nonnegative gates (Mamba, Griffin, RWKV) cannot compute PARITY.
]#leanfile("TC0VsUnboundedRNN.lean:152")

The proof uses eigenvalue analysis. Nonnegative matrices have nonnegative dominant eigenvalues. The state evolves monotonically in sign. But parity requires alternating sign with each 1-bit input. The dynamics cannot oscillate in the required way.

Linear SSMs fall _below_ TC⁰---they cannot even compute the function that separates AC⁰ from TC⁰.

#theorem("E88 Exceeds TC⁰")[
  E88 with $T$ timesteps has effective depth $D times T$. For any constant bound $C$, there exists sequence length $T$ such that $D times T > C$.
]#leanfile("TC0VsUnboundedRNN.lean:127")

E88's depth grows with sequence length. It is not constant-depth. Therefore it exceeds TC⁰ by definition.

== The Inverted Ranking

We can now complete the picture:

#simpletable(
  columns: 4,
  align: (left, center, center, center),
  [*Architecture*], [*Circuit Class*], [*PARITY*], [*Depth*],
  [Linear SSM], [< TC⁰], [No], [$D$],
  [Transformer], [= TC⁰], [Yes], [$D$],
  [E88], [> TC⁰], [Yes], [$D times T$],
  [E23], [= RE], [Yes], [Unbounded],
)

The popular ranking---"Transformers are more powerful than SSMs, which are more powerful than RNNs"---is based on training efficiency and benchmark performance. For _expressivity_, the ranking inverts completely.

E88, the "old" architecture descended from Elman networks, exceeds Transformers in computational class. Linear SSMs, the "modern" efficient architectures, fall below Transformers. The RNN that looked like a step backward is actually a step forward in computational power.

#keyinsight[
  The naive ranking conflates trainability with expressivity. Transformers train efficiently via parallelism. SSMs train efficiently via parallel scan. But E88, while harder to train, computes functions that neither can compute.
]

== What This Means

The circuit complexity perspective clarifies the stakes. TC⁰ is a well-studied computational class with known limitations. Transformers live there. Any function outside TC⁰ is forever beyond the reach of Transformers, no matter how large or how well-trained.

E88's escape to beyond TC⁰ is not a minor improvement---it crosses a classical complexity barrier. The separation is not empirical; it is the same separation that complexity theorists have studied for decades.

#centerrule

We have placed our architectural hierarchy in classical complexity theory. Linear SSMs fall below TC⁰. Transformers sit at TC⁰. E88 exceeds TC⁰ by achieving depth that grows with sequence length. The popular ranking inverts.

But computational class is not the whole story. What happens when models are allowed to write output and read it back? The answer involves an emergent tape.
