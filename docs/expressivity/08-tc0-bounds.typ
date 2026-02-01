// Section 8: TC0 Circuit Complexity Bounds

#import "traditional-math-style.typ": *

= Circuit Complexity and the Inverted Hierarchy

The hierarchy we have established among neural architectures has a classical counterpart in circuit complexity. The correspondence is illuminating---and the conclusion is surprising. The popular ranking of architectures by "power" inverts when we measure computational expressivity.

== The Boolean Circuit Hierarchy

Circuit complexity measures computational power by the resources a circuit needs. The classes form a nested hierarchy:

$ "NC"^0 subset.neq "AC"^0 subset.neq "TC"^0 subset.eq "NC"^1 subset.eq "P" $

#definition("TC⁰")[
  TC⁰ is the class of languages decidable by a _uniform_ family of Boolean circuits ${C_n}_(n in NN)$ satisfying:
  + _Constant depth_: $"depth"(C_n) = O(1)$
  + _Polynomial size_: $"size"(C_n) = n^(O(1))$
  + _Threshold gates_: Allowed gates include NOT, unbounded fan-in AND/OR, and MAJORITY (output 1 iff more than half of inputs are 1)
  + _Uniformity_: The description of $C_n$ is computable in $"DLOGTIME"$
]

NC⁰ consists of constant-depth circuits with bounded fan-in gates. AC⁰ allows unbounded fan-in AND and OR. TC⁰ adds threshold gates (MAJORITY). NC¹ allows logarithmic depth with bounded fan-in @barrington1989bounded. P is polynomial-time computation @sipser2012introduction.

#theorem("PARITY Separates AC⁰ from TC⁰")[
  $"PARITY" in "TC"^0 backslash "AC"^0$.
]

#proof-sketch[
  PARITY $in$ TC⁰: Compute PARITY by counting 1s modulo 2, which is expressible with a single threshold gate testing if the count exceeds half. PARITY $in.not$ AC⁰: This is Furst-Saxe-Sipser/Razborov-Smolensky. Any AC⁰ circuit computing PARITY on $n$ bits requires size $2^(Omega(n^(1/(d+1))))$ for depth $d$---super-polynomial for any constant $d$.
]

This is exactly the separation we have been studying---now placed in classical context.

== Where the Architectures Fall

The question is: where do Transformers, SSMs, and recurrent networks sit in this hierarchy?

#theorem("Transformers Are TC⁰-Bounded")[
  A Transformer with $D$ layers and saturated (hard) attention can be simulated by a TC⁰ circuit of depth $O(D)$.#leanfile("TC0Bounds.lean:152")
]

#proof-sketch[
  Consider a single attention head: $"Attn"(Q, K, V) = "softmax"(Q K^top) V$. With hard attention (one-hot softmax), this selects a single value based on the maximum dot product. The argmax is computable by $O(1)$ threshold gates (compare each dot product to every other). The feedforward layer $"FFN"(x) = W_2 dot.op "ReLU"(W_1 x + b_1) + b_2$ is a bounded composition of linear operations and ReLU. ReLU is a threshold: $"ReLU"(x) = x dot.op [x > 0]$. Total depth per layer: $O(1)$. Total depth for $D$ layers: $O(D)$. Since $D$ is fixed, the circuit has constant depth.
]

This means Transformers live at TC⁰---exactly at the boundary where threshold gates give power over AC⁰. They can compute parity. But their depth is constant.

#theorem("Linear SSMs Are Below TC⁰")[
  State-space models with nonnegative gates (Mamba, Griffin, RWKV) cannot compute PARITY.#leanfile("TC0VsUnboundedRNN.lean:152")
]

#proof-sketch[
  The state update is $h_t = A(x_t) h_(t-1) + b(x_t)$ where $A(x_t)$ has nonnegative entries (typical for gating mechanisms). By the Perron-Frobenius theorem, nonnegative matrices have a dominant eigenvalue that is real and nonnegative. The state evolves as:
  $ h_T = product_(t=1)^T A(x_t) h_0 + sum_(t=1)^T product_(s=t+1)^T A(x_s) b(x_t) $
  Each term involves products of nonnegative matrices, yielding nonnegative contributions. PARITY requires the output to _alternate_ sign with each 1-bit: the output should be positive for even parity, negative for odd. But the nonnegative matrix products cannot produce the oscillating sign structure that parity requires. Contradiction.
]

Linear SSMs fall _below_ TC⁰---they cannot even compute the function that separates AC⁰ from TC⁰.

#theorem("E88 Exceeds TC⁰")[
  E88 with $T$ timesteps has effective depth $D times T$. For any constant bound $C$, there exists sequence length $T$ such that $D times T > C$.#leanfile("TC0VsUnboundedRNN.lean:127")
]

#proof[
  The key observation is that each timestep in E88 applies a nonlinear function (tanh) to the previous state: $S_t = tanh(alpha S_(t-1) + delta v_t k_t^top)$. For a $D$-layer E88 processing $T$ timesteps, the total number of sequential nonlinear operations is $D times T$.

  TC⁰ is defined by _constant_ depth---the depth cannot grow with input size. But E88's depth $D times T$ grows linearly with sequence length $T$, which is part of the input size. For any constant bound $C$, we can choose $T > C / D$ to exceed it.

  Therefore, E88 with unbounded sequence length cannot be simulated by TC⁰ circuits.
]

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
