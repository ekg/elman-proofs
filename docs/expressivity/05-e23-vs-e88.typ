// Section 5: E23 vs E88

#import "traditional-math-style.typ": *

= Two Paths Beyond Linear

E88 is not the only architecture to escape linear-temporal limitations. E23 takes a different route---explicit tape memory rather than implicit saturation dynamics. The comparison illuminates what is possible and what is practical.

== E23: The Tape-Based Approach

E23 maintains explicit memory slots, accessed through attention-like addressing.

#definition("E23 Dynamics")[
  _Read_: $r_t = sum_(i=1)^N alpha_i h_"tape"^((i))$, an attention-weighted sum over $N$ tape slots.

  _Update_: $h_"work"' = tanh(W_h h_"work" + W_x x_t + r_t + b)$, incorporating the read result.

  _Write_: $h_"tape"^((j))' = (1 - beta_j) h_"tape"^((j)) + beta_j h_"work"'$, selectively updating tape slots.
]

With hard attention (winner-take-all addressing), this becomes a Turing machine simulator.

#theorem("E23 is UTM-Complete")[
  E23 with $N$ tape slots and hard attention simulates any Turing machine using at most $N$ tape cells.#leanfile("E23_DualMemory.lean:730") // e23_is_utm
]

The proof constructs the simulation explicitly: each tape slot holds one cell, the attention mechanism implements head movement, and the working state encodes the finite control. The correspondence is exact.

== E88: The Saturation-Based Approach

E88 achieves its expressivity without explicit tape. The matrix state and tanh saturation create implicit structure.

The approaches differ in their source of power. E23's memory is explicit, addressable, and permanent. Writing to tape slot $j$ leaves the content there indefinitely; it never decays. E88's memory is implicit, distributed across the matrix entries, and dynamically stable. Saturation prevents decay, but the stability is emergent rather than designed-in.

The capacity formulas differ accordingly. E23 with $N$ slots of dimension $D$ has $N times D$ capacity. E88 with $H$ heads of dimension $d$ has $H times d^2$ capacity. For typical parameters, E88's capacity is larger, but it is less flexibly addressable.

== The Trade-offs

Why would anyone choose E88 over E23, given that E23 is Turing-complete and E88 is not?

The answer lies in training dynamics. E23's tape creates memory bandwidth pressure: $N times D$ reads and writes per step, with discrete addressing decisions at each step. Hard attention is non-differentiable; replacing it with soft attention loses the exactness that made E23 Turing-complete.

E88's matrix operations, by contrast, are naturally differentiable. The tanh is smooth everywhere. The gradient flows through the recurrence without discontinuities. The operations are also GPU-friendly: dense matrix multiplications achieve high utilization, while E23's variable addressing patterns create irregular memory access.

#keyinsight[
  E23 is Turing-complete but hard to train. E88 is sub-UTM but trainable. The mechanisms that give E23 theoretical power---hard addressing, explicit tape---are exactly what make it difficult to optimize.
]

== The Expressivity-Trainability Frontier

This trade-off is a recurring theme in neural architecture design. Greater expressivity often comes with harder optimization. The frontier is not empty: E88 finds a point that exceeds linear-temporal models while remaining differentiable.

The choice depends on the task. For problems requiring exact symbolic manipulation---compiler verification, theorem proving, arbitrary algorithm execution---E23's Turing completeness is necessary. For problems where approximate computation suffices and training efficiency matters---language modeling, retrieval, pattern recognition---E88's position on the frontier is more favorable.

#centerrule

We have seen two escape routes from linear-temporal limitations. E23 achieves full Turing completeness through explicit tape, at the cost of training difficulty. E88 achieves bounded superiority through implicit saturation, while remaining differentiable. Both exceed what Mamba2 and linear attention can compute. Neither dominates the other across all dimensions.

The next question is: where do these architectures sit in the classical hierarchy of computational complexity? The answer requires connecting neural network expressivity to circuit complexity.
