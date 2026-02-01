// Section 6: The Computational Hierarchy

#import "traditional-math-style.typ": *

= The Computational Hierarchy

We have linear-temporal models that cannot compute parity, E88 that can, and E23 that simulates Turing machines.

== The Strict Hierarchy

#theorem("Strict Computational Hierarchy")[
  $ "Linear RNN" subset.neq "D-layer Linear-Temporal" subset.neq "E88" subset.neq "E23" $
]#leanfile("MultiLayerLimitations.lean:365")

A depth-$(D+1)$ function separates $D$-layer from $(D+1)$-layer. Running parity separates linear-temporal from E88. Unbounded computation separates E88 from E23.

== Two Barriers, Two Escapes

#theorem("Affine Barrier")[
  Linear-temporal outputs are affine. XOR is not: $"XOR"(0,0) + "XOR"(1,1) = 0 eq.not 2 = "XOR"(0,1) + "XOR"(1,0)$.
]#leanfile("LinearLimitations.lean:218")

E88 escapes through sign encoding. Parity is stored in whether $S$ is positive or negative.

#theorem("Continuity Barrier")[
  Linear-temporal outputs are continuous. Threshold is discontinuous.
]#leanfile("ExactCounting.lean:344")

E88 escapes through saturation. The tanh approaches a step function as input accumulates.

== E88's Constructions

#theorem("E88 Computes Parity")[
  With $alpha = 1$ and $delta = 2$: input 0 preserves the sign of $S$; input 1 flips the sign.
]#leanfile("TanhSaturation.lean:720")

Input 0 gives $S' = tanh(S)$. Input 1 gives $S' = tanh(S + 2)$, which crosses zero when $S < 0$.

#theorem("E88 Computes Soft Threshold")[
  Accumulated positive inputs drive $S$ toward $+1$. Accumulated negative inputs drive $S$ toward $-1$.
]#leanfile("TanhSaturation.lean:424")

Each input adds to the running sum inside the tanh. The sign indicates whether the sum exceeds threshold.

== The Complete Capability Table

#simpletable(
  columns: 5,
  align: (left, center, center, center, center),
  [*Capability*], [*Linear RNN*], [*D-layer Lin.*], [*E88*], [*E23*],
  [Running parity], [No], [No], [Yes], [Yes],
  [Running threshold], [No], [No], [Yes], [Yes],
  [Binary latching], [No], [No], [Yes], [Yes],
  [Arbitrary FSM], [No], [No], [Yes], [Yes],
  [UTM simulation], [No], [No], [No], [Yes],
)

These are theorems, verified in Lean 4. "No" entries have impossibility proofs; "Yes" entries have constructions.

#centerrule

Parity separates linear from E88. Unbounded computation separates E88 from E23.

Where do these architectures sit relative to TC⁰?
