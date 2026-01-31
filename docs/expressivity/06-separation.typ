// Section 6: The Computational Hierarchy

#import "traditional-math-style.typ": *

= The Computational Hierarchy

The pieces are now in place to state the complete picture. We have linear-temporal models that cannot compute parity. We have E88 that can. We have E23 that simulates Turing machines. How do these relate to each other, and to classical complexity theory?

== The Strict Hierarchy

#theorem("Strict Computational Hierarchy")[
  $ "Linear RNN" subset.neq "D-layer Linear-Temporal" subset.neq "E88" subset.neq "E23" $
  Each inclusion is proper: there exist functions computable at each level but not the previous one.
]#leanfile("MultiLayerLimitations.lean:365")

The witnesses are concrete. A depth-$(D+1)$ function separates $D$-layer linear-temporal from $(D+1)$-layer. Running parity separates any linear-temporal from E88. Arbitrary Turing machine computation separates E88 (bounded) from E23 (unbounded).

== Two Barriers, Two Escapes

The linear-temporal models face two fundamental barriers. Understanding them reveals why E88 succeeds where they fail.

#theorem("Affine Barrier")[
  Linear-temporal outputs are affine functions of their inputs. XOR is not affine: $"XOR"(0,0) + "XOR"(1,1) = 0 eq.not 2 = "XOR"(0,1) + "XOR"(1,0)$. Therefore running parity is impossible.
]#leanfile("LinearLimitations.lean:218")

E88 escapes through sign encoding. The parity is stored in whether $S$ is positive or negative. This is not an affine function of the input---it involves the nonlinear tanh.

#theorem("Continuity Barrier")[
  Linear-temporal outputs are continuous. Threshold is discontinuous. Therefore running threshold is impossible.
]#leanfile("ExactCounting.lean:344")

E88 escapes through saturation. The tanh approaches a step function as the input accumulates. While technically continuous, the transition becomes arbitrarily sharp---a soft threshold that approximates hard threshold arbitrarily well.

== E88's Constructions

The escape is constructive. We can write down the parameters that compute parity and threshold.

#theorem("E88 Computes Parity")[
  With $alpha = 1$ and $delta = 2$: input 0 preserves the sign of $S$, while input 1 flips the sign. The state sign encodes the running parity.
]#leanfile("TanhSaturation.lean:720")

The construction: input 0 gives $S' = tanh(S)$, preserving sign. Input 1 gives $S' = tanh(S + 2)$. When $S < 0$, this crosses zero to positive; when $S > 0$ and moderate, it grows toward $+1$. The dynamics implement XOR.

#theorem("E88 Computes Soft Threshold")[
  Accumulated positive inputs drive $S$ toward $+1$. Accumulated negative inputs drive $S$ toward $-1$. The transition steepens as accumulation grows.
]#leanfile("TanhSaturation.lean:424")

The construction: each input adds to the running sum inside the tanh. As the sum grows large and positive, $tanh$ saturates toward $+1$. As it grows large and negative, toward $-1$. The sign indicates whether the sum exceeds the threshold.

== The Complete Capability Table

We summarize what each computational class can and cannot do.

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

These are not empirical findings. They are theorems, verified in Lean 4. The "No" entries have proofs of impossibility; the "Yes" entries have explicit constructions.

#centerrule

The hierarchy is now established. Linear-temporal models form a proper subclass of E88, which forms a proper subclass of E23. The separations are witnessed by specific functions: parity separates linear from E88, and unbounded computation separates E88 from E23.

But there is a more familiar landmark in computational complexity: TC⁰, the class of constant-depth threshold circuits. Where do our architectures sit relative to this classical boundary? The answer is surprising.
