// Section 4: E88 Temporal Nonlinearity

#import "traditional-math-style.typ": *

= E88: Escaping the Linear Barrier

The answer to the question posed at the end of the previous section lies in two properties: _matrix state_ and _tanh saturation_. Together, they give E88 capabilities that linear-temporal models provably lack.

== Matrix State

Where Mamba2 maintains a state vector of dimension $n$, E88 maintains a state _matrix_ of dimension $d times d$.

#definition("E88 State Update")[
  $ S_t := tanh(alpha dot.op S_(t-1) + delta dot.op v_t k_t^top) $
  where $S in RR^(d times d)$ is matrix state, $alpha in (0, 2)$ is the retention coefficient, and $v_t k_t^top$ is the rank-1 outer product update from current input.
]

The capacity difference is immediate. For $d = 64$, each E88 head stores $64^2 = 4096$ values. Mamba2 stores 64--256 total across its state dimension. An 8-head E88 maintains 32,768 state values versus Mamba2's few hundred---a gap of two orders of magnitude.

But capacity alone does not explain the expressivity gap. A large linear system is still linear. The crucial ingredient is what happens inside the tanh.

== Tanh Saturation and the Bifurcation

The tanh function is bounded: its output lies in $(-1, 1)$. As the input grows large, tanh approaches $plus.minus 1$ and its derivative approaches zero. This _saturation_ regime is where E88's power emerges.

#lemma("Tanh Boundedness")[
  For all $x in RR$: $|tanh(x)| < 1$.
]#leanfile("TanhSaturation.lean:42")

#proof[
  Recall $tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))$. The numerator has magnitude strictly less than the denominator for all finite $x$.
]

#lemma("Tanh Derivative Vanishes at Saturation")[
  $tanh'(x) = 1 - tanh^2(x)$. As $|x| -> infinity$, we have $tanh'(x) -> 0$.
]#leanfile("TanhSaturation.lean:87")

#proof[
  The derivative formula is standard calculus. Since $|tanh(x)| -> 1$ as $|x| -> infinity$, we have $tanh'(x) = 1 - tanh^2(x) -> 0$.
]

This derivative property is key: perturbations to a saturated state have vanishing effect. The state is _stable_ near $plus.minus 1$.

#definition("Fixed Point of Tanh Recurrence")[
  A value $S^*$ is a fixed point of the map $f(S) = tanh(alpha S)$ if $S^* = tanh(alpha S^*)$.
]

#theorem("Zero Is Always a Fixed Point")[
  $S^* = 0$ is a fixed point for all $alpha$.
]

#proof[
  $tanh(alpha dot.op 0) = tanh(0) = 0$.
]

#theorem("Unique Fixed Point for $alpha <= 1$")[
  For $alpha <= 1$, zero is the unique fixed point, and it is globally attracting.
]#leanfile("AttentionPersistence.lean:156")

#proof-sketch[
  The map $f(S) = tanh(alpha S)$ is a contraction when $|f'(S)| < 1$ everywhere. At $S = 0$: $f'(0) = alpha$. For $alpha <= 1$, the slope at the fixed point is at most 1. Since $|tanh'(x)| <= 1$ everywhere and $|tanh(x)| < |x|$ for $x eq.not 0$, the Banach fixed-point theorem applies: every orbit converges to zero.
]

#theorem("Nonzero Fixed Points for $alpha > 1$")[
  For $alpha > 1$, there exist two additional fixed points $plus.minus S^*$ with $S^* > 0$. Zero becomes unstable (repelling), and $plus.minus S^*$ become stable (attracting).
]#leanfile("AttentionPersistence.lean:212")

#proof-sketch[
  For $alpha > 1$, the derivative at zero is $f'(0) = alpha > 1$, so zero is unstable. Consider $g(S) = tanh(alpha S) - S$. We have $g(0) = 0$, $g'(0) = alpha - 1 > 0$, and $lim_(S -> infinity) g(S) = 1 - infinity = -infinity$. By the intermediate value theorem, there exists $S^* > 0$ with $g(S^*) = 0$. By symmetry, $-S^*$ is also a fixed point. Stability follows from $|f'(plus.minus S^*)| < 1$, which holds because tanh is strictly concave on $(0, infinity)$.
]

This bifurcation is the mathematical core of E88's memory capability. When $alpha > 1$, the system has _bistability_: two stable states that the dynamics can settle into and remain in. A sufficiently strong input can push the state from one basin to the other, where it stays until pushed again.

In contrast, linear systems have no such structure. The recurrence $S_t = alpha S_(t-1)$ with $|alpha| < 1$ decays exponentially to zero. With $|alpha| >= 1$, the state explodes. There is no stable nonzero memory.

== Latching: The Memory Mechanism

The bifurcation theorem tells us that stable nonzero states exist. Latching tells us that E88 can use them.

#definition("Latched State")[
  A state $S$ is _latched_ if $|S| > 1 - epsilon$ for some small $epsilon > 0$ and the state remains in this region under the dynamics.
]

#theorem("E88 Latched State Persistence")[
  For $alpha > 1$, once $|S|$ enters a neighborhood of the stable fixed point $plus.minus S^*$, it remains there indefinitely in the absence of strong external input.
]#leanfile("TanhSaturation.lean:320")

#proof-sketch[
  The basin of attraction of $S^*$ contains all points $S$ with $S > 0$ and $S$ closer to $S^*$ than to 0. Since $|f'(S^*)| < 1$, orbits in the basin converge exponentially to $S^*$.
]

#theorem("Linear State Exponential Decay")[
  In a linear system $S_t = alpha S_(t-1)$ with $|alpha| < 1$: $S_T = alpha^T S_0 -> 0$ exponentially fast.
]

#proof[
  Direct computation: $|S_T| = |alpha|^T |S_0| -> 0$ as $T -> infinity$ since $|alpha| < 1$.
]

#theorem("E88 Latches; Linear Decays")[
  In E88: once $|S|$ approaches the stable fixed point, the state persists indefinitely. The tanh derivative vanishes at saturation, so perturbations have vanishing effect.

  In linear systems: without continuous reinforcement, the state decays to zero. There is no permanent memory.
]#leanfile("TanhSaturation.lean:360")

This is the mechanism behind E88's ability to "remember" a binary fact. Once the state saturates near the stable fixed point, it stays there. The model has latched onto a decision. In a linear-temporal model, the same decision would gradually fade.

== Computing Parity

With latching in hand, we can show that E88 computes running parity---the function that linear-temporal models provably cannot compute.

#theorem("E88 Computes Parity")[
  With $alpha = 1$ and $delta = 2$: an input of 0 preserves the sign of $S$ (since $tanh(S)$ has the same sign as $S$), while an input of 1 flips the sign (since $tanh(S + 2)$ crosses zero when $S$ is negative). The sign of $S$ encodes the running parity.
]#leanfile("TanhSaturation.lean:720") // e88_computes_running_parity

The construction is elegant. We use the sign of the state---positive or negative---to encode even or odd. Each 0-input preserves the sign. Each 1-input flips it. The state sign is the running XOR.

This is impossible for linear-temporal models. We proved it. Yet a single-layer E88 achieves it with a simple parameter choice.

== Computing Soft Threshold

Threshold requires a similar but distinct mechanism.

#theorem("E88 Computes Soft Threshold")[
  Accumulated positive signal drives $S$ toward $+1$. Accumulated negative signal drives $S$ toward $-1$. The transition between regimes is a soft step function, approaching a hard threshold as the accumulation grows.
]#leanfile("TanhSaturation.lean:424") // e88_can_count_mod

A linear-temporal model cannot produce a step. E88 can: the tanh saturation creates an increasingly sharp transition as the input accumulates.

== Head Independence

E88's multi-head structure provides an additional dimension of capacity.

#theorem("Parallel State Machines")[
  An $H$-head E88 is equivalent to $H$ independent state machines. Head $h$'s update depends only on its own state $S^((h))$ and its own input projection. The heads do not interact within a layer.
]#leanfile("MultiHeadTemporalIndependence.lean:129") // e88_head_update_independent

Each head can track a different binary fact, a different parity, a different threshold. The $H$-head system has $H$ independent bistable bits plus the full matrix state within each head.

== The Separation

We can now state the formal separation between E88 and linear-temporal models.

#simpletable(
  columns: 3,
  align: (left, center, center),
  [*Property*], [*E88*], [*Linear-Temporal*],
  [State], [Matrix $d^2$], [Vector $n$],
  [Dynamics], [Nonlinear (tanh)], [Linear],
  [Fixed points], [$0, plus.minus S^*$], [Decay or explosion],
  [Latching], [Yes], [No],
  [Threshold/Parity], [Yes], [No],
  [Composition depth], [$D times T$], [$D$],
)

The gap is not a matter of degree. It is a qualitative difference in computational class. Functions computable by a 1-layer E88 are provably impossible for any $D$-layer linear-temporal model. Both the matrix state capacity and the tanh nonlinearity are required: a large linear system cannot latch, and a small nonlinear system lacks the capacity to track complex state.

#centerrule

We have answered the question: E88 escapes the linear barrier through tanh saturation. The bifurcation at $alpha = 1$ creates bistable fixed points. Latching exploits this structure for permanent memory. Parity and threshold, impossible for linear-temporal models, become achievable with simple parameter choices.

The next question is: how far does this escape extend? E88 exceeds linear-temporal models. Does it exceed Transformers? Does it reach Turing completeness? The answer requires placing these architectures in the landscape of circuit complexity.
