// Section 4: E88 Temporal Nonlinearity

#import "traditional-math-style.typ": *

= E88: Escaping the Linear Barrier

Two properties give E88 capabilities that linear-temporal models provably lack: _matrix state_ and _tanh saturation_.

== Matrix State

Where Mamba2 maintains a state vector of dimension $n$, E88 maintains a state _matrix_ of dimension $d times d$.

#definition("E88 State Update")[
  $ S_t := tanh(alpha dot.op S_(t-1) + delta dot.op v_t k_t^top) $
  where $S in RR^(d times d)$ is matrix state, $alpha in (0, 2)$ is the retention coefficient, and $v_t k_t^top$ is the rank-1 outer product update from current input.
]

For $d = 64$, each E88 head stores $64^2 = 4096$ values. Mamba2 stores 64--256 total. An 8-head E88 maintains 32,768 state values versus Mamba2's few hundred.

But capacity alone does not explain the gap. A large linear system is still linear. The crucial ingredient is tanh saturation.

== Tanh Saturation and the Bifurcation

The tanh function is bounded in $(-1, 1)$. As input grows large, tanh approaches $plus.minus 1$ and its derivative approaches zero.

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
  Since $|tanh(x)| -> 1$ as $|x| -> infinity$, we have $tanh'(x) = 1 - tanh^2(x) -> 0$.
]

Perturbations to a saturated state have vanishing effect. The state is stable near $plus.minus 1$.

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
  For $alpha <= 1$, zero is the unique fixed point and is globally attracting.
]#leanfile("AttentionPersistence.lean:156")

#proof-sketch[
  At $S = 0$: $f'(0) = alpha <= 1$. Since $|tanh'(x)| <= 1$ and $|tanh(x)| < |x|$ for $x eq.not 0$, the Banach fixed-point theorem applies.
]

#theorem("Nonzero Fixed Points for $alpha > 1$")[
  For $alpha > 1$, there exist two additional fixed points $plus.minus S^*$ with $S^* > 0$. Zero becomes unstable; $plus.minus S^*$ are stable.
]#leanfile("AttentionPersistence.lean:212")

#proof-sketch[
  For $alpha > 1$, $f'(0) = alpha > 1$, so zero is unstable. Consider $g(S) = tanh(alpha S) - S$. We have $g(0) = 0$, $g'(0) = alpha - 1 > 0$, and $lim_(S -> infinity) g(S) = -infinity$. By the intermediate value theorem, there exists $S^* > 0$ with $g(S^*) = 0$. By symmetry, $-S^*$ is also a fixed point.
]

When $alpha > 1$, the system has bistability: two stable states. A strong input can push the state from one basin to the other, where it stays.

Linear systems have no such structure. The recurrence $S_t = alpha S_(t-1)$ with $|alpha| < 1$ decays to zero. With $|alpha| >= 1$, the state explodes.

== Latching: The Memory Mechanism

#definition("Latched State")[
  A state $S$ is _latched_ if $|S| > 1 - epsilon$ for small $epsilon > 0$ and remains in this region.
]

#theorem("E88 Latched State Persistence")[
  For $alpha > 1$, once $|S|$ enters a neighborhood of $plus.minus S^*$, it remains there absent strong external input.
]#leanfile("TanhSaturation.lean:320")

#proof-sketch[
  Since $|f'(S^*)| < 1$, orbits in the basin of attraction converge exponentially to $S^*$.
]

#theorem("Linear State Exponential Decay")[
  In a linear system $S_t = alpha S_(t-1)$ with $|alpha| < 1$: $S_T = alpha^T S_0 -> 0$ exponentially fast.
]

#proof[
  Direct computation: $|S_T| = |alpha|^T |S_0| -> 0$ as $T -> infinity$ since $|alpha| < 1$.
]

#theorem("E88 Latches; Linear Decays")[
  In E88: once $|S|$ approaches the stable fixed point, the state persists. The tanh derivative vanishes at saturation.

  In linear systems: the state decays to zero without continuous reinforcement.
]#leanfile("TanhSaturation.lean:360")

Once the state saturates near the stable fixed point, it stays there. In a linear-temporal model, the decision would fade.

== Computing Parity

#theorem("E88 Computes Parity")[
  With $alpha = 1$ and $delta = 2$: input 0 preserves the sign of $S$; input 1 flips the sign. The sign encodes running parity.
]#leanfile("TanhSaturation.lean:720")

The sign of the state encodes even or odd. Each 0-input preserves the sign, each 1-input flips it.

This is impossible for linear-temporal models. A single-layer E88 achieves it.

== Computing Soft Threshold

#theorem("E88 Computes Soft Threshold")[
  Accumulated positive signal drives $S$ toward $+1$. Accumulated negative signal drives $S$ toward $-1$. The transition approaches a hard threshold as accumulation grows.
]#leanfile("TanhSaturation.lean:424")

E88 creates an increasingly sharp transition as input accumulates.

== Head Independence

#theorem("Parallel State Machines")[
  An $H$-head E88 is equivalent to $H$ independent state machines. Heads do not interact within a layer.
]#leanfile("MultiHeadTemporalIndependence.lean:129")

Each head can track a different binary fact, parity, or threshold.

== The Separation

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

Functions computable by a 1-layer E88 are provably impossible for any $D$-layer linear-temporal model. Both matrix state capacity and tanh nonlinearity are required.

#centerrule

E88 escapes the linear barrier through tanh saturation. The bifurcation at $alpha = 1$ creates bistable fixed points. Latching exploits this for permanent memory.

Does E88 exceed Transformers? Does it reach Turing completeness?
