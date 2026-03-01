// Section 4: E88 — Our Novel Architecture

#import "traditional-math-style.typ": *

= E88: A Novel Recurrent Architecture

This section introduces E88, a recurrent architecture we developed to overcome the fundamental limitations of existing state-space models. Where prior work accepts linear temporal dynamics as a necessary trade-off for parallelism, E88 demonstrates that nonlinear temporal composition and practical efficiency can coexist.

== The Problem: Linear State Decay

The dominant paradigm in efficient sequence modeling—embodied by Mamba2 @dao2024mamba2 and related state-space models—relies on linear recurrence for computational efficiency:

$ h_t = A_t h_(t-1) + B_t x_t $

This structure enables parallel computation through associative scans. But it comes with a fundamental cost that existing work has not adequately addressed.

#theorem("Linear State Decay")[
  In any linear recurrence with contraction factor $|alpha| < 1$, state decays exponentially: $|h_T| = O(alpha^T |h_0|)$. Information from timestep $0$ is attenuated by factor $alpha^T$ at timestep $T$.#leanfile("TanhSaturation.lean:338")
]

For $alpha = 0.95$ and $T = 100$, initial information is attenuated by factor $0.95^100 approx 0.006$. After 500 steps, it is effectively gone. This is not a training failure—it is a mathematical consequence of linear dynamics.

#theorem("No Stable Binary Storage")[
  A linear recurrence cannot store a binary fact (e.g., "flag was set") indefinitely. Either the state decays to zero ($|alpha| < 1$) or it explodes ($|alpha| > 1$). There is no stable retention.#leanfile("TanhSaturation.lean:360")
]

These limitations are structural. No amount of training, parameter tuning, or architectural elaboration within the linear-temporal paradigm can escape them. The impossibility is proven in Lean.

== Our Solution: E88

E88 addresses these limitations through two key innovations: _matrix state_ and _tanh saturation_. Together, these create a recurrent architecture with fundamentally different computational properties.

#definition("E88 State Update")[
  $ S_t := tanh(alpha dot.op S_(t-1) + delta dot.op v_t k_t^top) $
  where $S in RR^(d times d)$ is matrix state, $alpha in (0, 2)$ is the retention coefficient, $delta$ is input scaling, and $v_t k_t^top$ is the rank-1 outer product update from current input.#leanfile("E88Definition.lean:127")
]

The design choices are deliberate:

*Matrix state* ($d times d$ rather than vector $d$): For $d = 64$, each E88 head maintains $64^2 = 4096$ state values. A comparable Mamba2 model stores 64--256 values total. This capacity difference is not merely quantitative—it enables qualitatively different computation.

*Tanh nonlinearity applied element-wise*: Unlike linear SSMs where state update is a linear function of previous state, E88's tanh creates nonlinear temporal composition. Each timestep adds computational depth.

*Retention coefficient $alpha in (0, 2)$*: This parameter controls the dynamics. For $alpha <= 1$, the system has a single stable fixed point at zero. For $alpha > 1$, a bifurcation creates two additional stable fixed points at $plus.minus S^*$. This bifurcation is the mechanism for binary storage.

== The Key Insight: Full State Accessibility

The matrix structure of E88 state enables a form of computation impossible in vector-state models.

Consider retrieval in a linear SSM. The state $h_T = sum_t A^(T-t) B x_t$ is a weighted sum of past inputs. Retrieving information about $x_s$ requires the model to "look through" this sum—the information is _entangled_ with all other inputs.

In E88, the state $S in RR^(d times d)$ can be queried through matrix multiplication. Given query $q$:

$ "output" = S dot.op q $

This is a _linear_ operation on the _nonlinear_ state. The matrix structure means different information can be stored in different "addresses" (rows/columns) and retrieved independently.

#theorem("E88 Rank Accumulation")[
  Each update $v_t k_t^top$ is rank-1, contributing $2d$ degrees of freedom. After $T$ timesteps with linearly independent inputs, the cumulative information is $T times 2d$ values. The tanh nonlinearity mixes these across all $d^2$ matrix entries. For $T >= d$, the state can achieve full rank $d$, utilizing the full $d^2$-dimensional state space.#leanfile("E88RankAccumulation.lean:161")
]

This contrasts sharply with linear matrix recurrences, where rank growth is bounded by $min(T, d)$ regardless of sequence length.#leanfile("E88RankAccumulation.lean:114")

#centerrule

== Tanh Saturation: The Memory Mechanism

The tanh function is bounded in $(-1, 1)$ with derivative $tanh'(x) = 1 - tanh^2(x)$. Two properties follow:

#lemma("Vanishing Gradient at Saturation")[
  As $|tanh(x)| -> 1$, the derivative $tanh'(x) -> 0$. Near saturation, perturbations have diminishing effect.#leanfile("TanhSaturation.lean:87")
]

#theorem("Bifurcation at $alpha = 1$")[
  For the map $f(S) = tanh(alpha S)$:
  - When $alpha <= 1$: Zero is the unique fixed point and is globally attracting.
  - When $alpha > 1$: Zero becomes unstable. Two new stable fixed points emerge at $plus.minus S^*$ where $S^* = tanh(alpha S^*)$.#leanfile("AttentionPersistence.lean:212")
]

#proof-sketch[
  At $S = 0$, $f'(0) = alpha$. For $alpha <= 1$, $|f'(0)| <= 1$ and Banach fixed-point applies. For $alpha > 1$, zero is unstable. The function $g(S) = tanh(alpha S) - S$ satisfies $g(0) = 0$, $g'(0) = alpha - 1 > 0$, and $lim_(S -> infinity) g(S) = -infinity$. By intermediate value theorem, a positive root $S^*$ exists. Symmetry gives $-S^*$.
]

This bifurcation is the foundation of E88's memory capability.

#definition("Latched State")[
  A state value $S_(i j)$ is _latched_ if $|S_(i j)|$ exceeds a threshold (e.g., $|S_(i j)| > 0.9$) and remains in this region under the E88 dynamics.
]

#theorem("E88 Latching Persistence")[
  For $alpha in (0.9, 1.1)$ and small input scaling $|delta| < 0.1$: once $|S_(i j)| > 0.9$, subsequent updates with bounded inputs maintain $|S_(i j)| > 0.85$. The latched state persists.#leanfile("TanhSaturation.lean:204")
]

Compare this to linear decay:

#theorem("Linear State Cannot Latch")[
  In a linear recurrence $S_t = alpha S_(t-1)$ with $|alpha| < 1$: $|S_T| = |alpha|^T |S_0| -> 0$ exponentially. No stable non-zero state exists.#leanfile("TanhSaturation.lean:360")
]

The contrast is stark. E88 can store a binary fact ("the flag was set") indefinitely through latching. Linear-temporal models cannot—the fact decays.

== Capabilities Enabled by Nonlinear Temporal Dynamics

E88's architecture enables computations that are provably impossible for linear-temporal models.

=== Running Parity

#theorem("E88 Computes Running Parity")[
  With appropriate parameters ($alpha = 1$, $delta = 2$ in a simplified 1D case): input 0 preserves the sign of $S$; input 1 flips the sign. The sign encodes running parity $x_1 xor x_2 xor dots xor x_t$.#leanfile("TanhSaturation.lean:720")
]

This is impossible for linear-temporal models at any depth. The proof that linear recurrence cannot compute XOR is in `LinearLimitations.lean:315`. Running parity is a sequence of $T$ XOR operations—requiring $T$ nonlinear compositions, which a $D$-layer linear-temporal model cannot provide when $T > 2^D$.

=== Soft Threshold Detection

#theorem("E88 Computes Soft Threshold")[
  Accumulated positive signal drives $S$ toward $+1$. Accumulated negative signal drives toward $-1$. The transition sharpens as accumulation grows, approaching a hard threshold.#leanfile("TanhSaturation.lean:424")
]

Linear-temporal models cannot compute threshold functions because linear functions are continuous and threshold functions are discontinuous.#leanfile("LinearLimitations.lean:107")

=== Exact Counting Modulo n

#theorem("E88 Counts Modulo n")[
  For small $n <= 8$, E88 can maintain $n$ distinct stable states, counting inputs modulo $n$. The tanh creates $n$ attractors at approximately $-1 + 2i/(n-1)$ for $i = 0, dots, n-1$.#leanfile("E88Definition.lean:276")
]

Linear recurrence accumulates unboundedly or decays—it cannot maintain distinct stable count values.

== Comparison with Prior Architectures

#simpletable(
  columns: 4,
  align: (left, center, center, center),
  [*Property*], [*E88*], [*E1H (Elman)*], [*Mamba2 / Linear SSM*],
  [State structure], [Matrix $d times d$], [Vector $d$], [Vector $n$],
  [State capacity/head], [$d^2$ scalars], [$d$ scalars], [$n$ scalars],
  [Temporal dynamics], [Nonlinear (tanh)], [Nonlinear (tanh)], [Linear ($A h + B x$)],
  [Fixed points], [$0, plus.minus S^*$ (bistable)], [$0, plus.minus h^*$ (bistable)], [Decay or explosion],
  [Binary latching], [Yes], [Yes], [No (provably impossible)],
  [Running parity], [Yes], [Yes], [No (provably impossible)],
  [Threshold detection], [Yes], [Yes], [No (provably impossible)],
  [State retention], [Indefinite (latching)], [Indefinite (latching)], [Exponential decay],
  [Composition depth/layer], [$T$ (sequence length)], [$T$ (sequence length)], [$1$ (collapses)],
  [Content-addressable retrieval], [Yes ($S dot.op q$)], [No (vector state)], [No],
  [State capacity vs E88], [$d^2$ (reference)], [$d$ (strictly less)], [$n$ (strictly less)],
)

The entries marked "provably impossible" are not empirical observations—they are mathematical facts verified in Lean. The capacity separation ($d^2 > d$ for $d >= 2$) and retrieval separation are proven in `E1HDefinition.lean` and `E88ExceedsE1HCapacity.lean`; the full three-way hierarchy is assembled in `ExpressivityHierarchy.lean`.#leanfile("ExpressivityHierarchy.lean:267")

== Head Independence and Parallelism

#theorem("Parallel Head Evolution")[
  An $H$-head E88 is equivalent to $H$ independent state machines. Head $h_1$'s state depends only on its own parameters $(alpha_(h_1), delta_(h_1), K_(h_1), V_(h_1))$ and the input sequence—not on other heads' parameters or states.#leanfile("E88Definition.lean:389")
]

This independence has two implications:

1. *Computational parallelism*: All heads can be computed simultaneously.
2. *Compositional capacity*: Each head tracks independent information. With $H$ heads, E88 can track $H$ separate binary facts, count $H$ different quantities, or compute $H$ independent parities.

== Attention Persistence

#definition("Alert State")[
  A head is in an _alert state_ if some state element is saturated: $exists (i,j): |S_(i j)| > 0.9$.
]

#theorem("Alert State Persistence")[
  Once a head enters an alert state, it remains alert across subsequent timesteps with bounded inputs. The saturated element's magnitude remains above threshold.#leanfile("E88Definition.lean:433")
]

This enables _event detection with memory_: a head can detect an event (entering alert state) and maintain awareness of it indefinitely. Linear-temporal models cannot implement this—any "detection" signal decays exponentially.

#centerrule

== Formal Guarantees

All claims in this section are backed by machine-verified proofs in Lean 4. The key formalizations:

- `E88Definition.lean`: Core architecture, state update, head independence
- `TanhSaturation.lean`: Bifurcation analysis, latching persistence, parity/threshold computation
- `E88RankAccumulation.lean`: Matrix rank growth under E88 dynamics
- `AttentionPersistence.lean`: Fixed point analysis, alert state persistence
- `LinearLimitations.lean`: Impossibility of threshold/XOR/parity for linear recurrence
- `LinearCapacity.lean`: Linear state as weighted sum, exponential decay
- `E1HDefinition.lean`: E1H (Elman) architecture with vector state; capacity comparison $d$ vs $d^2$
- `E1HTemporalTheorems.lean`: E1H temporal depth $T$, saturation regime, threshold capability
- `E88ExceedsE1HCapacity.lean`: Proof that E88 matrix state strictly exceeds E1H vector state
- `ExpressivityHierarchy.lean`: Full three-way hierarchy: Linear SSM $subset.neq$ E1H $subset.neq$ E88

The repository contains machine-verified proofs establishing two strict containments: linear SSMs cannot compute threshold or XOR (separation from E1H), and E1H vector state has strictly less capacity than E88 matrix state and cannot implement content-addressable retrieval (separation from E88).

== Summary

E88 is not an incremental modification of existing architectures. It represents a fundamentally different approach to recurrent computation:

1. *Matrix state* provides $d^2$ storage capacity vs linear models' $d$
2. *Tanh saturation* creates stable binary storage through bistable fixed points
3. *Nonlinear temporal composition* provides depth $T$ per layer vs depth $1$
4. *Full state accessibility* through $S dot.op q$ retrieval enables random access

The theoretical advantages are not conjectures—they are proven. Functions computable by single-layer E88 (parity, threshold, exact counting) are provably impossible for any-depth linear-temporal models.

The next section examines where E88 fits in the computational hierarchy: strictly more powerful than linear SSMs, comparable to bounded-depth circuits, but still short of full Turing completeness.
