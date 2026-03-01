// E88 Architecture Reference
// A focused description of the E88 recurrent architecture

#set document(title: "E88 Architecture", author: "Elman Project")
#set page(margin: 1in)
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

= E88: Matrix-State Recurrent Architecture

E88 is a recurrent neural network architecture with *matrix state* and *nonlinear temporal dynamics*. Each layer maintains a square state matrix that evolves through time via element-wise tanh.

== Core Update Rule

For each head $h$ with state $S in RR^(d times d)$:

$ S_t = tanh(alpha dot S_(t-1) + delta dot v_t k_t^top) $

where:
- $S_t in RR^(d times d)$ — state matrix at time $t$
- $alpha in (0, 2)$ — retention/decay coefficient
- $delta in RR$ — input scaling factor
- $k_t = K x_t in RR^d$ — key vector (projected from input)
- $v_t = V x_t in RR^d$ — value vector (projected from input)
- $v_t k_t^top in RR^(d times d)$ — rank-1 outer product update
- $tanh$ — applied element-wise to the matrix

== Output Computation

Given query $q_t = Q x_t in RR^d$:

$ y_t = S_t dot q_t in RR^d $

The output is a matrix-vector product: the state matrix applied to the query.

== Multi-Head Structure

An E88 layer with $H$ heads:

$ y_t = W_O [S_t^1 q_t^1 ; S_t^2 q_t^2 ; dots.c ; S_t^H q_t^H] $

Each head has independent state $S^h$, projections $(K^h, V^h, Q^h)$, and parameters $(alpha^h, delta^h)$. Heads do not interact within the recurrence—only at output projection.

== Parameter Count

Per head with state dimension $d$ and input dimension $d_"in"$:
- Key projection $K$: $d times d_"in"$
- Value projection $V$: $d times d_"in"$
- Query projection $Q$: $d times d_"in"$
- Scalars $alpha, delta$: $2$
- State $S$: $d^2$ (not a parameter, but runtime memory)

Total per head: $3 d dot d_"in" + 2$ parameters, $d^2$ state memory.

== Key Properties

=== Matrix State Capacity

State is $d times d = d^2$ values per head. For $d = 64$: 4,096 values per head. With 8 heads: 32,768 state values per layer.

=== Rank Accumulation

Each update $v_t k_t^top$ is rank-1. After $T$ steps, cumulative rank can reach $min(T, d)$. The tanh nonlinearity mixes information across all $d^2$ entries, enabling full utilization of state capacity for $T >= d$.

=== Tanh Saturation (Latching)

When $|S_(i j)|$ approaches 1, $tanh'(S_(i j)) -> 0$. The state becomes stable—resistant to perturbation. This enables *binary latching*: once a state element saturates, it stays saturated.

=== Bifurcation at $alpha = 1$

- $alpha <= 1$: Single stable fixed point at $S = 0$. State decays.
- $alpha > 1$: Zero becomes unstable. Two stable fixed points at $plus.minus S^*$ emerge. State can latch to either.

=== Nonlinear Temporal Composition

The tanh applies at every timestep. Unlike linear recurrence (which collapses to a weighted sum), E88's state $S_T$ is a *nested* nonlinear function:

$ S_T = tanh(alpha dot tanh(alpha dot tanh(dots) + dots) + delta dot v_T k_T^top) $

This provides compositional depth $T$ per layer (one nonlinearity per timestep).

== Comparison to E1H (Elman / Vector-State RNN)

E1H is a multi-head Elman RNN with vector state, sitting strictly between linear SSMs and E88 in the expressivity hierarchy.

#table(
  columns: 3,
  align: (left, center, center),
  stroke: 0.5pt,
  [*Property*], [*E88*], [*E1H (Elman)*],
  [State shape], [$d times d$ matrix per head], [$d$-vector per head],
  [State capacity], [$d^2$ scalars per head], [$d$ scalars per head],
  [Update], [$tanh(alpha S + delta v k^top)$], [$tanh(W_x x + W_h h + b)$],
  [Decay factor], [$alpha in (0, 2)$ scalar], [None (folded into $W_h$)],
  [Input coupling], [Rank-1 outer product], [Full matrix-vector product],
  [Retrieval], [$S dot.op q$ (content-addressable)], [Fixed linear projection],
  [Temporal depth], [$T$ per layer], [$T$ per layer],
)

For $d >= 2$: E88 stores $d^2 > d$ scalars per head, giving strictly more memory capacity. The content-addressable retrieval $S dot.op q$ enables key-based addressing impossible in E1H's vector state. These capacity and retrieval differences establish E88 $supset$ E1H strictly; see `ExpressivityHierarchy.lean`.

== Comparison to Linear State-Space Models

#table(
  columns: 3,
  align: (left, center, center),
  stroke: 0.5pt,
  [*Property*], [*E88*], [*Mamba2 / Linear SSM*],
  [State shape], [$d times d$ matrix], [$n$-dim vector],
  [Update], [$tanh(alpha S + delta v k^top)$], [$A h + B x$],
  [Temporal dynamics], [Nonlinear], [Linear],
  [State retention], [Latches at $plus.minus 1$], [Decays as $alpha^t$],
  [Composition depth], [$T$ per layer], [$1$ per layer],
)

== Expressivity Hierarchy

Machine-verified in Lean 4 (`ExpressivityHierarchy.lean`):

#align(center)[
  #block[
    #table(
      columns: 5,
      align: center,
      stroke: none,
      [*Linear SSM*], [$space subset.neq space$], [*E1H*], [$space subset.neq space$], [*E88*],
      [Vector state], [], [Vector state], [], [Matrix state],
      [$d$ scalars], [], [$d$ scalars], [], [$d^2$ scalars],
      [Linear update], [], [Nonlinear tanh], [], [Nonlinear tanh],
      [Depth 1/layer], [], [Depth $T$/layer], [], [Depth $T$/layer],
    )
  ]
]

Separation witnesses:
- *Linear SSM $subset.neq$ E1H*: Threshold and XOR functions are linearly uncomputable; E1H's tanh saturation regime ($alpha in (1, 2)$) implements both.
- *E1H $subset.neq$ E88*: E88 state $d^2 > d$ strictly (for $d >= 2$); E88 supports content-addressable retrieval $S dot.op q$ that vector-state E1H cannot implement.

== Pseudocode

```python
class E88Head:
    def __init__(self, d, d_in):
        self.K = Linear(d_in, d)  # Key projection
        self.V = Linear(d_in, d)  # Value projection
        self.Q = Linear(d_in, d)  # Query projection
        self.alpha = Parameter(1.0)  # Retention
        self.delta = Parameter(0.1)  # Input scale
        self.S = zeros(d, d)  # State matrix

    def forward(self, x):
        k = self.K(x)  # [d]
        v = self.V(x)  # [d]
        q = self.Q(x)  # [d]

        # State update: S = tanh(α·S + δ·v⊗k)
        self.S = tanh(self.alpha * self.S + self.delta * outer(v, k))

        # Output: S @ q
        return self.S @ q  # [d]

class E88Layer:
    def __init__(self, H, d, d_in):
        self.heads = [E88Head(d, d_in) for _ in range(H)]
        self.W_O = Linear(H * d, d_in)

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        return self.W_O(concat(head_outputs))
```

== Variants

=== Gated E88

Add input-dependent gating on the update:

$ g_t &= sigma(W_g x_t) \
S_t &= tanh(alpha dot S_(t-1) + g_t circle.stroked.small (delta dot v_t k_t^top)) $

The gate $g_t in (0,1)^(d times d)$ controls which state elements receive the update.

=== Decoupled Key-Value Dimensions

Allow rectangular state $S in RR^(m times n)$ with $m eq.not n$:
- Key dimension: $n$
- Value dimension: $m$
- Enables asymmetric capacity allocation

=== Multi-Pass E88

Run $k$ forward passes, each reading previous output:

$ S_t^((p)) = tanh(alpha S_(t-1)^((p)) + delta dot v_t^((p)) k_t^((p) top)) $

where $v_t^((p)), k_t^((p))$ depend on output from pass $p-1$. Achieves composition depth $k dot T$.

== Computational Complexity

Per timestep, per head:
- Key/Value/Query projection: $3 dot d dot d_"in"$
- Outer product: $d^2$
- State scaling + addition: $2 dot d^2$
- Tanh: $d^2$
- Output matmul: $d^2$

Total: $O(d dot d_"in" + d^2)$ per head per timestep.

For $H$ heads over sequence length $T$: $O(H T (d dot d_"in" + d^2))$

== Empirical Validation: CMA-ES Architecture Search at 32K

To test whether E88's theoretical advantages manifest empirically, we ran CMA-ES hyperparameter optimization across architectures at equal scale (~480M parameters). Each evaluation trains in two phases: 10 minutes at 512 tokens, then 10 minutes at 32K tokens. The search is 5-dimensional (width, heads, depth, learning rate) with 200+ evaluations per model.

=== Results at 32K Context

#table(
  columns: 4,
  align: (left, center, center, center),
  stroke: 0.5pt,
  [*Model*], [*Type*], [*Loss \@ 32K*], [*Config*],
  [*E88 (n=16)*], [Nonlinear sequential], [*1.1000*], [dim=1280, h=240, d=20],
  [FLA-GDN], [Linear associative scan], [1.1345], [dim=1664, exp=3, d=13],
  [Mamba2], [Linear SSM], [1.1882], [dim=1920, d_state=208, d=21],
)

E88 leads by 0.034 nats over FLA-GDN and 0.088 nats over Mamba2.

=== Ranking Inversion

At 512 tokens, E88 trailed both linear baselines by ~0.04 nats. At 32K, the ranking inverts. This matches the theoretical prediction: nonlinear temporal composition depth scales with $T$, so E88's advantage grows with sequence length.

=== Architectural Shifts at Long Context

CMA-ES found that models prefer different shapes at 32K vs 512:

#table(
  columns: 4,
  align: (left, center, center, left),
  stroke: 0.5pt,
  [*Model*], [*512 Optimal*], [*32K Optimal*], [*Shift*],
  [E88 (n=16)], [h=141, d=25], [h=240, d=20], [More heads, shallower],
  [Mamba2], [d_state=96, d=25], [d_state=208, d=21], [2$times$ state, shallower],
  [FLA-GDN], [exp=2, d=17], [exp=3, d=13], [Wider, shallower],
)

E88 scales by adding independent heads (141 $arrow$ 240) while _reducing_ layer depth (25 $arrow$ 20). The optimizer discovered that temporal composition depth (from tanh, scaling with $T$) substitutes for layer depth. Mamba2 doubles its state dimension but still trails — more state within linear dynamics does not compensate.

E88 benefited most from 32K-aware search: 0.130 nat improvement vs 0.044 for FLA-GDN. The 512-optimal config was severely suboptimal for long context.

== Summary

E88 is defined by three choices:

1. *Matrix state* — $S in RR^(d times d)$ instead of vector $h in RR^n$
2. *Outer product update* — $v k^top$ writes to state via rank-1 addition
3. *Element-wise tanh* — nonlinearity compounds across time

These combine to give: large state capacity ($d^2$), addressable storage (via $S q$ retrieval), persistent memory (via saturation/latching), and deep temporal composition (depth $T$ per layer).
