// Section 10: Multi-Pass RNN Model
// k-Pass Random Access, Tape Modification, and E88 Multi-Pass Computational Class

#let theorem(title, body) = block(
  fill: rgb("#f0f7ff"),
  stroke: rgb("#3366cc"),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
)[
  #strong[Theorem (#title)]#linebreak()
  #body
]

#let definition(title, body) = block(
  fill: rgb("#fff7f0"),
  stroke: rgb("#cc6633"),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
)[
  #strong[Definition (#title)]#linebreak()
  #body
]

#let lemma(title, body) = block(
  fill: rgb("#f7fff0"),
  stroke: rgb("#66cc33"),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
)[
  #strong[Lemma (#title)]#linebreak()
  #body
]

#let corollary(title, body) = block(
  fill: rgb("#f7f0ff"),
  stroke: rgb("#9933cc"),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
)[
  #strong[Corollary (#title)]#linebreak()
  #body
]

#let proof(body) = block(
  inset: (left: 1em),
)[
  _Proof._ #body #h(1fr) $square$
]

#let lean(code) = raw(code, lang: "lean4", block: true)

= Section 10: Multi-Pass RNN Model

_k-Pass Sequential Access, Soft Random Access, and Computational Efficiency_

== 10.1 Overview

The previous sections established that single-pass models face fundamental trade-offs:
- *Fixed-state RNNs* (E88, Mamba2): O(1) memory, limited to regular languages
- *Single-pass with feedback*: O(T) memory, DTIME(T) power, sequential tape access
- *Transformers with CoT*: O(T) memory, DTIME(T) power, random tape access via attention

This section formalizes a middle ground: *multi-pass RNNs* that re-process the input sequence multiple times. This architecture provides:

1. *O(k) soft random access*: With k passes, any position can be reached in O(k) sequential traversals
2. *Tape modification between passes*: Output from pass $i$ becomes input to pass $i+1$
3. *E88 multi-pass computational class*: Intermediate between single-pass and unbounded
4. *Practical efficiency trade-offs*: O(kT) sequential vs O(T) parallel

== 10.2 Multi-Pass Architecture

#definition("k-Pass RNN")[
  A *k-pass RNN* processes an input sequence $x_1, ..., x_T$ by making $k$ sequential passes over the data:

  $ "Pass" 1: quad h^1_t = f(h^1_(t-1), x_t) $
  $ "Pass" 2: quad h^2_t = f(h^2_(t-1), x_t, y^1_t) $
  $ dots.v $
  $ "Pass" k: quad h^k_t = f(h^k_(t-1), x_t, y^(k-1)_t) $

  where:
  - $h^i_t$ is the hidden state at timestep $t$ in pass $i$
  - $y^i_t$ is the output at timestep $t$ in pass $i$
  - Each pass reads the original input $x_t$ plus the previous pass's output $y^(i-1)_t$
]

#definition("Inter-Pass Tape")[
  The *inter-pass tape* $Y^i = (y^i_1, ..., y^i_T)$ is the sequence of outputs from pass $i$.

  Key property: Pass $i+1$ has *full read access* to $Y^i$ at each position, effectively creating a "tape" that can be modified between passes.
]

== 10.3 k-Pass Provides O(k) Soft Random Access

The fundamental insight: while a single-pass RNN can only access the current position, a k-pass RNN can *write markers* in early passes that guide later passes to specific positions.

#definition("Soft Random Access")[
  A model has *soft random access* with cost $c$ if, for any position $p$ in a sequence of length $T$:
  - The model can retrieve information from position $p$
  - The retrieval requires at most $c$ operations (or $c$ passes over the data)
]

#theorem("k-Pass RNN Achieves O(k) Soft Random Access")[
  A k-pass RNN can access any position $p$ in the input sequence with cost O(k) passes.

  *Construction*:
  1. *Pass 1 (mark phase)*: Write position markers at each location
  2. *Pass 2 (locate phase)*: Identify the target position $p$ and mark it specially
  3. *Pass 3 (retrieve phase)*: Copy the value at position $p$ to all subsequent positions

  For $k >= 3$, any single position can be accessed. More generally, $floor(k/3)$ independent accesses can be performed.
]

#proof[
  We construct the k-pass algorithm explicitly:

  *Pass 1*: For each position $t$, output the position index: $y^1_t = t$

  *Pass 2*: For each position $t$, compare with target $p$ (which can be computed from input or a query):
  $ y^2_t = cases(1 "if" t = p, 0 "otherwise") $
  This marks position $p$ with a special flag.

  *Pass 3*: Maintain a "carry" variable that latches when it sees the marker:
  $ y^3_t = cases(x_p "if" y^2_t = 1 "or" h^3_(t-1) "is carrying", h^3_(t-1) "otherwise") $

  After pass 3, position $p$'s value is available in the state for all subsequent computation.

  Since each phase requires one pass and we need 3 phases, the access cost is O(3) = O(k) for $k >= 3$.
]

#corollary("k-Pass Can Simulate k/3 Independent Random Accesses")[
  With $k$ passes, a multi-pass RNN can perform $floor(k/3)$ independent position lookups, each potentially to a different target position.
]

== 10.4 Tape Modification Between Passes

Unlike fixed-tape models (like traditional Turing machines where the tape persists), multi-pass RNNs can *completely rewrite* the inter-pass tape between passes.

#definition("Tape Modification Operations")[
  Between pass $i$ and pass $i+1$, the tape $Y^i$ can undergo:

  1. *Read*: $Y^i_t$ is read at position $t$ during pass $i+1$
  2. *Transform*: $Y^(i+1)_t = g(Y^i_t, x_t, h^(i+1)_t)$ for some function $g$
  3. *Insert (virtual)*: By writing longer outputs, simulate inserting new cells
  4. *Delete (virtual)*: By writing special "skip" markers, simulate deletion
]

#theorem("Tape Transformation Power")[
  A single pass can implement any computable transformation $g: {0,1}^T -> {0,1}^T$ on the tape contents, subject to the constraint that position $t$'s output can only depend on positions $1, ..., t$ (causality).

  *Lean formalization sketch*:
  #lean("def tapeTransform (g : (Fin T → Bool) → (Fin T → Bool))
    (causal : ∀ t, g(tape) t depends only on tape[0..t]) :
    CausalTapeTransform T")
]

#proof[
  The RNN state at position $t$ has access to:
  - All previous tape values $Y^i_1, ..., Y^i_(t-1)$ (via accumulation in state)
  - Current tape value $Y^i_t$
  - Original input $x_t$

  Any causal function of these can be computed with sufficient state capacity (by the universality of RNNs with nonlinear activation).
]

#lemma("Insert and Delete via Marking")[
  Insert and delete operations can be simulated with a 2-pass scheme:

  *Insert at position p*:
  - Pass A: Mark position $p$ with "insert here" flag
  - Pass B: When reading the tape, split the output to accommodate the insertion

  *Delete at position p*:
  - Pass A: Mark position $p$ with "delete" flag
  - Pass B: Skip over marked positions, compressing the effective tape
]

== 10.5 E88 Multi-Pass Computational Class

We now characterize the computational power of multi-pass E88 specifically.

#definition("MULTIPASS(k, T)")[
  The computational class *MULTIPASS(k, T)* consists of all decision problems solvable by a k-pass RNN on inputs of length T with:
  - Fixed state dimension (independent of T)
  - k sequential passes over the input
  - Each pass runs in O(T) time

  Total time complexity: O(kT).
]

#theorem("Single-Pass E88 ⊂ MULTIPASS(2, T)")[
  Every problem solvable by single-pass E88 with state dimension $n$ is also solvable by a 2-pass RNN with state dimension $O(n)$.

  Additionally, MULTIPASS(2, T) contains problems not solvable by any single-pass E88.
]

#proof[
  Containment: A single-pass E88 is trivially a special case of 2-pass where the second pass ignores the first pass's output.

  Strict containment: Consider the problem "Is position T/2's value equal to position T's value?"
  - Single-pass E88 must store position T/2's value for T/2 steps, requiring Ω(log T) bits of precision as decay occurs
  - 2-pass E88: Pass 1 writes all values to tape; Pass 2 compares positions T/2 and T directly
]

#theorem("MULTIPASS Hierarchy")[
  For fixed state dimension:
  $ "MULTIPASS"(1, T) subset.eq "MULTIPASS"(2, T) subset.eq dots.c subset.eq "MULTIPASS"(k, T) subset.eq dots.c subset.eq "DTIME"(T^2) $

  Each inclusion is strict for sufficiently large T.
]

#proof[
  The strict inclusions follow from the number of independent random accesses each can perform:
  - MULTIPASS(1, T): 0 random accesses (fully sequential)
  - MULTIPASS(3, T): 1 random access
  - MULTIPASS(3k, T): k random accesses

  Problems requiring $k+1$ independent random accesses separate MULTIPASS(3k, T) from MULTIPASS(3(k+1), T).
]

#definition("E88-MULTIPASS(k)")[
  *E88-MULTIPASS(k)* is the class of problems solvable by k-pass E88 with:
  - State update: $S^i_t = tanh(alpha S^i_(t-1) + delta k^i_t + gamma y^(i-1)_t)$
  - Inter-pass tape $Y^i$ with tanh-saturated outputs (approximately binary)
  - Fixed number of heads $H$
]

#theorem("E88-MULTIPASS(k) Computational Power")[
  E88-MULTIPASS(k) can compute:
  1. All regular languages (even single-pass can)
  2. Majority function (requires O(1) passes for approximate, O(log T) for exact)
  3. Palindrome detection (2 passes)
  4. Sorting (O(T) passes using bubble sort simulation)
  5. Pattern matching (O(1) passes for fixed patterns)
]

== 10.6 Comparison: Transformer O(T) Parallel vs RNN k-Pass O(kT) Sequential

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,
    align: (left, center, center, center),
    [*Model*], [*Access Pattern*], [*Time Complexity*], [*Parallelism*],
    [Transformer + CoT], [Random O(1)], [O(T) parallel], [O(T) parallel ops],
    [RNN + Feedback], [Sequential], [O(T) sequential], [O(1) parallel ops],
    [k-Pass RNN], [Soft random O(k)], [O(kT) sequential], [O(1) parallel ops],
    [k-Pass with O(T) state], [Random via state], [O(kT) sequential], [O(1) parallel ops],
  ),
  caption: [Comparison of access patterns and complexity.],
)

#theorem("Transformer vs k-Pass RNN Efficiency")[
  For problems requiring r random accesses:
  - *Transformer + CoT*: O(T) time (parallel attention)
  - *k-Pass RNN*: O(3rT) time (3 passes per access, sequential)

  The efficiency gap is a factor of O(r) in the number of passes, but the RNN trades parallelism for memory efficiency.
]

#proof[
  Transformer attention computes all pairwise similarities in parallel, providing O(1) access to any position but requiring O(T²) space for attention weights.

  k-Pass RNN accesses positions sequentially, requiring O(k) passes for each random access but only O(1) space for the state (plus O(T) for the inter-pass tape).
]

#lemma("When k-Pass RNN Matches Transformer")[
  For problems with O(1) random accesses (independent of T), k-Pass RNN achieves the same asymptotic power as Transformer + CoT:
  - Both in DTIME(T) for bounded k
  - Both can solve the same decision problems

  The difference is *efficiency*, not *computability*.
]

== 10.7 Practical Trade-offs

#definition("Hardware Efficiency Metric")[
  For a model processing sequence of length T:
  $ "Efficiency" = ("Problems solvable") / ("Hardware cost") $

  Where hardware cost includes:
  - Memory bandwidth: Transformers O(T²), RNNs O(T)
  - Compute: Transformers O(T²), RNNs O(kT)
  - Parallelism utilization: Transformers high, RNNs low
]

#theorem("Multi-Pass RNN Hardware Trade-off")[
  The optimal choice depends on the problem structure:

  *Choose Transformer when*:
  - Many random accesses needed (r = Ω(T))
  - Hardware has high parallelism (GPUs)
  - Memory bandwidth is not bottleneck

  *Choose k-Pass RNN when*:
  - Few random accesses needed (r = O(1))
  - Sequential processing is acceptable
  - Memory bandwidth is limited
  - Low-latency inference per token is needed
]

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,
    align: (left, center, center, center),
    [*Task*], [*Transformer + CoT*], [*k-Pass RNN*], [*Better Choice*],
    [Sorting], [O(T log T)], [O(T²) = O(T) passes × O(T)], [Transformer],
    [Palindrome], [O(T)], [O(2T)], [Equivalent],
    [Pattern match], [O(T)], [O(kT) for k = O(1)], [Equivalent],
    [Language model], [O(T²) per token], [O(T) per token], [RNN (inference)],
    [Counting], [O(T)], [O(T)], [RNN (simpler)],
    [Binary search (on tape)], [O(log T) accesses × O(1)], [O(log T) passes × O(T)], [Transformer],
  ),
  caption: [Task-specific efficiency comparison.],
)

== 10.8 The Multi-Pass Hierarchy

#theorem("Complete Computational Hierarchy with Multi-Pass")[
  The full hierarchy including multi-pass models:

  $ "Linear-REG" < "E88 (1-pass)" < "E88-MULTIPASS"(k) < "E88 + Feedback" equiv "Transformer + CoT" < "E23" $

  Where:
  - Linear-REG: Linear temporal models (Mamba2), cannot count
  - E88 (1-pass): Nonlinear temporal, can count mod n, O(1) memory
  - E88-MULTIPASS(k): O(k) soft random access, O(kT) time
  - E88 + Feedback / Transformer + CoT: Full random access, DTIME(T)
  - E23: Unbounded tape, Turing complete
]

#proof[
  Each separation is witnessed by concrete problems:

  1. *Linear-REG < E88 (1-pass)*: Running parity separates (Section 6)

  2. *E88 (1-pass) < E88-MULTIPASS(2)*: "Compare first and last element" requires storing first element for T steps. Single-pass E88 with fixed state has precision decay. 2-pass E88 writes first element to tape, then compares in pass 2.

  3. *E88-MULTIPASS(k) < E88 + Feedback*: For k fixed, problems requiring ω(k) random accesses cannot be solved by MULTIPASS(k) but can by feedback models with O(T) tape.

  4. *E88 + Feedback < E23*: Bounded tape vs unbounded tape separates via halting problem.
]

== 10.9 Connections to Classical Complexity

#theorem("Multi-Pass and Space-Bounded Complexity")[
  Multi-pass models connect to classical space complexity:

  - *1-pass, O(1) state*: L (log-space) with read-once input
  - *k-pass, O(1) state*: Similar to L with k-head read-only input tape
  - *Unbounded passes*: Similar to PSPACE (polynomial space)
]

#definition("Multi-Pass Streaming Model")[
  The *streaming model* in complexity theory closely matches multi-pass RNNs:

  - Input arrives as a stream, read left-to-right
  - Limited working memory (state)
  - Multiple passes allowed

  Classical results:
  - Equality testing requires Ω(log T) space or 2 passes
  - Frequency moments estimation: 1-pass with O(polylog T) space for approximation
  - Exact frequency: requires Ω(T) space or O(T) passes
]

#theorem("E88 Multi-Pass in Streaming Complexity")[
  E88-MULTIPASS(k) with state dimension n has:
  - Space complexity: O(n) = O(1) (fixed with respect to T)
  - Pass complexity: k

  This places E88-MULTIPASS(k) in the streaming complexity class STREAM[k, O(1)].
]

== 10.10 Summary

#figure(
  table(
    columns: 2,
    stroke: 0.5pt,
    align: (left, left),
    [*Property*], [*Multi-Pass RNN*],
    [Access type], [Soft random: O(k) passes per access],
    [Time complexity], [O(kT) sequential],
    [Space complexity], [O(1) state + O(T) inter-pass tape],
    [Parallelism], [Low (sequential processing)],
    [Tape modification], [Full rewrite between passes],
    [Computational class], [Between single-pass and DTIME(T)],
    [Best use case], [Few random accesses, memory-limited hardware],
  ),
  caption: [Summary of multi-pass RNN properties.],
)

The key insights:

1. *k passes provide k/3 random accesses*: Each access requires marking, locating, and retrieving phases.

2. *Tape modification is powerful*: Complete rewriting between passes enables complex algorithms without explicit memory.

3. *Trade-off is efficiency, not power*: For problems with bounded random accesses, k-pass RNN achieves the same computability as Transformer + CoT, but with different efficiency characteristics.

4. *Hardware alignment*: Multi-pass RNN is better suited for memory-bandwidth-limited scenarios, while Transformers excel with high parallelism.

5. *Hierarchy position*: E88-MULTIPASS(k) strictly contains single-pass E88 and is strictly contained in full feedback/CoT models, with the separation determined by the number of required random accesses.

This analysis shows that the choice between Transformer and multi-pass RNN should be driven by the specific problem structure and hardware constraints, not by computational power alone. Both achieve bounded Turing machine capability; they differ in how efficiently they use hardware resources for different access patterns.
