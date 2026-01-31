// Section 9: Output Feedback and Emergent Tape Memory
// Chain-of-Thought, Scratchpad, and Computational Classes

= Output Feedback and Emergent Tape

The previous sections analyzed _fixed-state_ models: architectures where the state dimension is constant regardless of input length. But what happens when a model can _write output_ and _read it back_? This creates an "emergent tape" that fundamentally changes computational power.

== The Core Insight

When a model can:
1. *Write* tokens/state to an output stream
2. *Read* those tokens back (via attention or recurrence)
3. Run for *T steps*

...it creates an emergent tape of length T. This is the mechanism behind chain-of-thought (CoT) reasoning, scratchpad computation, and autoregressive self-conditioning.

#block(
  fill: rgb("#f0fff7"),
  stroke: rgb("#33cc66"),
  inset: 12pt,
  radius: 4pt,
)[
  *Key Result (OutputFeedback.lean)*: Output feedback elevates any architecture to bounded Turing machine power.

  Even a simple linear RNN with output feedback can simulate a bounded TM, because the feedback creates an emergent tape of length T.
]

== Tape Types: Sequential vs Random Access

The mechanism for reading the tape determines efficiency, though not computational power:

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,
    align: (left, center, center),
    [*Architecture*], [*Tape Access*], [*Access Cost*],
    [RNN + Feedback], [Sequential], [$O(T)$ to reach position $p$],
    [Transformer + CoT], [Random], [$O(1)$ to any position],
  ),
  caption: [Both achieve DTIME(T), but with different access patterns.],
)

=== RNN Feedback: Sequential Tape

An RNN with output feedback creates a tape traversed sequentially:

$ h_t = f(h_(t-1), x_t, o_(t-1)) $

where $o_(t-1)$ is the previous output fed back as input. The model writes one cell per step and reads only the most recent output.

*Equivalent to*: A one-tape Turing machine reading left-to-right, with the head moving one cell per step.

=== Transformer CoT: Random Access Tape

A Transformer with chain-of-thought creates a tape with random access:

$ h_t = "Attention"(Q_t, K_(1:t), V_(1:t)) $

The attention mechanism can access any previous position in $O(1)$ time by learning appropriate query-key alignments.

*Equivalent to*: A RAM machine with $O(T)$ memory.

== The Computational Hierarchy

Output feedback creates a strict hierarchy:

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,
    align: (left, center, left),
    [*Architecture*], [*Memory*], [*Computational Class*],
    [Fixed Mamba2], [$O(1)$], [Linear-REG (no counting)],
    [Fixed E88], [$O(1)$], [Nonlinear-REG (can count mod $n$)],
    [E88 + Feedback], [$O(T)$], [DTIME(T) - bounded TM, sequential],
    [Transformer + CoT], [$O(T)$], [DTIME(T) - bounded TM, random],
    [E23 (unbounded tape)], [unbounded], [RE - Turing complete],
  ),
  caption: [Memory capacity determines computational class.],
)

Each level strictly contains the previous. The separations are witnessed by concrete problems.

== Separation: Fixed E88 vs E88+Feedback

The key separation is witnessed by *palindrome recognition*:

#block(
  fill: rgb("#fff7f0"),
  stroke: rgb("#cc6633"),
  inset: 12pt,
  radius: 4pt,
)[
  *Theorem (OutputFeedback.e88_feedback_exceeds_fixed_e88_palindrome)*:

  Palindrome recognition requires $Omega(n)$ memory. Fixed E88 has $O(1)$ memory. E88+feedback has $O(T)$ memory via output tape.

  *Algorithm for E88+Feedback*:
  - Write phase ($T\/2$ steps): Store each input bit as saturated output ($plus.minus 1$)
  - Verify phase ($T\/2$ steps): Compare current input with stored tape value
  - Accept iff all comparisons match
]

=== Why Fixed E88 Cannot Recognize Palindromes

The communication complexity argument:
1. Consider palindromes $u || "reverse"(u)$ for all $u in {0,1}^(n\/2)$
2. There are $2^(n\/2)$ such palindromes
3. To distinguish them, any machine needs $n\/2$ bits of memory
4. Fixed E88 with state dimension $d$ has $O(d)$ bits---constant, not growing with $n$
5. For $n > 2d$, fixed E88 fails

=== Why E88+Feedback Succeeds

With feedback, the output sequence becomes a tape:
- The tape grows to length $T$ (one cell per step)
- Saturated tanh outputs give reliable binary symbols
- Total memory: $O(T)$ bits
- For palindrome of length $n$, set $T = n$: $O(n) >= Omega(n\/2)$ ✓

== Chain-of-Thought Equals Explicit Tape

A fundamental equivalence:

#block(
  fill: rgb("#f0f7ff"),
  stroke: rgb("#3366cc"),
  inset: 12pt,
  radius: 4pt,
)[
  *Theorem (OutputFeedback.cot_equals_emergent_tape)*:

  CoT context length $T$ = explicit tape of length $T$. Both achieve DTIME(T) computational class.

  The "tape" emerges from:
  1. Token generation (write)
  2. Self-attention (read)
  3. Autoregressive conditioning (sequential access)
]

This explains *why CoT works*: it provides the working memory needed for algorithmic reasoning, without requiring architectural changes. The scratchpad is not a trick---it's a computationally necessary resource.

=== Information Capacity

The information capacity of T-step chain-of-thought:

$ "Capacity" = T times log_2(V) "bits" $

where $V$ is vocabulary size. For $V = 50000$, $T = 1000$:

$ "Capacity" approx 1000 times 15.6 approx 15600 "bits" $

This is enough for substantial algorithmic computation.

== Sequential vs Random Access Efficiency

Both E88+feedback and Transformer+CoT achieve DTIME(T), but with different constants:

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,
    align: (left, center, center, center),
    [*Problem*], [*Random Access*], [*Sequential Access*], [*Gap*],
    [Sorting $n$ elements], [$O(n log n)$], [$O(n^2 log n)$], [$n$],
    [Palindrome check], [$O(n)$], [$O(n)$], [1],
    [Pattern matching], [$O(n + m)$], [$O(n m)$], [$min(n,m)$],
    [Binary search], [$O(log n)$], [$O(n)$], [$n \/ log n$],
  ),
  caption: [Random access (attention) is more efficient for some algorithms.],
)

#block(
  fill: rgb("#f7f7ff"),
  stroke: rgb("#6666cc"),
  inset: 12pt,
  radius: 4pt,
)[
  *Theorem (OutputFeedback.cot_random_access_efficiency)*:

  For sorting $n$ elements:
  - Transformer+CoT: $O(n log n)$ operations (random access to tape)
  - RNN+feedback: $O(n^2 log n)$ operations (sequential tape traversal)

  The efficiency gap is a factor of $n$.
]

For problems where random access matters (sorting, searching), Transformer+CoT is more efficient. For problems that are naturally sequential (palindrome, FSM simulation), both are equivalent.

== Practical Implications

=== Why CoT Helps Complex Reasoning

Chain-of-thought is not just a prompting trick---it provides the *working memory* needed for multi-step reasoning:

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,
    align: (left, center, left),
    [*Task*], [*Memory Needed*], [*CoT Benefit*],
    [Multi-step arithmetic], [$O(n)$ for $n$-digit numbers], [Essential],
    [Logical deduction], [$O(k)$ for $k$ premises], [Helpful],
    [Code execution], [$O(n)$ for $n$ variables], [Essential],
    [Factoid recall], [$O(1)$], [Minimal],
  ),
  caption: [CoT matters most when tasks require working memory.],
)

=== The T-Bound is Fundamental

No matter the architecture, computation is bounded by steps $T$:

- $T$ steps = DTIME(T) computational power
- Cannot solve problems requiring $> T$ time
- Cannot use $> T$ tape cells

The only exception is E23-style unbounded tape, which achieves RE (Turing completeness). But for bounded $T$:

$ "Fixed state" subset.eq "E88+Feedback" equiv "Transformer+CoT" subset.eq "E23" $

=== When Feedback Matters

Feedback/CoT matters for:
- Algorithmic tasks (sorting, searching)
- Long reasoning chains ($> O(d)$ steps, where $d$ is state dimension)
- Counting beyond fixed state capacity
- Any task requiring $omega(1)$ working memory

Feedback/CoT is overkill for:
- Simple pattern matching
- Factoid retrieval
- Single-step classification
- Tasks within fixed-state capacity

== E88 with Feedback

E88's temporal nonlinearity combines well with feedback:

$ S_t = tanh(alpha S_(t-1) + delta k_t + gamma o_(t-1)) $

The tanh saturation creates reliable binary outputs that serve as tape symbols:
- $o_t approx +1$: bit value 1
- $o_t approx -1$: bit value 0
- Saturation prevents drift---written symbols remain stable

This makes E88+feedback particularly effective:
- Nonlinear dynamics for temporal decisions
- Saturated outputs for reliable tape symbols
- Hardware-efficient compute

== The Scratchpad Model

A more explicit formulation is the *scratchpad model*:

$ ("state", "scratchpad") arrow.r.bar ("new_state", "write_or_none") $

Each step:
1. Read current state and full scratchpad
2. Compute new state
3. Optionally append one cell to scratchpad

#block(
  fill: rgb("#f0fff7"),
  stroke: rgb("#33cc66"),
  inset: 12pt,
  radius: 4pt,
)[
  *Theorem (OutputFeedback.ScratchpadModel)*:

  Scratchpad capacity = max_length × cell_size bits. With $T$ steps and 1-bit cells, this gives $T$ bits of working memory.
]

This formalizes what language models do with CoT: they write intermediate results to the scratchpad (output context) and read them back via attention.

== Summary: The Emergent Tape Principle

#figure(
  table(
    columns: 2,
    stroke: 0.5pt,
    align: (left, left),
    [*Without Feedback*], [*With Feedback*],
    [Fixed $O(d)$ memory], [$O(T)$ emergent tape],
    [Regular languages], [Bounded TM power],
    [Immediate decisions], [Multi-step reasoning],
    [Pattern matching], [Algorithmic computation],
  ),
  caption: [Feedback transforms computational capability.],
)

The key insight: *output feedback creates emergent Turing-completeness* (up to the tape bound $T$).

This explains:
- Why chain-of-thought dramatically improves reasoning
- Why longer context helps complex tasks
- Why scratchpad training improves algorithmic capability
- Why E88+feedback can match Transformer+CoT for bounded computation

The hierarchy is complete:

$ "Fixed Mamba2" < "Fixed E88" < "E88+Feedback" equiv "Transformer+CoT" < "E23" $

Each separation is witnessed by a concrete problem:
1. Mamba2 $<$ E88: Running parity (linear cannot threshold)
2. E88 $<$ E88+Feedback: Palindromes ($O(1)$ vs $O(T)$ memory)
3. E88+Feedback $equiv$ Transformer+CoT: Both DTIME(T), differ in efficiency
4. CoT $<$ E23: Halting problem (bounded vs unbounded tape)

All theorems are formalized in `OutputFeedback.lean` with complete proofs.
