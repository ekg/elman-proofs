// Section 5: E23 vs E88
// Tape Memory vs Temporal Saturation

= E23 vs E88: Two Paths to Expressivity

Both E23 and E88 achieve computational power beyond linear-temporal models, but through fundamentally different mechanisms. Understanding this difference illuminates what makes architectures work on real hardware.

== The Two Mechanisms

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,
    align: (left, left, left),
    [*Aspect*], [*E23 (Dual Memory)*], [*E88 (Temporal Nonlinearity)*],
    [Memory], [Explicit tape + working memory], [Implicit in saturated state],
    [Persistence], [Tape never decays], [Tanh saturation creates stability],
    [Capacity], [$N times D$ (tape size)], [$H times D$ (head count × dim)],
    [Compute], [$O(N D + D^2)$ per step], [$O(H D^2)$ per step],
    [Theoretical class], [UTM (universal)], [Bounded but very expressive],
  ),
  caption: [E23 and E88 achieve expressivity through different mechanisms.],
)

== E23: The Tape-Based Approach

E23 (Dual Memory Elman) separates memory into two components:

$ "Tape": quad & h_"tape" in RR^(N times D) quad "(persistent, N slots)" $
$ "Working": quad & h_"work" in RR^D quad "(nonlinear, computation)" $

The dynamics:
+ *Read*: Attention over tape slots, weighted sum into working memory
+ *Update*: $h_"work"' = tanh(W_h h_"work" + W_x x + "read" + b)$
+ *Write*: Replacement write to tape via attention: $(1-alpha) dot "old" + alpha dot "new"$

=== E23's Theoretical Strength

#block(
  fill: rgb("#f0fff7"),
  stroke: rgb("#33cc66"),
  inset: 12pt,
  radius: 4pt,
)[
  *Theorem (E23_DualMemory.lean)*: E23 is Turing-complete (UTM class).

  The proof relies on three capabilities:
  1. Nonlinearity (tanh in working memory)
  2. Content-based addressing (attention for routing)
  3. Persistent storage (tape with no decay)
]

With hard attention (one-hot), replacement write becomes exact slot replacement---precisely Turing machine semantics.

=== E23's Practical Weakness

Despite theoretical universality, E23 struggles on real hardware:

*Memory bandwidth*: Each step reads and potentially writes all $N$ tape slots. Even with sparse attention, the tape must be kept in memory. For $N = 64, D = 1024$, the tape is $64 times 1024 times 4 = 256"KB"$ per sequence---significant at scale.

*Attention overhead*: Computing attention scores over $N$ slots adds $O(N D)$ compute per step. This compounds with sequence length.

*Training instability*: The replacement write $(1-alpha) dot "old" + alpha dot "new"$ creates long gradient paths through the tape. Information written early affects reads much later, creating vanishing/exploding gradient issues.

*Discrete vs continuous*: Theoretical UTM requires discrete operations (exact slot addressing). Soft attention approximates this but introduces errors that accumulate.

== E88: The Saturation Approach

E88 uses temporal nonlinearity instead of explicit tape:

$ S_t = tanh(alpha S_(t-1) + delta k_t^top) $

where $S in RR^(H times D)$ (H heads, D dimensions).

=== How Saturation Creates Memory

The key insight: tanh saturation creates stable fixed points.

#block(
  fill: rgb("#f0f7ff"),
  stroke: rgb("#3366cc"),
  inset: 12pt,
  radius: 4pt,
)[
  *Theorem (TanhSaturation.lean)*: For $|S| approx 1$, the derivative $tanh'(S) = 1 - tanh^2(S) approx 0$.

  This means small perturbations to a saturated state cause negligible change. The state "latches" at $plus.minus 1$.
]

This creates implicit binary memory:
- State near $+1$: represents "fact is true"
- State near $-1$: represents "fact is false"
- Saturation prevents drift---the state persists without explicit storage

=== E88's Practical Strengths

*Hardware efficiency*: E88's computation is $O(H D^2)$ per step---matrix multiplications that GPUs excel at. No separate tape access, no attention over memory slots.

*Gradient flow*: The recurrence $S_t = tanh(alpha S_(t-1) + delta k_t)$ has bounded gradients. Unlike E23's tape, there's no separate storage creating long gradient paths.

*Parallelization*: While inherently sequential in $t$, E88 can parallelize across heads $H$. Each head runs independent dynamics (proven in MultiHeadTemporalIndependence.lean).

*Natural batching*: State is fixed-size $H times D$ regardless of "memory requirements." No dynamic allocation, no variable-length tape.

== The Core Trade-off

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,
    align: (center, center, center),
    [], [*E23*], [*E88*],
    [Memory capacity], [Explicit: $N times D$], [Implicit: $H times D$ "soft bits"],
    [Precision], [Can be exact (hard attn)], [Approximate (saturation)],
    [Hardware fit], [Poor (memory-bound)], [Good (compute-bound)],
    [Training], [Hard (long gradients)], [Easier (bounded)],
    [Theoretical power], [UTM], [Sub-UTM but very expressive],
  ),
  caption: [E23 trades practical efficiency for theoretical power.],
)

== Why E88 Wins in Practice

*The memory bottleneck*: Modern accelerators (GPUs, TPUs) are compute-bound, not memory-bound. E23's tape creates memory bandwidth pressure; E88's matrix operations are compute-dense.

*The precision illusion*: E23's "exact" addressing requires hard attention, which is non-differentiable. In practice, soft attention is used, losing the precision advantage.

*Gradient scaling*: E23's tape creates $O(T)$ gradient paths (information written at step 1 affects reads at step $T$). E88's saturation naturally bounds gradient magnitude.

*Capacity scaling*: Need more memory? E23 requires larger tape (more memory bandwidth). E88 adds more heads (more parallel compute)---the right direction for modern hardware.

== When E23 Might Be Preferred

E23's explicit tape could be valuable for:
- *Interpretability*: Tape contents are directly inspectable
- *Guaranteed persistence*: Information never decays (vs E88's "almost never")
- *Exact retrieval*: When approximate recall is unacceptable
- *Theoretical analysis*: UTM equivalence enables formal reasoning

But these advantages rarely outweigh E88's practical benefits in typical ML settings.

== The Deeper Lesson

E23 and E88 represent two philosophies:

*E23*: "Memory should be explicit and addressable, like a Turing machine tape."

*E88*: "Memory should emerge from dynamics, stable states encoding information."

The success of E88 suggests that for neural networks, the second philosophy aligns better with:
- How gradient-based learning works
- How modern hardware is designed
- How information needs to be stored (approximately, not exactly)

E23 is theoretically beautiful. E88 is practically effective. The proofs we've developed explain _why_: the mechanisms that give E23 its power (explicit tape, hard addressing) are exactly the mechanisms that make it hard to train and deploy.
