// Section 10: Multi-Pass RNNs

#import "traditional-math-style.typ": *

= Multi-Pass RNNs: A Middle Path

Between fixed state and full output feedback lies an intermediate option: re-processing the input multiple times. Multi-pass RNNs trade computation for improved access patterns, achieving a form of soft random access without generating output.

== The Multi-Pass Idea

An RNN with fixed state processes the sequence once, left to right. But suppose we let it process the sequence twice, or three times, or $k$ times. Each pass can carry information forward, building on what previous passes learned.

#theorem("k-Pass Random Access")[
  A $k$-pass RNN can access position $p$ in a sequence of length $T$ using 3 passes: (1) mark positions on the first pass, (2) locate the target position on the second pass, (3) retrieve the value on the third pass. Therefore $k$ passes provide $floor(k\/3)$ effective random accesses.
]#leanfile("MultiPass.lean:878")

The construction mimics how humans scan a document. First pass: note where relevant information appears. Second pass: find the specific location needed. Third pass: retrieve the content. Each "random access" costs three sequential passes.

== The Multi-Pass Hierarchy

More passes means more access capability, forming a strict hierarchy.

#theorem("Strict Multi-Pass Hierarchy")[
  $ "MULTIPASS"(1, T) subset.neq "MULTIPASS"(2, T) subset.neq dots.c subset.neq "DTIME"(T^2) $
  where $"MULTIPASS"(k, T)$ is the class of functions computable by a $k$-pass RNN on sequences of length $T$.
]#leanfile("MultiPass.lean:958")

The limit $"DTIME"(T^2)$ comes from the observation that $O(T)$ passes, each taking $O(T)$ time, gives $O(T^2)$ total computation---matching quadratic attention.

== E88 Multi-Pass: Depth Through Time

When we apply multi-pass to E88 specifically, the temporal nonlinearity compounds across passes. Each pass adds $T$ to the compositional depth, creating a multiplicative advantage over linear-temporal models.

#theorem("E88 Multi-Pass Depth")[
  An E88 RNN with $k$ passes over a sequence of length $T$ achieves compositional depth $k times T$. Each pass contributes $T$ nested tanh applications, accumulating across passes to create $k times T$ total depth.
]#leanfile("E88MultiPass.lean:149")

This is the fundamental expressivity advantage. A linear-temporal model like Mamba2 collapses temporally at each layer, giving effective depth $k$ regardless of sequence length. E88's tanh nonlinearity prevents this collapse: the state at the end of pass $i$ is a $T$-fold nested composition of tanh, and pass $i+1$ applies tanh $T$ more times on top of that.

#theorem("E88 Exceeds Linear-Temporal Multi-Pass")[
  For any $k > 0$ and $T > 1$, E88 with $k$ passes has compositional depth $k times T$, which strictly exceeds the depth $k$ of a linear-temporal $k$-pass RNN.
]#leanfile("E88MultiPass.lean:212")

The gap is multiplicative in sequence length. For typical sequences ($T approx 1000$), even a single-pass E88 has 1000× the compositional depth of a linear-temporal model's single pass.

== Tape Modification and Learned Traversal

Between passes, the RNN can modify working memory—not just carrying state forward, but actively transforming an external tape. This enables iterative refinement algorithms that progressively build solutions.

#theorem("Tape Modification Operations")[
  A multi-pass RNN with tape modification can perform: (1) insertions that grow working memory, (2) deletions that shrink/clean working memory, (3) rewrites that modify intermediate results, (4) content-based head movement for adaptive traversal.
]#leanfile("MultiPass.lean:1075")

The tape serves as external memory. On pass 1, the RNN might mark positions of interest. On pass 2, it inserts computed values at those positions. On pass 3, it reads the augmented tape and deletes temporary markers. Each pass sees the modified tape from the previous pass, enabling progressive computation.

#remark(none)[
  Learned traversal patterns allow data-dependent access. Rather than always scanning left-to-right, the RNN can learn to move backward when it encounters certain symbols, skip forward to find targets, or bounce between positions. This converts sequential access into _adaptive_ sequential access—still not random, but no longer rigid.
]

With unbounded tape modification across passes, the computational power approaches a Turing machine. Each pass executes one TM step: read current cell, update state, write to tape, move head. The multi-pass architecture thus sits between streaming RNNs (no tape) and full Turing machines (arbitrary computation).

== Transformer vs Multi-Pass: The Memory-Parallelism Trade-off

The comparison reveals a fundamental trade-off in sequence processing.

#simpletable(
  columns: 4,
  align: (left, center, center, center),
  [*Model*], [*Access*], [*Time Complexity*], [*Memory Bandwidth*],
  [Transformer+CoT], [Random $O(1)$], [$O(T)$ parallel], [$O(T^2)$],
  [RNN+Feedback], [Sequential], [$O(T)$], [$O(T)$],
  [$k$-Pass RNN], [Soft random], [$O(k T)$], [$O(k T)$],
)

A Transformer achieves $O(1)$ random access through attention, at the cost of $O(T^2)$ memory bandwidth (the attention matrix). An RNN with feedback has only sequential access but linear bandwidth. A $k$-pass RNN interpolates: $floor(k\/3)$ random accesses with $O(k T)$ bandwidth.

#theorem("E88 vs Transformer Depth Comparison")[
  For sequence length $T$ and Transformer depth $D$, an E88 single-pass has compositional depth $T$. When $T > D$ (typical: $T > 32$), E88 exceeds Transformer depth. With $k$ passes, E88 achieves depth $k times T$, which grows linearly with sequence length while Transformer depth remains constant at $D$.
]#leanfile("E88MultiPass.lean:238")

The contrast is sharp. Transformers have _constant depth_ regardless of sequence length—this is the defining property of TC⁰. E88 has _depth proportional to $T$_—this exceeds TC⁰ for long sequences. Multi-pass E88 with $k$ passes achieves depth $k times T$, creating a computational class that can grow arbitrarily large.

#keyinsight[
For problems requiring a bounded number of random accesses, multi-pass RNNs match Transformers computationally while using $O(d)$ memory vs $O(T^2)$. For problems requiring unbounded random accesses, Transformers win through parallelism. For algorithmic tasks requiring deep sequential composition, E88 multi-pass provides depth that no fixed-layer Transformer can match.
]

== RNN k-Pass vs Transformer CoT: A Formal Comparison

Chain-of-thought (CoT) adds computation tokens to Transformers, letting them "think step by step." How does this compare to multi-pass RNNs, which iterate over the input $k$ times?

#theorem("Main Comparison: Memory vs Depth")[
  For RNN with $k$ passes over sequence length $T$, state dimension $d$, and Transformer with $D$ layers:

  1. RNN achieves $T\/3$ soft random accesses with $k=T$ passes (vs $T$ full random accesses for Transformer)
  2. RNN uses $O(k times T)$ total operations (vs $O(D times T^2)$ for Transformer with attention)
  3. RNN uses $O(d)$ memory (vs $O(T^2)$ for Transformer attention matrices)
  4. RNN has sequential depth $k times T$ (vs constant depth $D$ for Transformer)
]#leanfile("MultiPass.lean:2004")

The key insight: RNN $k$-pass trades parallelism for memory efficiency. A Transformer can process all $T$ positions simultaneously (parallelism $T$), but must store $O(T^2)$ attention weights. An RNN processes one position at a time (parallelism 1), but uses only $O(d)$ state memory.

#observation(none)[
  Transformer CoT doesn't change the depth class. Adding $C$ CoT tokens increases the effective width to $T+C$, enabling $C$ additional "reasoning steps" in parallel. But the circuit depth remains $D$—constant in the sequence length. The Transformer is still TC⁰, regardless of how many CoT tokens we add.
]

For E88 specifically, the multi-pass depth advantage is even more pronounced:

#theorem("E88 Multi-Pass Hierarchy")[
  For $k > 0$ and $T > D$:

  $ "Linear-temporal" k"-pass" subset.neq "E88" k"-pass" $

  with compositional depth $k$ vs $k times T$. Furthermore, for $k times T > D$:

  $ "TC"^0 ("Transformer," op("depth") D) subset.neq "E88" k"-pass" (op("depth") k times T) $

  E88 exceeds both linear-temporal multi-pass (by factor $T$) and Transformers (when depth $k times T > D$).
]#leanfile("E88MultiPass.lean:435")

== The Extended Hierarchy

We can now place multi-pass RNNs in the complete picture.

$ "Mamba2" subset.neq "E88" subset.neq "E88-MULTIPASS" subset.neq "E88+Feedback" equiv "Transformer+CoT" subset.neq "E23" $

Multi-pass E88 sits between single-pass E88 and E88 with full output feedback. It exceeds single-pass because multiple passes enable random-access-like capabilities. It falls short of full feedback because the number of passes is fixed at architecture time, not adaptive at runtime.

The practical implications follow from the theory. For tasks where $k < T\/D$ passes suffice, multi-pass RNNs are more memory-efficient than Transformers. For tasks requiring full random access, Transformers win. For tasks requiring compositional depth $> D$ (deep sequential reasoning), E88 multi-pass exceeds what any fixed-layer Transformer can compute.

#centerrule

Multi-pass processing offers a practical middle ground. When the number of required random accesses is known and bounded, multi-pass RNNs achieve the necessary computation with less memory overhead than full attention. The hierarchy continues to refine: each architectural choice trades off computation, memory, and capability in different ways.

