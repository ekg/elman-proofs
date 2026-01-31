// Section 10: Multi-Pass RNNs

#import "traditional-math-style.typ": *

= Multi-Pass RNNs: A Middle Path

Between fixed state and full output feedback lies an intermediate option: re-processing the input multiple times. Multi-pass RNNs trade computation for improved access patterns, achieving a form of soft random access without generating output.

== The Multi-Pass Idea

An RNN with fixed state processes the sequence once, left to right. But suppose we let it process the sequence twice, or three times, or $k$ times. Each pass can carry information forward, building on what previous passes learned.

#theorem("k-Pass Random Access")[
  A $k$-pass RNN can access position $p$ in a sequence of length $T$ using 3 passes: (1) mark positions on the first pass, (2) locate the target position on the second pass, (3) retrieve the value on the third pass. Therefore $k$ passes provide $floor(k\/3)$ effective random accesses.
]#leanfile("MultiPass.lean:164")

The construction mimics how humans scan a document. First pass: note where relevant information appears. Second pass: find the specific location needed. Third pass: retrieve the content. Each "random access" costs three sequential passes.

== The Multi-Pass Hierarchy

More passes means more access capability, forming a strict hierarchy.

#theorem("Strict Multi-Pass Hierarchy")[
  $ "MULTIPASS"(1, T) subset.neq "MULTIPASS"(2, T) subset.neq dots.c subset.neq "DTIME"(T^2) $
  where $"MULTIPASS"(k, T)$ is the class of functions computable by a $k$-pass RNN on sequences of length $T$.
]#leanfile("MultiPass.lean:183")

The limit $"DTIME"(T^2)$ comes from the observation that $O(T)$ passes, each taking $O(T)$ time, gives $O(T^2)$ total computation---matching quadratic attention.

== Transformer vs Multi-Pass

The comparison clarifies the trade-offs.

#simpletable(
  columns: 4,
  align: (left, center, center, center),
  [*Model*], [*Access*], [*Time Complexity*], [*Memory Bandwidth*],
  [Transformer+CoT], [Random $O(1)$], [$O(T)$ parallel], [$O(T^2)$],
  [RNN+Feedback], [Sequential], [$O(T)$], [$O(T)$],
  [$k$-Pass RNN], [Soft random], [$O(k T)$], [$O(k T)$],
)

A Transformer achieves $O(1)$ random access through attention, at the cost of $O(T^2)$ memory bandwidth (the attention matrix). An RNN with feedback has only sequential access but linear bandwidth. A $k$-pass RNN interpolates: $floor(k\/3)$ random accesses with $O(k T)$ bandwidth.

For problems requiring a bounded number of random accesses, multi-pass RNNs match Transformers computationally while using less memory. For problems requiring unbounded random accesses, Transformers win. For problems requiring only sequential access, single-pass RNNs suffice.

== The Extended Hierarchy

We can now place multi-pass RNNs in the complete picture.

$ "Mamba2" subset.neq "E88" subset.neq "E88-MULTIPASS" subset.neq "E88+Feedback" equiv "Transformer+CoT" subset.neq "E23" $

Multi-pass E88 sits between single-pass E88 and E88 with full output feedback. It exceeds single-pass because multiple passes enable random-access-like capabilities. It falls short of full feedback because the number of passes is fixed at architecture time, not adaptive at runtime.

#centerrule

Multi-pass processing offers a practical middle ground. When the number of required random accesses is known and bounded, multi-pass RNNs achieve the necessary computation with less memory overhead than full attention. The hierarchy continues to refine: each architectural choice trades off computation, memory, and capability in different ways.

But all these theoretical results face a sobering empirical reality. Despite E88's provably greater computational power, Mamba2 outperforms it on language modeling benchmarks. The next section confronts this gap between theory and practice.
