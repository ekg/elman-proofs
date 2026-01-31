// Section 14: Composition Depth Analysis

#import "traditional-math-style.typ": *

= Composition Depth in Practice

The theoretical gap between linear-temporal models and E88 is real. But does it matter for tasks people actually care about? The answer depends on how much composition depth those tasks require.

== The Distribution of Depth

Human-generated text follows a heavy-tailed distribution in composition depth. Most text is simple; some is moderately complex; a small fraction requires deep sequential reasoning.

#simpletable(
  columns: 4,
  align: (left, center, center, center),
  [*Domain*], [*Typical Depth*], [*Maximum*], [*$D=32$ Sufficient?*],
  [Syntactic parsing], [2--3], [~7], [Yes],
  [Semantic composition], [3--4], [~10], [Yes],
  [Discourse structure], [4--6], [~15], [Yes],
  [Simple programs], [2--5], [~10], [Yes],
  [Recursive algorithms], [10--30], [~100], [Partially],
  [Interpreters], [50--200], [unbounded], [No],
  [Formal proofs], [100+], [unbounded], [No],
)

For syntax and simple semantics, a 32-layer model provides more than enough depth. For recursive code execution or proof verification, no fixed-depth model suffices.

== The Uncanny Valley of Reasoning

This distribution creates a distinctive failure mode. Large language models produce fluent, confident output across all complexity levels. But when the required depth exceeds what the architecture provides, the output degrades---not obviously wrong, just subtly broken.

#definition("The Depth Barrier")[
  If a function requires composition depth $N$ and a model has capacity $D < N$, the model cannot compute the function. Failures manifest as: correct for small instances, degraded for medium instances, random for large instances.
]

The model does not announce that it has exceeded its depth. It confabulates. The output looks like reasoning but misses steps, inverts implications, or hallucinates connections. This is the _uncanny valley_: systems that appear intelligent but fail in ways that surprise us.

The failure modes are predictable from the theory. _State tracking_: after $D$ updates, the model loses track of accumulated state. _Long reasoning chains_: the model skips steps or inverts causality. _Negation blindness_: running parity of negations---a simple counting task---is impossible for linear-temporal models regardless of depth.

== Where Depth Matters

#keyinsight[
  Natural language follows a heavy-tailed distribution. Most text requires depth 2--5. Occasional complex text requires 10--30. Rare deep reasoning requires 50+. Linear-temporal models perform well on the bulk of the distribution. E88's advantage manifests in the tail---exactly where reasoning fails and chain-of-thought becomes necessary.
]

This explains why linear-temporal models achieve good perplexity on language modeling benchmarks: they handle the bulk of the distribution well. It also explains why they fail unexpectedly on reasoning tasks: those tasks live in the tail where the architectural limitation bites.

== Practical Guidance

A rough heuristic for architecture selection based on task depth:

_Depth ≤ 10_: Any modern architecture works. Syntax, simple semantics, pattern matching.

_Depth 10--30_: A $D=32$ model handles most cases. Complex discourse, moderately nested code.

_Depth 30--100_: E88 or chain-of-thought required. Multi-step proofs, recursive algorithm traces.

_Depth > 100_: External tools required. Compilers, theorem provers, debuggers.

The boundary at $D=32$ is not magical---it reflects current practice where 32-layer models are common. As models scale, the boundary shifts, but the qualitative picture remains: beyond some depth, fixed-depth linear-temporal models fail.

#centerrule

The composition depth perspective clarifies when the theoretical hierarchy matters. For the bulk of natural language, linear-temporal models suffice. For the tail of complex reasoning, they do not. The appendix provides specific examples across task types.
