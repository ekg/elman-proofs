// Section 7: Practical Implications

#import "traditional-math-style.typ": *

= Practical Implications

The theorems we have established are mathematically precise. But mathematics alone does not build systems. When does the hierarchy matter for practice?

== Matching Architecture to Task

The fundamental insight translates into a design principle: match the architecture's computational class to the task's requirements.

#definition("Task Classification")[
  _Pattern aggregation_: tasks that combine information from multiple positions without sequential dependencies. Examples include language modeling (predicting the next token), retrieval (finding relevant documents), and classification (mapping sequence to label). Linear-temporal models suffice.

  _Sequential decision tasks_: tasks where each decision depends on previous decisions. Examples include counting (tracking a running sum), state tracking (simulating a finite automaton), and parity computation. Temporal nonlinearity is mathematically required.
]

The classification is not always obvious from the task description. "Summarize this document" seems like pattern aggregation, but if the summary requires tracking which points have been covered, it becomes sequential. "Answer this question" may be aggregation or may require multi-step reasoning. The depth of the required computation determines the architectural need.

== When Linear Suffices

For most natural language processing with $D >= 32$ layers, linear-temporal models suffice. Linear-temporal models are strictly less expressive, but natural language tasks rarely need that expressivity.

Most sentences require parsing depth 2--5. Complex center-embedded constructions push to 7--10. The extreme tail may reach 20--25.

A 32-layer model provides 32 levels of composition, exceeding natural language requirements almost everywhere.

Parallel scan processes entire sequences simultaneously. Nonlinear recurrence proceeds step by step. For training at scale, throughput dominates.

== When the Gap Matters

_Depth-constrained deployment_: A 4-layer model on device has only 4 levels of composition. Tasks requiring 5 or more become impossible for linear-temporal architectures.

_Algorithmic reasoning_: Counting objects, tracking state across a narrative, following a procedure.

_State machine simulation_: Parsing context-free grammars, simulating finite automata, tracking dialogue state.

#keyinsight[
  Adding layers is curvature in the wrong dimension. Depth adds nonlinearity _between_ layers; temporal nonlinearity adds it _through_ time. These are orthogonal. A 64-layer Mamba2 still cannot compute running parity. A 1-layer E88 can.
]

== Separating Benchmarks

If we want to measure the expressivity gap empirically, we need benchmarks designed to exercise it.

#simpletable(
  columns: 3,
  align: (left, center, center),
  [*Benchmark*], [*E88*], [*Linear-Temporal*],
  [Running parity], [$approx 100%$], [$approx 50%$ (random)],
  [Running threshold], [Sharp transition], [Smooth sigmoid],
  [FSM simulation], [Exact match], [Degrades with state count],
)

On running parity, the prediction is stark: E88 can achieve near-perfect accuracy with appropriate training, while linear-temporal models cannot exceed chance regardless of training budget. The impossibility is mathematical.

== Design Principles

The theory yields practical guidance.

_Match architecture to task._ Use linear-temporal models for pattern aggregation where throughput matters. Use nonlinear-temporal models for sequential decision tasks where correctness matters.

_Accept that depth does not substitute for temporal nonlinearity._ If a task requires temporal decisions, adding layers does not help. The solution is architectural, not parametric.

_Recognize saturation as a feature._ E88's tanh saturation is not numerical instability---it is the mechanism enabling persistent memory. Attempts to "fix" saturation by clipping or normalization may destroy the expressivity advantage.

_Consider hardware alignment._ E88's matrix operations achieve high GPU utilization. Explicit tape memory (E23) creates irregular access patterns. The choice of escape route affects not just expressivity but efficiency.

#centerrule

The practical implications are clear. For most language tasks at scale, linear-temporal models suffice and train faster. For algorithmic tasks, constrained deployments, and exact computation requirements, the theoretical hierarchy predicts empirical outcomes. The art is recognizing which regime applies.

