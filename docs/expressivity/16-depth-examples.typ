// Section 16: Appendix - Composition Depth Reference

#import "traditional-math-style.typ": *

= Appendix: Composition Depth Reference

This appendix provides concrete estimates for the composition depth required by various tasks. Use it to predict where linear-temporal models will succeed and where they will fail.

== Depth by Task Type

#simpletable(
  columns: 4,
  align: (left, center, center, left),
  [*Task*], [*Depth*], [*$D=32$?*], [*Notes*],
  [Simple word prediction], [1--2], [Yes], [Context matching],
  [Relative clause resolution], [4], [Yes], [Binding dependency],
  [Triple center-embedding], [8], [Yes], [Nested clause parsing],
  [50 negations], [50], [No], [Running parity problem],
  [$n$-digit addition], [$n$], [If $n lt.eq 32$], [Carry propagation],
  [10×10 multiplication], [~100], [No], [Full digit-by-digit],
  [5-step deduction], [6], [Yes], [Each step builds on previous],
  [50-step proof], [51], [No], [Sequential dependency chain],
  [`fib(10)` evaluation], [~18], [Yes], [Recursive call depth],
  [`fib(50)` evaluation], [~99], [No], [Deep recursion tree],
  [20-iteration loop], [21], [Yes], [State update per iteration],
  [100-iteration loop], [101], [No], [Exceeds typical depth],
  [100-char DFA simulation], [~100], [No], [State transition per char],
)

The pattern is clear: tasks with sequential dependencies accumulate depth linearly with sequence length. When that length exceeds $D$, linear-temporal models fail.

== Architecture Selection by Domain

#simpletable(
  columns: 3,
  align: (left, center, left),
  [*Domain*], [*Linear SSM Sufficient?*], [*Recommendation*],
  [Casual conversation], [Yes], [Mamba2 for throughput],
  [Technical writing], [Yes], [Mamba2 for throughput],
  [Mathematical proofs], [Partially], [E88 or chain-of-thought],
  [Simple code completion], [Yes], [Mamba2 for throughput],
  [Complex algorithm tracing], [No], [E88 with external tools],
  [Formal verification], [No], [E88 + proof assistant],
)

== The Decision Procedure

Given a task, estimate its composition depth. Compare to the available model depth $D$.

If depth $lt.eq D$: any architecture works; choose based on throughput.

If depth $> D$ but bounded: E88 or chain-of-thought can help.

If depth unbounded: external tools are necessary.

The theory does not tell us what a model _will_ learn. It tells us what a model _cannot_ learn. Use the tables to identify tasks where the theoretical limits bind.
