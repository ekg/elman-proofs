// Section 16: Appendix - Composition Depth Examples
// Comprehensive Reference Table

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

#let example(title, body) = block(
  fill: rgb("#fff0f7"),
  stroke: rgb("#cc3399"),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
)[
  #strong[Example: #title]#linebreak()
  #body
]

= Appendix: Composition Depth Examples

_Comprehensive Reference for Depth Requirements Across Domains_

This appendix provides a detailed catalog of composition depth requirements for various computational tasks. Use this as a reference for predicting where linear-temporal architectures (Mamba2, FLA) will succeed or fail relative to nonlinear-temporal architectures (E88).

== A.1 Depth Calculation Methodology

#definition("How to Calculate Composition Depth")[
  For a computation $f(x)$:

  1. *Identify the critical path*: The longest chain of dependent operations from input to output

  2. *Count nonlinear operations*: Only operations that cannot be expressed as $y = A x + b$ for some matrix $A$ and vector $b$

  3. *Account for parallelism*: Operations that can execute in parallel count as depth 1 together

  *Linear operations* (don't add depth):
  - Addition, subtraction
  - Scalar multiplication
  - Matrix-vector products
  - Convolutions (linear filters)

  *Nonlinear operations* (add depth):
  - Comparisons ($<$, $>$, $=$)
  - Thresholding (if-then-else)
  - Multiplication of two variables
  - XOR, AND, OR on non-trivial inputs
  - Activation functions (tanh, ReLU, etc.)
]

== A.2 Natural Language Examples

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    align: (left, left, center, center, left),
    [*Task*], [*Example*], [*Depth*], [*D=32?*], [*Critical Operation*],
    [Word prediction], ["The cat sat on the ..."], [1-2], [$checkmark$], [Pattern match],
    [Simple sentence], ["She runs fast."], [2], [$checkmark$], [Subject-verb agreement],
    [Compound sentence], ["He ran and she walked."], [3], [$checkmark$], [Conjunction],
    [Relative clause], ["The man who ran won."], [4], [$checkmark$], [Clause embedding],
    [2-level embedding], ["I know that you said that..."], [5], [$checkmark$], [Nested clauses],
    [3-level embedding], ["A said B said C said..."], [7], [$checkmark$], [Deep nesting],
    [5-level embedding], ["...5 levels of quotes..."], [11], [$checkmark$], [Very deep nesting],
    [Center embedding $times$3], ["The cat the dog the rat bit chased ran"], [8], [$checkmark$], [Center embedding],
    [Pronoun chain $times$5], ["He gave it to her after she..."], [6-8], [$checkmark$], [Coreference],
    [Negation $times$3], ["not unlikely to not fail"], [3], [$checkmark$], [Negation parity],
    [Negation $times$7], ["not not not not not not not"], [7], [$checkmark$], [Parity (within budget)],
    [Negation $times$50], [50 nested negations], [50], [$times$], [Running parity],
  ),
  caption: [Composition depth for natural language tasks.],
)

== A.3 Arithmetic and Mathematical Examples

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    align: (left, left, center, center, left),
    [*Task*], [*Example*], [*Depth*], [*D=32?*], [*Critical Operation*],
    [Single addition], [23 + 45], [1], [$checkmark$], [Carry propagation (1 digit)],
    [Multi-digit add], [789 + 456], [3], [$checkmark$], [Carry chain],
    [10-digit add], [1234567890 + ...], [10], [$checkmark$], [Long carry chain],
    [100-digit add], [100-digit numbers], [100], [$times$], [Very long carry chain],
    [Single multiply], [7 $times$ 8], [1], [$checkmark$], [Lookup / pattern],
    [2$times$2 multiply], [23 $times$ 45], [4-6], [$checkmark$], [Partial products + adds],
    [3$times$3 multiply], [123 $times$ 456], [9-12], [$checkmark$], [More partials],
    [10$times$10 multiply], [10-digit $times$ 10-digit], [~100], [$times$], [Full long multiplication],
    [Division], [1234 $div$ 56], [10-20], [$checkmark$], [Iterative subtraction],
    [Long division], [100-digit $div$ 50-digit], [~200], [$times$], [Many iterations],
    [Square root], [$sqrt(1234)$], [10-15], [$checkmark$], [Newton iteration],
    [Modular exp], [$3^{100} mod 7$], [~100], [$times$], [Repeated squaring],
  ),
  caption: [Composition depth for arithmetic tasks.],
)

== A.4 Logical Reasoning Examples

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    align: (left, left, center, center, left),
    [*Task*], [*Example*], [*Depth*], [*D=32?*], [*Critical Operation*],
    [Modus ponens], [A $->$ B, A, therefore B], [2], [$checkmark$], [Rule application],
    [2-step deduction], [A$->$B, B$->$C, A $therefore$ C], [3], [$checkmark$], [Chain rule],
    [5-step deduction], [5-step logical chain], [6], [$checkmark$], [Longer chain],
    [10-step proof], [10-step deduction], [11], [$checkmark$], [Long chain],
    [30-step proof], [Complex theorem], [31], [Borderline], [Very long chain],
    [50-step proof], [Major theorem], [51], [$times$], [Exceeds depth],
    [Syllogism], [All X are Y, all Y are Z...], [3], [$checkmark$], [Set inclusion],
    [Resolution], [CNF satisfiability], [varies], [varies], [Clause resolution],
    [SAT (3 vars)], [$(a or b) and (not a or c)$], [3-5], [$checkmark$], [Case analysis],
    [SAT (10 vars)], [10-variable CNF], [10-20], [$checkmark$], [Deeper case analysis],
    [SAT (50 vars)], [50-variable CNF], [50-200], [$times$], [Exponential worst case],
  ),
  caption: [Composition depth for logical reasoning tasks.],
)

== A.5 Algorithm Execution Examples

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    align: (left, left, center, center, left),
    [*Task*], [*Example*], [*Depth*], [*D=32?*], [*Critical Operation*],
    [Linear search (10)], [Find x in 10 elements], [10], [$checkmark$], [Sequential compare],
    [Linear search (100)], [Find x in 100 elements], [100], [$times$], [100 comparisons],
    [Binary search (100)], [Find x in sorted 100], [7], [$checkmark$], [$log_2(100)$ comparisons],
    [Bubble sort (10)], [Sort 10 elements], [~45], [$times$], [$O(n^2)$ compares],
    [Merge sort (10)], [Sort 10 elements], [~34], [$times$], [$O(n log n)$],
    [Merge sort (100)], [Sort 100 elements], [~700], [$times$], [Much larger],
    [Fibonacci(10)], [`fib(10)`], [~18], [$checkmark$], [Recursive calls],
    [Fibonacci(20)], [`fib(20)`], [~38], [$times$], [More calls],
    [Factorial(5)], [`5!`], [9], [$checkmark$], [5 multiplies + 5 compares],
    [Factorial(20)], [`20!`], [~39], [$times$], [20 multiplies + 20 compares],
    [GCD], [`gcd(48, 18)`], [~6], [$checkmark$], [Euclidean steps],
    [Prime check (small)], [Is 97 prime?], [~10], [$checkmark$], [Trial division],
    [Prime check (large)], [Is 1000003 prime?], [~1000], [$times$], [Many trials],
  ),
  caption: [Composition depth for algorithm execution.],
)

== A.6 Code Understanding Examples

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    align: (left, left, center, center, left),
    [*Task*], [*Example*], [*Depth*], [*D=32?*], [*Critical Operation*],
    [Variable trace], [x=1; x=x+1; print(x)], [2], [$checkmark$], [Sequential update],
    [If-else], [if (a>b) x=1 else x=2], [2], [$checkmark$], [Branch + assign],
    [Loop (5 iters)], [for i in range(5): x+=1], [6], [$checkmark$], [5 iterations + init],
    [Loop (20 iters)], [for i in range(20): x+=1], [21], [$checkmark$], [More iterations],
    [Loop (100 iters)], [for i in range(100): x+=1], [101], [$times$], [Too many iterations],
    [Nested loop 3$times$3], [Outer $times$ inner], [10], [$checkmark$], [3 $times$ 3 + overhead],
    [Nested loop 10$times$10], [Outer $times$ inner], [~100], [$times$], [100 iterations],
    [Function call], [def f(x): return x+1; f(5)], [3], [$checkmark$], [Call + body + return],
    [3 nested calls], [f(g(h(x)))], [~9], [$checkmark$], [3 $times$ 3],
    [10 nested calls], [f1(f2(...f10(x)))], [~30], [Borderline], [10 calls deep],
    [Recursion depth 5], [Recursive function], [~15], [$checkmark$], [5 stack frames],
    [Recursion depth 20], [Deeper recursion], [~60], [$times$], [20 frames],
  ),
  caption: [Composition depth for code understanding.],
)

== A.7 State Machine Examples

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    align: (left, left, center, center, left),
    [*Task*], [*Example*], [*Depth*], [*D=32?*], [*Critical Operation*],
    [2-state toggle], [ON/OFF after 5 inputs], [5], [$checkmark$], [5 state updates],
    [2-state toggle], [ON/OFF after 50 inputs], [50], [$times$], [Running parity],
    [3-state machine], [A$->$B$->$C$->$A cycle], [varies], [varies], [State transitions],
    [Counter (mod 3)], [Count inputs mod 3], [varies], [varies], [Modular arithmetic],
    [Regex matching], [`a*b+c?` on input], [~len], [If short], [NFA simulation],
    [DFA (5 states)], [Process 20 chars], [~20], [$checkmark$], [State per char],
    [DFA (5 states)], [Process 100 chars], [~100], [$times$], [Many transitions],
    [PDA simulation], [Balanced parens check], [~depth], [If shallow], [Stack operations],
    [Turing machine], [Arbitrary TM], [$infinity$], [$times$], [Unbounded],
  ),
  caption: [Composition depth for state machine tasks.],
)

== A.8 Running Parity and XOR

#definition("Running Parity")[
  Running parity over sequence $x_1, ..., x_T$:
  $ y_t = x_1 xor x_2 xor ... xor x_t $

  *Key property*: Each XOR is a nonlinear operation. Running parity of length $T$ requires exactly $T-1$ sequential XOR operations.

  *Implication*: Any task that reduces to running parity inherits its depth requirements.
]

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    align: (left, left, center, center, left),
    [*Task*], [*Instance*], [*Depth*], [*D=32?*], [*Notes*],
    [XOR of 2], [$a xor b$], [1], [$checkmark$], [Single XOR],
    [XOR of 5], [$a xor b xor c xor d xor e$], [4], [$checkmark$], [4 XORs],
    [XOR of 32], [32 binary values], [31], [Borderline], [Matches D exactly],
    [XOR of 100], [100 binary values], [99], [$times$], [Far exceeds D],
    [Parity check], [Is sum even?], [$T-1$], [If $T <= 32$], [Reduces to XOR],
    [Toggle count], [Even/odd toggles?], [$T-1$], [If $T <= 32$], [Same as parity],
    [Balanced brackets], [Equal opens/closes?], [~$T$], [If $T <= 32$], [Similar structure],
  ),
  caption: [Running parity depth requirements.],
)

== A.9 Threshold and Counting

#definition("Running Threshold")[
  Running threshold count at threshold $tau$:
  $ y_t = cases(1 "if" sum_(s <= t) x_s > tau, 0 "otherwise") $

  *Key property*: The threshold decision is discontinuous. Linear-temporal models are continuous functions and cannot represent discontinuities exactly.
]

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    align: (left, left, center, center, left),
    [*Task*], [*Instance*], [*Depth*], [*D=32?*], [*Notes*],
    [Count to 3], [Output 1 when count $>= 3$], [3+], [See below], [Needs exact threshold],
    [Running max], [Track max so far], [~$T$], [If $T <= D$], [Decision per position],
    [Running min], [Track min so far], [~$T$], [If $T <= D$], [Same as max],
    [Threshold alert], [Alert when sum $> k$], [Absorbing], [$checkmark$/$times$], [Once triggered, stays],
    [Count occurrences], [How many 'a's?], [$T$], [If $T <= D$], [Increment per match],
    [Exact count], [Exactly 5 'a's?], [$T + 1$], [If $T <= D$], [Count + final check],
  ),
  caption: [Threshold and counting depth requirements.],
)

== A.10 Summary Table: Architecture Selection Guide

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    align: (left, center, center, center, left),
    [*Domain*], [*Typical Depth*], [*Max Depth*], [*Linear SSM?*], [*Recommendation*],
    [Casual text], [2-5], [10], [$checkmark$], [Mamba2 fine],
    [Technical writing], [3-8], [15], [$checkmark$], [Mamba2 fine],
    [Legal documents], [5-10], [20], [$checkmark$], [Mamba2 OK],
    [Math proofs], [10-50], [200+], [Partial], [E88 or CoT],
    [Code (simple)], [5-15], [30], [$checkmark$], [Mamba2 OK],
    [Code (complex)], [20-100], [1000+], [$times$], [E88 + tools],
    [Formal verification], [50-500], [$infinity$], [$times$], [E88 + proof assistant],
    [Algorithm design], [10-50], [100+], [Partial], [E88 or hybrid],
    [Puzzle solving], [10-30], [50+], [Partial], [E88 + search],
    [Game playing], [5-20], [100+], [Partial], [E88 for deep games],
  ),
  caption: [Architecture selection based on depth requirements.],
)

#block(
  fill: rgb("#f0f7ff"),
  stroke: rgb("#3366cc"),
  inset: 12pt,
  radius: 4pt,
)[
  *Quick Reference*:

  - *Depth $<= 10$*: Any architecture works
  - *Depth 10-30*: D=32 layer models handle most cases
  - *Depth 30-100*: E88 or multi-pass (CoT) required
  - *Depth $> 100$*: External tools or specialized systems required

  *Rule of Thumb*: Count the longest chain of dependent decisions/operations. If it exceeds your model's depth $D$, expect failures.
]

