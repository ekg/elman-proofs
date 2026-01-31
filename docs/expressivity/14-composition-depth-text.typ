// Section 14: Composition Depth in Human Text
// How Deep Does Natural Language Go?

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

#let observation(title, body) = block(
  fill: rgb("#f0fff7"),
  stroke: rgb("#33cc66"),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
)[
  #strong[Observation (#title)]#linebreak()
  #body
]

#let finding(body) = block(
  fill: rgb("#f7f7ff"),
  stroke: rgb("#6666cc"),
  inset: 12pt,
  radius: 4pt,
)[#body]

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

= Composition Depth in Human Text

_Analyzing the Compositional Structure of Natural Language and Code_

This section examines how much _composition depth_ human-generated text actually requires. If natural language rarely exceeds the composition depth available in D-layer linear-temporal models, then the theoretical expressivity gap (E88 > Mamba2) may be irrelevant in practice.

== 14.1 Defining Composition Depth

#definition("Composition Depth")[
  The *composition depth* of a computation is the longest chain of dependent nonlinear operations from input to output.

  For a function $f = g_n compose g_(n-1) compose ... compose g_1$ where each $g_i$ is nonlinear, the composition depth is $n$.

  *Key insight*: Linear operations don't add depth---they collapse into single matrix multiplications. Only nonlinearities contribute to depth.
]

#definition("Per-Token Composition Depth")[
  For a sequence model processing tokens $x_1, ..., x_T$ and producing output $y_T$, the *per-token composition depth* is the number of nonlinear operations on the path from any input token $x_t$ to $y_T$.

  *Linear-temporal models*: Depth = $D$ (layer count), independent of $T$
  *E88-style models*: Depth = $D times T$ (layers $times$ timesteps)
]

== 14.2 Composition Depth in Natural Language

=== 14.2.1 Syntactic Depth

Natural language syntax has bounded depth:

#example("Syntactic Composition")[
  Consider the sentence:
  #quote["The cat that the dog that the rat bit chased ran away."]

  This has embedding depth 3:
  - Level 0: "The cat ... ran away"
  - Level 1: "the dog ... chased [the cat]"
  - Level 2: "the rat bit [the dog]"

  Human comprehension degrades rapidly beyond depth 3-4 for center-embedded clauses.
]

#observation("Chomsky Hierarchy and Human Limits")[
  While natural language is theoretically context-free (or mildly context-sensitive), _actual usage_ stays within bounds:

  - Average sentence depth: 2-3 levels
  - Maximum practical depth: ~7 (hard to parse)
  - Center-embedding limit: ~3 before incomprehensible

  A D=32 layer model provides far more depth than syntactic structure requires.
]

=== 14.2.2 Semantic Depth

Semantic composition involves building meaning from parts:

#example("Semantic Composition")[
  "The quick brown fox jumps over the lazy dog."

  Semantic composition:
  1. "quick brown" modifies "fox" (depth 1)
  2. "quick brown fox" is the jumper (depth 2)
  3. "lazy" modifies "dog" (depth 1)
  4. "lazy dog" is jumped over (depth 2)
  5. The entire event is composed (depth 3)

  Total semantic depth: ~3-4 nonlinear compositions.
]

#definition("Semantic Composition Functions")[
  Common semantic compositions:
  - *Modification*: adj(noun) $->$ modified noun
  - *Predication*: verb(subject, object) $->$ event
  - *Quantification*: quantifier(set) $->$ scoped meaning
  - *Negation*: not(proposition) $->$ negated proposition

  Each adds one level of nonlinear composition.
]

=== 14.2.3 Discourse Depth

Multi-sentence reasoning adds composition:

#example("Discourse Composition")[
  "John went to the store. He bought milk. He paid with cash."

  Discourse composition:
  1. "John" $->$ referent for "He" (depth 1)
  2. "store" $->$ context for "bought" (depth 2)
  3. "milk" $->$ referent for transaction (depth 3)
  4. Cash payment implies physical store (depth 4)

  Tracking this across sentences: ~4-6 composition levels.
]

== 14.3 Composition Depth in Code

Programming languages exhibit higher composition depth than natural language:

#example("Code Composition: Shallow")[
  ```python
  x = a + b * c
  ```

  Composition depth: 2
  1. `b * c` (multiplication)
  2. `a + ...` (addition)

  Linear-temporal models handle this easily.
]

#example("Code Composition: Deep")[
  ```python
  def fib(n):
      if n <= 1:
          return n
      return fib(n-1) + fib(n-2)
  ```

  Composition depth for `fib(10)`:
  - Recursive call depth: 10
  - Each call: 2-3 operations
  - Total depth: ~20-30

  This approaches D=32 layer limits.
]

#example("Code Composition: Very Deep")[
  ```python
  def eval_expr(expr, env):
      if is_atom(expr):
          return lookup(expr, env)
      elif is_lambda(expr):
          return Closure(expr, env)
      elif is_application(expr):
          func = eval_expr(get_func(expr), env)
          arg = eval_expr(get_arg(expr), env)
          return apply_func(func, arg)
  ```

  For deeply nested expressions like:
  `((((f a) b) c) d)` with each requiring closure evaluation:

  Composition depth: ~4 $times$ (recursion depth per eval) $times$ (expression nesting)

  Can easily exceed 100+ for realistic programs.
]

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,
    align: (left, center, center, center),
    [*Domain*], [*Typical Depth*], [*Maximum Depth*], [*D=32 Sufficient?*],
    [Syntax], [2-3], [~7], [$checkmark$],
    [Semantics], [3-4], [~10], [$checkmark$],
    [Discourse], [4-6], [~15], [$checkmark$],
    [Simple code], [2-5], [~10], [$checkmark$],
    [Recursive code], [10-30], [~100], [$checkmark$/partial],
    [Interpreter], [50-200], [unbounded], [$times$],
    [Formal proofs], [100+], [unbounded], [$times$],
  ),
  caption: [Composition depth requirements by domain.],
)

== 14.4 Detailed Examples with Depth Analysis

=== 14.4.1 Example: Pronoun Resolution

#example("Pronoun Resolution Chain")[
  _"Alice told Bob that she saw him at the store where they met."_

  Resolution chain:
  1. "Alice" establishes referent (depth 1)
  2. "Bob" establishes referent (depth 1)
  3. "she" resolves to Alice (depth 2: match gender + recency)
  4. "him" resolves to Bob (depth 2: match gender + recency)
  5. "the store" introduces location (depth 2)
  6. "where they met" links to store (depth 3)
  7. "they" resolves to {Alice, Bob} (depth 4: plural + context)

  *Total sequential decisions*: 7
  *Nonlinear compositions*: ~4-5

  A D=32 model can handle this with depth to spare.
]

=== 14.4.2 Example: Arithmetic Word Problem

#example("Multi-Step Arithmetic")[
  _"A store has 3 boxes. Each box contains 4 bags. Each bag has 5 apples. A customer buys 2 boxes. How many apples does the customer have?"_

  Computation:
  1. Parse: 3 boxes, 4 bags/box, 5 apples/bag (depth 1)
  2. Total apples: $3 times 4 times 5 = 60$ (depth 2-3)
  3. Customer buys 2 boxes: $2 times 4 times 5 = 40$ (depth 2-3)
  4. Answer: 40 (depth 4)

  *Critical observation*: Steps 2 and 3 require _sequential_ multiplication. Each multiplication is a nonlinear operation. But there are only 3-4 multiplications total.

  A D=32 model can represent this. The question is whether it _learns_ to.
]

=== 14.4.3 Example: Logical Deduction

#example("Syllogistic Reasoning")[
  _"All mammals are animals. All dogs are mammals. Fido is a dog. Is Fido an animal?"_

  Deduction chain:
  1. Parse rule: mammal $->$ animal (depth 1)
  2. Parse rule: dog $->$ mammal (depth 1)
  3. Parse fact: Fido $in$ dog (depth 1)
  4. Apply rule 2: Fido $in$ mammal (depth 2: modus ponens)
  5. Apply rule 1: Fido $in$ animal (depth 3: modus ponens)
  6. Answer: Yes (depth 4)

  *Total composition depth*: 4

  *But*: Each modus ponens requires a _threshold decision_ (does the rule apply?). Linear-temporal models cannot implement exact threshold. D=32 provides 32 such decisions, enough for simple syllogisms but potentially insufficient for long deduction chains.
]

=== 14.4.4 Example: Code Execution Trace

#example("Factorial Execution")[
  ```
  def factorial(n):
      if n <= 1: return 1
      return n * factorial(n-1)

  factorial(5)
  ```

  Execution trace:
  ```
  factorial(5)
    -> 5 * factorial(4)      [decision: 5 > 1]
    -> 5 * (4 * factorial(3)) [decision: 4 > 1]
    -> 5 * (4 * (3 * factorial(2))) [decision: 3 > 1]
    -> 5 * (4 * (3 * (2 * factorial(1)))) [decision: 2 > 1]
    -> 5 * (4 * (3 * (2 * 1))) [decision: 1 <= 1]
    -> 120
  ```

  *Decisions*: 5 threshold comparisons
  *Multiplications*: 4
  *Total nonlinear depth*: ~9

  For `factorial(100)`:
  - Decisions: 100
  - Multiplications: 99
  - Total depth: ~199

  This _exceeds_ D=32. A 32-layer linear-temporal model cannot trace `factorial(100)` exactly.

  *E88 advantage*: With temporal nonlinearity, a 1-layer E88 could potentially track the state across all 100 steps.
]

== 14.5 The Temporal vs Depth Trade-off

=== 14.5.1 Depth in Layers vs Depth in Time

#definition("Depth Allocation")[
  Given a computation requiring $N$ nonlinear operations:

  *Linear-temporal (D layers)*: Can compute $N$ if $N <= D$.
  Depth comes from layer stacking only.

  *E88 (D layers, T timesteps)*: Can compute $N$ if $N <= D times T$.
  Depth comes from both layers and temporal nonlinearity.
]

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,
    align: (left, center, center, center),
    [*Computation*], [*Depth Needed*], [*D=32 Linear*], [*D=1 E88, T=1000*],
    [Pronoun resolution], [5], [$checkmark$], [$checkmark$],
    [Word problem], [4], [$checkmark$], [$checkmark$],
    [Syllogism], [4], [$checkmark$], [$checkmark$],
    [`factorial(10)`], [19], [$checkmark$], [$checkmark$],
    [`factorial(50)`], [99], [$times$], [$checkmark$],
    [`factorial(100)`], [199], [$times$], [$checkmark$],
    [Interpreter], [500+], [$times$], [$checkmark$ (if T adequate)],
  ),
  caption: [Depth requirements vs available depth by architecture.],
)

=== 14.5.2 When Does Temporal Depth Matter?

#observation("The Crossover Point")[
  Temporal nonlinearity becomes essential when:

  $ "Required depth" > D times 2^D $

  For D=32: crossover at ~$32 times 2^(32)$, effectively never for practical sequences.

  *But*: This assumes the depth can be _distributed_ across layers optimally. In practice:
  - Mamba2 must learn to use all 32 layers effectively
  - Each layer adds parameters and training complexity
  - Some computations require _sequential_ steps that can't parallelize across layers
]

== 14.6 Case Study: Running Parity in Text

=== 14.6.1 Does Running Parity Occur Naturally?

#example("Quasi-Parity in Language")[
  _"The door was open. Then it was closed. Then open again. Then closed. What is the current state?"_

  This is running XOR:
  - State $<-$ State XOR (open/closed toggle)
  - After 4 toggles from "open": open $->$ closed $->$ open $->$ closed
  - Answer: closed

  *Key*: This requires _exact_ tracking of the parity of toggles.
]

#observation("Rarity of Exact Parity")[
  Running parity in natural text is:
  1. *Rare*: Most state changes don't require exact toggle counting
  2. *Approximate*: "Several times" often suffices without exact count
  3. *Short*: When it occurs, typically 2-4 toggles, not 100

  For typical text, linear-temporal models may approximate well enough. The theoretical limitation (cannot compute exact parity) may not manifest as practical failure.
]

=== 14.6.2 When Parity Matters

Parity becomes critical in:

1. *Negation stacking*: "He didn't say he wouldn't not do it."
   - Parity of negations determines meaning
   - Beyond ~3 negations, humans struggle too

2. *Transaction logs*: "Credit, debit, credit, debit, credit."
   - Net effect depends on parity
   - Financial systems need exact tracking

3. *Game state*: "He moved up, then down, then up, then down..."
   - Final position depends on parity
   - Important for game-playing AI

#finding[
  *Assessment*: Running parity is theoretically important but practically rare in natural text. The E88 vs Mamba2 separation on parity may not translate to language modeling benchmarks but could matter for:
  - Code execution
  - Game playing
  - Financial reasoning
  - Any domain requiring exact state tracking
]

== 14.7 Concrete Examples Table

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    align: (left, left, center, center, center),
    [*Example*], [*Description*], [*Depth*], [*D=32 OK?*], [*E88 Needed?*],
    ["The cat sat."], [Simple sentence], [2], [$checkmark$], [No],
    [Nested relative clauses $times$3], [Complex syntax], [6], [$checkmark$], [No],
    ["He said she said he said..."], [Quote embedding], [varies], [$checkmark$ for $<=$ 10], [If deep],
    [5-step word problem], [Arithmetic reasoning], [8-10], [$checkmark$], [No],
    [8-step proof], [Logical deduction], [15-20], [$checkmark$], [Borderline],
    [`fib(20)` trace], [Recursive execution], [~40], [$times$], [Yes],
    [`quicksort([1..100])`], [Algorithm trace], [~700], [$times$], [Yes],
    [Full program execution], [Interpreter simulation], [1000+], [$times$], [Yes],
    [4 XOR toggles], [Running parity], [4], [$checkmark$ per layer], [Exact: Yes],
    [100 XOR toggles], [Long running parity], [100], [$times$], [Yes],
  ),
  caption: [Composition depth requirements for specific examples.],
)

== 14.8 Summary

#finding[
  *Key Findings*:

  1. *Natural language* typically requires depth 2-10, well within D=32 limits
  2. *Simple code* requires depth 5-20, usually within limits
  3. *Complex algorithms* require depth 50-1000+, exceeding D=32
  4. *Running parity* is rare in natural text but critical for exact reasoning
  5. *The gap matters* for code execution, formal reasoning, and exact state tracking
]

#block(
  fill: rgb("#f0f7ff"),
  stroke: rgb("#3366cc"),
  inset: 12pt,
  radius: 4pt,
)[
  *The Depth Distribution Hypothesis*:

  The composition depth of natural language follows a heavy-tailed distribution:
  - Most text: depth 2-5 (easily handled by D=32)
  - Occasional complex text: depth 10-30 (stretches D=32)
  - Rare deep reasoning: depth 50+ (exceeds D=32)

  Linear-temporal models (Mamba2) perform well on the bulk of the distribution. E88's advantage manifests in the tail---exactly the cases where reasoning fails and chain-of-thought becomes necessary.
]

The next section explores how this depth limitation manifests as the "uncanny valley" of LLM reasoning: models that _appear_ to reason but fail on tasks requiring genuine compositional depth.

