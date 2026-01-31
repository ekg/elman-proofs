// Section 15: The Uncanny Valley of Reasoning
// Why LLMs Fail at Deep Thought

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

#let failure(title, body) = block(
  fill: rgb("#fff0f0"),
  stroke: rgb("#cc3333"),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
)[
  #strong[Failure Mode: #title]#linebreak()
  #body
]

= The Uncanny Valley of Reasoning

_Why Language Models Appear to Reason But Fundamentally Cannot_

This section analyzes the "uncanny valley" phenomenon in LLM reasoning: models produce outputs that _look_ like reasoning but fail on tasks requiring genuine sequential logic. We connect this phenomenon to the composition depth limitations established in earlier sections.

== 15.1 The Uncanny Valley Phenomenon

#definition("Uncanny Valley of Reasoning")[
  The *uncanny valley of reasoning* describes the gap between:
  1. *Apparent capability*: LLMs produce fluent, structured, reasoning-like output
  2. *Actual capability*: LLMs fail on problems requiring $>D$ sequential steps

  This creates systems that appear intelligent but fail in surprising, often frustrating ways.
]

#observation("The Pattern")[
  LLMs excel at:
  - Recognizing reasoning _patterns_ seen in training
  - Producing syntactically correct reasoning _text_
  - Shallow composition (2-5 steps)

  LLMs fail at:
  - Novel combinations of reasoning steps
  - Deep composition (10+ steps)
  - Tasks requiring exact sequential logic
]

#finding[
  *The Core Claim*: The uncanny valley arises from a mismatch between pattern recognition (what LLMs do well) and compositional computation (what reasoning requires).

  LLMs have been trained on text that _describes_ reasoning, not on the computation _underlying_ reasoning. They learn the surface form, not the deep structure.
]

== 15.2 Architectural Explanation

=== 15.2.1 The Depth Bottleneck

Recall from Section 14:

#definition("Composition Depth Limits")[
  *Transformers* (D layers): Composition depth = $D$

  *Linear-temporal SSMs* (D layers): Composition depth = $D$

  *E88-style* (D layers, T steps): Composition depth = $D times T$

  For D=32 (typical), transformers and linear SSMs can represent at most 32 sequential nonlinear operations. Any computation requiring more is _provably impossible_.
]

#theorem("The Depth Barrier")[
  Let $f$ be a function requiring composition depth $N$. If $N > D$:
  - No D-layer transformer can compute $f$
  - No D-layer linear-temporal SSM can compute $f$
  - Both may produce _approximations_ that appear correct on training distribution

  The failure manifests as:
  - Correct answers for small instances (depth $< D$)
  - Degraded accuracy for larger instances (depth $approx D$)
  - Random-ish behavior for large instances (depth $>> D$)
]

=== 15.2.2 Why It Looks Like Reasoning

LLMs produce reasoning-like output because:

1. *Training data contains reasoning traces*: Step-by-step solutions in textbooks, StackOverflow, etc.

2. *Pattern matching suffices for shallow cases*: "If the problem looks like X, the solution looks like Y" works for depth $< D$.

3. *Fluency masks incapability*: The model generates grammatically correct, topically relevant text even when the logic is wrong.

#example("Shallow vs Deep")[
  *Shallow* (within depth budget):
  _"What is 23 + 45?"_
  Model: "23 + 45 = 68" #sym.checkmark

  The pattern "number + number = result" is well-represented.

  *Deep* (beyond depth budget):
  _"If Alice has 3 apples, gives 1 to Bob, Bob gives half to Carol, Carol gives 1 back to Alice, and Alice gives all but 1 to Dave, how many does Alice have?"_

  Model may: track state incorrectly, lose count, or produce a plausible-sounding wrong answer.
]

== 15.3 Failure Modes

=== 15.3.1 State Tracking Failures

#failure("Lost State")[
  *Task*: Track a counter through multiple operations.

  _"Start with 0. Add 3. Multiply by 2. Subtract 4. Add 1. What's the result?"_

  Correct: 0 $->$ 3 $->$ 6 $->$ 2 $->$ 3

  LLM failure mode: Loses track of intermediate value, produces 7 (dropped the subtract) or 5 (dropped the multiply).

  *Root cause*: Each operation requires updating state. After $D$ operations, the model cannot track the full sequence.
]

#failure("State Confusion")[
  *Task*: Track multiple entities with changing states.

  _"Alice is in the kitchen. Bob is in the garden. Alice moves to the garden. Bob moves to the kitchen. Where is Alice?"_

  LLM failure mode: Confuses Alice and Bob after swaps, says "Alice is in the kitchen."

  *Root cause*: Distinguishing entities with swapped properties requires maintaining separate state channels with update history.
]

=== 15.3.2 Logical Deduction Failures

#failure("Chain Breaking")[
  *Task*: Follow a long logical chain.

  _"If A then B. If B then C. If C then D. If D then E. Given A, what can we conclude about E?"_

  Correct: A $->$ B $->$ C $->$ D $->$ E #sym.checkmark

  LLM failure mode for longer chains:
  - Skips steps: "A implies C" (missing B)
  - Inverts: "E implies A"
  - Hallucinates: "A implies F" (where F wasn't mentioned)

  *Root cause*: Each modus ponens step requires a nonlinear decision. A 10-step chain requires 10 sequential decisions.
]

#failure("Negation Blindness")[
  *Task*: Handle nested negations.

  _"It is not the case that John did not fail to avoid not missing the train."_

  Correct: Parse the 5 negations (not, not, fail, avoid, miss) to determine: John missed the train.

  LLM failure mode: Counts negations incorrectly, or defaults to most common interpretation.

  *Root cause*: This is running parity---provably impossible for linear-temporal models.
]

=== 15.3.3 Mathematical Reasoning Failures

#failure("Carry Propagation")[
  *Task*: Multi-digit arithmetic with carries.

  _"What is 789 + 456?"_

  Correct: 9+6=15 (write 5, carry 1), 8+5+1=14 (write 4, carry 1), 7+4+1=12. Answer: 1245.

  LLM failure mode for larger numbers:
  - Drops carries: 789 + 456 = 1135
  - Wrong carry position: 789 + 456 = 1345
  - Completely wrong: 789 + 456 = 1200 (rounding)

  *Root cause*: Each carry is a threshold decision. N-digit addition requires N threshold decisions.
]

#failure("Proof Step Omission")[
  *Task*: Construct a multi-step proof.

  _"Prove that the square of an odd number is odd."_

  Correct: Let n = 2k+1. Then n#super[2] = (2k+1)#super[2] = 4k#super[2] + 4k + 1 = 2(2k#super[2] + 2k) + 1 = 2m + 1, which is odd.

  LLM failure mode:
  - Skips the algebra: "Since n is odd, n#super[2] is obviously odd."
  - Wrong justification: "Odd times odd is odd, so done."
  - Correct answer, wrong proof: Gets lucky on the pattern.

  *Root cause*: The proof requires sequential algebraic steps, each depending on the previous.
]

== 15.4 The Pattern Recognition vs Computation Distinction

#definition("Pattern Recognition")[
  *Pattern recognition* maps input patterns to output patterns based on training distribution:
  $ f: "pattern" -> "pattern" $

  Characteristics:
  - Parallel: All input features processed simultaneously
  - Associative: Similar inputs yield similar outputs
  - Interpolative: Works well within training distribution
]

#definition("Computation")[
  *Computation* executes a sequence of dependent operations:
  $ f = g_n compose g_(n-1) compose ... compose g_1 $

  Characteristics:
  - Sequential: Step $i$ depends on step $i-1$
  - Exact: Must get each step correct
  - Generalizing: Works on novel inputs if algorithm is correct
]

#observation("The Fundamental Mismatch")[
  LLMs are trained to _minimize loss on next token prediction_, which rewards:
  - Pattern matching (what token usually follows?)
  - Fluency (what sounds natural?)
  - Coherence (what maintains topic?)

  This does NOT reward:
  - Correctness of intermediate steps
  - Consistency of state across tokens
  - Validity of logical inferences

  The training objective is misaligned with reasoning capability.
]

== 15.5 Chain-of-Thought as a Workaround

=== 15.5.1 How CoT Helps

Chain-of-thought prompting asks the model to generate intermediate steps:

#example("CoT Improvement")[
  *Without CoT*:
  _"If Alice has 8 apples and gives half to Bob, how many does she have?"_
  Model: "4" (correct by pattern matching)

  *With CoT*:
  _"Think step by step."_
  Model: "Alice has 8. Half of 8 is 4. She gives 4 to Bob. She has 8 - 4 = 4 apples."

  The intermediate steps are _generated as tokens_, allowing the model to use them as "external memory."
]

#theorem("CoT as Depth Extension")[
  Chain-of-thought extends composition depth by:

  1. *Externalizing state*: Intermediate values written as tokens become input to subsequent generation
  2. *Adding passes*: Each generated step is a new "forward pass" through the model
  3. *Trading breadth for depth*: Sequence length increases, but each segment stays within depth D

  Effective depth with CoT: $D times P$ where $P$ is the number of "steps" generated.
]

=== 15.5.2 Limitations of CoT

#failure("CoT Error Propagation")[
  *Problem*: If any intermediate step is wrong, subsequent steps are corrupted.

  _"What is 37 $times$ 28?"_
  Model with CoT: "37 $times$ 28 = 37 $times$ 30 - 37 $times$ 2 = 1110 - 74 = 1036"

  Correct: 37 $times$ 28 = 1036 #sym.checkmark

  But if the model makes an error:
  "37 $times$ 28 = 37 $times$ 30 - 37 $times$ 2 = 1110 - 72 = 1038" #sym.crossmark

  The error in 37 $times$ 2 = 72 propagates to the final answer.
]

#failure("CoT Doesn't Help Parallel Requirements")[
  *Problem*: Some computations need multiple simultaneous states.

  Tracking 5 characters in a story, each with their own location, inventory, and relationships, requires 5 parallel state channels. CoT linearizes this, but the model still needs to maintain coherence across the linear trace.

  *Root cause*: CoT increases depth but not width. Problems requiring $W > 1$ parallel channels remain difficult.
]

== 15.6 Why LLMs Fail at Deep Thought

=== 15.6.1 The Three Failures

#finding[
  LLMs fail at deep thought due to three compounding issues:

  1. *Architectural limitation*: Fixed depth $D$ bounds composition depth
  2. *Training limitation*: Objective rewards fluency, not correctness
  3. *Generalization limitation*: Training distribution doesn't cover novel compositions
]

=== 15.6.2 The Composition Gap

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,
    align: (left, center, center, center),
    [*Task Type*], [*Depth Needed*], [*D=32 Model*], [*Failure Mode*],
    [2-step reasoning], [2], [$checkmark$], [N/A],
    [5-step reasoning], [5], [$checkmark$], [Rare errors],
    [10-step reasoning], [10], [$checkmark$], [Occasional errors],
    [20-step reasoning], [20], [Partial], [Frequent errors],
    [50-step reasoning], [50], [$times$], [Systematic failure],
    [Running parity (any length)], [$T$], [$times$], [~50% accuracy],
  ),
  caption: [Failure modes by composition depth.],
)

=== 15.6.3 The Emergent Behavior Illusion

#observation("Why It Feels Like Progress")[
  As models scale:
  1. They see more training examples of reasoning patterns
  2. They can pattern-match more complex shallow cases
  3. Their failures become more sophisticated (plausible wrong answers)

  This creates an illusion of "emergent reasoning" when what's actually happening is:
  - Better pattern matching
  - More training data coverage
  - Smoother interpolation in feature space

  The fundamental depth limit remains: composition depth $<= D$.
]

== 15.7 Implications for AI Safety and Alignment

=== 15.7.1 Unpredictable Failure Modes

#failure("Confident Wrongness")[
  Models produce wrong answers with high confidence because:
  1. The answer _pattern_ looks correct
  2. The fluency is indistinguishable from correct answers
  3. The model has no mechanism to verify its own reasoning

  This is especially dangerous for:
  - Critical decisions based on LLM output
  - Long reasoning chains where errors compound
  - Novel situations outside training distribution
]

=== 15.7.2 The Verification Problem

#observation("Why Verification is Hard")[
  Verifying a reasoning chain requires:
  - Checking each step individually (linear in chain length)
  - Understanding the dependencies between steps
  - Detecting subtle errors that maintain surface coherence

  LLMs _generating_ verification have the same depth limits as LLMs generating reasoning. Verification is not easier than generation.

  *Solution*: External verification tools (proof assistants, code execution, calculators) that perform exact computation.
]

== 15.8 Architectural Paths Forward

=== 15.8.1 Temporal Nonlinearity (E88-style)

#finding[
  *E88 approach*: Add nonlinearity to the temporal dimension.

  $ S_(t+1) = tanh(alpha S_t + delta k_t^T) $

  This gives composition depth $D times T$ instead of $D$, potentially addressing the depth bottleneck for sequential reasoning.

  *Trade-off*: Slower training (sequential instead of parallel).

  *Status*: Theoretically sound, empirically unproven at scale for reasoning tasks.
]

=== 15.8.2 Multi-Pass Processing

#finding[
  *Multi-pass approach*: Run the model multiple times, feeding output back as input.

  Effective depth: $D times P$ where $P$ is the number of passes.

  *Implementation*: Chain-of-thought, scratchpad, etc.

  *Trade-off*: Linear slowdown in inference time.

  *Status*: Proven effective (CoT improves reasoning), but error propagation limits gains.
]

=== 15.8.3 Hybrid Architectures

#finding[
  *Hybrid approach*: Combine linear-temporal (fast) with nonlinear-temporal (expressive).

  - Use Mamba2-style for bulk processing
  - Add E88-style heads for reasoning-critical paths
  - Route based on detected task complexity

  *Trade-off*: Architectural complexity, routing overhead.

  *Status*: Theoretical proposal, not yet validated.
]

=== 15.8.4 External Tools

#finding[
  *Tool use approach*: Delegate computation to external systems.

  - Calculator for arithmetic
  - Proof assistant for logical deduction
  - Code executor for algorithms
  - Search engine for factual recall

  *Trade-off*: Requires reliable tool interfaces, error handling, and orchestration.

  *Status*: Practical and widely deployed (GPT-4 + Code Interpreter, etc.).
]

== 15.9 The Fundamental Tension

#block(
  fill: rgb("#f0f7ff"),
  stroke: rgb("#3366cc"),
  inset: 12pt,
  radius: 4pt,
)[
  *The Core Tension*:

  Pattern recognition and computation are fundamentally different operations:

  *Pattern recognition*:
  - Input $->$ Output in one pass
  - Parallel, fast, efficient
  - Works on distribution, fails on novel combinations

  *Computation*:
  - Input $->$ Step 1 $->$ Step 2 $->$ ... $->$ Output
  - Sequential, slow, exact
  - Works on novel inputs if algorithm is correct

  Current LLMs are pattern recognizers pretending to be computers. The pretense works when the pattern is in the training data. It fails when genuine computation is required.
]

== 15.10 Summary

#figure(
  table(
    columns: 2,
    stroke: 0.5pt,
    align: (left, left),
    [*Phenomenon*], [*Explanation*],
    [Fluent wrong answers], [Pattern matching on surface form, not logic],
    [State tracking failures], [Depth limit exceeded for sequential updates],
    [Long chain breakage], [Each step requires depth; long chains exceed budget],
    [Negation blindness], [Running parity is provably impossible],
    [CoT improvement], [Externalizes state as tokens, extends effective depth],
    [Confident errors], [Fluency indistinguishable from correctness],
    [Scaling helps (a bit)], [More patterns memorized, same depth limit],
  ),
  caption: [Summary of uncanny valley phenomena.],
)

#block(
  fill: rgb("#fff0f0"),
  stroke: rgb("#cc3333"),
  inset: 12pt,
  radius: 4pt,
)[
  *The Uncanny Valley Explained*:

  LLMs fail at deep thought because they are _architecture-limited_ to composition depth $D$ (typically 32). This is not a training data problem, not an optimization problem, and not a scale problem---it is a _fundamental representational limitation_.

  The solution requires:
  1. *Architectural change*: Temporal nonlinearity (E88) or multi-pass processing
  2. *External computation*: Tools for exact arithmetic, logic, search
  3. *Honest acknowledgment*: These systems approximate reasoning but do not implement it

  The gap between appearing to reason and actually reasoning is the uncanny valley. Crossing it requires moving from pattern recognition to genuine sequential computation.
]

