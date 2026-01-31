// Section 3: The Linear-Temporal Limitation

#import "traditional-math-style.typ": *

= The Linear-Temporal Limitation

Armed with the mathematical machinery, we can now catalogue what linear-temporal models cannot do. The results are not merely theoretical curiosities---they apply directly to Mamba2, Fast Linear Attention, Gated Delta Networks, and every architecture built on linear state-space foundations.

== The Architectures in Question

To see why these limitations matter, we must first recognize how widespread linear temporal dynamics are. The modern sub-quadratic revolution rests on a shared insight: if the state evolution is linear in the hidden state, the entire sequence can be processed in parallel via associative scan.

Mamba2's core recurrence is $h_t = A h_(t-1) + B x_t$, which unfolds to $h_T = sum_t A^(T-t) B x_t$. The matrices $A$ and $B$ may depend on the input at each step, but the state remains a linear function of past states. This is what enables the parallel scan---and what constrains the computation.

Fast Linear Attention computes the output as $"Output" = q dot.op (sum_i k_i times.o v_i)$, a linear combination of key-value outer products. Gated Delta Networks update state as $S' = S + (v - S k) k^top$, which is linear in $S$. Despite their different motivations---selective state spaces, linear attention, delta rule learning---they share the same fundamental constraint.

== The Threshold Barrier

Running threshold is perhaps the simplest function that linear-temporal models cannot compute. The task: output 1 when the cumulative sum of inputs exceeds a threshold, output 0 otherwise.

#theorem("Running Threshold")[
  No $D$-layer linear-temporal model computes running threshold. The output of such a model is a continuous function of its inputs; threshold is discontinuous.
]#leanfile("ExactCounting.lean:344")

The proof is almost too simple. A linear combination of inputs is continuous. Threshold has a jump discontinuity. Continuous functions cannot equal discontinuous ones. Adding layers does not help---the composition of linear-temporal layers with nonlinear activations produces a continuous output, not a discontinuous one.

This impossibility extends to any function with a hard decision boundary. Binary classification, exact counting, flip detection---all require the output to jump between discrete values. Linear-temporal models can only approximate such jumps with smooth transitions.

== The Parity Barrier

Running parity presents a different obstacle. The task: at each position, output the XOR of all inputs so far. Unlike threshold, parity is not about discontinuity---it is about the algebraic structure of XOR.

#theorem("Running Parity")[
  No linear-temporal model computes $y_t = x_1 xor dots xor x_t$. Parity violates the affine identity: $f(0,0) + f(1,1) = 0 eq.not 2 = f(0,1) + f(1,0)$.
]#leanfile("RunningParity.lean:200")

The proof invokes the same algebraic argument we saw for XOR on two inputs. Affine functions satisfy $f(a) + f(b) = f(c) + f(d)$ when $a + b = c + d$. Parity does not. Therefore parity is not affine, and linear-temporal outputs are affine.

Parity has a distinguished role in complexity theory. It is the canonical function separating AC⁰ from TC⁰---adding threshold gates to constant-depth circuits allows parity to be computed. For linear-temporal models, parity is impossible at _any_ depth.

== The Capability Boundary

We can summarize the landscape in a single table. The separation is stark: functions that seem computationally trivial---parity, threshold, state machine simulation---are provably beyond the reach of linear-temporal architectures.

#simpletable(
  columns: 4,
  align: (left, center, center, center),
  [*Task*], [*Why Impossible*], [*D-layer Linear*], [*1-layer E88*],
  [Running threshold], [Discontinuous], [No], [Yes],
  [Running parity], [Non-affine], [No], [Yes],
  [FSM simulation], [State count], [Limited], [Full],
)

The rightmost column anticipates the next section. E88, with a single layer of nonlinear temporal dynamics, computes all three. The contrast is not empirical---it is mathematical. These are theorems, not benchmarks.

== When Linear Suffices

The limitations we have catalogued do not doom linear-temporal models. For many practical tasks, linear suffices.

Natural language has a characteristic depth distribution. Most sentences require parsing depth 2--5; complex center-embedded clauses push to 7--10; the extreme tail reaches 20--25. A 32-layer model with $D = 32$ exceeds most natural language requirements. The linear-temporal scan processes the entire sequence in parallel, achieving throughput that sequential nonlinear recurrence cannot match.

The limitation matters when depth is constrained (embedded systems, latency-sensitive applications), when tasks inherently require temporal decisions (counting, parity, state tracking), or when algorithmic reasoning is needed (following explicit procedures, simulating automata). For these problems, no amount of linear-temporal depth suffices. The gap is fundamental.

#centerrule

We have established the boundary. Linear-temporal models---Mamba2, FLA, GDN, and their relatives---are mathematically constrained. Threshold, parity, and state tracking lie beyond their reach, regardless of depth. The constraint is not a bug to be fixed; it is a consequence of the design choice to make time flow linearly.

The question becomes: how does E88 escape? What is it about $S_t = tanh(alpha S_(t-1) + delta v_t k_t^top)$ that crosses the boundary we have just drawn?
