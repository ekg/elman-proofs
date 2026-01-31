// Section 2: Mathematical Foundations

#import "traditional-math-style.typ": *

= Mathematical Foundations

Before we can prove what linear systems cannot do, we need the language to describe what they _are_. The central concept is _recurrence linearity_: whether the new state depends linearly or nonlinearly on the previous state.

== Linear Recurrent Systems

The simplest recurrent model updates its hidden state by a matrix multiplication plus an input term.

#definition("Linear RNN")[
  A linear RNN has dynamics $h_t = A h_(t-1) + B x_t$ and output $y_t = C h_t$, where $A in RR^(n times n)$, $B in RR^(n times m)$, $C in RR^(k times n)$.
]

This innocuous-looking equation has a powerful consequence. By unrolling the recurrence, we obtain a closed form.

#theorem("State as Weighted Sum")[
  For a linear RNN starting from $h_0 = 0$: $h_T = sum_(t=0)^(T-1) A^(T-1-t) B x_t$.
]#leanref("LinearCapacity.lean:72", "linear_state_is_sum")

#proof[
  Expand the recurrence: $h_1 = B x_0$, $h_2 = A B x_0 + B x_1$, and so on. The general term is a weighted sum of past inputs, with weights given by powers of $A$.
]

This closed form is the key to everything that follows. The state $h_T$ is a _linear combination_ of inputs. The output $y_T = C h_T$ is therefore also linear in the input sequence. No matter how long the sequence, no matter how cleverly we choose $A$, $B$, $C$, the output remains additive and homogeneous: $y(alpha x + beta z) = alpha y(x) + beta y(z)$.

== The Impossibility Results

Linearity is a severe constraint. Consider the threshold function: output 1 if the sum of inputs exceeds some threshold $tau$, output 0 otherwise. This is discontinuous---it jumps from 0 to 1 at the boundary. Linear functions are continuous. Therefore:

#theorem("Threshold Impossibility")[
  No linear RNN computes $"thresh"_tau$: output 1 if $sum_t x_t > tau$, else 0.
]#leanref("LinearLimitations.lean:107", "linear_cannot_threshold")

#proof[
  A linear output has the form $y(x) = g dot x$ for some fixed vector $g$. At inputs above threshold, $y = 1$ regardless of the exact sum. But if $y = 1$ for all $x$ with $sum x_i > tau$, and the output is linear, then $g = 0$. This contradicts $y = 1$ for any input. The function cannot be realized.
]

The argument for XOR is different. XOR is not continuous in the right sense---it is a Boolean function. But it fails linearity in a more algebraic way.

#theorem("XOR Is Not Affine")[
  No affine function equals XOR on ${0,1}^2$.
]#leanref("LinearLimitations.lean:218", "xor_not_affine")

#proof[
  Suppose $f(x,y) = a x + b y + c$ agrees with XOR. Then $f(0,0) = 0$ gives $c = 0$. $f(0,1) = 1$ gives $b = 1$. $f(1,0) = 1$ gives $a = 1$. But then $f(1,1) = a + b + c = 2 eq.not 0 = "XOR"(1,1)$.
]

== Depth Does Not Help

A natural response is to add layers. Perhaps a deep stack of linear-temporal layers can overcome the limitation? This hope is unfounded.

#definition("Multi-Layer Linear-Temporal Model")[
  A $D$-layer model with linear temporal dynamics within each layer and nonlinear activations (ReLU, GeLU) between layers.
]

#theorem("Depth Cannot Compensate")[
  For any $D >= 1$, a $D$-layer linear-temporal model cannot compute threshold, parity, or XOR.
]#leanref("MultiLayerLimitations.lean:231", "multilayer_cannot_running_threshold")

The intuition is geometric. Each layer aggregates features linearly across time. Stacking adds vertical depth---nonlinearity between layers---but not horizontal depth---nonlinearity through time. These are orthogonal directions. Depth is nonlinearity in the wrong dimension.

== The Composition Depth Gap

We can now state the central technical result precisely.

#theorem("Composition Depth Gap")[
  For a $D$-layer model processing sequences of length $T$:
  #h(1em) Linear temporal dynamics: total composition depth $D$.
  #h(1em) Nonlinear temporal dynamics: total composition depth $D times T$.
]#leanref("RecurrenceLinearity.lean:229", "e1_more_depth_than_minGRU")

The depth of a computation is the longest chain of nonlinear operations from input to output. In a linear-temporal model, only the interlayer activations are nonlinear---$D$ of them. In a nonlinear-temporal model, each timestep adds nonlinearity _within_ a layer, for a total of $D times T$.

== Architecture Classification

Which architectures fall where? The classification follows directly from the recurrence equation.

#proposition("Classification")[
  _Linear in $h$_: MinGRU, MinLSTM, Mamba2 SSM. All have the form $h_t = A(x_t) h_(t-1) + b(x_t)$---linear in $h$, even if $A$ and $b$ depend on the input.

  _Nonlinear in $h$_: E1, E88, standard RNN, LSTM, GRU. All have the form $h_t = sigma(W h_(t-1) + V x_t)$---the previous state passes through a nonlinearity.
]#leanref("RecurrenceLinearity.lean:148,171", "e1_is_nonlinear_in_h, mamba2_is_linear_in_h")

Input-dependent gating does not change the classification. In Mamba2, $A(x_t)$ and $B(x_t)$ depend on the current input, but the recurrence $h_t = A(x_t) h_(t-1) + B(x_t) x_t$ is still linear in $h$. The state at time $T$ is still a weighted sum of inputs.

#centerrule

We have established the mathematical foundation. Linear recurrent systems are constrained to compute linear functions of their input history. No amount of depth escapes this constraint---depth adds nonlinearity between layers, not through time. The next section examines the consequences: which functions are provably impossible for linear-temporal models, and why this matters for practical architectures.
