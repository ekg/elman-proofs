# Q1: Multi-Layer SSM Expressivity Bounds

**Date:** 2026-01-29
**Status:** Analysis Complete
**Summary:** Depth does NOT compensate for linear temporal dynamics. D-layer Mamba2 cannot compute functions requiring true temporal nonlinearity.

---

## 1. The Central Question

From `OPEN_QUESTIONS_TEMPORAL_VS_DEPTH.md`:

> **Q1: Does depth compensate for linear temporal dynamics?**
> Can a D-layer Mamba2 compute functions that require temporal nonlinearity?

### The Hypothesis Under Investigation

**Optimistic hypothesis:** Stacking D layers of linear-temporal dynamics, with nonlinear inter-layer connections, might achieve the same expressivity as a single layer with nonlinear temporal dynamics.

**Counter-argument raised:** "The features computed by layer L at position t include nonlinearly-mixed information from positions 0..t via layers 1..L-1. Maybe this indirection provides temporal expressivity?"

---

## 2. Formal Framework

### 2.1 Definitions from Existing Proofs

From `LinearCapacity.lean`, a linear RNN's state evolution is:

```
h_T = Σ_{t=0}^{T-1} A^{T-1-t} B x_t
```

The output `y = C h_T` is a **linear function** of the entire input sequence.

From `LinearLimitations.lean`, we have two key impossibility results:

1. **`linear_cannot_threshold`**: Linear RNNs cannot compute threshold functions
2. **`linear_cannot_xor`**: Linear RNNs cannot compute XOR over sequence positions

### 2.2 Multi-Layer Linear-Temporal Model

Define a D-layer model where:
- Each layer L has state `h^L_t` evolving **linearly in time**
- Layer L receives layer L-1's output as input
- **Nonlinear activations** (SiLU, etc.) are applied between layers

The key observation: **Within each layer**, for fixed input from below:
```
h^L_T = Σ_{t=0}^{T-1} A_L^{T-1-t} B_L (input^L_t)
```

---

## 3. The Main Theorem

### Theorem: Multi-Layer Linear-Temporal Limitation

**Statement:** For any D ≥ 1, a D-layer model where each layer has linear temporal dynamics cannot compute functions that require **temporal nonlinearity within a layer**.

**Proof Sketch:**

Consider a single layer L at depth d. Let `I^L_t` denote its input at time t (which comes from layer L-1).

**Step 1: Layer-wise linearity.**
The output of layer L at time T is:
```
y^L_T = C_L · h^L_T = C_L · Σ_{t=0}^{T-1} A_L^{T-1-t} B_L · I^L_t
```
This is a **linear function** of the sequence `(I^L_0, I^L_1, ..., I^L_{T-1})`.

**Step 2: Composition of linear-in-sequence functions.**
Let `f_L: (ℝ^{d_in})^T → ℝ^{d_out}` be the function computed by layer L mapping input sequence to output at position T.

Each `f_L` is **linear** (affine if we include bias) in its input sequence.

**Step 3: The indirection doesn't help.**
Consider the counter-argument: layer L's input `I^L_t` includes nonlinearly-transformed information from earlier positions via layers 1..L-1.

**Critical observation:** While `I^L_t` is a nonlinear function of `(x_0, ..., x_t)`, the output `y^L_T` is still:
- A **linear combination** of `I^L_0, I^L_1, ..., I^L_{T-1}`
- Which is **not** equivalent to a nonlinear temporal composition

**Step 4: The threshold function counter-example.**

Consider the **running threshold function**:
```
τ_t = 1 if Σ_{s≤t} x_s > θ, else 0
```

This requires comparing a running sum against a threshold **at each timestep**.

**Claim:** No D-layer linear-temporal model can compute τ_t for all t simultaneously.

**Proof by induction on D:**

*Base case (D=1):* By `linear_cannot_threshold` from `LinearLimitations.lean`, a single linear layer cannot compute threshold functions.

*Inductive case (D→D+1):*
Assume the top layer D+1 receives input sequence `I^{D+1}_0, ..., I^{D+1}_{T-1}` from layer D.

The top layer's output at position t is:
```
y^{D+1}_t = C · Σ_{s=0}^{t-1} A^{t-1-s} B · I^{D+1}_s
```

This is a **linear function** of `I^{D+1}` values up to t.

For the model to compute `τ_t`, we need:
```
τ_t = linear_function(I^{D+1}_0, ..., I^{D+1}_{t-1})
```

But `τ_t` is a **step function** (discontinuous). A linear function is continuous.

Therefore, **no linear-in-time top layer can output τ_t**, regardless of how sophisticated the inputs from lower layers are.

∎

---

## 4. Why the Counter-Argument Fails

The counter-argument was:

> "The features computed by layer L at position t include nonlinearly-mixed information from positions 0..t via layers 1..L-1. Maybe this indirection provides temporal expressivity?"

### 4.1 The Distinction: Feature Nonlinearity vs. Temporal Nonlinearity

Consider what happens at position t in layer D:

1. **From below (layers 1..D-1):** The input `I^D_t` is a nonlinear function of `x_0, ..., x_t`
2. **Within layer D:** The state update is `h^D_t = A · h^D_{t-1} + B · I^D_t`

The nonlinearity from below allows **complex features** at each position t.
But the linear temporal dynamics prevent **nonlinear integration across time within layer D**.

### 4.2 Concrete Example

Suppose layers 1..D-1 compute:
- `I^D_t = relu(W · x_t)` (nonlinear feature of current input)

Layer D then computes:
- `h^D_T = Σ A^{T-1-t} B · relu(W · x_t)`

This is a linear combination of **individual nonlinear features**—not a nonlinear composition over time.

**What E88 can do that Mamba2 cannot:**

E88 update: `S := tanh(αS + δk^T)`

At time T:
```
S_T = tanh(α · tanh(α · tanh(...) + ...) + ...)
```

This is a **T-deep composition of tanh**. Each tanh can implement a decision boundary. T compositions can implement T sequential decisions.

Mamba2 update: `S_T = Σ α^{T-t} · k_t v_t^T`

This is a **single linear combination**. No matter how complex `k_t` and `v_t` are (from lower layers), the combination itself is linear.

---

## 5. Formal Statement for Lean

The following theorem would extend `LinearLimitations.lean`:

```lean
/-- A multi-layer model where each layer has linear temporal dynamics -/
structure MultiLayerLinearTemporal (D n m k : ℕ) where
  /-- Transition matrix for each layer -/
  A : Fin D → Matrix (Fin n) (Fin n) ℝ
  /-- Input projection for each layer -/
  B : Fin D → Matrix (Fin n) (Fin m) ℝ
  /-- Output projection for each layer -/
  C : Fin D → Matrix (Fin k) (Fin n) ℝ
  /-- Inter-layer nonlinearity (applied pointwise) -/
  σ : ℝ → ℝ

/-- The function computed by a multi-layer linear-temporal model -/
noncomputable def MultiLayerLinearOutput
    (model : MultiLayerLinearTemporal D n m k)
    (T : ℕ)
    (inputs : Fin T → (Fin m → ℝ)) : Fin k → ℝ :=
  sorry  -- Recursive definition through layers

/-- Multi-layer linear-temporal models cannot compute threshold functions -/
theorem multilayer_linear_cannot_threshold (D : ℕ) (τ : ℝ) (T : ℕ) (hT : T ≥ 1) :
    ¬ ∃ (model : MultiLayerLinearTemporal D n 1 1),
      ∀ inputs, MultiLayerLinearOutput model T inputs = thresholdFunction τ T inputs := by
  sorry  -- Proof by induction on D using linear_cannot_threshold
```

**Key insight for proof:** The top layer must output a **discontinuous function** (threshold), but its output is a continuous (linear) function of its inputs, regardless of what those inputs are.

---

## 6. Connection to RecurrenceLinearity.lean

From `RecurrenceLinearity.lean`:

```lean
/-- Within-layer composition depth -/
def within_layer_depth (r : RecurrenceType) (seq_len : Nat) : Nat :=
  match r with
  | RecurrenceType.linear => 1           -- Collapses regardless of seq_len
  | RecurrenceType.nonlinear => seq_len  -- Grows with sequence length

/-- Total composition depth = layers × within-layer depth -/
def total_depth (r : RecurrenceType) (layers seq_len : Nat) : Nat :=
  layers * within_layer_depth r seq_len
```

For **D-layer Mamba2** (linear temporal):
```
total_depth = D × 1 = D
```

For **D-layer E88** (nonlinear temporal):
```
total_depth = D × T = D·T
```

**The depth gap is a factor of T (sequence length)!**

This explains why E88sh16n32 nearly matches fla-gdn at 1.40 loss despite being sequential—it has T-fold more compositional depth per layer.

---

## 7. Practical Implications

### 7.1 What This Means for Mamba2

Mamba2's advantages come from:
1. **Selectivity:** Input-dependent A(x) provides dynamic routing
2. **State expansion:** Higher-dimensional state space
3. **Depth:** More layers can still help (D compositions)

But Mamba2 fundamentally **cannot** express certain functions that E88 can with a single layer.

### 7.2 The Trade-off

| Property | Mamba2 (D layers) | E88 (D layers) |
|----------|-------------------|----------------|
| Within-layer depth | 1 | T (seq length) |
| Total depth | D | D·T |
| Parallelizable | Yes (scan) | No (sequential) |
| Throughput | High | Lower |

**Conclusion:** E88's sequential nature is the **cost** of temporal nonlinearity, which buys **T-fold more compositional depth** per layer.

### 7.3 When Does This Matter?

The limitation matters for tasks requiring **temporal decision sequences**:
- Running threshold detection
- Temporal XOR: `y_t = x_t XOR x_{t-k}`
- State machines with irreversible transitions
- Counting with saturation

For language modeling, empirical evidence suggests:
- Mamba2 matches Transformers (both have sufficient depth)
- The "missing" temporal nonlinearity may not be critical for language
- Or depth compensation is sufficient in practice (D ≥ 32)

---

## 8. Summary

### Answer to Q1

**No, depth does NOT fully compensate for linear temporal dynamics.**

A D-layer model with linear temporal dynamics has total composition depth D.
A D-layer model with nonlinear temporal dynamics has total composition depth D·T.

There exist functions (threshold, XOR) that:
- 1-layer E88 can compute (with sufficient T)
- No finite-depth Mamba2 can compute

### Formalization Status

| Component | Status |
|-----------|--------|
| Single-layer impossibility | ✓ Proven in `LinearLimitations.lean` |
| Multi-layer theorem | Stated, proof sketch provided |
| RecurrenceLinearity connection | ✓ Proven in `RecurrenceLinearity.lean` |

### Recommended Next Steps

1. **Q2:** Find explicit separation example (threshold with temporal output)
2. **Q3:** Empirical test on synthetic tasks
3. **Formalize** `multilayer_linear_cannot_threshold` in Lean

---

## Appendix: Key File References

- `OPEN_QUESTIONS_TEMPORAL_VS_DEPTH.md` — Problem statement
- `ElmanProofs/Expressivity/LinearLimitations.lean:107` — `linear_cannot_threshold`
- `ElmanProofs/Expressivity/LinearCapacity.lean:72` — `linear_state_is_sum`
- `ElmanProofs/Architectures/RecurrenceLinearity.lean:215` — `within_layer_depth`
- `ElmanProofs/Information/LinearVsNonlinear.lean:77` — `linear_rnn_collapses`
- `E88_EXPANSION_FINDINGS.md` — E88 architecture details
