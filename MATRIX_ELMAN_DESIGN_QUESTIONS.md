# Matrix Elman Design Questions: E70-E73 vs Originals

## Overview

E70-E73 are matrix-state analogs of successful vector-state models. Two have discrepancies worth investigating:

| Model | Inspiration | Status |
|-------|-------------|--------|
| **E70** | E42 | **DISCREPANCY**: Has tanh, but E42 has NO tanh |
| E71 | E67 | OK - sigmoid gate is intentional in both |
| E72 | E68 | OK - sigmoid gating is intentional in both |
| **E73** | E1 | **DISCREPANCY**: Has sigmoid on z, but E1 has no pre-gating |

---

## E70: The Tanh Problem (More Severe)

### Current E70 Implementation

```python
S = decay * S + outer(v, k)   # Linear accumulation
S = tanh(S)                   # <-- THIS IS WRONG
out = (S @ q) * silu(S @ q)   # Self-gating output
```

### Original E42 (What It Should Match)

```python
h = W @ x + W @ h + b         # Linear recurrence, NO TANH!
output = h * silu(h)          # Self-gating (only nonlinearity)
# Stability via spectral normalization: ||W|| < 1
```

### The Problem

E42's entire design philosophy is **linear recurrence**:
- NO tanh on hidden state
- Spectral norm keeps ||W|| < 1 for stability
- Self-gating `h * silu(h)` is the ONLY nonlinearity

E70 adds tanh which destroys this property. The tanh:
1. Breaks linearity of the recurrence
2. Is redundant since `decay < 1` already bounds growth
3. Removes E42's key differentiator from other models

### E42's Stability Mechanism (Important!)

E42 uses spectral norm on W to bound BOTH:
- `W @ h` (recurrence) - contracts because ||W|| < 1
- `W @ x` (input) - bounded contribution

For E70 to be a true analog, we need to bound both parts:
- `decay * S` - bounded by decay < 1 ✓
- `outer(v, k)` - **currently unbounded!** ✗

The problem: `||outer(v, k)|| = ||v|| * ||k||` can grow without bound if W_k, W_v aren't constrained.

### Proposed Fix: E70n (No Tanh + Spectral Norm)

```python
# E70n: True E42 analog
W_k = spectral_norm(W_k, radius=0.999)  # bound ||k||
W_v = spectral_norm(W_v, radius=0.999)  # bound ||v||

k = W_k @ x  # ||k|| ≤ 0.999 * ||x||
v = W_v @ x  # ||v|| ≤ 0.999 * ||x||

S = decay * S + outer(v, k)   # Linear accumulation, NO tanh
out = (S @ q) * silu(S @ q)   # Self-gating provides nonlinearity

# Stability:
# - decay < 1 bounds recurrence (like ||W|| < 1 for h term)
# - spectral norm bounds ||outer(v,k)|| ≤ sr² * ||x||² (like ||W|| < 1 for x term)
```

### Alternative Approaches

1. **Spectral norm on W_k, W_v** - closest to E42 (recommended)
2. **Normalize v, k directly** (RMS norm) - simpler but different mechanism
3. **Just decay < 1** - might work in practice but less principled

### Why This Matters

E42 is one of the best-performing vector models. Its success suggests linear recurrence + self-gating is a powerful combination. If E70 adds tanh, we're not testing that hypothesis for matrix state.

To properly test: remove tanh AND add spectral norm on projections.

---

## E73: The Sigmoid Question (Less Severe)

### Current E73 Implementation

```python
z = sigmoid(W_z @ x + b_z)       # [B, n_state] in (0, 1)
S_mod = S * z.unsqueeze(1)       # Column modulation
S = tanh(S_mod + outer(v, k))    # Bounded output
```

### Original E1 (What It Should Match)

```python
h = tanh(W_h @ h + W_x @ x + b)  # No pre-gating sigmoid!
```

E1 has **no pre-gating sigmoid**. It just:
1. Takes linear combination of old state and new input
2. Applies tanh to bound the result

### Analysis

**Why sigmoid might be unnecessary:**

1. **Tanh already bounds output**: After `S = tanh(...)`, S is in (-1, 1) regardless of z

2. **Sigmoid restricts expressivity**: z forced to (0, 1) means:
   - S can only be scaled down (never amplified)
   - No sign flips possible through z
   - Column modulation is strictly attenuating

3. **E1 doesn't need it**: The successful vector-state model has no pre-gating

**Why sigmoid might help:**

1. Smoother gradients during training
2. Forget-gate interpretation (GRU-style)
3. Prevents explosive intermediate values

### Proposed Variant: E73n (No Sigmoid)

```python
# E73n: True E1 analog
z = W_z @ x + b_z                 # Unbounded
S_mod = S * z.unsqueeze(1)        # Can amplify, attenuate, or flip
S = tanh(S_mod + outer(v, k))     # tanh bounds final result
```

---

## Recommendations

### Priority 1: Fix E70 (Remove Tanh + Add Spectral Norm)

This is the more severe issue. E42's linear recurrence is a key architectural choice, not an oversight. E70 should test whether that property transfers to matrix state.

Create E70n variant:
```python
W_k = spectral_norm(W_k, radius=0.999)
W_v = spectral_norm(W_v, radius=0.999)
S = decay * S + outer(v, k)   # NO tanh
out = (S @ q) * silu(S @ q)
```

### Priority 2: Test E73 Variants

Less severe since tanh still provides nonlinearity. But worth testing:
- E73 (current): sigmoid on z
- E73n: no sigmoid on z

### Benchmarking

Run 100M parameter comparison:
- E70 vs E70n (tanh vs no-tanh)
- E73 vs E73n (sigmoid vs no-sigmoid)

If the "n" variants match or beat originals, the extra nonlinearities are unnecessary complexity.

---

## Summary for Elman Agents

**E70 has tanh that E42 doesn't have.** E42's whole point is linear recurrence with stability through spectral norm. E70 should:
1. Remove the tanh (breaks linearity)
2. Add spectral norm on W_k, W_v (bounds the input contribution)
3. Keep decay < 1 (bounds the recurrence)

The tanh is doing the job that spectral norm should do. Replace it with the proper mechanism.

**E73 has sigmoid that E1 doesn't have.** E1 just uses tanh on the combined state+input. The sigmoid pre-gating may be unnecessary since tanh already bounds everything.

E71 and E72 are fine - their sigmoid gates are intentional and match the original E67/E68 designs.
