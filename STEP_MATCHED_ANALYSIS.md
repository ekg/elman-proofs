# Step-Matched Analysis: What Makes Mamba2 Better?

**Date**: 2026-01-08

## The Experiment

To eliminate confounds from different training times/tokens, we ran all models for exactly **1000 steps × 32 batch × 512 tokens = 16.4M tokens**.

| Model | Params | Loss | Time | Tok/s |
|-------|--------|------|------|-------|
| **Mamba2** | 402M | **1.4189** | 11.6 min | 23.6K |
| E1 (exp2.5) | 399M | 1.5084 | 9.5 min | 28.9K |
| minLSTM | 399M | 1.6190 | 13.0 min | 21.0K |
| E16 s4 | 400M | 1.6249 | 12.0 min | 22.8K |
| minGRU | 398M | 1.6291 | 13.2 min | 20.7K |

## The Key Finding

**Mamba2 learns 0.09 nats better than E1 on identical data.**

This means Mamba2's advantage is NOT just from seeing more tokens due to parallel scan speed. It genuinely learns more efficiently per token.

## What Linear vs Nonlinear Gets Wrong

The naive hypothesis: "Linear recurrence (Mamba2) enables parallel scan, so it's faster, so it sees more data, so it wins."

This is **incomplete**. Consider:
- minLSTM is linear: `h = f ⊙ h + i ⊙ x̃`
- minGRU is linear: `h = (1-z) ⊙ h + z ⊙ x̃`
- Both are worse than E1 (nonlinear)!

If linear was simply better, minLSTM should beat E1. It doesn't.

## What Actually Makes Mamba2 Special

### 1. Selectivity Architecture (Input-Dependent Dynamics)

**Mamba2**:
```
A, B, C, dt = projections(x)  # ALL input-dependent
h = discretize(A, dt) @ h + discretize(B, dt) @ x
y = C @ h
```

**minLSTM**:
```
f, i = sigmoid(W_f @ x), sigmoid(W_i @ x)  # Gates input-dependent
h = f ⊙ h + i ⊙ tanh(W @ x)  # But structure is fixed
```

**E1**:
```
h = tanh(W_h @ h + W_x @ x)  # Structure fixed
y = h * silu(z)  # Output gating
```

Mamba2's selectivity is **richer** - the entire state transition matrix A changes with input, not just scalar gates.

### 2. State Dimension and Structure

| Model | State Dim | Structure |
|-------|-----------|-----------|
| Mamba2 | d_state=128, nheads=~27 | Grouped diagonal + head structure |
| E1 | d_inner = dim × expansion | Dense matrix W_h |
| minLSTM | d_inner = dim | Element-wise (diagonal) |
| E16 | d_state = d_inner × 4 | Diagonal A, dense B/C |

Mamba2's state is structured with **multiple heads** and **grouped operations**, giving it more expressivity per parameter than simple diagonal or dense approaches.

### 3. Initialization (S4D Legacy)

Mamba2 inherits initialization from S4/S4D designed for long-range dependencies:
- HiPPO-inspired A matrix init
- Careful dt (discretization step) initialization
- This matters for learning dynamics

### 4. Discretization (dt Mechanism)

Mamba2 uses continuous-time discretization:
```
A_discrete = exp(dt * A)
B_discrete = (A_discrete - I) / A * B
```

This creates a specific inductive bias for temporal dynamics that simple RNN updates lack.

## Why E1 is Still Interesting

Despite losing on loss, E1 has advantages:

1. **Faster per step** (28.9K vs 23.6K tok/s) - 22% faster!
2. **Simpler** - no parallel scan, no discretization
3. **Depth-scalable** - Mamba2 collapses beyond d=22, E1 scales gracefully
4. **Memory efficient** - No intermediate states for parallel scan

## The Real Question

Is there a way to get Mamba2's per-token efficiency with E1's simplicity?

Candidates to explore:
1. **Richer selectivity in E1** - Make W_h input-dependent like Mamba2's A
2. **Better initialization** - Apply S4D-style init to E1
3. **State structuring** - Multi-head E1 with grouped operations
4. **Hybrid approaches** - Nonlinear core + selective gating wrapper

## BF16 Verification

All models ran in bf16. Mamba2's official implementation uses bf16 throughout. The comparison is fair on precision.

## Implications for Nonlinear RNN Research

The hard truth: **Nonlinearity alone doesn't beat linear SSMs.**

What matters more:
- Selectivity mechanism design
- State structure and capacity
- Initialization for the task
- How gradients flow through the architecture

The path forward isn't "add more nonlinearity" but "design better selectivity and state dynamics."

## Raw Numbers

```
E1:       loss=1.5084, time=567.7s, tok/s=28.9K
E16 s4:   loss=1.6249, time=718.8s, tok/s=22.8K
Mamba2:   loss=1.4189, time=693.6s, tok/s=23.6K
minGRU:   loss=1.6291, time=790.9s, tok/s=20.7K
minLSTM:  loss=1.6190, time=778.8s, tok/s=21.0K
```

All at ~400M params, 1000 steps, 32 batch, 512 chunk, AdamW lr=1e-4.
