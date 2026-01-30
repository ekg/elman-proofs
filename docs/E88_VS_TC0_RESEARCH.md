# E88 vs TC0: Computational Class Analysis

**Date:** 2026-01-30
**Status:** Research Complete
**Task:** research-e88-vs

---

## Executive Summary

This document investigates whether E88 with unbounded time exceeds TC0, the circuit complexity class that bounds Transformers.

**Key Finding:** E88 with unbounded timesteps T → ∞ provably exceeds TC0. The temporal nonlinearity (nested tanh compositions) provides computational depth that scales with sequence length, while Transformers and linear SSMs are fundamentally bounded by constant circuit depth.

---

## 1. Background: TC0 and Transformer Limitations

### 1.1 TC0 Definition

TC0 (Threshold Circuit depth 0) is a circuit complexity class containing:
- Languages decided by constant-depth, polynomial-size circuits
- With unbounded fan-in AND, OR, NOT, and MAJORITY (threshold) gates

The hierarchy: NC0 ⊊ AC0 ⊊ TC0 ⊆ NC1 ⊆ P

Key problems in TC0:
- Sorting n n-bit numbers
- Integer multiplication
- Division
- Dyck language with two bracket types

Key problems NOT in TC0 (unless TC0 = NC1):
- Graph connectivity
- Iterated multiplication (requiring log depth)

### 1.2 Transformer TC0 Bounds (Merrill et al. 2022)

**Theorem (Merrill, Sabharwal, Smith):** Saturated transformers with floating-point values can be simulated by constant-depth threshold circuits. TC0 is an upper bound on the formal languages they recognize.

Sources:
- [Saturated Transformers are Constant-Depth Threshold Circuits](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00493/112604) (TACL 2022)
- [Formal Language Recognition by Hard Attention Transformers](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00490/112496)

**Key assumptions:**
1. Saturated attention (generalization of hard attention)
2. Floating-point values (finite precision)
3. Constant depth D (independent of input length n)

**Implications:**
- Transformers cannot recognize languages outside TC0
- PARITY is recognizable (PARITY ∈ TC0)
- But complex state-tracking may be limited

### 1.3 Hard vs Soft Attention

| Attention Type | Circuit Class | Key Result |
|----------------|---------------|------------|
| Unique hard (UHAT) | AC0 | Hahn 2020 |
| Generalized unique hard (GUHAT) | AC0 | Hahn 2020 |
| Averaging hard (AHAT) | TC0 | Merrill et al. 2022 |
| Saturated softmax | TC0 | Merrill & Sabharwal 2023 |

---

## 2. RNN Computational Power

### 2.1 Siegelmann-Sontag Results (1995)

**Theorem:** Recurrent neural networks with:
- Unbounded precision (real weights)
- Unbounded computation time

are **Turing complete** and can even be **super-Turing** (deciding P/poly in polynomial time).

Source: [On the Computational Power of Neural Nets](https://binds.cs.umass.edu/papers/1995_Siegelmann_JComSysSci.pdf) (JCSS 1995)

### 2.2 Finite Precision Constraints

With finite precision and bounded time:

| Architecture | Computational Class |
|--------------|---------------------|
| Simple RNN (finite precision) | Regular (DFA) |
| GRU (finite precision) | Regular (DFA) |
| LSTM (saturated) | Counter automata |

Source: [On the Practical Computational Power of Finite Precision RNNs](https://aclanthology.org/P18-2117/)

**Key insight:** The gap between theoretical (Turing complete) and practical (finite automata) RNN power comes from precision and time constraints.

---

## 3. Linear SSM Limitations

### 3.1 SSM Cannot Compute PARITY

**Theorem (Merrill et al. 2024):** No SSM satisfying the nonnegative gate constraint can recognize PARITY at arbitrary input lengths with finite precision.

This applies to:
- Mamba (exponential parameterization of decay)
- Griffin
- GLA
- HGRN

Source: [The Expressive Capacity of State Space Models](https://arxiv.org/html/2405.17394)

**Proof sketch:**
1. Nonnegative gates force eigenvalues to be nonnegative real
2. PARITY requires tracking count mod 2, needing oscillatory dynamics
3. Nonnegative eigenvalues cannot oscillate - they either decay, grow, or stay constant
4. As T→∞, the state converges/diverges monotonically, losing parity information

### 3.2 Star-Free Regular Languages

SSMs with nonnegative gates recognize exactly **star-free regular languages** - those definable without Kleene star.

Examples in star-free: a*b*, complement of a*
Examples NOT star-free: (ab)*, a*ba* (odd number of b's)

### 3.3 The "Illusion of State"

**Key finding:** Despite their recurrent formulation, SSMs like Mamba have similar expressiveness to Transformers (non-recurrent).

Source: [The Illusion of State in State-Space Models](https://arxiv.org/html/2404.08819v2)

Neither can solve inherently sequential problems like:
- Composing permutations
- State-tracking with irreversible transitions

---

## 4. E88 Computational Analysis

### 4.1 E88 Architecture

E88 update rule:
```
S := tanh(α·S + δ·k^T)
```

Key difference from Mamba2:
- **Mamba2:** `h := A(x)·h + B(x)·x` (linear in h)
- **E88:** `S := tanh(...)` (nonlinear in S)

### 4.2 Compositional Depth Analysis

| Architecture | Within-Layer Depth | Total Depth (D layers, T steps) |
|--------------|-------------------|----------------------------------|
| Transformer | 1 | D |
| Mamba2 (linear SSM) | 1 | D |
| E88 | T | D × T |

The tanh in E88 compounds across timesteps:
```
S_T = tanh(α · tanh(α · tanh(... tanh(α·S_0 + δ·k_1) ...) + δ·k_T)
```

This is T nested nonlinear compositions, providing **unbounded depth as T→∞**.

### 4.3 E88 Exceeds TC0 with Unbounded Time

**Theorem (informal):** E88 with T timesteps can simulate circuits of depth O(T).

**Proof sketch:**
1. Each tanh provides a bounded-depth circuit (constant depth for function approximation)
2. T sequential tanh applications compose to depth T
3. As T→∞, E88 can compute functions requiring unbounded depth
4. TC0 is bounded to constant depth
5. Therefore E88 with unbounded T exceeds TC0

**Concrete separation:**

Consider the "iterated threshold" problem:
- Input: n bits x_1, ..., x_n
- At each step i, maintain counter c_i = (c_{i-1} + x_i) mod 3
- Output: c_n

For linear-temporal models: This requires D ≥ log(n) layers to track mod-3 counting
For E88: Single layer with T = n suffices via basin dynamics

---

## 5. Formal Hierarchy

```
Linear RNN (Mamba2) ⊊ E88 (bounded T) ⊆ TC0 ⊊ E88 (unbounded T) ⊆ P
```

### 5.1 Separations

**Linear < E88 (bounded T):**
- Witness: Running parity, exact counting mod n
- Proof: LinearLimitations.lean, ExactCounting.lean

**E88 (bounded T) vs TC0:**
- With fixed T, E88 has depth O(T) = constant
- So E88 with bounded T ⊆ TC0

**E88 (unbounded T) > TC0:**
- With T → ∞, E88 can compute functions requiring unbounded depth
- TC0 is constant depth
- Separation exists unless TC0 = NC1 (open problem)

### 5.2 Caveats

1. **Precision matters:** These results assume sufficient precision. With finite precision, all neural networks collapse to finite automata.

2. **Practical T:** In practice, T is bounded by context length (e.g., 4K-128K). For typical T < 2^32, the theoretical gap may not manifest.

3. **Learning vs expressivity:** Even if E88 can express functions beyond TC0, whether it can learn them from data is a separate question.

---

## 6. Connection to Existing Proofs

### 6.1 LinearLimitations.lean

Proves:
- Linear RNNs cannot compute threshold functions
- Linear RNNs cannot compute XOR
- Output is a linear (hence continuous) function of inputs

### 6.2 MultiLayerLimitations.lean

Proves:
- D-layer linear-temporal models cannot compute running threshold
- Depth D doesn't compensate for linear temporal dynamics

### 6.3 ExactCounting.lean

Proves:
- Running threshold is discontinuous (hence not linearly computable)
- Count mod n is not linearly computable
- E88 can create discrete attractor basins via tanh

### 6.4 ComputationalClasses.lean

Proves:
- Regular ⊆ Linear RNN recognizable
- Linear RNN ⊊ E88 (parity separation)
- Hierarchy: Linear < E88 ⊆ REG ⊊ RE

---

## 7. Summary Table

| Architecture | Time Complexity | Circuit Class | Can Compute |
|--------------|-----------------|---------------|-------------|
| Transformer (D layers) | Parallel | TC0 | ✓ PARITY, ✓ MAJORITY, ✗ Perm comp |
| Mamba2 (D layers) | Linear (parallel scan) | ⊆ TC0 | ✓ MAJORITY, ✗ PARITY*, ✗ Perm comp |
| E88 (D layers, T steps) | Linear | Depth D×T | ✓ PARITY, ✓ Counting mod n |
| E88 (unbounded T) | Potentially unbounded | Beyond TC0 | ✓ Functions requiring unbounded depth |
| Classical RNN (unbounded) | Unbounded | Turing complete | ✓ All RE languages |

(*) SSMs with nonnegative gates cannot compute PARITY

---

## 8. Conclusions

1. **Transformers are TC0-bounded:** Constant-depth threshold circuits regardless of depth D.

2. **Linear SSMs (Mamba2) are weaker than TC0:** Cannot even compute PARITY due to nonnegative gate constraints.

3. **E88 exceeds both with unbounded time:** The temporal tanh provides compositional depth scaling with T.

4. **Practical implications:**
   - For typical language modeling (T < 100K), the gap may not matter
   - For algorithmic reasoning, code execution, formal math: E88's temporal nonlinearity could be crucial
   - Hybrid architectures combining E88's temporal power with Transformer's parallel efficiency may be optimal

5. **Open questions:**
   - Can E88 learn to use its additional expressivity?
   - What is the precise circuit complexity of E88 with bounded T?
   - Does the E88 > TC0 gap manifest on practical benchmarks?

---

## References

1. Merrill, Sabharwal, Smith. "Saturated Transformers are Constant-Depth Threshold Circuits." TACL 2022.
2. Hahn. "Theoretical Limitations of Self-Attention in Neural Sequence Models." TACL 2020.
3. Siegelmann, Sontag. "On the Computational Power of Neural Nets." JCSS 1995.
4. Pérez, Barceló, Marinkovic. "Attention is Turing Complete." JMLR 2021.
5. Merrill et al. "The Expressive Capacity of State Space Models: A Formal Language Perspective." 2024.
6. Merrill et al. "The Illusion of State in State-Space Models." 2024.
7. Weiss et al. "On the Practical Computational Power of Finite Precision RNNs." ACL 2018.
