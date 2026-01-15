# Information Capacity and Rank Accumulation in Matrix State RNNs

## Executive Summary

Matrix state RNNs use a state S of shape [n_state, n_state] = n^2 parameters, but updates are rank-1 outer products outer(v, k). This analysis examines the fundamental capacity limits and practical implications of this architecture.

**Key findings:**
1. Rank accumulation is controlled by decay: without decay, rank grows linearly until full rank at T = n_state
2. With decay alpha, effective rank saturates at approximately 1/(1-alpha) for long sequences
3. Information bottleneck: 2n write bandwidth per step is insufficient to fully utilize n^2 capacity in bounded time
4. DeltaNet's delta rule provides more efficient rank utilization than simple decay
5. For n_state=64, approximately 32-64 steps needed to "fill" memory, but decay limits utilization

---

## 1. Rank Accumulation Analysis

### 1.1 The Basic Update

Each timestep contributes a rank-1 matrix:

```
S_T = alpha^T * S_0 + sum_{t=1}^{T} alpha^{T-t} * outer(v_t, k_t)
```

This is a sum of T rank-1 matrices with exponentially decaying weights.

### 1.2 Rank Bounds Without Decay (alpha = 1)

Without decay, after T timesteps:

```
S_T = S_0 + sum_{t=1}^{T} outer(v_t, k_t)
```

**Theorem:** rank(S_T) <= min(rank(S_0) + T, n_state)

**Proof:** Each outer(v_t, k_t) has rank 1. The rank of a sum of matrices is at most the sum of ranks. Therefore rank(S_T) <= rank(S_0) + T. Rank cannot exceed n_state (the matrix dimension).

**Corollary:** To reach full rank from S_0 = 0:
- Best case: T = n_state steps (if all v_t, k_t are linearly independent)
- Expected case: ~n_state steps (assuming random v_t, k_t)
- Worst case: never (if updates are collinear)

### 1.3 Rank Dynamics With Decay (alpha < 1)

With decay, the situation is more complex. The state at time T is:

```
S_T = alpha^T * S_0 + sum_{t=1}^{T} alpha^{T-t} * outer(v_t, k_t)
```

**Key insight:** The decay weights form a geometric series: alpha^0, alpha^1, ..., alpha^{T-1}

**Effective rank analysis:** Consider the contribution of past updates. Update from time t has weight alpha^{T-t}. For old updates (large T-t), the contribution becomes negligible.

**Effective memory horizon:** Updates older than tau_eff = 1/(1-alpha) contribute < 1/e of their original weight.

| Decay (alpha) | tau_eff | Effective rank bound |
|---------------|---------|---------------------|
| 0.9 | 10 | ~10 |
| 0.95 | 20 | ~20 |
| 0.99 | 100 | ~100 |
| 0.999 | 1000 | ~1000 |

**Theorem (Effective Rank Saturation):** For alpha < 1 and T >> 1/(1-alpha), the expected rank of S_T approaches a stationary distribution with mean:

```
E[rank(S_T)] ~ min(n_state, 1/(1-alpha))
```

**Proof sketch:** Consider the singular value decomposition perspective. Each outer product contributes magnitude ||v|| * ||k|| to one singular value direction. With decay, these contributions decay exponentially. The steady-state "energy" in each direction depends on the balance between new contributions and decay.

### 1.4 When Does S Reach Full Rank?

**Without decay:** T >= n_state steps (assuming general position)

**With decay:** Full rank (rank = n_state) requires:
```
1/(1-alpha) >= n_state
alpha >= 1 - 1/n_state
```

For n_state = 64:
- Need alpha >= 0.984 for potential full rank
- At alpha = 0.9, max effective rank ~ 10 (only 15% of capacity)
- At alpha = 0.99, max effective rank ~ 64 (can reach full)

**Implication:** Decay fights against capacity utilization. Strong decay (small alpha) creates a "forgetful" memory that never reaches full capacity.

---

## 2. Information Bottleneck Analysis

### 2.1 Bandwidth Calculations

| Component | Parameters per timestep |
|-----------|------------------------|
| Write key (k) | n_state |
| Write value (v) | n_state |
| **Total write** | **2 * n_state** |
| Read query (q) | n_state |
| State matrix (S) | n_state^2 |

### 2.2 The Capacity Utilization Problem

**Question:** Is 2n write bandwidth sufficient to utilize n^2 capacity?

**Analysis:**

To write n^2 independent values, we need at least n^2 / (2n) = n/2 timesteps.

But with decay, information written at timestep t decays as alpha^{T-t}. The "effective information" that survives is:

```
I_eff = sum_{t=1}^{T} 2n * alpha^{T-t} = 2n * (1 - alpha^T)/(1 - alpha)
```

For T -> infinity:
```
I_eff = 2n / (1 - alpha)
```

**Theorem (Information Capacity Bound):** The maximum information that can be stored in steady state is:

```
I_max = 2n / (1 - alpha) bits
```

For this to equal the matrix capacity n^2:
```
2n / (1 - alpha) = n^2
alpha = 1 - 2/n
```

For n_state = 64:
- Need alpha >= 0.969 to potentially fill capacity
- At alpha = 0.9, only ~20% of n^2 capacity is accessible
- At alpha = 0.99, ~150% (full capacity plus some redundancy)

### 2.3 Information Flow Diagram

```
Input x_t
    |
    v
[Projections: W_k, W_v, W_q]  <-- 3n^2 learned params
    |
    +-- k_t (n dims) --+
    |                  |
    +-- v_t (n dims) --+--> outer(v_t, k_t) --> [S] <-- n^2 state
    |                                            |
    +-- q_t (n dims) ----------------------> matmul
                                                |
                                                v
                                          output (n dims)
```

**Bottleneck location:** The outer product creates a rank-1 update. Even with full n^2 state, only 2n-1 degrees of freedom are updated per step.

---

## 3. Comparison to DeltaNet

### 3.1 DeltaNet Update Rule

```
S_t = S_{t-1} + outer(v_t, k_t) - outer(S_{t-1} @ k_t, k_t)
    = S_{t-1} + outer(v_t - S_{t-1} @ k_t, k_t)
```

This is the "delta rule": update the value associated with key k_t to be v_t.

### 3.2 Rank Properties

**DeltaNet preserves or increases rank selectively:**

If S_{t-1} @ k_t = v_t (key already maps to correct value):
- Delta = 0
- No change to S, rank preserved

If S_{t-1} @ k_t != v_t:
- Correction applied
- Rank can increase by 1 (if k_t is new direction)
- Or rank preserved (if k_t is existing direction)

**Key difference from decay:**
- Decay: ALL past information fades uniformly
- DeltaNet: ONLY updates what needs updating

### 3.3 Capacity Utilization

**DeltaNet is more capacity-efficient:**

| Property | Simple Decay | DeltaNet |
|----------|--------------|----------|
| Old info decay | Exponential | Selective |
| Rank growth | Limited by decay | Up to full |
| Interference | Growing | Controlled |
| Retrieval fidelity | Degrades with time | Maintained for stored pairs |

**Proof:** DeltaNet implements exact associative memory for stored k-v pairs. If we write (k_1, v_1), then (k_2, v_2), and query k_1:

- **DeltaNet:** Returns v_1 exactly (if k_1, k_2 orthogonal)
- **Decay:** Returns alpha * v_1 + noise (information lost)

### 3.4 The Gated Delta Rule

Full form from our Lean formalization (GatedDeltaRule.lean):

```
S_t = alpha_t * S_{t-1} * (I - beta_t * outer(k_t, k_t)) + beta_t * outer(v_t, k_t)
```

This combines:
1. Global decay (alpha_t)
2. Selective erasure (I - beta * outer(k, k) erases along k direction)
3. Selective write (outer(v, k))

**Interpretation:** "Forget globally by alpha, but specifically erase what we're about to write, then write the new value."

---

## 4. Retrieval Fidelity and Interference

### 4.1 Classic Associative Memory Theory (Hopfield Networks)

For a matrix storing T patterns via outer products:

```
S = sum_{t=1}^{T} outer(v_t, k_t)
```

When querying with k_s:

```
S @ k_s = sum_{t=1}^{T} v_t * (k_t^T @ k_s)
        = v_s * ||k_s||^2 + sum_{t != s} v_t * (k_t^T @ k_s)
                ^^^^^^^       ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                signal        interference (crosstalk)
```

### 4.2 Signal-to-Interference Ratio

Assuming ||k_t|| = 1 for all t and k_t drawn randomly:

- **Signal:** ||v_s||
- **Interference:** sum of T-1 terms, each ~ O(1/sqrt(n)) magnitude

**SNR ~ sqrt(n) / sqrt(T-1)**

For reliable retrieval (SNR > 1):
```
T < n
```

**Theorem (Hopfield Capacity):** A matrix of size n x n can reliably store O(n) random patterns.

### 4.3 Impact of Decay on Retrieval

With decay factor alpha, patterns older than tau have weight alpha^tau:

**Recent patterns:** High weight, good SNR
**Old patterns:** Low weight, poor SNR

This creates a "recency bias" - the matrix naturally remembers recent information better.

**Effective capacity:** ~1/(1-alpha) patterns with good retrieval, instead of n.

### 4.4 Improving Retrieval: Key Orthogonalization

If keys are made orthogonal (k_i^T @ k_j = 0 for i != j):

```
S @ k_s = v_s * ||k_s||^2 + 0  (no interference!)
```

**Approaches:**
1. **Learned projections:** Train W_k to produce orthogonal keys
2. **Explicit orthogonalization:** Gram-Schmidt on keys
3. **Random high-dimensional:** Random vectors in high dimensions are nearly orthogonal

For random unit vectors in R^n:
```
E[k_i^T @ k_j] = 0
Var[k_i^T @ k_j] = 1/n
```

With n = 64 and T = 10 patterns, interference is ~10/64 ~ 15% of signal.

---

## 5. Practical Implications for n_state = 64

### 5.1 Parameter Counts

| Component | Parameters |
|-----------|------------|
| State matrix S | 64 x 64 = 4096 |
| Write per step (v, k) | 64 + 64 = 128 |
| Read per step (q) | 64 |

### 5.2 Steps to "Fill" Memory

**Without decay:**
- Minimum: 64 steps (one per rank direction)
- Expected: ~64 steps (random vectors span space)

**With decay alpha = 0.9:**
- Effective horizon: 10 steps
- Max effective rank: ~10
- **Only 10/64 = 15% of capacity is utilized**

**With decay alpha = 0.99:**
- Effective horizon: 100 steps
- Max effective rank: ~64 (saturated)
- **Can utilize full capacity**

### 5.3 Decay vs. Capacity Tradeoff

| Decay (alpha) | Eff. Rank | Capacity Used | Retrieval Quality |
|---------------|-----------|---------------|-------------------|
| 0.8 | 5 | 8% | Best for recent |
| 0.9 | 10 | 15% | Good for recent |
| 0.95 | 20 | 31% | Moderate |
| 0.99 | 64+ | 100% | Poor differentiation |
| 1.0 | 64 (full) | 100% | Interference issues |

**Optimal decay depends on task:**
- Short-range dependencies: Higher decay (lower alpha), faster forgetting
- Long-range dependencies: Lower decay (higher alpha), better capacity

### 5.4 Comparison to E1 (Vector State)

| Architecture | State Params | Update Params | Capacity | Throughput |
|--------------|--------------|---------------|----------|------------|
| E1 (vector) | 64 | 64 | 64 | High (simple) |
| Matrix (n=64) | 4096 | 128 | 64x more | 5-6x slower |

From E14 benchmarks:
- E1: 31.1K tok/s
- E14 (Matrix): 5.7K tok/s

**The 64x more capacity comes at 5-6x throughput cost.**

---

## 6. Theoretical Capacity Limits

### 6.1 Shannon Capacity Perspective

For a state matrix S with entries in [-1, 1] (after normalization):

**Naive capacity:** 4096 * log(precision) bits

But the rank constraint limits this. With rank r:
- Actually storing: 2 * n * r - r^2 parameters (for SVD: U, Sigma, V)
- For full rank r = n: n^2 (but only ~2n^2 from parameterized form)

### 6.2 Learning-Theoretic Capacity

What functions can a matrix-state RNN compute?

**Key insight from our SCALING_THEORY.md:** Nonlinearity provides composition depth. But matrix state updates are LINEAR in S:

```
S_t = alpha * S_{t-1} + outer(v_t, k_t)
```

This is LINEAR in S, even though v_t = f(x_t) can be nonlinear in x.

**Implication:** Matrix state alone doesn't add composition depth. The nonlinearity must come from:
1. Input projections (W_k, W_v with nonlinear activation)
2. Output processing (after S @ q)
3. Stacking multiple layers

### 6.3 Connection to Linear Attention

Linear attention is exactly matrix-state RNN with:
- S_t = S_{t-1} + outer(V_t, K_t)  (no decay, alpha = 1)
- Output: S_t @ Q_t

Our analysis explains why linear attention struggles with long contexts:
- Without decay: interference grows with T
- With decay: capacity limited to 1/(1-alpha) items

**This is why Mamba uses selective state spaces instead of pure linear attention.**

---

## 7. Design Recommendations

### 7.1 For Maximum Capacity Utilization

1. **Use high alpha (>= 1 - 1/n_state):** Allows full rank accumulation
2. **Use DeltaNet-style updates:** Selective erasure before write
3. **Orthogonalize keys:** Reduce interference
4. **Match decay to task:** Short context = high decay, long context = low decay

### 7.2 For Maximum Throughput

1. **Keep n_state small:** Quadratic memory cost
2. **Use vector state (E1-style):** 5-6x faster than matrix
3. **If using matrix state:** Consider gradient checkpointing to reduce memory

### 7.3 For Best Tradeoff

Based on our empirical results (E14_MATRIX_STATE_IMPLEMENTATION.md):

**Matrix state is NOT recommended for general language modeling:**
- 5-6x slower than E1
- Higher memory usage
- No significant quality improvement at tested scales

**Matrix state may be useful for:**
- Retrieval-intensive tasks
- Long-range dependency modeling
- Tasks requiring explicit key-value storage

---

## 8. Summary

### Key Equations

**Rank after T steps (no decay):**
```
rank(S_T) = min(rank(S_0) + T, n_state)
```

**Effective rank with decay:**
```
rank_eff ~ min(n_state, 1/(1-alpha))
```

**Information capacity:**
```
I_max = 2n / (1 - alpha) bits
```

**Associative memory capacity:**
```
T_reliable < n_state (for random keys)
```

### Key Insights

1. **Decay limits capacity:** alpha = 0.9 with n = 64 uses only 15% of capacity
2. **Bandwidth is the bottleneck:** 2n per step, need n/2 steps for full rank
3. **DeltaNet is more efficient:** Selective update > uniform decay
4. **Interference grows with stored patterns:** O(sqrt(T/n)) noise ratio
5. **Matrix state trades throughput for capacity:** 5-6x slower, 64x more state

### When to Use Matrix State

| Use Case | Recommendation |
|----------|----------------|
| General LM | Vector state (E1) |
| Retrieval-heavy | Matrix state + DeltaNet |
| Long context | Matrix state + high alpha |
| Maximum speed | Vector state (E1) |

---

## References

- `E14_MATRIX_STATE_IMPLEMENTATION.md` - Empirical benchmarks
- `MatrixStateRNN.lean` - Formal complexity analysis
- `GatedDeltaRule.lean` - DeltaNet formalization
- `SSM_RESIDUAL_CONNECTION.md` - Connection to SSMs
- `SCALING_THEORY.md` - Depth vs width analysis
- Hopfield, J.J. (1982). Neural networks and physical systems with emergent collective computational abilities.
- Schlag et al. (2021). Linear Transformers Are Secretly Fast Weight Programmers.
- Yang et al. (2024). Gated Linear Attention Transformers with Hardware-Efficient Training.

---

*Generated: 2026-01-15*
