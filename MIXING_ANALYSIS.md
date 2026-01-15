# Mixing Time and Channel Interaction in Matrix State RNNs

## Executive Summary

Matrix state RNNs (E70-E73) use a state tensor S of shape [n_state, n_state] instead of a vector h of dimension d. This analysis examines how information flows through this matrix structure and answers the critical question: **Is there any cross-row or cross-column mixing, and does it matter?**

**Key Finding**: Current matrix state designs have **zero mixing between rows and zero mixing between columns**. The "mixing diameter" is infinite - S[0,0] can NEVER influence S[n-1,n-1] through internal dynamics. However, this may be a feature rather than a bug: the outer product structure provides a different form of expressivity (associative memory) that compensates for lack of mixing.

---

## Part 1: Operations Analysis

### 1.1 Read Operation: `out = S @ q`

```
S @ q = [n_state, n_state] @ [n_state] = [n_state]
out[i] = sum_j S[i,j] * q[j]
```

**Information flow**:
- Row i's output depends on all columns j within that row
- **No row-to-row interaction**: Row 0 and row 1 never mix
- The query q weights columns uniformly across all rows

**Gradient flow**:
```
d_out[i]/d_S[i,j] = q[j]
d_out[i]/d_S[k,j] = 0  for k != i
```

Each row receives gradients independently. Training can adjust rows independently based on what each row should output.

### 1.2 Write Operation: `outer(v, k)` = v[:, None] * k[None, :]

```
outer(v, k)[i,j] = v[i] * k[j]
```

**Information flow**:
- Each cell S[i,j] receives v[i] * k[j]
- The update is **rank-1**: only one "mode" of information per timestep
- **No mixing**: Cell (i,j) is updated independently of cell (i',j')

**Gradient flow**:
```
d_outer[i,j]/d_v[i] = k[j]
d_outer[i,j]/d_k[j] = v[i]
d_outer[i,j]/d_v[i'] = 0  for i' != i
d_outer[i,j]/d_k[j'] = 0  for j' != j
```

The outer product creates a separable update pattern. Each row is scaled by its v[i], each column by its k[j], but there's no cross-term.

### 1.3 Decay/Gate: `alpha * S`

```
S_new[i,j] = alpha[i,j] * S[i,j]
```

**Information flow**:
- Each cell decays independently
- Even if alpha is position-dependent alpha[i,j], there's no mixing
- This is element-wise: S[i,j] only affects S_new[i,j]

**Note on E70-E73 variants**:
- E70: Uses `decay[:,:,None] * H` where decay broadcasts over the state dimension
- E71-E73: Similar element-wise gating patterns

### 1.4 Tanh: `tanh(S)`

```
S_new[i,j] = tanh(S[i,j])
```

**Information flow**:
- Pure element-wise nonlinearity
- **Zero mixing**: S[i,j] affects only S_new[i,j]
- Provides boundedness but no spatial structure

---

## Part 2: Mixing Diameter Analysis

### 2.1 Definition

**Mixing diameter**: The minimum number of timesteps T such that information at S[i,j] can influence S[k,l] for any (i,j) and (k,l).

### 2.2 Computation for Current E70-E73

Let's trace information flow through the operations:

**Timestep t=0**:
- S[0,0] has some value x

**Timestep t=1**:
```
S_new = decay * S + outer(v, k)
S_new[i,j] = decay[i,j] * S[i,j] + v[i] * k[j]
```

- S_new[0,0] depends on S[0,0]
- S_new[n-1,n-1] depends on S[n-1,n-1]
- **No cross-cell dependency!**

**Timestep t=T (any T)**:

By induction, if operations at each step don't mix cells, then after T steps:
- S_T[i,j] depends only on S_0[i,j] and inputs x_1, ..., x_T
- S_T[k,l] is **completely independent** of S_0[i,j] for (i,j) != (k,l)

**Conclusion: Mixing diameter = infinity**

S[0,0] can NEVER influence S[n-1,n-1] through internal state dynamics alone.

### 2.3 Indirect Mixing via External Path

There IS an indirect mixing pathway through the network:

```
S[i,j] --> out[i] --> external network --> x_t --> v, k, q --> S[k,l]
```

This requires:
1. Reading from row i (gets out[i])
2. Processing through external layers (MLP, layer norm, etc.)
3. Re-projecting to v, k, q at the next timestep
4. Writing affects all cells

**But this is NOT internal mixing** - it relies on the full model architecture, not the matrix state dynamics.

---

## Part 3: Comparison to Vector State RNNs

### 3.1 Vector State Mixing: W @ h

In classic Elman/GRU with h in R^d:
```
h_new = tanh(W_h @ h + W_x @ x + b)
```

The matrix multiply W_h @ h provides **full mixing**:
```
h_new[i] = sum_j W_h[i,j] * h[j]
```

- **Every dimension mixes**: h[0] influences h_new[n-1] through W_h[n-1, 0]
- **Mixing diameter = 1**: Any h[i] can affect any h_new[j] in ONE step
- The weight matrix W_h has d^2 parameters precisely to enable this mixing

### 3.2 The Parameter Allocation Difference

| Architecture | State Size | Mixing Params | Per-Cell State |
|-------------|------------|---------------|----------------|
| Vector (E1) | d | d^2 (W_h) | 1 param/cell mixing |
| Matrix (E70) | n^2 | 0 | No mixing |

The matrix state trades mixing parameters for state capacity:
- E1: d state cells, d^2 mixing parameters
- E70: n^2 state cells, 0 mixing parameters

### 3.3 What This Means

**Vector state (E1)**:
- Each timestep, the entire state gets "scrambled" by W_h
- Information propagates everywhere in 1 step
- State is a compressed representation that integrates all history

**Matrix state (E70-E73)**:
- Each cell is an independent "slot" that accumulates outer products
- No internal scrambling - cells maintain independence
- State is an associative memory that stores key-value pairs

---

## Part 4: What Operations WOULD Provide Mixing?

### 4.1 Matrix-Matrix Multiply: S @ S or S^T @ S

```
(S @ S)[i,j] = sum_k S[i,k] * S[k,j]
```

This WOULD provide mixing:
- (S @ S)[0, n-1] depends on S[0,k] for all k AND S[k, n-1] for all k
- Full matrix interaction in one operation
- **Mixing diameter would be 1**

**Cost**: O(n^3) per timestep, which is prohibitive for large n

### 4.2 Transpose Operations: S + S^T

```
(S + S^T)[i,j] = S[i,j] + S[j,i]
```

This provides **row-column interaction**:
- Cell (i,j) depends on both original (i,j) and (j,i)
- Creates symmetry or anti-symmetry patterns

**Cost**: O(n^2), same as current operations

### 4.3 Attention Over Positions: softmax(S @ S^T) @ S

This is essentially self-attention where S is the [n, n] "sequence":
```
A = softmax(S @ S^T / sqrt(n))  # [n, n] attention weights
S_new = A @ S                    # [n, n] output
```

**Mixing**: Each row attends to all other rows, full row mixing in 1 step

**Cost**: O(n^3), prohibitive

### 4.4 Depthwise Convolution Over Rows/Columns

```
S_new[i,:] = conv1d(S[i,:], kernel)  # Mix within each row
S_new[:,j] = conv1d(S[:,j], kernel)  # Mix within each column
```

**Mixing**: Local mixing within rows/columns, not full mixing
**Cost**: O(n^2 * k) where k is kernel size

### 4.5 Row/Column Permutation or Shuffle

```
S_new[i,j] = S[pi(i), sigma(j)]
```

This could provide structured mixing via learned permutations, but it's not differentiable in standard form.

---

## Part 5: Is Lack of Mixing a Problem?

### 5.1 The Associative Memory Perspective

Matrix state with outer product updates implements **linear associative memory**:

```
S = sum_t v_t @ k_t^T  (accumulated outer products)
out = S @ q           (retrieval via query)
```

This is mathematically equivalent to linear attention's key-value memory:
```
S_t = S_{t-1} + k_t^T @ v_t
o_t = q_t @ S_t
```

**Key insight**: Linear attention (RetNet, RWKV, Mamba-style) is highly successful WITHOUT internal state mixing. The matrix accumulates associations; mixing would corrupt the stored patterns.

### 5.2 When Mixing Might Help

Mixing could help when:
1. **Feature interaction**: If detecting patterns requires combining information from different "slots"
2. **Compression**: If the state needs to consolidate information rather than accumulate it
3. **Non-linear history encoding**: If simple outer products can't capture required patterns

### 5.3 When Mixing Hurts

Mixing could hurt when:
1. **Interference**: Mixing stored associations corrupts retrieval
2. **Gradient flow**: More operations = more gradient path complexity
3. **Computational cost**: Mixing operations are expensive

### 5.4 Evidence from Experiments

From E14/E20/E21 experiments:
- E14 (matrix state): 5.7K tok/s, 17.4GB memory
- E20 (matrix state + heads): OOM
- E21 (MIMO, some mixing structure): 10K tok/s but poor loss

The matrix state approaches are **memory/compute bound**, not expressivity bound. Adding mixing would make this worse, not better.

---

## Part 6: Computational Graph Analysis

### 6.1 Information Flow Over T Timesteps

```
t=0:  S_0[i,j] initialized
t=1:  S_1[i,j] = decay * S_0[i,j] + v_1[i] * k_1[j]
t=2:  S_2[i,j] = decay * S_1[i,j] + v_2[i] * k_2[j]
...
t=T:  S_T[i,j] = decay^T * S_0[i,j] + sum_{t=1}^{T} decay^{T-t} * v_t[i] * k_t[j]
```

**Observation**: S_T[i,j] is a **decayed sum of rank-1 updates**, all at position (i,j). It never sees information from other positions.

### 6.2 Gradient Flow Back Through Time

For loss L depending on final output:
```
dL/dS_0[i,j] = dL/dS_T[i,j] * dS_T/dS_0[i,j]
            = dL/dS_T[i,j] * decay^T
```

- Gradient at each cell flows independently
- No gradient mixing between cells
- This is actually GOOD for optimization: each cell has clear gradient signal

### 6.3 Comparison to Vector State Gradient

For vector state with W_h @ h:
```
dL/dh_0 = dL/dh_T * W_h^T * ... * W_h^T  (T times)
        = dL/dh_T * (W_h^T)^T
```

- Eigenvalues of W_h determine gradient behavior
- Vanishing/exploding gradients if eigenvalues != 1
- The mixing that helps expressivity hurts gradient flow

---

## Part 7: Theoretical Summary

### 7.1 Key Properties of Current Design

| Property | Matrix State (E70-E73) | Vector State (E1) |
|----------|----------------------|-------------------|
| Mixing diameter | Infinity | 1 |
| Gradient coupling | None (independent cells) | Full (W_h couples all) |
| State capacity | n^2 cells | d cells |
| Memory model | Associative (linear attention) | Holistic (compressed) |
| Computational model | Key-value storage | Feature transformation |

### 7.2 The Fundamental Tradeoff

**Vector state**: Uses O(d^2) parameters for mixing (W_h), providing feature transformation at each step. State is a compressed representation.

**Matrix state**: Uses O(d^2) state capacity instead of mixing parameters. State is an explicit key-value store. No internal transformation.

### 7.3 Why This Design Makes Sense

The matrix state design is inspired by **linear attention mechanisms** (RetNet, RWKV, Mamba):

1. **Linear attention accumulates, doesn't mix**: S = sum(k^T @ v) stores associations
2. **Query retrieval is linear**: out = S @ q retrieves by similarity
3. **Success of this pattern**: RWKV-14B, Mamba-2, Griffin all use this without internal mixing

The matrix state RNN extends this to a richer structure with gating/nonlinearity, but preserves the core accumulation principle.

---

## Part 8: Recommendations

### 8.1 Don't Add Mixing

The current design without mixing is **intentional and principled**:
- Matches successful linear attention architectures
- Preserves gradient flow (no W_h eigenvalue issues)
- Keeps operations O(n^2), not O(n^3)

### 8.2 If Mixing Is Needed, Consider Lightweight Options

If experiments show mixing would help:
1. **Row/column attention with linear kernel**: O(n^2) instead of O(n^3)
2. **Sparse mixing**: Only mix subset of cells
3. **Periodic mixing**: Apply S @ S every K steps, not every step

### 8.3 Alternative: Multiple Heads

Instead of mixing within one large matrix, use **multiple smaller matrices** (heads):
```
S^h for h = 1, ..., H
Each S^h is [n/H, n/H]
Concatenate outputs across heads
```

This provides:
- Cross-head mixing via output projection
- Smaller per-head matrices (faster operations)
- Head specialization (different decay rates, patterns)

This is exactly what E20 attempted (before OOM issues).

### 8.4 Focus on Other Bottlenecks

The real issues with E70-E73 are:
1. **Memory**: Storing S for all timesteps for backprop
2. **Throughput**: Sequential updates can't parallelize
3. **Parameter efficiency**: May need more training to utilize capacity

Mixing is not the bottleneck. Address memory/throughput first.

---

## Conclusion

**Matrix state RNNs (E70-E73) have zero internal mixing** - the mixing diameter is infinite. Information at S[i,j] can never reach S[k,l] through internal dynamics alone.

**This is by design**, not a flaw:
- Matches linear attention architectures (RWKV, RetNet, Mamba)
- Preserves clean gradient flow
- Keeps operations efficient (O(n^2))

**The outer product structure provides associative memory capacity** as compensation for lack of mixing. Each cell independently accumulates key-value associations; mixing would interfere with this storage mechanism.

**Recommendation**: Don't add mixing operations. The architecture's expressivity comes from its key-value storage capacity, not from internal state transformation. If more expressivity is needed, consider:
1. More layers (more transformation opportunity)
2. Multiple heads (cross-head mixing via projection)
3. Better gating mechanisms (selective attention to stored associations)

---

*Analysis completed: 2026-01-15*
*For the elman-proofs project*
