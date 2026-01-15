# Jacobian Spectrum Analysis for Matrix State Elman Models E70-E73

## Overview

This document analyzes the Jacobian $\frac{\partial S_t}{\partial S_{t-1}}$ for matrix state recurrences E70-E73. We treat the matrix state $S \in \mathbb{R}^{n \times m}$ as a flattened vector $\text{vec}(S) \in \mathbb{R}^{nm}$ to compute the Jacobian as a $(nm) \times (nm)$ matrix.

**Key notation**:
- $S \in \mathbb{R}^{n \times m}$: matrix state
- $\text{vec}(S)$: column-major flattening of $S$ into $\mathbb{R}^{nm}$
- $\otimes$: Kronecker product
- $\odot$: Hadamard (element-wise) product
- $\rho(J)$: spectral radius of Jacobian $J$

**Key identity** (for Jacobians of matrix expressions):
$$\frac{\partial \text{vec}(AXB)}{\partial \text{vec}(X)} = B^\top \otimes A$$

---

## E70: Linear Accumulation + tanh

### Update Rule
```python
S_t = tanh(decay * S_{t-1} + outer(v, k))
```
where $\text{outer}(v, k) = v k^\top \in \mathbb{R}^{n \times m}$.

### Jacobian Derivation

Let $Z_t = \text{decay} \cdot S_{t-1} + v k^\top$ (pre-activation).

The element-wise tanh gives:
$$S_t = \tanh(Z_t)$$

Using the chain rule:
$$\frac{\partial \text{vec}(S_t)}{\partial \text{vec}(S_{t-1})} = \frac{\partial \text{vec}(\tanh(Z_t))}{\partial \text{vec}(Z_t)} \cdot \frac{\partial \text{vec}(Z_t)}{\partial \text{vec}(S_{t-1})}$$

**Term 1**: $\frac{\partial \text{vec}(\tanh(Z_t))}{\partial \text{vec}(Z_t)}$

Since tanh is applied element-wise:
$$\frac{\partial \text{vec}(\tanh(Z_t))}{\partial \text{vec}(Z_t)} = \text{diag}(1 - \tanh^2(Z_t))$$

where $\text{diag}(\cdot)$ creates a diagonal matrix from the flattened $(1 - \tanh^2(Z_t))$.

**Term 2**: $\frac{\partial \text{vec}(Z_t)}{\partial \text{vec}(S_{t-1})}$

Since $Z_t = \text{decay} \cdot S_{t-1} + vk^\top$ and $vk^\top$ doesn't depend on $S_{t-1}$:
$$\frac{\partial \text{vec}(Z_t)}{\partial \text{vec}(S_{t-1})} = \text{decay} \cdot I_{nm}$$

**Full Jacobian**:
$$\boxed{J_{\text{E70}} = \text{decay} \cdot \text{diag}(1 - \tanh^2(Z_t))}$$

This is a **diagonal matrix** with entries $\text{decay} \cdot (1 - \tanh^2(z_{ij}))$ for each element.

### Spectral Radius

Since $J_{\text{E70}}$ is diagonal:
$$\rho(J_{\text{E70}}) = \text{decay} \cdot \max_{i,j} (1 - \tanh^2(z_{ij}))$$

**Analysis**:
- $1 - \tanh^2(z) \in (0, 1]$ with maximum 1 at $z = 0$
- When $z_{ij} \to 0$: $\rho(J) \to \text{decay}$
- When $|z_{ij}| \to \infty$: $\rho(J) \to 0$ (saturation)

**Conditions**:
- $\rho < 1$ always (since decay $< 1$ and $(1 - \tanh^2) \leq 1$)
- $\rho \approx \text{decay}$ near origin
- $\rho \ll 1$ in saturation regime

### Gradient Flow Implications

Over $T$ timesteps, the gradient magnitude scales as:
$$\left\|\frac{\partial S_T}{\partial S_0}\right\| \leq \prod_{t=1}^{T} \rho(J_t) \leq \text{decay}^T$$

**Key insight**: E70 has **guaranteed gradient vanishing** because:
1. The decay factor ensures $\rho < 1$ always
2. Tanh saturation further reduces the spectral radius
3. Gradients decay at least as fast as $\text{decay}^T$

This is both a **stability guarantee** and a **gradient bottleneck**.

---

## E71: State-Dependent Gate (Multiplicative)

### Update Rule
```python
retrieved = S @ k                              # [n]
alpha = sigmoid(W_alpha @ x + d_alpha * retrieved)  # [n]
S_t = alpha[:, None] * S_{t-1} + (1 - alpha)[:, None] * outer(v, k)
```

Here $\alpha \in \mathbb{R}^n$ is broadcast across columns.

### Jacobian Derivation

This is more complex because $\alpha$ depends on $S$.

Let:
- $r = S k$ (retrieved, $\in \mathbb{R}^n$)
- $a = \sigma(W_\alpha x + d_\alpha r)$ where $\sigma$ is sigmoid

The update in matrix form:
$$S_t = \text{diag}(\alpha) S_{t-1} + \text{diag}(1-\alpha) v k^\top$$

**Dependency graph**:
$$S_{t-1} \to r \to \alpha \to S_t$$
$$S_{t-1} \to S_t \quad \text{(direct term)}$$

**Computing $\frac{\partial \alpha}{\partial S_{t-1}}$**:

First, $\frac{\partial r}{\partial \text{vec}(S_{t-1})}$:
$$r_i = \sum_j S_{ij} k_j = (I_n \otimes k^\top) \text{vec}(S)$$

So:
$$\frac{\partial r}{\partial \text{vec}(S)} = I_n \otimes k^\top \in \mathbb{R}^{n \times nm}$$

Next, $\frac{\partial \alpha}{\partial r}$:
$$\frac{\partial \alpha_i}{\partial r_j} = d_\alpha \cdot \sigma'(\cdot) \cdot \delta_{ij} = d_\alpha \cdot \alpha_i(1-\alpha_i) \cdot \delta_{ij}$$

So:
$$\frac{\partial \alpha}{\partial r} = d_\alpha \cdot \text{diag}(\alpha \odot (1-\alpha))$$

Chain rule:
$$\frac{\partial \alpha}{\partial \text{vec}(S)} = d_\alpha \cdot \text{diag}(\alpha \odot (1-\alpha)) \cdot (I_n \otimes k^\top)$$

**Computing the full Jacobian**:

$$\text{vec}(S_t) = (\alpha \otimes I_m) \odot \text{vec}(S_{t-1}) + \text{vec}((1-\alpha)[:,\text{None}] \cdot vk^\top)$$

More carefully in column-major:
$$\text{vec}(S_t) = (I_m \otimes \text{diag}(\alpha)) \text{vec}(S_{t-1}) + \text{terms with } (1-\alpha)$$

The Jacobian has two parts:

**Part 1** (direct term):
$$\frac{\partial}{\partial \text{vec}(S_{t-1})} [(I_m \otimes \text{diag}(\alpha)) \text{vec}(S_{t-1})] = I_m \otimes \text{diag}(\alpha) + \text{correction from } \alpha(S)$$

**Part 2** (indirect through $\alpha$):
This requires computing how changes in $S_{t-1}$ affect $\alpha$, then how $\alpha$ affects both terms.

**Full Jacobian** (after detailed calculation):
$$\boxed{J_{\text{E71}} = I_m \otimes \text{diag}(\alpha) + \underbrace{[\text{vec}(S_{t-1}) - \text{vec}(vk^\top)] \otimes \left(d_\alpha \cdot \alpha \odot (1-\alpha)\right) \cdot (I_n \otimes k^\top)}_{\text{state-dependent coupling}}}$$

The first term is block-diagonal. The second term creates **cross-column coupling** through the retrieval mechanism.

### Spectral Radius Analysis

The Jacobian is **not diagonal** due to the state-dependent gate.

**Bounding the spectral radius**:

1. **First term**: $\|I_m \otimes \text{diag}(\alpha)\| = \max_i \alpha_i \leq 1$

2. **Second term**: The rank-1 correction has spectral norm:
   $$\|[\text{vec}(S - vk^\top)] \cdot [\text{retrieval gradient}]^\top\| \leq \|S - vk^\top\|_F \cdot |d_\alpha| \cdot \max(\alpha \odot (1-\alpha)) \cdot \|k\|$$

Since $\alpha_i(1-\alpha_i) \leq 1/4$:
$$\text{correction norm} \leq \frac{|d_\alpha|}{4} \cdot \|k\| \cdot \|S - vk^\top\|_F$$

**Conditions for $\rho < 1$**:
- When $\alpha \approx 0$: Jacobian $\approx 0$, very stable
- When $\alpha \approx 1$: Jacobian $\approx I$, marginal stability
- The correction term can push $\rho > 1$ if:
  - $|d_\alpha|$ is large
  - $\|k\|$ is large
  - $\|S - vk^\top\|_F$ is large

**Critical observation**: Unlike E70, **E71 can have $\rho > 1$** when the state-dependent feedback is strong enough.

### Gradient Flow Implications

1. **Potential for gradient preservation**: When $\alpha \approx 1$, gradients propagate almost unchanged
2. **Potential for gradient explosion**: The state-dependent term can amplify gradients
3. **Input-dependent dynamics**: The spectral radius varies with the input sequence

This is similar to LSTM/GRU gradient dynamics where the gate allows selective gradient flow.

---

## E72: State-Dependent Value Gate

### Update Rule
```python
retrieved = S @ k                    # [n]
g = sigmoid(d_g * retrieved)         # [n]
v_gated = v * g                      # [n]
S_t = alpha * S_{t-1} + (1 - alpha) * outer(v_gated, k)
```

Here $\alpha$ is **not state-dependent** (comes from input only).

### Jacobian Derivation

The state-dependence enters through $g$, which modulates the value $v$.

**Dependency graph**:
$$S_{t-1} \to r \to g \to v_{\text{gated}} \to S_t$$

But note: $v_{\text{gated}} k^\top$ appears with coefficient $(1-\alpha)$, not in the recurrent term $\alpha S_{t-1}$.

**Key insight**: The direct recurrence term is just $\alpha S_{t-1}$, giving:
$$\frac{\partial (\alpha S_{t-1})}{\partial \text{vec}(S_{t-1})} = \alpha \cdot I_{nm}$$

The $(1-\alpha) v_{\text{gated}} k^\top$ term contributes:
$$\frac{\partial \text{vec}((1-\alpha) v_{\text{gated}} k^\top)}{\partial \text{vec}(S_{t-1})}$$

Computing this:
- $\frac{\partial g}{\partial r} = d_g \cdot g \odot (1-g)$ (diagonal)
- $\frac{\partial r}{\partial \text{vec}(S)} = I_n \otimes k^\top$
- $\frac{\partial v_{\text{gated}}}{\partial g} = \text{diag}(v)$
- $\frac{\partial \text{vec}(v_{\text{gated}} k^\top)}{\partial v_{\text{gated}}} = k \otimes I_n$

Chain rule:
$$\frac{\partial \text{vec}(v_{\text{gated}} k^\top)}{\partial \text{vec}(S)} = (1-\alpha) \cdot (k \otimes I_n) \cdot \text{diag}(v) \cdot d_g \cdot \text{diag}(g \odot (1-g)) \cdot (I_n \otimes k^\top)$$

**Full Jacobian**:
$$\boxed{J_{\text{E72}} = \alpha \cdot I_{nm} + (1-\alpha) \cdot d_g \cdot (k \otimes \text{diag}(v \odot g \odot (1-g))) \cdot (I_n \otimes k^\top)}$$

Simplifying the second term:
$$(k \otimes D_v) (I_n \otimes k^\top) = k k^\top \otimes D_v$$

where $D_v = \text{diag}(v \odot g \odot (1-g))$.

So:
$$\boxed{J_{\text{E72}} = \alpha \cdot I_{nm} + (1-\alpha) \cdot d_g \cdot (k k^\top) \otimes D_v}$$

### Spectral Radius Analysis

The structure is: **scaled identity + rank-1 correction** (in the Kronecker sense).

**Eigenvalue analysis**:

The matrix $(kk^\top) \otimes D_v$ has:
- Eigenvalues: $\|k\|^2 \cdot \lambda_i(D_v)$ for the non-zero eigenvalue of $kk^\top$
- Zero eigenvalues for the null space of $kk^\top$

So the eigenvalues of $J_{\text{E72}}$ are:
- $\alpha$ with multiplicity $(m-1) \cdot n$ (from null space of $kk^\top$)
- $\alpha + (1-\alpha) \cdot d_g \cdot \|k\|^2 \cdot (v_i g_i (1-g_i))$ for $i = 1, \ldots, n$

**Spectral radius**:
$$\rho(J_{\text{E72}}) = \max\left(\alpha, \max_i \left|\alpha + (1-\alpha) d_g \|k\|^2 v_i g_i(1-g_i)\right|\right)$$

**Conditions**:
- $\rho < 1$: Guaranteed when $\alpha < 1$ and the correction term is small
- $\rho = 1$: When $\alpha = 1$ (pure memory, no update)
- $\rho > 1$: Possible when $(1-\alpha) d_g \|k\|^2 \max(|v_i| g_i(1-g_i))$ is large enough

Since $g_i(1-g_i) \leq 1/4$:
$$|\text{correction}| \leq \frac{(1-\alpha) |d_g| \|k\|^2 \|v\|_\infty}{4}$$

### Gradient Flow Implications

1. **Mostly diagonal**: The Jacobian is close to $\alpha I$ when the correction is small
2. **Controlled explosion risk**: Bounded by $g(1-g) \leq 1/4$
3. **Value-dependent**: Large values $|v_i|$ increase the spectral radius

E72 is **more stable than E71** because:
- The state-dependent term only affects the value, not the gate
- The gate modulation is multiplicative (bounded by sigmoid derivative)

---

## E73: State Inside tanh

### Update Rule
```python
z = sigmoid(W_z @ x + b_z)           # [m] (column modulation)
S_mod = S * z[None, :]               # Element-wise column scaling
S_t = tanh(S_mod + outer(v, k))
```

### Jacobian Derivation

Note: $z$ depends only on $x$, not on $S$. So the state-dependence is simpler.

Let $Z = S_{t-1} \odot \mathbf{1}_n z^\top + v k^\top$ (pre-activation).

The element-wise scaling gives:
$$(S \cdot z[None,:])_{ij} = S_{ij} \cdot z_j$$

In vectorized form with column-major ordering:
$$\text{vec}(S \cdot z[None,:]) = (z \otimes I_n) \odot \text{vec}(S) = \text{diag}(\text{vec}(\mathbf{1}_n z^\top)) \text{vec}(S)$$

More simply:
$$\frac{\partial \text{vec}(S \cdot z[None,:])}{\partial \text{vec}(S)} = \text{diag}(\text{tile}(z, n))$$

where $\text{tile}(z, n)$ repeats $z$ for each of the $n$ rows.

**Full Jacobian**:
$$\boxed{J_{\text{E73}} = \text{diag}(1 - \tanh^2(Z)) \cdot \text{diag}(\text{tile}(z, n))}$$

This is a **diagonal matrix** with entries $(1 - \tanh^2(Z_{ij})) \cdot z_j$.

### Spectral Radius

$$\rho(J_{\text{E73}}) = \max_{i,j} \left[(1 - \tanh^2(Z_{ij})) \cdot z_j\right]$$

Since $z_j = \sigma(\cdot) \in (0, 1)$ and $(1 - \tanh^2) \in (0, 1]$:

$$\rho(J_{\text{E73}}) < 1 \quad \text{always}$$

**More precisely**:
- Maximum possible: approaches 1 when $z_j \to 1$ and $Z_{ij} \to 0$
- Minimum: approaches 0 when either $z_j \to 0$ or $|Z_{ij}| \to \infty$

### Gradient Flow Implications

Like E70, E73 has **guaranteed gradient vanishing**:
$$\left\|\frac{\partial S_T}{\partial S_0}\right\| \leq \prod_{t=1}^{T} \max_j(z_j^{(t)}) \cdot \max_{i,j}(1-\tanh^2(Z_{ij}^{(t)}))$$

However, E73 has an additional control knob through $z$:
- The sigmoid gate $z$ can selectively **forget** certain columns
- When $z_j \approx 0$, column $j$ contributes negligibly to future states

This is similar to LSTM/GRU forget gate behavior but applied column-wise.

---

## Comparison with Linear Recurrence (Gated DeltaNet Style)

### Linear Update Rule
```python
S_t = alpha * S_{t-1} + outer(v, k)
```
where $\alpha$ is input-dependent but **not state-dependent**.

### Jacobian
$$J_{\text{linear}} = \alpha \cdot I_{nm}$$

This is simply a **scaled identity matrix**.

### Spectral Radius
$$\rho(J_{\text{linear}}) = \alpha$$

**Conditions**:
- $\rho < 1$ when $\alpha < 1$
- $\rho = 1$ when $\alpha = 1$ (pure memory)
- $\rho > 1$ when $\alpha > 1$ (explosive, typically avoided)

### Gradient Flow

Over $T$ timesteps:
$$\frac{\partial S_T}{\partial S_0} = \prod_{t=1}^{T} \alpha_t \cdot I_{nm}$$

Gradient magnitude:
$$\left\|\frac{\partial S_T}{\partial S_0}\right\| = \prod_{t=1}^{T} \alpha_t$$

**Key properties**:
1. **Predictable**: Gradient magnitude is exactly the product of gates
2. **Controllable**: $\alpha$ directly controls gradient flow
3. **No saturation effects**: Unlike tanh models, no additional attenuation
4. **Associative**: Enables parallel scan computation

---

## Summary Comparison Table

| Model | Jacobian Structure | Spectral Radius | Gradient Bound | Key Property |
|-------|-------------------|-----------------|----------------|--------------|
| **E70** | Diagonal | $\text{decay} \cdot (1-\tanh^2)$ | $< \text{decay}^T$ | Always stable, vanishes |
| **E71** | Block-diag + rank correction | Can exceed 1 | Unbounded | State-dependent, can explode |
| **E72** | Identity + Kronecker correction | Usually $< 1$ | Mostly controlled | Value-gated, moderate |
| **E73** | Diagonal | $z \cdot (1-\tanh^2)$ | $< 1^T = 1$ | Column forget gate |
| **Linear** | Scaled identity | $\alpha$ | $\prod \alpha_t$ | Fully controllable |

### Stability Ranking (most to least stable)

1. **E70**: Double contraction (decay AND tanh) guarantees $\rho < \text{decay} < 1$
2. **Linear**: Single contraction, predictable dynamics
3. **E73**: Sigmoid + tanh give $\rho < 1$ but less aggressive
4. **E72**: Mostly stable but value-dependent correction
5. **E71**: State-dependent gate can cause $\rho > 1$

### Expressivity Ranking (most to least expressive)

1. **E71**: Full state-dependent gating, richest dynamics
2. **E72**: State-dependent value modulation
3. **E73**: Column-wise forget gate (input-dependent)
4. **Linear**: Input-dependent scalar gate only
5. **E70**: Effectively memoryless due to strong contraction

### Practical Recommendations

1. **For long sequences** (gradient flow critical):
   - Prefer **Linear** (Gated DeltaNet): controllable gradients, parallel scan
   - **E71** with careful initialization: state-dependent but risky

2. **For stability** (avoiding NaN/explosion):
   - **E70** is safest but may underfit
   - **E73** is a good middle ground

3. **For expressivity**:
   - **E71** offers the most flexible dynamics
   - But requires careful regularization ($|d_\alpha|$ small, $\|k\|$ bounded)

4. **Initialization guidance**:
   - E70: Initialize decay $\approx 0.9$, values near zero
   - E71: Initialize $\alpha \approx 0.5$, small $d_\alpha$
   - E72: Initialize $d_g$ small, normalize $v, k$
   - E73: Initialize $z$ bias for $z \approx 0.5$

---

## Mathematical Appendix

### Kronecker Product Properties Used

1. $(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$
2. $(A \otimes B)^\top = A^\top \otimes B^\top$
3. $\text{vec}(AXB) = (B^\top \otimes A)\text{vec}(X)$
4. Eigenvalues of $A \otimes B$ are $\lambda_i(A) \cdot \mu_j(B)$

### Sigmoid Derivative Bound

$$\sigma'(x) = \sigma(x)(1-\sigma(x)) \leq \frac{1}{4}$$

with equality at $x = 0$.

### Tanh Derivative Bound

$$\tanh'(x) = 1 - \tanh^2(x) \leq 1$$

with equality at $x = 0$, and $\tanh'(x) \to 0$ as $|x| \to \infty$.

---

*Generated: 2026-01-15*
