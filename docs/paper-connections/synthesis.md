# Synthesis: Why Nonlinear Recurrence Matters

## Three Papers, One Story

Three recent papers — LongHorn (NeurIPS 2024), "The Illusion of State" and related TC⁰ work (2024-2025), and "Temporal Superposition" (ICLR 2026 Oral) — converge on a unified explanation of our empirical and formal results. Each attacks the problem from a different angle, but they tell the same story: **linear state evolution is a fundamental limitation, and tanh-based nonlinear recurrence provides a qualitatively different representational geometry.**

---

## The Unified Picture

### Level 1: What linear SSMs optimize (LongHorn)

LongHorn shows that every principled SSM — Linear Attention, RetNet, GLA, DeltaNet, Longhorn itself — is the solution to a **convex online learning problem** on the hidden state:

```
s_t = argmin_s { stability(s, s_{t-1}) + plasticity(s, x_t) }
```

Different architectures correspond to different loss functions:

| Architecture | Loss on hidden state | Update rule |
|---|---|---|
| Linear Attention | Linear: `-⟨s^T k, x⟩` | `s_t = s_{t-1} + k·x^T` (accumulate) |
| RetNet | + L2 regularization `‖s‖²` | `s_t = γ·s_{t-1} + k·x^T` (decay + accumulate) |
| DeltaNet | Quadratic regression `‖s^T k - x‖²` | `s_t = (I - β·k·k^T)·s_{t-1} + β·v·k^T` (erase + write) |
| Longhorn | Same as DeltaNet, closed-form | Same structure, bounded step size |
| Mamba | No principled objective | Ad-hoc learned A matrix |

**The critical constraint:** Convex optimization over the state means every solution is **linear in `s_{t-1}`**. The framework literally cannot produce tanh, sigmoid, or any nonlinear function of the previous state.

**E88 without tanh IS RetNet.** The linearized E88 update `S_t = α·S_{t-1} + δ·v·k^T` is exactly RetNet's update. Our ablation at 512 tokens showing "tanh contributes nothing" is consistent: at short context, E88 operates within the LongHorn framework. The tanh is present but the model doesn't use it.

### Level 2: What linear SSMs cannot compute (TC⁰ bounds)

Six independent papers prove that all models with linear state evolution — every architecture derivable from LongHorn's framework — are confined to **TC⁰** (constant-depth threshold circuits):

- Merrill et al.: SSMs are in TC⁰ (Theorem 4.2, 4.4)
- Li et al.: Mamba with selectivity is in DLOGTIME-uniform TC⁰ (Theorem 4.5)
- Sarrof et al.: Nonneg-eigenvalue SSMs recognize exactly star-free regular languages (Theorem 4)

This means they **cannot solve:**
- Parity (XOR over bit sequences)
- Permutation composition (S₅ word problem, NC¹-complete)
- Boolean formula evaluation
- Chess state tracking (NC¹-complete via reduction to S₅)

Three escape routes exist:

| Escape | Mechanism | Parallelizable? | Our architecture |
|---|---|---|---|
| Negative eigenvalues | Extend [0,1] → [-1,1] | Yes (scan) | — |
| Householder products | DeltaProduct | Yes (scan) | — |
| Temporal nonlinearity | tanh/sgn in recurrence | No (sequential) | **E88, E1H** |

**Our contribution fills a gap:** No paper proves that any specific differentiable nonlinearity suffices. They use theoretical constructions (sgn) or structural modifications (Householder, eigenvalue remapping). Our Lean proofs are the first machine-verified analysis showing **tanh specifically suffices** for parity, threshold, counting, and binary latching.

### Level 3: Why nonlinearity helps geometrically (Temporal Superposition)

The ICLR 2026 Oral provides the geometric mechanism: tanh creates an **interference-free space** — a region of activation space where features can be stored without interfering with the output or with each other.

Three regimes:

| Regime | Temporal demand vs capacity | Nonlinearity role | Our observation |
|---|---|---|---|
| **Sparse** (low demand) | D >> features to track | Creates interference-free space, sharp forgetting | E88 at 32K with many heads |
| **Dense** (high demand) | D << features to track | Irrelevant — model is "effectively linear" | E88 at 512 tokens (tanh ablation = 0.00 nats) |
| **Transition** | D ≈ features | Phase transition in spectral radius and geometry | ? |

The key mechanisms:

- **Sharp forgetting** (nonlinear): Old features are decisively moved to the interference-free region (tanh saturation at ±1). No decay, no interference.
- **Smooth forgetting** (linear): Old features spiral toward the origin via exponential decay (α^T). Residuals accumulate, interference grows with sequence length.

**This IS our bifurcation / binary latching mechanism**, viewed from a different angle:

| Our terminology | Temporal superposition terminology |
|---|---|
| Bistable fixed points at ±S* | Interference-free space near tanh saturation |
| Binary latching | Sharp forgetting (decisive state placement) |
| Saturation-based write protection (tanh' ≈ 0) | Feature insensitivity in interference-free region |
| Linear state decay (α^T attenuation) | Smooth forgetting |

---

## Five Empirical Results Explained

### 1. Tanh ablation at 512 tokens (Δloss = 0.00 nats)

**LongHorn**: E88 without tanh IS RetNet. At short context, the convex online learning solution suffices.

**TC⁰ bounds**: Tasks at 512 tokens don't require parity/state-tracking, so TC⁰ expressivity is sufficient.

**Temporal superposition**: At 512 tokens, temporal demand is low. Each head is in the "effectively linear" regime where nonlinear RNNs adopt linear-like strategies. The interference-free space exists but isn't needed.

### 2. 32K ranking inversion (E88 > Mamba2 by 0.088 nats)

**LongHorn**: RetNet-like decay attenuates by α^T. At T=32K, early information is attenuated by α^32000 — effectively gone. E88's tanh saturation creates stable fixed points that persist indefinitely.

**TC⁰ bounds**: 32K tokens creates tasks requiring deep temporal composition. Linear SSMs hit the TC⁰ ceiling. E88's composition depth is D×T = 25×32768 = 819,200 vs Mamba2's D = 25.

**Temporal superposition**: At 32K, Mamba2's linear state is forced into destructive temporal superposition — exponential decay creates massive interference among accumulated residuals. E88 scales by adding independent heads (h=187 at 32K vs h=141 at 512), each operating in the sparse regime with sharp forgetting.

### 3. Many small heads > few large heads (68×16 > fewer×larger)

**Temporal superposition**: 68 heads × 16-dim state keeps each head in the **sparse regime** (low interference, near-orthogonal features). Fewer larger heads risk the **dense regime** where interference is inevitable and the model becomes "effectively linear" — negating the advantage of having nonlinearity at all.

**Additionally**: Independent heads provide zero cross-head interference by construction. Gradient isolation means each head's learning is localized and simple. This is why the effect is both theoretically sound and empirically easier to train.

### 4. E88 scales by adding heads, not depth, at 32K

**Temporal superposition**: Each new head contributes its own interference-free space with independent sharp forgetting. Total capacity scales as N_heads × D² with zero cross-head interference. This is the only scaling strategy that keeps each unit in the sparse regime.

**TC⁰ bounds**: Depth adds layers of composition. But E88 already has D×T composition depth from temporal nonlinearity. Additional depth is less valuable than additional parallel state capacity. The bottleneck at 32K is memory (how many features to track), not composition depth.

### 5. Expressivity hierarchy: Linear SSM ⊊ E1H ⊊ E88

**All three frameworks agree:**

| Separation | LongHorn view | TC⁰ view | Temporal superposition view |
|---|---|---|---|
| Linear SSM → E1H | Convex → non-convex optimization on state | TC⁰ → exceeds TC⁰ | No interference-free space → D-dim interference-free space |
| E1H → E88 | Same nonlinearity, different state capacity | Same complexity class, different capacity | D-dim → D²-dim interference-free space + content-addressable retrieval |

---

## What Our Work Uniquely Contributes

Across all three bodies of literature, our work fills specific gaps:

### 1. Tanh as a specific sufficient nonlinearity

No paper proves that tanh suffices for state tracking. They use sgn (not differentiable), Householder products (structural, not pointwise), or eigenvalue remapping (stays linear). Our Lean proofs provide the first machine-verified analysis of a practical, differentiable, trainable nonlinearity.

### 2. Matrix state vs vector state separation

No paper addresses the E1H → E88 separation. Merrill et al.'s RNN-SSM uses vector state. Grazzi et al.'s modifications apply to vector-state models. DeltaProduct's state is a matrix but evolves linearly. The combination of matrix state AND nonlinear temporal dynamics is unique to E88 in the literature.

### 3. Bifurcation analysis

None of the papers analyze fixed-point dynamics of tanh saturation. Our proofs about bistable fixed points at α > 1 and latching persistence provide mechanistic understanding of how E88 stores discrete facts. The temporal superposition paper describes the same phenomenon geometrically ("interference-free space") but does not formalize the dynamical systems mechanism.

### 4. Empirical validation at scale

The TC⁰ papers are purely theoretical. LongHorn tests at small scale. The temporal superposition paper uses toy tasks. Our CMA-ES experiments at 480M parameters with real language modeling provide the empirical bridge between theory and practice:
- Tanh ablation directly tests whether nonlinearity matters
- 32K ranking inversion shows when it starts mattering
- Head count scaling shows how E88 exploits its architectural advantage

---

## Architectural Implications

### The hybrid architecture thesis

LongHorn provides optimal linear state updates (principled, parallelizable). E88 provides nonlinear state updates (expressive, sequential). The natural next step: **a hybrid that uses LongHorn-style layers for pattern matching and E88-style layers for state tracking.**

This is supported by:
- **DeltaProduct** shows that parallelizable state tracking is possible but requires complex structures (products of Householder reflections)
- **Temporal superposition** shows that nonlinearity is only needed in the sparse regime (specific layers/heads that track state)
- **Our CMA-ES results** show that at 512 tokens, linear updates suffice (tanh ablation = 0.00 nats), suggesting most layers could be linear

### The online learning perspective on architecture design

LongHorn's framework suggests a design principle: **choose the loss function on the hidden state, then derive the update rule.** For linear SSMs, this is well-understood. For nonlinear recurrence, the open question is: what non-convex loss function would yield tanh-like updates as its (approximate) minimizer?

One candidate: bounded regression with a log-barrier constraint:
```
L_t(s) = ‖s - s_{t-1}‖² + β_t‖s^T k_t - x_t‖² - λ Σ log(1 - s_ij²)
```

The barrier term `-log(1 - s²)` enforces `s ∈ (-1, 1)`, and the KKT conditions of this constrained optimization would produce updates resembling tanh. This is speculative but provides a principled path toward deriving E88's update from optimization principles.

---

## Summary

| Paper | What it explains | What it can't explain |
|---|---|---|
| **LongHorn** | Why all principled SSMs are linear in state; why E88-without-tanh = RetNet; the convex ceiling | Why tanh helps (lies outside the framework) |
| **TC⁰ bounds** | Why linear SSMs cannot compute parity/state-tracking; what complexity class E88 escapes | Which specific nonlinearity works; empirical performance |
| **Temporal superposition** | Why tanh helps geometrically (interference-free space); why many heads win; why tanh is unused at 512 tokens | The formal complexity-theoretic framing; the specific mathematical constructions |
| **Our work** | Tanh specifically suffices (Lean proofs); matrix > vector state; 32K ranking inversion; head scaling; batch size dynamics | The online learning derivation of E88; the full complexity class of E88 |

The three papers provide the theoretical framework. Our Lean proofs provide the formal verification. Our CMA-ES experiments provide the empirical validation. Together, they form a complete picture: **linear state evolution is a convex optimization ceiling (LongHorn) that confines models to TC⁰ (complexity theory) by denying them interference-free representational geometry (temporal superposition). Tanh breaks all three barriers simultaneously.**

---

## References

### The three papers
- **LongHorn**: Liu et al. (NeurIPS 2024). [arXiv:2407.14207](https://arxiv.org/abs/2407.14207)
- **The Illusion of State**: Merrill, Petty, Sabharwal (2024). [arXiv:2404.08819](https://arxiv.org/abs/2404.08819)
- **Temporal Superposition**: Sharma, Proca, Prieto, Mediano (ICLR 2026 Oral). [OpenReview](https://openreview.net/forum?id=7cMzTpbJHC)

### Supporting papers
- Unlocking State-Tracking via Negative Eigenvalues: Grazzi et al. (ICLR 2025 Oral). [arXiv:2411.12537](https://arxiv.org/abs/2411.12537)
- DeltaProduct: Grazzi et al. (2025). [arXiv:2502.10297](https://arxiv.org/abs/2502.10297)
- Computational Limits of SSMs: Li, Guo, Chen (2024). [arXiv:2412.06148](https://arxiv.org/abs/2412.06148)
- Expressive Capacity of SSMs: Sarrof, Veitsman, Hahn (2024). [arXiv:2405.17394](https://arxiv.org/abs/2405.17394)
- DeltaNet: Yang et al. (NeurIPS 2024). [arXiv:2406.06484](https://arxiv.org/abs/2406.06484)
- TTT: Sun et al. (2024). [arXiv:2407.04620](https://arxiv.org/abs/2407.04620)
- Structured Sparse Transitions: (2025). [arXiv:2509.22284](https://arxiv.org/abs/2509.22284)
