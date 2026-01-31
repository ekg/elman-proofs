# Sorry Inventory - ElmanProofs/Expressivity

**Current Total: 21 sorries** (down from 57 at initial scan)

Updated: 2026-01-30

---

## Summary by File

| File | Count | Primary Root Cause |
|------|-------|-------------------|
| ExactCounting.lean | 9 | Existence proofs needing constructions |
| AttentionPersistence.lean | 7 | Incorrect theorem statements (missing constraints) |
| TanhSaturation.lean | 5 | Missing numerical bounds + incorrect statements |

---

## ROOT CAUSE ANALYSIS

### CATEGORY 1: Incorrect Theorem Statements (~10 sorries)

Several theorems are stated too broadly and are **mathematically false** without additional constraints:

**A) Missing `α > 1` hypothesis:**
- `alert_basin_nonempty` (TanhSaturation.lean:1051, 1122): For α ≤ 1, tanh(αx) iterates converge to 0, so alertBasin is EMPTY.

**B) Missing `θ < S*(α)` constraint:**
- `tanh_recur_preserves_alert` (AttentionPersistence.lean:1167, 1182): For θ close to 1 and α close to 1, the fixed point S*(α) may be < θ.
- `alert_forward_invariant` (AttentionPersistence.lean:1363, 1382): Same issue.
- Example: α=1.01, θ=0.9 → S*≈0.1 < θ, theorem is FALSE.

**C) Input assumption gap:**
- `strong_input_triggers_alert` (AttentionPersistence.lean:1804, 1847): For certain S values, α*S + δ*input falls in range where tanh(arg) < θ.

**Fix strategy:** Revise theorem statements with explicit hypotheses:
- Add `(hα_gt : α > 1)` where needed
- Weaken θ constraint to `θ < 0.76` (guaranteed below any fixed point for α > 1)

### CATEGORY 2: Missing Numerical Bounds (~2 sorries)

- `tanh(0.575) > 0.5` (TanhSaturation.lean:279)
- `latched_state_robust` (AttentionPersistence.lean:1396)

These require tanh bounds not directly in Mathlib.

**Fix strategy:** Prove via exp definition:
- `tanh(x) = (exp(2x) - 1)/(exp(2x) + 1)`
- `tanh(0.575) > 0.5 ⟺ exp(1.15) > 3`
- Use `exp(x) > 1 + x + x²/2` for lower bounds

### CATEGORY 3: Existence Proofs Needing Constructions (~9 sorries)

All in ExactCounting.lean:
- `linear_rnn_continuous_per_t` (line 251): Need Mathlib matrix continuity API
- `linear_rnn_continuous_output` (line 259): Same
- `exact_count_detection_not_linear` (line 334): Need discontinuity argument
- `count_mod_2_not_linear` (line 363): Connect to XOR impossibility
- `count_mod_3_not_linear` (line 372): Similar
- `e88_count_mod_2` (line 391): Construct specific α, δ, init
- `e88_count_mod_3_existence` (line 406): Construct basin parameters
- `tanh_multiple_fixed_points` (line 436): IVT-based proof
- `tanh_basin_of_attraction` (line 443): Contraction argument
- `e88_threshold_count` (line 455): Induction on inputs
- `latched_threshold_persists` (line 482): Numerical bounds

**Fix strategy:** These are provable with careful construction. Priority is the continuity lemmas since they unlock downstream proofs.

---

## Detailed Sorry Locations

### AttentionPersistence.lean (7 sorries)

| Line | Theorem | Root Cause | Fix |
|------|---------|------------|-----|
| 1167 | `tanh_recur_preserves_alert` (branch) | θ ≥ 0.76 not handled | Add θ < 0.76 hypothesis |
| 1182 | `tanh_recur_preserves_alert` (branch) | α close to 1 fails | Add α > 1.5 or θ small |
| 1363 | `alert_forward_invariant` (branch) | Same as 1167 | Add θ < 0.76 hypothesis |
| 1382 | `alert_forward_invariant` (branch) | Same as 1182 | Adjust constraints |
| 1396 | `latched_state_robust` | Numerical bounds | Prove via exp |
| 1804 | `strong_input_triggers_alert` | Theorem false for some S | Revise statement |
| 1847 | `strong_input_triggers_alert` | Same | Revise statement |

### ExactCounting.lean (9 sorries)

| Line | Theorem | Root Cause | Fix |
|------|---------|------------|-----|
| 251 | `linear_rnn_continuous_per_t` | Mathlib API | Use Continuous.matrix_mulVec |
| 259 | `linear_rnn_continuous_output` | Mathlib API | Same |
| 334 | `exact_count_detection_not_linear` | Discontinuity argument | Build from threshold |
| 363 | `count_mod_2_not_linear` | XOR connection | Use LinearLimitations |
| 372 | `count_mod_3_not_linear` | Similar | Affine impossibility |
| 391 | `e88_count_mod_2` | Construction | α=1.5, δ=2 works |
| 406 | `e88_count_mod_3_existence` | Construction | Define interval basins |
| 436 | `tanh_multiple_fixed_points` | IVT | g(x) = tanh(αx) - x |
| 443 | `tanh_basin_of_attraction` | Contraction | |f'(x*)| < 1 argument |
| 455 | `e88_threshold_count` | Induction | Track state lower bound |
| 482 | `latched_threshold_persists` | Numerical | Adjust parameters |

### TanhSaturation.lean (5 sorries)

| Line | Theorem | Root Cause | Fix |
|------|---------|------------|-----|
| 279 | `tanh_temporal_saturation` | Numerical | Prove tanh(0.575) > 0.5 |
| 1051 | `alert_basin_nonempty` (α > 1 branch) | Missing θ < S* | Add hypothesis |
| 1122 | `alert_basin_nonempty` (α ≤ 1 branch) | Theorem FALSE | Add α > 1 hypothesis |
| 1148 | `alert_state_robust` (trivial branch) | Edge case arg = 0 | Handle explicitly |
| 1160 | `alert_state_robust` (main branch) | Needs α ≥ 1 | Add hypothesis |

---

## Recommended Fix Strategy

### Priority 1: Fix Theorem Statements (unlocks ~10 sorries)

The key insight is that several theorems are **mathematically false** as stated. Fixing them requires:

1. **Add `α > 1` hypothesis** to `alert_basin_nonempty` (TanhSaturation.lean)
2. **Add `θ < 0.76` constraint** to forward invariance theorems
3. **Add bounded S constraint** to `strong_input_triggers_alert`

Once statements are correct, many proofs become straightforward.

### Priority 2: Complete Numerical Bounds (unlocks ~2 sorries)

Create or extend NumericalBounds.lean:
- `tanh_gt_half_of_ge_0575 : ∀ x ≥ 0.575, tanh x > 0.5`
- Based on: `exp(1.15) > 3` which can be proved from `exp(1) > 2.718`

### Priority 3: Continuity Proofs (unlocks downstream)

Use Mathlib's:
- `Continuous.matrix_mulVec`
- `continuous_finset_sum`
- `ContinuousLinearMap` composition

### Priority 4: Existence Constructions

Each can be done independently:
- Parity: Show sign-flip dynamics with specific parameters
- Mod 3: Define interval basins [-0.9, -0.3], [-0.3, 0.3], [0.3, 0.9]
- Fixed points: IVT on g(x) = tanh(αx) - x
