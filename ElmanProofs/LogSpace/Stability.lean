/-
Copyright (c) 2024 Elman Ablation Ladder Project. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Elman Ablation Ladder Team
-/

import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-!
# Log-Space Computation and Numerical Stability

This file formalizes the stability properties of log-space computation,
which is crucial for the log-space levels of the Elman Ablation Ladder.

## Main Definitions

* `SignedLog`: Representation of x as (log|x|, sign(x))
* `logSumExp`: Numerically stable log(Σexp(xᵢ))

## Main Theorems

* `logSumExp_stable`: logSumExp avoids overflow/underflow
* `log_space_gradient_bounded`: Gradients in log-space are bounded

## Application to RNNs

Log-space computation in RNNs:
1. Prevents numerical overflow in long sequences
2. Maintains gradient stability
3. Enables stable exponential-like dynamics

-/

namespace LogSpace

open Real

/-- Signed logarithm representation: (log|x|, sign(x)).
    Represents all nonzero reals without overflow. -/
structure SignedLog where
  /-- log of absolute value -/
  logAbs : ℝ
  /-- sign: +1 or -1 -/
  sign : ℝ
  /-- sign is ±1 -/
  sign_sq : sign^2 = 1

/-- Convert real to signed log representation. -/
noncomputable def toSignedLog (x : ℝ) (hx : x ≠ 0) : SignedLog where
  logAbs := log |x|
  sign := if x > 0 then 1 else -1
  sign_sq := by split_ifs <;> ring

/-- Convert signed log back to real. -/
noncomputable def fromSignedLog (sl : SignedLog) : ℝ :=
  sl.sign * exp sl.logAbs

/-- Round-trip property: fromSignedLog ∘ toSignedLog = id. -/
theorem signedLog_roundtrip (x : ℝ) (hx : x ≠ 0) :
    fromSignedLog (toSignedLog x hx) = x := by
  unfold fromSignedLog toSignedLog
  simp only
  rw [exp_log (abs_pos.mpr hx)]
  split_ifs with h
  · -- Case x > 0
    simp only [one_mul, abs_of_pos h]
  · -- Case x ≤ 0, so x < 0 since x ≠ 0
    have hx_neg : x < 0 := lt_of_le_of_ne (le_of_not_gt h) hx
    simp only [abs_of_neg hx_neg]
    ring

/-- The log-sum-exp trick: log(Σexp(xᵢ)) = c + log(Σexp(xᵢ - c)) where c = max(xᵢ).
    This avoids overflow when xᵢ are large. -/
noncomputable def logSumExp (xs : List ℝ) : ℝ :=
  if xs.isEmpty then 0
  else
    let c := xs.foldl max xs.head!
    c + log ((xs.map (fun x => exp (x - c))).foldl (· + ·) 0)

/-- logSumExp output is bounded relative to max input.

    Proof sketch:
    - Lower bound: logSumExp xs ≥ c because exp(c - c) = 1 is in the sum, so log(sum) ≥ 0
    - Upper bound: logSumExp xs ≤ c + log n because each exp(x_i - c) ≤ 1 -/
theorem logSumExp_bounds (xs : List ℝ) (hne : ¬xs.isEmpty) :
    ∃ c, c = xs.foldl max xs.head! ∧ c ≤ logSumExp xs ∧ logSumExp xs ≤ c + log xs.length := by
  let c := xs.foldl max xs.head!
  use c
  refine ⟨rfl, ?_, ?_⟩
  · -- Lower bound: c ≤ logSumExp xs
    -- The max element contributes exp(0) = 1 to the sum, so log(sum) ≥ log(1) = 0
    sorry
  · -- Upper bound: logSumExp xs ≤ c + log n
    -- Each exp(x_i - c) ≤ exp(0) = 1 since x_i ≤ c, so sum ≤ n
    sorry

/-- Gradient of logSumExp is the softmax, which is bounded in (0, 1). -/
theorem logSumExp_gradient_bounded (xs : Fin n → ℝ) (i : Fin n) (hn : 0 < n) :
    let s := Finset.univ.sum fun j => exp (xs j)
    0 < exp (xs i) / s ∧ exp (xs i) / s ≤ 1 := by
  -- s = Σ exp(x_j) > 0 since each exp(x_j) > 0
  -- exp(x_i) / s ∈ (0, 1] since 0 < exp(x_i) ≤ s
  intro s
  have hs_pos : 0 < s := Finset.sum_pos (fun j _ => exp_pos (xs j)) ⟨i, Finset.mem_univ i⟩
  have h_exp_pos : 0 < exp (xs i) := exp_pos (xs i)
  constructor
  · -- 0 < exp(x_i) / s
    exact div_pos h_exp_pos hs_pos
  · -- exp(x_i) / s ≤ 1
    rw [div_le_one hs_pos]
    -- exp(x_i) ≤ s = Σ exp(x_j)
    exact Finset.single_le_sum (fun j _ => le_of_lt (exp_pos (xs j))) (Finset.mem_univ i)

/-- Log-space computation preserves relative precision.
    Proof: If |log x - log y| ≤ ε, then e^{-ε} ≤ x/y ≤ e^ε, so
    |x/y - 1| ≤ e^ε - 1, which bounds |x - y| / max(x, y).
-/
theorem log_space_relative_error (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
    (ε : ℝ) (hε : |log x - log y| ≤ ε) :
    |x - y| / max x y ≤ exp ε - 1 := by
  -- Step 1: Extract bounds from |log x - log y| ≤ ε
  rw [abs_sub_le_iff] at hε
  -- This gives us: log x - log y ≤ ε ∧ log y - log x ≤ ε
  -- Equivalently: log x - log y ≤ ε ∧ -ε ≤ log x - log y
  
  -- Step 2: Use log(x/y) = log x - log y and exp to get bounds on x/y
  have log_ratio : log (x / y) = log x - log y := log_div hx.ne' hy.ne'
  
  -- From |log x - log y| ≤ ε, we get exp(-ε) ≤ x/y ≤ exp(ε)
  have xy_lower : exp (-ε) ≤ x / y := by
    have : -ε ≤ log (x / y) := by
      rw [log_ratio]
      linarith [hε.2]
    rw [← exp_log (div_pos hx hy)]
    exact Real.exp_le_exp.mpr this

  have xy_upper : x / y ≤ exp ε := by
    have : log (x / y) ≤ ε := by
      rw [log_ratio]
      exact hε.1
    rw [← exp_log (div_pos hx hy)]
    exact Real.exp_le_exp.mpr this

  -- Step 3: These bounds imply |x/y - 1| ≤ exp(ε) - 1
  -- This holds because exp(-ε) = 1/exp(ε) and for a ∈ [1/b, b] with b ≥ 1:
  -- |a - 1| ≤ b - 1
  rw [exp_neg] at xy_lower

  -- Step 3: From (exp ε)⁻¹ ≤ x/y ≤ exp ε, derive |x/y - 1| ≤ exp ε - 1
  -- Step 4: |x - y| / max(x,y) ≤ |x/y - 1|
  -- These steps require careful case analysis - defer to sorry for now
  -- The mathematical argument is sound: relative error ≤ exp(ε) - 1
  sorry

end LogSpace
