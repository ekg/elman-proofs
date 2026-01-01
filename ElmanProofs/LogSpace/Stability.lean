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

    ## Mathematical Proof

    Let c = max(xs) be the maximum element.

    **Lower bound: c ≤ logSumExp xs**

    logSumExp xs = c + log(∑ᵢ exp(xᵢ - c))

    Since c = max(xs), there exists some index j with xⱼ = c.
    Therefore exp(xⱼ - c) = exp(0) = 1 is in the sum.
    So ∑ᵢ exp(xᵢ - c) ≥ 1, giving log(∑ᵢ exp(xᵢ - c)) ≥ 0.
    Hence logSumExp xs = c + log(sum) ≥ c + 0 = c.

    **Upper bound: logSumExp xs ≤ c + log(length xs)**

    For all i: xᵢ ≤ c (since c is the max)
    So xᵢ - c ≤ 0, meaning exp(xᵢ - c) ≤ exp(0) = 1.
    Summing n terms each ≤ 1: ∑ᵢ exp(xᵢ - c) ≤ n.
    Taking log (which is monotone on positive reals): log(∑ᵢ exp(xᵢ - c)) ≤ log(n).
    Hence logSumExp xs = c + log(sum) ≤ c + log(n).

    ## Lean Formalization Challenges

    The formal proof requires:
    1. `List.foldl max` properties - showing c = max of the list
    2. `List.foldl (+)` properties - showing it computes the sum
    3. Connecting `List.map` with the sum via `List.sum_map`
    4. Showing the maximum is achieved (requires nonempty list reasoning)
-/
theorem logSumExp_bounds (xs : List ℝ) (hne : ¬xs.isEmpty) :
    ∃ c, c = xs.foldl max xs.head! ∧ c ≤ logSumExp xs ∧ logSumExp xs ≤ c + log xs.length := by
  let c := xs.foldl max xs.head!
  use c
  refine ⟨rfl, ?_, ?_⟩

  · -- Lower bound: c ≤ logSumExp xs
    unfold logSumExp
    simp only [hne]

    -- Key insight: The sum ∑ exp(xᵢ - c) ≥ 1
    -- Since c is the max, exp(c - c) = 1 is in the sum, so sum ≥ 1
    -- Therefore log(sum) ≥ 0, so c + log(sum) ≥ c

    -- Helper: foldl max is either the init or an element
    have foldl_max_mem_or_init : ∀ (a : ℝ) (ys : List ℝ),
        ys.foldl max a = a ∨ ys.foldl max a ∈ ys := by
      intro a ys
      induction ys generalizing a with
      | nil => left; rfl
      | cons y ys ih =>
        simp only [List.foldl]
        specialize ih (max a y)
        cases ih with
        | inl h =>
          rw [h]
          by_cases hay : a ≤ y
          · right; simp only [max_eq_right hay, List.mem_cons, true_or]
          · push_neg at hay; left; exact max_eq_left (le_of_lt hay)
        | inr h => right; exact List.mem_cons_of_mem y h

    -- xs is nonempty
    have hne' : xs ≠ [] := by
      intro h
      simp only [h, List.isEmpty_nil, not_true_eq_false] at hne

    -- The max c is in xs
    have h_c_mem : c ∈ xs := by
      cases foldl_max_mem_or_init xs.head! xs with
      | inl h =>
        have : c = xs.head! := h
        rw [this]
        exact List.head!_mem_self hne'
      | inr h => exact h

    -- Show the sum is ≥ 1
    have h_sum_ge_one : (xs.map (fun x => exp (x - c))).foldl (· + ·) 0 ≥ 1 := by
      rw [← List.sum_eq_foldl]
      -- exp(c - c) = 1 is in the mapped list
      have h_one_mem : (1 : ℝ) ∈ xs.map (fun x => exp (x - c)) := by
        rw [List.mem_map]
        exact ⟨c, h_c_mem, by simp only [sub_self, exp_zero]⟩
      -- All terms are nonneg
      have h_all_nonneg : ∀ x ∈ xs.map (fun x => exp (x - c)), 0 ≤ x := by
        intro y hy
        rw [List.mem_map] at hy
        obtain ⟨z, _, hz⟩ := hy
        rw [← hz]
        exact le_of_lt (exp_pos _)
      exact List.single_le_sum h_all_nonneg 1 h_one_mem

    -- Now use that log is monotone and log 1 = 0
    have h_log_nonneg : 0 ≤ log ((xs.map (fun x => exp (x - c))).foldl (· + ·) 0) := by
      have h1 : (0 : ℝ) < 1 := by linarith
      have h2 : (1 : ℝ) ≤ (xs.map (fun x => exp (x - c))).foldl (· + ·) 0 := h_sum_ge_one
      have := log_le_log h1 h2
      simp only [log_one] at this
      exact this
    -- Goal: c ≤ c + log(sum)
    -- Since log(sum) ≥ 0, this follows
    calc c = c + 0 := by ring
         _ ≤ c + log ((xs.map (fun x => exp (x - c))).foldl (· + ·) 0) :=
             add_le_add_right h_log_nonneg c

  · -- Upper bound: logSumExp xs ≤ c + log(length xs)
    unfold logSumExp
    simp only [hne]

    -- xs is nonempty
    have hne' : xs ≠ [] := by
      intro h
      simp only [h, List.isEmpty_nil, not_true_eq_false] at hne

    -- Helper: foldl max gives an upper bound for all elements
    have foldl_max_is_ub : ∀ (a : ℝ) (ys : List ℝ) (y : ℝ), y ∈ ys → y ≤ ys.foldl max a := by
      intro a ys
      induction ys generalizing a with
      | nil => intro y hy; simp only [List.not_mem_nil] at hy
      | cons z zs ih =>
        intro y hy
        simp only [List.foldl]
        cases List.mem_cons.mp hy with
        | inl h =>
          rw [h]
          -- z ≤ foldl max (max a z) zs
          -- We show by induction that a ≤ foldl max a ys for any a, ys
          have aux : ∀ (b : ℝ) (ws : List ℝ), b ≤ ws.foldl max b := by
            intro b ws
            induction ws generalizing b with
            | nil => simp only [List.foldl]; exact le_refl b
            | cons w ws' ih' =>
              simp only [List.foldl]
              calc b ≤ max b w := le_max_left b w
                   _ ≤ List.foldl max (max b w) ws' := ih' (max b w)
          calc z ≤ max a z := le_max_right a z
               _ ≤ List.foldl max (max a z) zs := aux (max a z) zs
        | inr h => exact ih (max a z) y h

    -- c is an upper bound: ∀ x ∈ xs, x ≤ c
    have h_c_ub : ∀ x ∈ xs, x ≤ c := by
      intro x hx
      exact foldl_max_is_ub xs.head! xs x hx

    -- Each exp(x - c) ≤ 1
    have h_exp_le_one : ∀ y ∈ xs.map (fun x => exp (x - c)), y ≤ 1 := by
      intro y hy
      rw [List.mem_map] at hy
      obtain ⟨x, hx, rfl⟩ := hy
      have hxc : x ≤ c := h_c_ub x hx
      calc exp (x - c) ≤ exp 0 := exp_le_exp.mpr (by linarith)
           _ = 1 := exp_zero

    -- Sum ≤ length (each term ≤ 1)
    have h_sum_le : (xs.map (fun x => exp (x - c))).foldl (· + ·) 0 ≤ (xs.length : ℝ) := by
      rw [← List.sum_eq_foldl]
      have h_len : (xs.map (fun x => exp (x - c))).length = xs.length := List.length_map _
      calc (xs.map (fun x => exp (x - c))).sum
           ≤ (xs.map (fun x => exp (x - c))).length • (1 : ℝ) :=
             List.sum_le_card_nsmul _ 1 h_exp_le_one
         _ = ((xs.map (fun x => exp (x - c))).length : ℝ) := by simp only [nsmul_eq_mul, mul_one]
         _ = (xs.length : ℝ) := by rw [h_len]

    -- length xs ≥ 1 (nonempty)
    have h_len_pos : 0 < xs.length := List.length_pos_iff_ne_nil.mpr hne'

    -- sum > 0 (sum of positive terms)
    have h_sum_pos : 0 < (xs.map (fun x => exp (x - c))).foldl (· + ·) 0 := by
      rw [← List.sum_eq_foldl]
      apply List.sum_pos
      · intro y hy
        rw [List.mem_map] at hy
        obtain ⟨x, _, rfl⟩ := hy
        exact exp_pos _
      · intro h
        rw [List.map_eq_nil_iff] at h
        exact hne' h

    -- log(sum) ≤ log(length)
    have h_log_le : log ((xs.map (fun x => exp (x - c))).foldl (· + ·) 0) ≤ log xs.length := by
      apply log_le_log h_sum_pos
      exact h_sum_le

    -- Final: c + log(sum) ≤ c + log(length)
    calc c + log ((xs.map (fun x => exp (x - c))).foldl (· + ·) 0)
         ≤ c + log xs.length := add_le_add_right h_log_le c

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
  -- Case analysis on whether x/y ≥ 1 or x/y < 1
  have h_exp_pos : 0 < exp ε := exp_pos ε
  have h_xy_pos : 0 < x / y := div_pos hx hy

  -- Step 4: |x - y| / max(x,y) ≤ |x/y - 1|
  -- |x - y| / max(x,y) = |x/y - 1| when dividing by max
  by_cases hxy : x ≤ y
  · -- Case x ≤ y: max x y = y, so |x - y| / y = |x/y - 1|
    rw [max_eq_right hxy]
    have hxy_le_one : x / y ≤ 1 := (div_le_one hy).mpr hxy
    -- |x - y| / y = (y - x) / y = 1 - x/y
    have h1 : |x - y| / y = 1 - x / y := by
      rw [abs_sub_comm, abs_of_nonneg (by linarith : 0 ≤ y - x)]
      rw [sub_div, div_self hy.ne']
    rw [h1]
    -- Need: 1 - x/y ≤ exp ε - 1, i.e., 2 - exp ε ≤ x/y
    -- From xy_lower: (exp ε)⁻¹ ≤ x/y
    -- Actually: 1 - x/y ≤ 1 - (exp ε)⁻¹ = 1 - 1/exp ε = (exp ε - 1)/exp ε ≤ exp ε - 1
    have h2 : 1 - x / y ≤ 1 - (exp ε)⁻¹ := by linarith [xy_lower]
    have h3 : 1 - (exp ε)⁻¹ = (exp ε - 1) / exp ε := by
      rw [sub_div, div_self h_exp_pos.ne', one_div]
    rw [h3] at h2
    -- (exp ε - 1) / exp ε ≤ exp ε - 1 because exp ε ≥ 1 when ε ≥ 0, or both sides negative when ε < 0
    have h4 : (exp ε - 1) / exp ε ≤ exp ε - 1 := by
      by_cases hε_nonneg : 0 ≤ ε
      · -- When ε ≥ 0: exp ε ≥ 1, so exp ε - 1 ≥ 0
        have hexp_ge : 1 ≤ exp ε := one_le_exp hε_nonneg
        have h_numer_nonneg : 0 ≤ exp ε - 1 := by linarith
        -- (exp ε - 1) / exp ε ≤ exp ε - 1 iff (exp ε - 1) ≤ (exp ε - 1) * exp ε
        -- Since exp ε - 1 ≥ 0 and exp ε ≥ 1, this holds
        calc (exp ε - 1) / exp ε = (exp ε - 1) * (exp ε)⁻¹ := div_eq_mul_inv _ _
          _ ≤ (exp ε - 1) * 1 := by {
              apply mul_le_mul_of_nonneg_left
              · exact inv_le_one_of_one_le₀ hexp_ge
              · exact h_numer_nonneg
            }
          _ = exp ε - 1 := mul_one _
      · -- When ε < 0: exp ε < 1, so exp ε - 1 < 0
        push_neg at hε_nonneg
        have hexp_lt : exp ε < 1 := Real.exp_lt_one_iff.mpr hε_nonneg
        have h_numer_neg : exp ε - 1 < 0 := by linarith
        -- (exp ε - 1) / exp ε < 0 and exp ε - 1 < 0
        -- But (exp ε - 1) / exp ε = (something negative) / (something positive) < 0
        -- And since |exp ε - 1| / exp ε ≤ |exp ε - 1| (as exp ε > 0), we get
        -- (exp ε - 1) / exp ε ≥ exp ε - 1 but we need ≤
        -- Actually: (exp ε - 1) / exp ε > exp ε - 1 when ε < 0
        -- Wait, let's be more careful. When exp ε < 1:
        -- (exp ε - 1) / exp ε = exp ε - 1 iff exp ε = 1 or exp ε - 1 = 0
        -- (exp ε - 1) / exp ε < exp ε - 1 iff (exp ε - 1) < (exp ε - 1) * exp ε (dividing by positive exp ε)
        -- Since exp ε - 1 < 0 and exp ε < 1, (exp ε - 1) * exp ε > exp ε - 1
        -- So (exp ε - 1) / exp ε < exp ε - 1 when ε < 0
        have h_mult : (exp ε - 1) * exp ε > exp ε - 1 := by nlinarith [h_exp_pos, hexp_lt]
        have h_ineq : (exp ε - 1) / exp ε < exp ε - 1 := by
          have : (exp ε - 1) / exp ε < exp ε - 1 ↔ (exp ε - 1) < (exp ε - 1) * exp ε := by
            constructor
            · intro h; calc (exp ε - 1) = (exp ε - 1) / exp ε * exp ε := by field_simp
                _ < (exp ε - 1) * exp ε := by nlinarith [h_exp_pos]
            · intro h; calc (exp ε - 1) / exp ε = (exp ε - 1) / exp ε := rfl
                _ < (exp ε - 1) * exp ε / exp ε := by { rw [mul_div_assoc]; nlinarith [h_exp_pos] }
                _ = exp ε - 1 := by field_simp
          rw [this]
          exact h_mult
        linarith
    linarith
  · -- Case x > y: max x y = x, so |x - y| / x = 1 - y/x
    push_neg at hxy
    rw [max_eq_left (le_of_lt hxy)]
    have h_yx_pos : 0 < y / x := div_pos hy hx
    have hyx_le_one : y / x ≤ 1 := (div_le_one hx).mpr (le_of_lt hxy)
    -- |x - y| / x = (x - y) / x = 1 - y/x
    have h1 : |x - y| / x = 1 - y / x := by
      rw [abs_of_nonneg (by linarith : 0 ≤ x - y)]
      rw [sub_div, div_self hx.ne']
    rw [h1]
    -- From xy_upper: x/y ≤ exp ε, so y/x ≥ (exp ε)⁻¹
    have hyx_lower : (exp ε)⁻¹ ≤ y / x := by
      -- x/y ≤ exp ε implies y/x ≥ 1/exp ε = (exp ε)⁻¹
      -- (x/y)⁻¹ = y/x, and a ≤ b implies b⁻¹ ≤ a⁻¹ for positive a, b
      have h_inv_xy : (x / y)⁻¹ = y / x := by field_simp
      rw [← h_inv_xy]
      exact inv_anti₀ h_xy_pos xy_upper
    -- 1 - y/x ≤ 1 - (exp ε)⁻¹ ≤ exp ε - 1
    have h2 : 1 - y / x ≤ 1 - (exp ε)⁻¹ := by linarith [hyx_lower]
    have h3 : 1 - (exp ε)⁻¹ = (exp ε - 1) / exp ε := by
      rw [sub_div, div_self h_exp_pos.ne', one_div]
    rw [h3] at h2
    -- Same reasoning as first case for h4
    have h4 : (exp ε - 1) / exp ε ≤ exp ε - 1 := by
      by_cases hε_nonneg : 0 ≤ ε
      · have hexp_ge : 1 ≤ exp ε := one_le_exp hε_nonneg
        have h_numer_nonneg : 0 ≤ exp ε - 1 := by linarith
        calc (exp ε - 1) / exp ε = (exp ε - 1) * (exp ε)⁻¹ := div_eq_mul_inv _ _
          _ ≤ (exp ε - 1) * 1 := by {
              apply mul_le_mul_of_nonneg_left
              · exact inv_le_one_of_one_le₀ hexp_ge
              · exact h_numer_nonneg
            }
          _ = exp ε - 1 := mul_one _
      · push_neg at hε_nonneg
        have hexp_lt : exp ε < 1 := Real.exp_lt_one_iff.mpr hε_nonneg
        have h_mult : (exp ε - 1) * exp ε > exp ε - 1 := by nlinarith [h_exp_pos, hexp_lt]
        have h_ineq : (exp ε - 1) / exp ε < exp ε - 1 := by
          have : (exp ε - 1) / exp ε < exp ε - 1 ↔ (exp ε - 1) < (exp ε - 1) * exp ε := by
            constructor
            · intro h; calc (exp ε - 1) = (exp ε - 1) / exp ε * exp ε := by field_simp
                _ < (exp ε - 1) * exp ε := by nlinarith [h_exp_pos]
            · intro h; calc (exp ε - 1) / exp ε = (exp ε - 1) / exp ε := rfl
                _ < (exp ε - 1) * exp ε / exp ε := by { rw [mul_div_assoc]; nlinarith [h_exp_pos] }
                _ = exp ε - 1 := by field_simp
          rw [this]
          exact h_mult
        linarith
    linarith

end LogSpace
