# Mathlib Matrix/Linear Continuity API Reference

## Summary

This document catalogs the relevant Mathlib APIs for proving continuity of matrix and linear operations, essential for proving that linear RNN outputs are continuous functions of inputs.

## Core Files

### 1. `Mathlib/Topology/Instances/Matrix.lean`
**Primary reference for matrix continuity.**

Key lemmas:
- `Continuous.matrix_mul`: Matrix multiplication is continuous
  ```lean
  theorem Continuous.matrix_mul [Fintype n] [Mul R] [AddCommMonoid R] [ContinuousAdd R]
      [ContinuousMul R] {A : X → Matrix m n R} {B : X → Matrix n p R} (hA : Continuous A)
      (hB : Continuous B) : Continuous fun x => A x * B x
  ```

- `Continuous.matrix_mulVec`: Matrix-vector multiplication is continuous
  ```lean
  theorem Continuous.matrix_mulVec [NonUnitalNonAssocSemiring R] [ContinuousAdd R] [ContinuousMul R]
      [Fintype n] {A : X → Matrix m n R} {B : X → n → R} (hA : Continuous A) (hB : Continuous B) :
      Continuous fun x => A x *ᵥ B x
  ```

- `Continuous.matrix_elem`: Element access is continuous
  ```lean
  theorem Continuous.matrix_elem {A : X → Matrix m n R} (hA : Continuous A) (i : m) (j : n) :
      Continuous fun x => A x i j
  ```

- `Continuous.matrix_transpose`: Transpose is continuous
- `Continuous.matrix_trace`: Trace is continuous (requires `ContinuousAdd`)
- `Continuous.matrix_det`: Determinant is continuous (requires `IsTopologicalRing`)
- `Continuous.dotProduct`: Dot product is continuous

Instances:
- `ContinuousMul (Matrix n n R)`: Square matrices form a topological monoid
- `IsTopologicalSemiring (Matrix n n R)`: Square matrices form a topological semiring
- `IsTopologicalRing (Matrix n n R)`: Square matrices form a topological ring

### 2. `Mathlib/Topology/Algebra/Monoid.lean`
**Finite sum continuity.**

Key lemma:
```lean
@[to_additive (attr := continuity, fun_prop)]
theorem continuous_finset_prod {f : ι → X → M} (s : Finset ι) :
    (∀ i ∈ s, Continuous (f i)) → Continuous fun a => ∏ i ∈ s, f i a
```

The additive version is `continuous_finset_sum`:
```lean
theorem continuous_finset_sum {f : ι → X → M} (s : Finset ι) :
    (∀ i ∈ s, Continuous (f i)) → Continuous fun a => ∑ i ∈ s, f i a
```

### 3. `Mathlib/Topology/Algebra/Module/FiniteDimension.lean`
**Automatic continuity for finite-dimensional linear maps.**

Key theorem:
```lean
theorem LinearMap.continuous_of_finiteDimensional [T2Space E] [FiniteDimensional 𝕜 E]
    (f : E →ₗ[𝕜] F') : Continuous f
```

This is extremely powerful: **Any linear map from a finite-dimensional T2 space over a complete nontrivially normed field is automatically continuous.**

Related:
- `continuous_equivFun_basis`: Basis isomorphism is continuous
- `LinearMap.toContinuousLinearMap`: Converts `E →ₗ[𝕜] F'` to `E →L[𝕜] F'`

### 4. `Mathlib/Topology/Algebra/Module/Basic.lean`
**Topological modules and continuous scalar multiplication.**

Key structures:
- `ContinuousSMul R M`: Scalar multiplication is continuous
- `ContinuousConstSMul R M`: Scalar multiplication by constants is continuous

### 5. `Mathlib/Topology/Algebra/Monoid/Defs.lean`
**Continuous addition/multiplication.**

Key classes:
- `ContinuousAdd M`: Addition is continuous
- `ContinuousMul M`: Multiplication is continuous

Key lemmas:
- `Continuous.add`: Composition with addition
- `Continuous.mul`: Composition with multiplication

## Application to Linear RNN Continuity

For a linear RNN: `h_t = A * h_{t-1} + B * x_t`, the output at position t is:
```
y_t = C * h_t = C * (∑_{s=0}^{t-1} A^{t-1-s} * B * x_s)
```

### Proof Strategy

1. **Element access is continuous**: `continuous_apply` for finite product topology
2. **Matrix-vector product is continuous**: `Continuous.matrix_mulVec`
3. **Finite sums are continuous**: `continuous_finset_sum`
4. **Compositions are continuous**: Standard composition rules

### Required Imports

```lean
import Mathlib.Topology.Instances.Matrix  -- Matrix continuity
import Mathlib.Topology.Algebra.Monoid    -- continuous_finset_sum
import Mathlib.Topology.Algebra.Module.FiniteDimension  -- automatic continuity
```

## Example Proof Pattern

```lean
-- For linear state is sum form:
-- h_t = ∑_{s=0}^{t-1} A^{t-1-s} * B * x_s

theorem linear_rnn_output_continuous {d_state d_in d_out T : ℕ}
    (A : Matrix (Fin d_state) (Fin d_state) ℝ)
    (B : Matrix (Fin d_state) (Fin d_in) ℝ)
    (C : Matrix (Fin d_out) (Fin d_state) ℝ)
    (t : Fin T) :
    Continuous fun (x : Fin T → Fin d_in → ℝ) =>
      C *ᵥ (∑ s in Finset.range t, A^(t-1-s) *ᵥ (B *ᵥ x s)) := by
  -- 1. Each term in the sum is continuous in x
  have h1 : ∀ s ∈ Finset.range t,
      Continuous fun x => A^(t-1-s) *ᵥ (B *ᵥ x s) := fun s _ => by
    apply Continuous.matrix_mulVec continuous_const
    apply Continuous.matrix_mulVec continuous_const
    exact continuous_apply s
  -- 2. The sum is continuous
  have h2 : Continuous fun x => ∑ s in Finset.range t, A^(t-1-s) *ᵥ (B *ᵥ x s) := by
    exact continuous_finset_sum _ h1
  -- 3. Final multiplication is continuous
  exact Continuous.matrix_mulVec continuous_const h2
```

## Notes

- For `ℝ^n` with product topology, all finite-dimensional linear maps are continuous
- The topology on `Matrix m n R` is the product topology (Pi type)
- `continuous_finset_sum` requires `ContinuousAdd` which holds for `ℝ^n`
- Matrix multiplication builds on `continuous_finset_sum` internally
