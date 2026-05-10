/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic
import ElmanProofs.Architectures.M2RNNComparison

/-!
# Online Keyed Memory

This module factors out the part shared by Gated DeltaNet and E88/NDM:
delta-correcting associative memory.

The important point for the architecture comparison is that exact overwrite is
not E88-specific. It is already present in the ideal delta rule used by GDN:

`S' = S + k (v - S^T k)^T`.

E88/NDM uses the same error-correcting pre-activation, then applies a nonlinear
full-state map (tanh in the current implementation). The later separation target
is therefore not "delta memory versus GDN"; it is linear/scan-compatible delta
memory versus nonlinear recurrent delta memory, and raw-write/fixed-transition
matrix RNNs versus delta-correcting memory.
-/

namespace OnlineMemory

open Matrix BigOperators

abbrev KeyVec (K : Nat) := M2RNNComparison.Vec K
abbrev ValueVec (V : Nat) := M2RNNComparison.Vec V
abbrev Memory (K V : Nat) := M2RNNComparison.MatState K V

variable {K V : Nat}

/-! ## Abstract Online Memory Spec -/

/-- Inner product on key vectors. -/
def keyDot (k q : KeyVec K) : Real :=
  Finset.univ.sum fun i => k i * q i

/-- Read a value vector from memory by query key. -/
def read (S : Memory K V) (q : KeyVec K) : ValueVec V :=
  M2RNNComparison.queryReadout S q

/-- Retrieval error at key `k`: the correction needed to make `read S k = v`. -/
def correction (S : Memory K V) (k : KeyVec K) (v : ValueVec V) : ValueVec V :=
  v - read S k

/-- Ideal delta-correcting write:

`S' = S + k (v - read S k)^T`.
-/
def linearDeltaWrite
    (S : Memory K V) (k : KeyVec K) (v : ValueVec V) : Memory K V :=
  S + M2RNNComparison.outerKV k (correction S k v)

/-- Gated DeltaNet's ideal write with unit write gate and no global decay. -/
def gdnIdealWrite
    (S : Memory K V) (k : KeyVec K) (v : ValueVec V) : Memory K V :=
  linearDeltaWrite S k v

/-- E88/NDM pre-nonlinearity write with unit decay. -/
def ndmPreTanhWrite
    (S : Memory K V) (k : KeyVec K) (v : ValueVec V) : Memory K V :=
  linearDeltaWrite S k v

/-- E88/NDM's nonlinear full-state version of the delta write. -/
noncomputable def ndmNonlinearWrite
    (S : Memory K V) (k : KeyVec K) (v : ValueVec V) : Memory K V :=
  M2RNNComparison.matrixTanh (linearDeltaWrite S k v)

/-! ## Delta-Memory Semantics -/

/-- Reading an outer-product write returns the written value scaled by key
overlap. -/
theorem read_outerKV
    (k q : KeyVec K) (e : ValueVec V) :
    read (M2RNNComparison.outerKV k e) q =
      fun j => keyDot k q * e j := by
  ext j
  simp [read, M2RNNComparison.queryReadout, M2RNNComparison.outerKV,
    keyDot, Matrix.mulVec, dotProduct]
  calc
    Finset.univ.sum (fun x : Fin K => k x * e j * q x)
        = Finset.univ.sum (fun x : Fin K => (k x * q x) * e j) := by
          apply Finset.sum_congr rfl
          intro i _
          ring
    _ = (Finset.univ.sum fun i : Fin K => k i * q i) * e j := by
          rw [Finset.sum_mul]
    _ = (Finset.univ.sum fun i : Fin K => k i * q i) * e j := rfl

/-- Delta write changes a query in proportion to its overlap with the write key. -/
theorem read_linearDeltaWrite
    (S : Memory K V) (k q : KeyVec K) (v : ValueVec V) :
    read (linearDeltaWrite S k v) q =
      read S q + fun j => keyDot k q * correction S k v j := by
  ext j
  simp [linearDeltaWrite, read, correction, M2RNNComparison.queryReadout,
    M2RNNComparison.outerKV, keyDot, Matrix.mulVec, dotProduct]
  calc
    Finset.univ.sum
        (fun x : Fin K => (S x j + k x * (v j - Finset.univ.sum (fun x => S x j * k x))) * q x)
        =
      Finset.univ.sum
        (fun x : Fin K =>
          S x j * q x + (k x * q x) * (v j - Finset.univ.sum (fun x => S x j * k x))) := by
          apply Finset.sum_congr rfl
          intro i _
          ring
    _ =
      Finset.univ.sum (fun x : Fin K => S x j * q x) +
      Finset.univ.sum
        (fun x : Fin K => (k x * q x) * (v j - Finset.univ.sum (fun x => S x j * k x))) := by
          rw [Finset.sum_add_distrib]
    _ =
      Finset.univ.sum (fun x : Fin K => S x j * q x) +
      (Finset.univ.sum fun i : Fin K => k i * q i) *
        (v j - Finset.univ.sum (fun x => S x j * k x)) := by
          rw [Finset.sum_mul]

/-- Unit-key delta write exactly overwrites the addressed readout. -/
theorem linearDeltaWrite_exact_overwrite
    (S : Memory K V) (k : KeyVec K) (v : ValueVec V)
    (hk : keyDot k k = 1) :
    read (linearDeltaWrite S k v) k = v := by
  rw [read_linearDeltaWrite]
  ext j
  simp [correction, hk]

/-- Orthogonal queries are preserved by a delta write. -/
theorem linearDeltaWrite_preserves_orthogonal_query
    (S : Memory K V) (k q : KeyVec K) (v : ValueVec V)
    (horth : keyDot k q = 0) :
    read (linearDeltaWrite S k v) q = read S q := by
  rw [read_linearDeltaWrite]
  ext j
  simp [horth]

/-- In the idealized no-decay, unit-write setting, GDN and E88/NDM share the
same delta-correcting pre-nonlinearity. -/
theorem gdn_and_ndm_share_ideal_delta_write
    (S : Memory K V) (k : KeyVec K) (v : ValueVec V) :
    gdnIdealWrite S k v = ndmPreTanhWrite S k v := by
  rfl

/-- The ideal overwrite theorem applies to both GDN's delta core and E88/NDM's
pre-tanh delta core. -/
theorem shared_delta_core_exact_overwrite
    (S : Memory K V) (k : KeyVec K) (v : ValueVec V)
    (hk : keyDot k k = 1) :
    read (gdnIdealWrite S k v) k = v ∧
    read (ndmPreTanhWrite S k v) k = v := by
  constructor <;> exact linearDeltaWrite_exact_overwrite S k v hk

/-! ## Resource Interpretation Hooks -/

/-- A model family has one-step exact overwrite semantics for orthogonal keys if
its write operation satisfies exact overwrite and non-target preservation.

This is intentionally an operational spec, not a full architecture definition.
It lets later modules compare which update families implement the spec directly
and which need extra heads, dimensions, layers, or recurrent steps. -/
structure OneStepOverwriteSpec (K V : Nat) where
  write : Memory K V → KeyVec K → ValueVec V → Memory K V
  exactOverwrite :
    ∀ (S : Memory K V) (k : KeyVec K) (v : ValueVec V),
      keyDot k k = 1 → read (write S k v) k = v
  preservesOrthogonal :
    ∀ (S : Memory K V) (k q : KeyVec K) (v : ValueVec V),
      keyDot k q = 0 → read (write S k v) q = read S q

/-- The ideal delta rule directly satisfies one-step overwrite semantics. -/
def linearDeltaOneStepOverwriteSpec (K V : Nat) : OneStepOverwriteSpec K V where
  write := linearDeltaWrite
  exactOverwrite := by
    intro S k v hk
    exact linearDeltaWrite_exact_overwrite S k v hk
  preservesOrthogonal := by
    intro S k q v horth
    exact linearDeltaWrite_preserves_orthogonal_query S k q v horth

end OnlineMemory
