/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/

/-!
# Quarantined Research Sketches

This module is intentionally not imported by `ElmanProofs.lean`.

It collects modules that still contain unfinished `sorry` placeholders or
explicit research axioms. They may be useful as theorem roadmaps, but they are
not part of the trusted formal surface until each placeholder is either proved,
weakened to a true theorem, or moved into prose as a conjecture.

This file is a catalog rather than an import aggregate. Some historical sketch
files no longer elaborate under the current Lean/mathlib version, so importing
them here would make the catalog itself fail to build.

Current groups:

* explicit-assumption sketches: `Information.LanguageComplexity`,
  `Information.CompositionDepth`, `Architectures.SparseAttention`;
* learning-dynamics sketches: `Learning.*`;
* historical TC0/parity-centered sketches: `TC0Bounds`,
  `TC0VsUnboundedRNN`, `Section08_TC0CircuitComplexityBounds`,
  `ComputationalClasses`, `ExactCounting`, and `RunningParity`;
* expressivity sketches with placeholder proofs: `E88Definition`,
  `E88RankAccumulation`, `E88VariantClarification`, `MemoryCapacity`,
  `MultiPass`, and `StateAccessibility`.
-/
