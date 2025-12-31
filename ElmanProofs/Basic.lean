/-
Copyright (c) 2024 Elman Ablation Ladder Project. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Elman Ablation Ladder Team
-/

import Mathlib.Analysis.Normed.Group.Basic
import Mathlib.Analysis.Normed.Operator.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.Topology.MetricSpace.Contracting

/-!
# Elman Ablation Ladder: Formal Foundations

This library provides formal proofs about the stability, convergence, and learning
efficiency of recurrent neural network architectures, with focus on:

1. **Dynamical Systems**: Lyapunov stability, attractors, basins of attraction
2. **RNN Dynamics**: Recurrence maps, spectral radius constraints, contraction
3. **Activation Functions**: Lipschitz properties, boundedness, monotonicity
4. **Memory**: Hopfield-style attractors, capacity bounds, retrieval guarantees
5. **Gradient Flow**: Convergence rates, loss landscape geometry
6. **Log-Space**: Numerical stability in logarithmic representation

## Main Results

- `RNN.contraction_implies_unique_attractor`: Spectral radius < 1 implies unique fixed point
- `Activation.lipschitz_preserves_contraction`: Lipschitz activations maintain stability
- `GatedRNN.selective_memory`: Delta gates enable input-dependent attractor switching
- `LogSpace.bounded_gradient`: Log-space computation prevents gradient explosion

## References

* Elman, J.L. (1990). "Finding Structure in Time"
* Hochreiter & Schmidhuber (1997). "Long Short-Term Memory"
* Gu & Dao (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
-/

-- This file serves as the entry point, importing core modules
