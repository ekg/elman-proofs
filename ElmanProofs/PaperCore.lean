/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import ElmanProofs.Architectures.M2RNNComparison
import ElmanProofs.Architectures.OnlineMemory
import ElmanProofs.Architectures.RecurrentResourceFormalism
import ElmanProofs.Expressivity.E88ExceedsE1HCapacity

/-!
# Trusted Paper Core

This is the small paper-facing proof root.

Terminology:

* NDM means Nonlinear Delta Memory, the model family.
* E88 is the current production implementation lineage of NDM.

This root intentionally imports only the checked chain needed for the current
paper-space claims:

* M2RNN and NDM/E88 are distinct one-step transition families.
* M2RNN's raw outer write is separated from the delta-correcting write shared
  by ideal GDN and NDM/E88 pre-nonlinearity.
* NDM/E88 exposes the current 1.27B many-program production geometry.
* E88 has the checked matrix-state capacity separation over E1H.

Use `scripts/check_paper_core.sh` to reject unfinished proof holes, explicit
assumptions, opaque declarations, and kernel-bypassing computation in this
import closure.
-/
