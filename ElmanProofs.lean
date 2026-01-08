-- Elman Ablation Ladder: Formal Proofs Library
-- Main entry point importing all modules

import ElmanProofs.Basic

-- Dynamical Systems
import ElmanProofs.Dynamics.Lyapunov
import ElmanProofs.Dynamics.Contraction

-- RNN Dynamics
import ElmanProofs.RNN.Recurrence

-- Activation Functions
import ElmanProofs.Activations.Lipschitz

-- Stability Theory
import ElmanProofs.Stability.SpectralRadius

-- Memory and Attractors
import ElmanProofs.Memory.Attractor

-- Gradient Flow
import ElmanProofs.Gradient.Flow

-- Log-Space Computation
import ElmanProofs.LogSpace.Stability

-- Expressivity Theory
import ElmanProofs.Expressivity.SpectralLowRank
import ElmanProofs.Expressivity.ScalingLaws
import ElmanProofs.Expressivity.GradientDynamics
import ElmanProofs.Expressivity.ExpressivityGradientTradeoff

-- Architecture Formalizations
import ElmanProofs.Architectures.DepthScaling
import ElmanProofs.Architectures.E1_GatedElman
import ElmanProofs.Architectures.E10_MultiscaleEMA
import ElmanProofs.Architectures.Mamba2_SSM
import ElmanProofs.Architectures.ExpressivityGap
import ElmanProofs.Architectures.RecurrenceLinearity

-- Information Theory of Language
import ElmanProofs.Information.LanguageComplexity
import ElmanProofs.Information.CompositionDepth
import ElmanProofs.Information.LinearVsNonlinear
