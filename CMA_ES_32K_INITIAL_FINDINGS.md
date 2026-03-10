# CMA-ES Architecture Search at 32K: Initial Findings

**Date:** Feb 27, 2026
**Status:** Search in progress (3/6 models complete)

## Setup

CMA-ES architecture search where each evaluation runs a 2-phase progressive pipeline:
- Phase 1: 10 min training @ 512 tokens (bs=16) -> save checkpoint
- Phase 2: 10 min training @ 32K tokens (resume from Phase 1) -> report loss

CMA-ES optimizes dim, depth, heads, lr within ~480M params (+-10%). Fitness = Phase 2 loss only.

## Results So Far

| Model | Type | Best Loss @ 32K | Config | Params |
|-------|------|-----------------|--------|--------|
| **E88 n16** | Nonlinear sequential | **1.1000** | dim=1280, h=187, d=25 | 485M |
| **FLA-GDN** | Linear associative scan | **1.1345** | dim=1664, exp=3, d=13 | 433M |
| **Mamba2** | Linear SSM | **1.1882** | dim=1920, d_state=208, d=21 | 482M |

E88 n16 search still running (warm start 2/3). E88 n32, E1H n16, E1H n32 pending.

## Key Finding

**E88 (nonlinear sequential RNN) beats both linear baselines at 32K context.**

At 512 tokens, E88 trailed Mamba2 and FLA-GDN by ~0.04 nats. At 32K, E88 leads by 0.034 nats over FLA-GDN and 0.088 nats over Mamba2. The ranking has inverted.

This is consistent with the theoretical prediction: nonlinear-in-time recurrence achieves compositional depth that scales linearly with sequence length. At short sequences the advantage is negligible. At 32K bytes, the depth-of-computation advantage manifests empirically.

## Architectural Shifts at 32K

Models prefer different architectures at 32K vs 512:

| Model | 512 Optimal | 32K Optimal | Change |
|-------|-------------|-------------|--------|
| FLA-GDN | exp=2, d=17, h=24 | exp=3, d=13, h=12 | Wider, shallower |
| Mamba2 | d_state=96, d=25 | d_state=208, d=21 | 2x state, shallower |
| E88 n16 | h=141, d=25 | h=187, d=25 | More heads, same depth |

All models prefer wider/more state at long context. E88 scales by adding more independent memory heads (187 heads of 16x16 nonlinear matrices).

## Comparison to Manual Progressive Experiment (Feb 23)

The CMA-ES search significantly improves over our earlier progressive training with 512-optimized configs:

| Model | Manual Progressive | CMA-ES 32K | Improvement |
|-------|-------------------|------------|-------------|
| FLA-GDN | 1.179 | 1.135 | -0.044 |
| Mamba2 | 1.217 | 1.188 | -0.029 |
| E88 n16 | 1.230 | 1.100 | -0.130 |

E88 benefits the most from 32K-optimized architecture (0.13 nat improvement). The 512-optimal config (h=141) was significantly suboptimal for 32K -- more heads (187) works better.

## Remaining Searches

- E88 n32: Will show if n16 or n32 heads are better at 32K
- E1H n16/n32: Multi-head E1, which matched Mamba2 at 32K in manual experiments
