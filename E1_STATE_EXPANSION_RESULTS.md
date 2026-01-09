# E1 State Expansion Results

## Hypothesis

Mamba2's d_state parameter (internal state dimension) might provide expressive power that E1 lacks. E1's hidden state dimension equals d_inner, while Mamba2 has a separate state dimension.

## Approach

E1 already has a `expansion` parameter: `d_inner = dim * expansion`. By increasing expansion while reducing dim to maintain ~400M total params, we can test whether a wider hidden state helps.

## Experiment

Tested E1 at ~400M params with different expansion values:
- expansion=1.0: d_inner = dim = 1760
- expansion=1.5: d_inner = 1968, dim = 1312
- expansion=2.0: d_inner = 2112, dim = 1056
- expansion=2.5: d_inner = 2200, dim = 880

## Results (batch=16, 1500 steps, Schedule-Free AdamW)

| Model | Params | Loss | Tok/s | Memory | Gap to Mamba2 |
|-------|--------|------|-------|--------|---------------|
| **Mamba2** | 402M | **1.50** | 23.4K | 8.1GB | - |
| E1 exp2.5 | 403M | 1.54 | 13.9K | 10.3GB | 0.04 nats |
| E1 exp2.0 | 406M | 1.54 | 14.5K | 10.3GB | 0.04 nats |
| E1 exp1.5 | 403M | 1.56 | 15.2K | 9.9GB | 0.06 nats |
| E1 exp1.0 | 403M | 1.58 | 15.7K | 9.7GB | 0.08 nats |

## Key Findings

### 1. State Expansion Closes Half the Gap!
- E1 exp1.0: 0.08 nats behind Mamba2
- E1 exp2.5: 0.04 nats behind Mamba2
- **Wider d_inner reduced the gap by 50%**

### 2. Diminishing Returns Above exp=2.0
- exp1.0 → exp1.5: 0.02 nats improvement
- exp1.5 → exp2.0: 0.02 nats improvement
- exp2.0 → exp2.5: <0.01 nats improvement

### 3. Throughput-Quality Tradeoff
- Expansion hurts throughput (15.7K → 13.9K, -11%)
- But improves loss (1.58 → 1.54)
- At exp2.0, best quality-speed tradeoff

### 4. Memory Cost is Modest
- exp1.0: 9.7GB
- exp2.5: 10.3GB (+6%)
- Much better than E10/E11/E12 which had 2x memory

## Analysis

State expansion helps because:
1. Wider hidden state = more capacity per layer
2. Even with smaller dim, the RNN dynamics benefit from more internal state
3. No extra GEMMs needed (unlike E12)

The remaining 0.04 nats gap might come from:
1. Mamba2's parallel associative scan (vs sequential)
2. Different normalization schemes
3. d_state vs d_inner (Mamba2's d_state=128 is fixed, small)

## Recommendation

**Use expansion=2.0 for E1** as default:
- 0.04 nats behind Mamba2 (vs 0.08 at exp=1.0)
- Throughput: 14.5K tok/s (still competitive)
- Memory: 10.3GB (acceptable)

## Next Steps

The remaining 0.04 nat gap is small. To close it:
1. Try Mamba2's initialization scheme on E1
2. Test different normalization (RMSNorm vs LayerNorm)
3. Investigate if parallel scan implementation would help

Date: 2026-01-08
