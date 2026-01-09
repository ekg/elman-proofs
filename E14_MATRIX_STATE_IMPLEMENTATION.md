# E14: Matrix State Elman Implementation Summary

## Overview

E14 (Matrix State Elman) implements the architecture from `MATRIX_STATE_ELMAN.md`, trading **weight capacity for state capacity**. Instead of a vector hidden state h ∈ ℝ^d, it uses a **matrix hidden state H ∈ ℝ^(d×k)**.

## Mathematical Formulation

Per timestep update:
```
key = tanh(W_key @ x)           # [B, d], provides nonlinearity
value = W_val @ x               # [B, k]
decay = sigmoid(W_decay @ x)    # [B, d], input-dependent forgetting
H_new = decay[:,:,None] * H + key[:,:,None] * value[:,None,:]  # outer product update
query = W_query @ x             # [B, k]
pre_out = H_new @ query         # [B, d], matrix-vector multiply
output = pre_out * silu(z)      # gated output
```

## Key Properties

1. **State capacity**: d×k dynamic parameters (vs d for E1)
2. **Same computational cost**: O(d² + dk) when k=d, same as E1
3. **Associative memory**: Outer product accumulates key-value pairs like linear attention
4. **No extra GEMMs in recurrence**: Element-wise ops are O(dk) per step

## Implementation Files

### PyTorch Reference (COMPLETE & TESTED)
- `/home/erikg/elman/elman/models/matrix_state_elman.py`
- `MatrixStateElmanCell` - core cell with forward and forward_single_step
- `MatrixStateElman` - full layer with Mamba2-style in_proj/out_proj
- Tests pass for shapes, gradients, numerical gradient check

### CUDA Kernel (COMPLETE, NEEDS REBUILD)
- `/home/erikg/elman/elman/cuda/lib/matrix_state_elman_gpu.cu.cc`
- Forward and backward kernels for BF16 and generic types
- Pre-computes all projections (W_key @ x, W_val @ x, etc.) in batched GEMMs
- Element-wise kernels for state update and gated output

### Header Declarations
- `/home/erikg/elman/elman/cuda/lib/hasty/elman_ladder.h`
- Added `MatrixStateElmanForward<T>` and `MatrixStateElmanBackward<T>`

### Python Bindings
- `/home/erikg/elman/elman/cuda/pytorch/elman_ladder.cc`
- `matrix_state_elman_forward()` and `matrix_state_elman_backward()`

### Build Configuration
- `/home/erikg/elman/elman/cuda/Makefile` - added `lib/matrix_state_elman_gpu.o`

## Build Status

- CUDA kernel compiles and validates successfully
- PyTorch extension links to CUDA kernels
- **CUDA and PyTorch implementations match mathematically** (forward and backward validated)

## Benchmark Results (Jan 8, 2026)

| Model | Params | Throughput | Memory |
|-------|--------|------------|--------|
| E1 | 49.7M | 31.1K tok/s | 2.6GB |
| E14 | 20.3M | 5.7K tok/s | 17.4GB |
| Mamba2 | 50.9M | 92.6K tok/s | 2.3GB |

**Key finding**: E14 is ~5-6x slower than E1 due to:
1. Sequential element-wise state update (can't parallelize across time)
2. Massive memory for storing H_all [T+1, B, d, k] for backward
3. Element-wise kernels are memory-bound

## Analysis

E14 trades **weight parameters** for **state capacity**:
- E1: d vector state, d² weight parameters (W_h)
- E14: d×k matrix state, O(d²+dk) weight parameters

The matrix state gives more associative memory capacity (can store d×k key-value pairs) but at the cost of:
- Sequential recurrence (no parallel scan possible)
- O(dk) memory per timestep per layer (vs O(d) for E1)

## Potential Optimizations

1. **Gradient checkpointing**: Trade compute for memory by recomputing forward during backward
2. **Fused kernels**: Combine state update + output into single kernel
3. **Chunked BPTT**: Split long sequences into chunks
4. **Smaller d_state**: Use k < d for memory/speed tradeoff

## Parameter Count Comparison

For dim=d, expansion=1.0 (d_inner=d), d_state=k:

| Model | Weights per layer | State per layer |
|-------|------------------|-----------------|
| E1 | ~5d² (W_x, W_h, in/out proj) | d (vector h) |
| E14 | ~4d² + 4dk (projections) | dk (matrix H) |

When k=d, E14 has similar weight count but d² state capacity.

## Test Results

```
Test 1: Cell Forward Pass Shapes ✓
Test 2: Single Step vs Sequence Forward Consistency ✓
Test 3: Gradient Flow ✓
Test 4: Numerical Gradient Check ✓
Test 5: Full Layer Forward/Backward ✓
Test 6: Matrix State Update Math ✓
Test 7: Decay Initialization ✓
```

Date: 2026-01-08
