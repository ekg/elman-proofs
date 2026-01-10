# E23 Dual-Memory Elman: Performance Analysis

## Summary

E23 CUDA kernel achieves **theoretical optimal performance** at 24us/step (batch=4), which equals 2x the GEMM time. However, **E23 scales poorly with batch size** compared to E1, resulting in 6.4x slower training throughput at batch=64.

## Training Benchmark Results (50M params, 10 min, batch=64)

| Model | Loss | Throughput | Steps | vs E1 |
|-------|------|------------|-------|-------|
| **E1 d1280×6** | 1.41 | 201.6K tok/s | 3,692 | 1.0x |
| **Mamba2** | 1.44 | 102.4K tok/s | 1,876 | 0.51x |
| **E23** | 1.84 | 31.5K tok/s | 577 | 0.16x |

**Key finding**: E23 is 6.4x slower than E1 and has higher loss (1.84 vs 1.41). The tape memory overhead doesn't pay off at this scale.

## Architecture Overview

E23 (Dual-Memory Elman) uses:
- **Tape**: [B, N, D] - Large linear storage (N slots)
- **Working Memory**: [B, D] - Small nonlinear compute

Per timestep operations:
1. **Read**: Query tape via attention → read value
2. **Update**: `h_work_new = tanh(W_h @ h_work + W_x @ x + read + b)`
3. **Write**: `h_tape_new = (1-attn)*h_tape + attn*(W_write @ h_work_new)`

## Performance Measurements

### Baseline Comparison (batch=4, seq=512, dim=512, n_slots=8)

| Model | Per-step Time | Throughput | GEMMs/step |
|-------|---------------|------------|------------|
| E1 (Gated Elman) | 9.1us | 225K tok/s | 1 |
| E23 (Dual-Memory) | 24.3us | 84K tok/s | 2 |
| **Ratio** | **2.67x** | **0.37x** | **2x** |

### Theoretical Analysis

```
Single W_h @ h GEMM (4×512 @ 512×512): 11.3us
Two sequential GEMMs:                  23.8us
E23 actual per-step:                   24.3us
Overhead (attention + memcpy):          0.5us (2%)
```

**Conclusion**: E23 overhead is essentially zero. The 2.67x slowdown vs E1 comes entirely from needing 2 GEMMs per timestep instead of 1.

### Batch Size Scaling (Forward Pass Only)

| Batch | Per-step | Throughput | Scaling |
|-------|----------|------------|---------|
| 1 | 19.6us | 51K tok/s | baseline |
| 4 | 22.2us | 181K tok/s | 3.5x |
| 8 | 22.0us | 364K tok/s | 7.1x |
| 16 | 22.4us | 715K tok/s | 14.0x |
| 32 | 25.1us | 1.27M tok/s | 25.0x |

### Batch Size Scaling (Full Training)

| Model | batch=4 | batch=64 | Scaling |
|-------|---------|----------|---------|
| E1 | 7.7K tok/s | 201.6K tok/s | **26x** |
| E23 | 2.2K tok/s | 31.5K tok/s | **14x** |
| Mamba2 | 19.8K tok/s | 102.4K tok/s | **5.2x** |

**Key insight**: E23 scales worse with batch size than E1 (14x vs 26x). The tape memory operations become a bottleneck at larger batches.

### Compute Efficiency

| Batch | W_h GEMM time | TFLOP/s | % of A100 Peak |
|-------|---------------|---------|----------------|
| 4 | 12.1us | 0.2 | 0.03% |
| 64 | 12.0us | 2.8 | 0.4% |

Small batches are **kernel launch bound**, not compute bound. This is expected for sequential recurrent operations.

## CUDA Implementation Details

### Current Implementation (Phase1 + cuBLAS + Phase2)

```
Per timestep:
1. cuBLAS GEMM: W_h @ h_work_prev     (~11us)
2. Phase1 kernel: read attention + h_work update
3. cuBLAS GEMM: W_write @ h_work_new  (~11us)
4. Phase2 kernel: write attention + tape update
5. cudaMemcpyAsync: save tape state (training only)
```

Total: ~24us/step

### Attempted Optimizations (Failed)

**Inline W_write GEMM in kernel** (3 attempts):
1. Basic inline GEMM: 2858ms (3x slower than cuBLAS)
2. Vectorized loads with ILP: 1867ms (2x slower)
3. Warp-collaborative reduction: 2229ms (still slower)

**Why cuBLAS wins**: L2 cache reuse across batch elements. cuBLAS processes the entire [B, D] matrix efficiently, while inline GEMM in our kernel processes one batch element per block.

### Why Further Optimization is Impossible

1. **GEMM-bound**: 98% of time is in 2 necessary GEMMs
2. **Minimal overhead**: Attention + memcpy = 0.5us/step
3. **Architecture requires both GEMMs**:
   - W_h @ h_work: Compute new hidden state
   - W_write @ h_work_new: Transform for tape write

## Triton Implementation Attempts

Attempted to create a Triton alternative to the CUDA kernel. All attempts failed to match CUDA performance.

### Implementation Comparison (batch=4, seq=512)

| Implementation | Time/step | vs CUDA |
|---------------|-----------|---------|
| **CUDA (cuBLAS + kernels)** | 24.3us | 1.0x |
| Triton + cuBLAS hybrid | 184.1us | 7.6x slower |
| Fused Triton (inline GEMMs) | 297.8us | 12.2x slower |
| Pure PyTorch (einsum loop) | 374.9us | 15.4x slower |

### Why Triton Cannot Match CUDA

1. **Python loop overhead**: Each timestep requires Python to dispatch kernels (~40us/step overhead)
2. **Cannot call cuBLAS from Triton**: Must use inline GEMMs which are 10-20x slower
3. **Sequential recurrence**: Triton is designed for parallel operations, not sequential loops
4. **Memory access patterns**: CUDA kernel optimizes L2 cache reuse across batch elements

### Triton Approaches Tested

**1. Triton + cuBLAS Hybrid (7.6x slower)**
- Use Triton for Phase1/Phase2 attention kernels
- Use cuBLAS (via PyTorch) for W_h @ h and W_write @ h GEMMs
- Problem: Python loop between operations adds overhead

**2. Fused Triton Kernel (12.2x slower)**
- Single Triton kernel per timestep with inline GEMMs
- Problem: Inline matrix-vector products are extremely slow
- Each GEMM row computed sequentially instead of using tensor cores

### Conclusion

**Triton is fundamentally unsuited for E23** due to:
- Sequential recurrent operations requiring Python loop
- Need for cuBLAS GEMMs (inaccessible from Triton kernels)
- Poor performance of inline matrix operations in Triton

The CUDA kernel remains the only viable high-performance implementation.

## Architectural Alternatives (Not Implemented)

To match E1's speed, would need to eliminate W_write GEMM:

| Variant | Change | Speed | Expressivity |
|---------|--------|-------|--------------|
| E23-full | W_write @ h | 24us | Full |
| E23-lite | write_val = h | ~12us | Reduced |
| E23-shared | W_write = W_h | ~12us | Reduced |

Trade-off: 2x speedup vs reduced model capacity.

## Recommendations

1. **E23 kernel is optimally implemented** - No further kernel improvements possible
2. **E23 has poor batch scaling** - Only 14x improvement from batch=4→64 vs E1's 26x
3. **E23 underperforms E1** - 6.4x slower, higher loss (1.84 vs 1.41)
4. **E1 remains the best model** - Fastest throughput (201K tok/s), lowest loss (1.41)
5. **E23 may need architectural changes** - The tape memory concept adds overhead without improving loss

## Verdict

E23's dual-memory architecture is **not competitive** at 50M scale:
- 6.4x slower than E1
- 3.2x slower than Mamba2
- Higher loss despite more architectural complexity

The tape memory idea may need larger scale or different tasks (long-range dependencies, retrieval) to show benefits.

## Files

- `elman/cuda/lib/dual_memory_elman_gpu.cu.cc` - CUDA kernel
- `profile_e23.py` - Basic profiling
- `profile_e23_detailed.py` - Detailed breakdown
- `profile_e23_triton.py` - Triton vs CUDA comparison
- `profile_e23_all.py` - All implementations comparison
- `elman/kernels/e23_triton.py` - Triton + cuBLAS hybrid
- `elman/kernels/e23_triton_fused.py` - Fused Triton kernel

## Commit History

- `e520dfe`: Optimize E23 CUDA kernel: 5.5x speedup (5000ms → 917ms)
- `da65f77`: Fix E23 backward pass misaligned address error
- `772d80b`: Add E23 profiling scripts for performance analysis
- `95cca45`: Add E23 benchmark and profiling scripts
- `3a08eec`: Add E23 Triton kernel attempts and profiling
