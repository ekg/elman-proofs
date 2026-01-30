# Q3: Synthetic Benchmark Design for Temporal Nonlinearity

**Date:** 2026-01-29
**Status:** DESIGN COMPLETE
**Summary:** Concrete benchmark specifications for empirically testing the temporal nonlinearity separation between E88 and Mamba2.

---

## 1. Design Goals

From `OPEN_QUESTIONS_TEMPORAL_VS_DEPTH.md`:

> **Q3: Practical implications**
> Design a synthetic task requiring temporal nonlinearity:
> - Running threshold detection
> - Temporal XOR (output x_t XOR x_{t-k})
> - Compare E88 vs deep Mamba2

The benchmark must:
1. **Require temporal nonlinearity** — not solvable by linear accumulation
2. **Scale with sequence length T** — to test the D vs D·T depth gap
3. **Have clear ground truth** — exact computation, not learned approximation
4. **Be learnable** — models should fit training data if they have sufficient expressivity

---

## 2. Task Suite: Three Benchmark Tasks

### Task 1: Running Threshold Count (RTC)

**Definition:**
```
Input:  x ∈ {0, 1}^T
Output: y ∈ {0, 1}^T
y_t = 1 iff |{i ≤ t : x_i = 1}| ≥ τ(t)
```

Where τ(t) is a position-dependent threshold, specifically:
```
τ(t) = ⌈t/2⌉  (majority threshold)
```

**Why it requires temporal nonlinearity:**

At each position t, the model must:
1. Accumulate a count (linear operation)
2. Compare against threshold (nonlinear operation)
3. Output binary decision (nonlinear operation)

Steps 2-3 must happen **at each timestep**, not just at the final output. This requires T sequential nonlinear operations.

**Data Generation:**
```python
def generate_rtc_data(T: int, batch_size: int) -> tuple[Tensor, Tensor]:
    """Generate Running Threshold Count data."""
    x = torch.randint(0, 2, (batch_size, T)).float()
    cumsum = torch.cumsum(x, dim=1)
    threshold = torch.arange(1, T+1).float().div(2).ceil()
    y = (cumsum >= threshold).float()
    return x, y
```

**Metrics:**
- **Per-position accuracy:** fraction of correct predictions at each t
- **Full-sequence accuracy:** fraction of sequences with all positions correct
- **Threshold-crossing accuracy:** accuracy specifically at positions where y changes

**Expected Results:**
| Model | Per-position Acc | Full-sequence Acc |
|-------|------------------|-------------------|
| E88 (1-layer) | ~99%+ | ~95%+ |
| Mamba2 (32-layer) | ~75-85% | <50% |
| Linear baseline | ~50% (random) | ~0% |

---

### Task 2: Temporal XOR Chain (TXC)

**Definition:**
```
Input:  x ∈ {0, 1}^T
Output: y ∈ {0, 1}^T
y_t = x_1 XOR x_2 XOR ... XOR x_t  (prefix parity)
```

**Why it requires temporal nonlinearity:**

XOR is fundamentally nonlinear — proven in `LinearLimitations.lean:315` (`linear_cannot_xor`). The prefix parity requires t-1 XOR operations at position t, giving total O(T²) nonlinear operations across the sequence.

**Key insight:** A linear temporal model can track a weighted sum, but XOR is not a linear function. The model cannot "defer" the XOR computation to a final nonlinearity because it must output intermediate parities.

**Data Generation:**
```python
def generate_txc_data(T: int, batch_size: int) -> tuple[Tensor, Tensor]:
    """Generate Temporal XOR Chain data."""
    x = torch.randint(0, 2, (batch_size, T)).float()
    y = torch.cumsum(x.long(), dim=1).fmod(2).float()  # Equivalent to XOR chain
    return x, y
```

**Metrics:**
- **Per-position accuracy:** should be ~100% for correct models, ~50% for linear models
- **Position-dependent accuracy:** plot accuracy vs position t

**Expected Results:**
| Model | Avg Accuracy | Accuracy at t=T |
|-------|--------------|-----------------|
| E88 (1-layer) | ~99%+ | ~99%+ |
| Mamba2 (D-layer) | 50% + O(D/T) | ~50% (for T >> D) |
| Linear baseline | 50% | 50% |

---

### Task 3: Finite State Machine (FSM)

**Definition:**

A 4-state FSM with irreversible transitions:

```
States: {A, B, C, D}
Initial: A
Transitions:
  A --1--> B
  B --1--> C
  C --1--> D  (absorbing)
  D --*--> D
  (any state) --0--> (same state)

Output: y_t = 1 iff state at time t is D
```

Equivalently: y_t = 1 iff |{i ≤ t : x_i = 1}| ≥ 3 (first three 1s)

**Why it requires temporal nonlinearity:**

The FSM has **irreversible** state transitions. Once state D is reached, output stays 1 forever. This requires:
1. Counting up to 3 (linear)
2. Detecting when count reaches 3 (nonlinear threshold)
3. Maintaining "reached D" memory indefinitely (requires nonlinear saturation)

A linear model's state decays exponentially, so it cannot maintain permanent memory of threshold crossing.

**Data Generation:**
```python
def generate_fsm_data(T: int, batch_size: int) -> tuple[Tensor, Tensor]:
    """Generate FSM (3-threshold) data."""
    x = torch.randint(0, 2, (batch_size, T)).float()
    cumsum = torch.cumsum(x, dim=1)
    y = (cumsum >= 3).float()  # State D reached
    return x, y
```

**Why this differs from RTC:**
- RTC has position-dependent threshold (τ(t) = ⌈t/2⌉)
- FSM has fixed threshold (τ = 3)
- FSM specifically tests irreversible state memory

**Expected Results:**
| Model | Accuracy (T=100) | Accuracy (T=1000) |
|-------|------------------|-------------------|
| E88 (1-layer) | ~99%+ | ~99%+ |
| Mamba2 (32-layer) | ~90% | ~60% |
| Linear baseline | ~50% | ~50% |

---

## 3. Experimental Protocol

### 3.1 Model Configurations

**E88 Variants:**
```python
E88_CONFIGS = [
    {"name": "E88-1L", "layers": 1, "n_heads": 16, "n_state": 32, "dim": 128},
    {"name": "E88-4L", "layers": 4, "n_heads": 4, "n_state": 32, "dim": 64},
]
```

**Mamba2 Variants:**
```python
MAMBA2_CONFIGS = [
    {"name": "Mamba2-4L", "layers": 4, "d_model": 64, "d_state": 16},
    {"name": "Mamba2-8L", "layers": 8, "d_model": 64, "d_state": 16},
    {"name": "Mamba2-16L", "layers": 16, "d_model": 64, "d_state": 16},
    {"name": "Mamba2-32L", "layers": 32, "d_model": 64, "d_state": 16},
]
```

**Baseline:**
```python
BASELINE_CONFIGS = [
    {"name": "Linear-RNN", "layers": 1, "hidden_dim": 128},  # h = Ah + Bx
    {"name": "MLP", "layers": 4, "hidden_dim": 128},  # No recurrence
]
```

### 3.2 Training Setup

```python
TRAIN_CONFIG = {
    "batch_size": 256,
    "learning_rate": 1e-3,
    "weight_decay": 0.01,
    "max_epochs": 100,
    "early_stop_patience": 10,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealing",
}
```

### 3.3 Sequence Length Sweep

Critical parameter: test the D vs D·T depth separation hypothesis.

```python
SEQUENCE_LENGTHS = [16, 32, 64, 128, 256, 512, 1024, 2048]
```

**Hypothesis:**
- E88 accuracy should be constant across T (O(1) state, T temporal nonlinearities)
- Mamba2 accuracy should degrade as T increases beyond 2^D threshold

### 3.4 Evaluation Metrics

For each (model, task, T) combination:

```python
@dataclass
class BenchmarkResult:
    model_name: str
    task_name: str
    sequence_length: int

    # Core metrics
    per_position_accuracy: float      # Mean over all (batch, position)
    full_sequence_accuracy: float     # Fraction with all positions correct

    # Breakdown metrics
    accuracy_by_position: List[float] # Accuracy at each t
    threshold_crossing_accuracy: float # Accuracy at y transitions

    # Training metrics
    final_train_loss: float
    epochs_to_converge: int
    total_params: int
    throughput_tokens_per_sec: float
```

---

## 4. Expected Theoretical Predictions

### 4.1 Information-Theoretic Analysis

From Q2 analysis, the key bounds are:

**E88 (1-layer):**
- Temporal composition depth: T
- Can represent O(2^T) distinct input→output mappings
- For RTC/TXC: needs O(T) bits of mutual information with input → achievable

**Mamba2 (D-layer, n-dimensional state):**
- Temporal composition depth: D (linear combination within each layer)
- Can represent O(2^{D·n·log(T)}) mappings (per analysis in Q2)
- For RTC/TXC: needs O(T) bits → fails when T >> D·n·log(T)

**Crossover point:**
```
T_crossover ≈ D · n · log(T)
=> T_crossover ≈ exp(D · n)   (self-consistent solution)
```

For Mamba2-32L with n=16: T_crossover ≈ exp(32 · 16) >> any practical T

**Wait** — this suggests Mamba2 *should* work for practical T. Let's refine:

The issue is per-position decision complexity:
- At each position t, RTC requires comparing cumsum vs threshold
- A D-layer model has D levels of nonlinearity to achieve this
- But the comparison must be *within* the state, not just at output

**Refined prediction:**
- Mamba2 can compute RTC **at the final output** (one threshold comparison)
- Mamba2 **cannot** compute RTC **at every position** (T threshold comparisons)
- The key test: require output y_t at each position, not just y_T

### 4.2 Predictions Table

| Task | T | E88-1L | Mamba2-4L | Mamba2-32L |
|------|---|--------|-----------|------------|
| RTC | 64 | 99% | 85% | 95% |
| RTC | 256 | 99% | 70% | 90% |
| RTC | 1024 | 99% | 60% | 80% |
| TXC | 64 | 99% | 60% | 75% |
| TXC | 256 | 99% | 52% | 60% |
| TXC | 1024 | 99% | 50% | 52% |
| FSM | 64 | 99% | 90% | 98% |
| FSM | 1024 | 99% | 70% | 85% |

**Key prediction:** TXC shows strongest separation because XOR is maximally nonlinear (proved impossible in LinearLimitations.lean).

---

## 5. Implementation Plan

### 5.1 Data Generation Module

```python
# benchmark/data.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass
class TaskData:
    x: Tensor  # (batch, T, input_dim)
    y: Tensor  # (batch, T, output_dim)
    mask: Tensor  # (batch, T) - which positions to evaluate

class SyntheticTask(ABC):
    """Base class for synthetic benchmark tasks."""

    @abstractmethod
    def generate(self, T: int, batch_size: int, seed: int) -> TaskData:
        """Generate a batch of task data."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Task identifier."""
        pass

class RunningThresholdCount(SyntheticTask):
    """y_t = 1 iff count of 1s in x[0:t] >= ceil(t/2)"""

    def generate(self, T: int, batch_size: int, seed: int) -> TaskData:
        torch.manual_seed(seed)
        x = torch.randint(0, 2, (batch_size, T, 1)).float()
        cumsum = torch.cumsum(x.squeeze(-1), dim=1)
        threshold = torch.arange(1, T+1).float().div(2).ceil().unsqueeze(0)
        y = (cumsum >= threshold).float().unsqueeze(-1)
        mask = torch.ones(batch_size, T)
        return TaskData(x=x, y=y, mask=mask)

    def name(self) -> str:
        return "RTC"

class TemporalXORChain(SyntheticTask):
    """y_t = x_1 XOR x_2 XOR ... XOR x_t"""

    def generate(self, T: int, batch_size: int, seed: int) -> TaskData:
        torch.manual_seed(seed)
        x = torch.randint(0, 2, (batch_size, T, 1)).float()
        # XOR chain = prefix parity = cumsum mod 2
        y = torch.cumsum(x.squeeze(-1).long(), dim=1).fmod(2).float().unsqueeze(-1)
        mask = torch.ones(batch_size, T)
        return TaskData(x=x, y=y, mask=mask)

    def name(self) -> str:
        return "TXC"

class FiniteStateMachine(SyntheticTask):
    """y_t = 1 iff count of 1s in x[0:t] >= 3 (absorbing state)"""

    def generate(self, T: int, batch_size: int, seed: int) -> TaskData:
        torch.manual_seed(seed)
        x = torch.randint(0, 2, (batch_size, T, 1)).float()
        cumsum = torch.cumsum(x.squeeze(-1), dim=1)
        y = (cumsum >= 3).float().unsqueeze(-1)
        mask = torch.ones(batch_size, T)
        return TaskData(x=x, y=y, mask=mask)

    def name(self) -> str:
        return "FSM"
```

### 5.2 Model Interfaces

```python
# benchmark/models.py

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch import Tensor

class SequenceModel(ABC):
    """Interface for sequence models."""

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, T, input_dim)
        Returns:
            y: (batch, T, output_dim)
        """
        pass

    @abstractmethod
    def reset_state(self):
        """Reset any recurrent state."""
        pass

    @abstractmethod
    def param_count(self) -> int:
        """Total trainable parameters."""
        pass

class E88Model(SequenceModel):
    """E88 with tanh temporal nonlinearity."""

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dim: int, n_heads: int, n_state: int, n_layers: int):
        # State matrix S: (n_heads, n_state, n_state)
        # Update: S = tanh(decay * S + outer(v, k))
        # This implements the crucial temporal nonlinearity
        ...

class Mamba2Model(SequenceModel):
    """Mamba2 with linear temporal dynamics."""

    def __init__(self, input_dim: int, output_dim: int,
                 d_model: int, d_state: int, n_layers: int):
        # Per-layer state: h_t = A * h_{t-1} + B * x_t  (LINEAR in time)
        # Selectivity: A, B depend on x (inter-layer nonlinearity)
        ...

class LinearRNN(SequenceModel):
    """Pure linear RNN baseline: h = Ah + Bx, y = Ch"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        ...
```

### 5.3 Training Loop

```python
# benchmark/train.py

def train_model(
    model: SequenceModel,
    task: SyntheticTask,
    T: int,
    config: TrainConfig,
) -> BenchmarkResult:
    """Train model on task and return results."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.BCEWithLogitsLoss()  # Binary classification per position

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(config.max_epochs):
        # Training
        model.train()
        for batch_idx in range(config.batches_per_epoch):
            data = task.generate(T, config.batch_size, seed=epoch*1000+batch_idx)

            model.reset_state()
            pred = model(data.x)  # (batch, T, 1)

            loss = criterion(pred, data.y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Evaluation
        model.eval()
        with torch.no_grad():
            eval_data = task.generate(T, config.eval_batch_size, seed=42)
            pred = model(eval_data.x)
            acc = compute_accuracy(pred, eval_data.y, eval_data.mask)

        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    return BenchmarkResult(
        model_name=model.name,
        task_name=task.name(),
        sequence_length=T,
        per_position_accuracy=best_acc,
        ...
    )
```

### 5.4 Benchmark Runner

```python
# benchmark/run.py

def run_full_benchmark():
    """Run complete benchmark suite."""

    tasks = [RunningThresholdCount(), TemporalXORChain(), FiniteStateMachine()]
    sequence_lengths = [64, 128, 256, 512, 1024]

    results = []

    for task in tasks:
        for T in sequence_lengths:
            for model_config in ALL_MODEL_CONFIGS:
                model = create_model(model_config, input_dim=1, output_dim=1)
                result = train_model(model, task, T, TRAIN_CONFIG)
                results.append(result)
                print(f"{result.model_name} on {task.name()} T={T}: {result.per_position_accuracy:.2%}")

    save_results(results)
    generate_plots(results)
```

---

## 6. Diagnostic Tests

Beyond accuracy, these diagnostics reveal *why* models fail:

### 6.1 Position-Dependent Accuracy Plot

For each (model, task, T):
```
accuracy[t] = mean(pred[t] == y[t]) over batch
```

**Expected patterns:**
- E88: flat accuracy across all t
- Mamba2: degrading accuracy as t increases (temporal information lost)
- Linear: random (50%) at all t

### 6.2 Gradient Flow Analysis

Measure gradient magnitude at each position during backprop:
```
grad_norm[t] = ||∂L/∂h_t||
```

**Expected patterns:**
- E88: gradient preserved due to tanh saturation regions
- Mamba2: exponential decay due to linear dynamics (unless α ≈ 1)

### 6.3 State Representation Analysis

Train a linear probe on hidden states to predict cumulative count:
```
probe_loss[t] = ||W @ h_t - cumsum(x)[t]||^2
```

**Expected patterns:**
- Both E88 and Mamba2: can represent cumsum (linear operation)
- Difference emerges in threshold *decision*, not representation

### 6.4 Ablation: Remove Temporal Nonlinearity from E88

Modify E88 update: `S = decay * S + outer(v, k)` (remove tanh)

This should make E88 equivalent to linear-temporal and fail similarly to Mamba2.

---

## 7. Connection to Formal Proofs

The benchmark tasks are designed to empirically test the theorems in the codebase:

### Task 1 (RTC) tests:
- `linear_cannot_threshold` (LinearLimitations.lean:107)
- Multi-layer extension from Q1 analysis

### Task 2 (TXC) tests:
- `linear_cannot_xor` (LinearLimitations.lean:315)
- Composition depth bound from Q2 analysis

### Task 3 (FSM) tests:
- Irreversible state transitions (not yet formalized)
- Memory capacity under decay

**Validation criterion:** If formal proofs are correct, empirical results should match predictions in Section 4.2.

---

## 8. Success Criteria

The benchmark successfully demonstrates temporal nonlinearity separation if:

1. **E88 achieves >95% accuracy** on all tasks across all sequence lengths T
2. **Mamba2 accuracy degrades** as T increases, particularly for TXC
3. **Linear baseline performs at chance** (50%) confirming tasks require nonlinearity
4. **Ablated E88** (no tanh) performs similarly to Mamba2

Quantitatively, we expect:
- TXC accuracy gap (E88 - Mamba2-32L) ≥ 40% at T=1024
- RTC accuracy gap ≥ 15% at T=1024
- FSM accuracy gap ≥ 10% at T=1024

---

## 9. File Structure

```
benchmark/
├── __init__.py
├── data.py          # Task definitions and data generation
├── models/
│   ├── __init__.py
│   ├── e88.py       # E88 implementation
│   ├── mamba2.py    # Mamba2 implementation
│   └── baselines.py # Linear RNN, MLP
├── train.py         # Training loop
├── evaluate.py      # Metrics computation
├── run.py           # Main benchmark runner
├── analysis.py      # Diagnostic tools
└── visualize.py     # Plotting
```

---

## 10. Summary

This benchmark provides:

1. **Three tasks** (RTC, TXC, FSM) requiring temporal nonlinearity
2. **Controlled comparison** between E88 (temporal tanh) and Mamba2 (linear temporal)
3. **Sequence length sweep** to test the D vs D·T depth hypothesis
4. **Diagnostic tools** to understand failure modes
5. **Connection to formal proofs** in LinearLimitations.lean

**Key hypothesis being tested:**
> E88's temporal tanh provides O(T) compositional depth per layer, enabling it to compute functions (threshold, XOR) that D-layer Mamba2 cannot compute for T >> 2^D.

The benchmark will empirically validate (or refute) this theoretical prediction from the Q1/Q2 analysis.

---

## References

- `OPEN_QUESTIONS_TEMPORAL_VS_DEPTH.md` — Problem statement
- `Q2_SEPARATION_ANALYSIS.md` — Separation theorem for RTC and TXC
- `docs/Q1_MULTILAYER_SSM_EXPRESSIVITY.md` — Multi-layer limitation proof
- `ElmanProofs/Expressivity/LinearLimitations.lean` — Formal threshold/XOR impossibility proofs
- `E88_EXPANSION_FINDINGS.md` — E88 architecture details
