# Experimental Methodology for Comparing Neural Architectures

**Date:** 2026-01-31
**Purpose:** Best practices for comparing neural architectures with different expressivity properties
**Context:** E88 vs. Mamba2/SSMs expressivity separation

---

## Executive Summary

This document presents research-based methodology for fairly comparing neural architectures, particularly when comparing models with different theoretical expressivity (e.g., E88 with temporal nonlinearity vs. Mamba2 with linear-temporal dynamics). The methodology addresses parameter matching, hyperparameter control, benchmark selection, statistical testing, and presentation standards expected by top-tier reviewers.

**Key Insight:** Theory tells us what is possible with unlimited resources. Experiments tell us what happens with finite resources on specific tasks. Both perspectives are essential.

---

## 1. Parameter Matching and Fair Comparison

### 1.1 The Parameter Matching Problem

Different architectures achieve comparable capacity through different mechanisms:
- **Mamba2**: Linear SSM with d_state=96, expand=2 (hidden dim is 2× model dim)
- **E88**: Nonlinear recurrence with n_heads=68, n_state=16 per head (1088 total state), expansion=1.0
- **FLA-GDN**: Gated Delta with expansion=2, n_heads=24

### 1.2 Recommended Approach: CMA-ES Hyperparameter Optimization

**Method:** Use evolutionary optimization (CMA-ES) to search hyperparameter space for each architecture independently, constrained to similar parameter counts.

**Why this works:**
1. Each architecture explores its own optimal configuration
2. Parameter count constraint ensures fair comparison
3. Allows different architectures to use different depths, widths, heads
4. Avoids forcing architectures into suboptimal configurations

**Implementation (from Section 11 experiments):**
```python
# Define parameter budget (e.g., 480-500M params)
param_range = (480_000_000, 500_000_000)

# For each architecture, CMA-ES optimizes:
for architecture in [Mamba2, E88, FLA_GDN, Transformer]:
    search_space = architecture.get_search_space()
    best_config = cmaes.optimize(
        objective=lambda config: train_and_evaluate(config),
        search_space=search_space,
        constraint=lambda config: param_range[0] <= count_params(config) <= param_range[1],
        evaluations=120  # or architecture-specific (some need more)
    )
```

**Fixed hyperparameters (shared across all):**
- Learning rate: 3e-4 (Adam)
- Training time: 10 minutes per configuration
- Dataset: Same train/val/test split
- Batch size: Same effective batch (batch × gradient accumulation)
- Sequence length: Same context window

**Architecture-specific hyperparameters (optimized independently):**
- Depth (number of layers)
- Hidden dimension
- Number of heads
- State dimension
- Expansion factor
- Activation functions (if architecture allows choice)

### 1.3 Alternative: Grid Search with Post-Hoc Matching

When CMA-ES is too expensive, use exhaustive grid search and select similarly-performing models:

**Method:**
1. Define grid for each architecture (depth × hidden_dim × heads)
2. Train all configurations
3. For each architecture, identify the Pareto frontier (loss vs. params)
4. Select configurations with similar param counts from the frontier

**Advantage:** Deterministic, reproducible
**Disadvantage:** Exponential search space, may miss optimal configs

---

## 2. Standard Benchmarks for Expressivity

### 2.1 Two Types of Benchmarks

**Expressivity-Critical Tasks** (Theory predicts separation):
- Test whether architectural differences manifest empirically
- Linear-temporal models should fail; nonlinear-temporal should succeed

**Practical Tasks** (No theoretical prediction):
- Language modeling, real-world datasets
- May not exercise expressivity differences
- Winner depends on optimization, throughput, data

### 2.2 Expressivity-Critical Benchmark Suite

Based on theoretical separation (OPEN_QUESTIONS_RESOLUTION.md), these tasks require temporal nonlinearity:

| Task | Definition | Why Linear SSMs Fail | Why E88 Succeeds |
|------|------------|---------------------|------------------|
| **Running Parity** | y_t = x_1 ⊕ x_2 ⊕ ... ⊕ x_t | XOR is not linear (LinearLimitations.lean:315) | Tanh enables sign-flip dynamics |
| **Running Threshold Count** | y_t = 1 iff count(x[0:t] == 1) ≥ τ | Linear cannot threshold (LinearLimitations.lean:107) | Nested tanh creates quantization |
| **Temporal XOR Chain** | y_t = x_t ⊕ x_{t-k} | Requires T nonlinear compositions for T timesteps | O(T) compositional depth per layer |
| **Finite State Machine** | y_t = 1 iff count ≥ 3 (absorbing) | Cannot implement irreversible transitions | Tanh saturation creates absorbing states |

**Dataset Specification:**
- Sequence lengths: T ∈ {64, 128, 256, 512, 1024, 2048}
- Binary input: x_t ∈ {0, 1}
- Binary output: y_t ∈ {0, 1}
- Training: 50k sequences per length
- Validation: 10k sequences per length
- Test: 10k sequences per length
- Metric: Per-timestep accuracy (not just final position)

**Expected Results (from Q3_SYNTHETIC_BENCHMARK_DESIGN.md):**

| Task | T | E88-1L | Mamba2-32L | Gap | Theoretical Prediction |
|------|---|--------|------------|-----|------------------------|
| Running Parity | 1024 | 99% | 50% | 49% | ✓ Separation proven |
| Running Threshold | 1024 | 99% | 80% | 19% | ✓ Separation proven |
| Temporal XOR | 1024 | 99% | 52% | 47% | ✓ Separation proven |
| FSM (absorbing) | 1024 | 99% | 85% | 14% | ✓ Separation proven |

**Success Criteria:**
1. E88 achieves >95% accuracy on all tasks across all T
2. Mamba2 accuracy degrades as T increases (shows depth limitation)
3. Linear baseline (no temporal nonlinearity) performs at chance (50%)
4. Ablated E88 (tanh → linear) matches Mamba2 performance

### 2.3 Practical Benchmark Suite

**Language Modeling:**
- Dataset: Standardized corpus (e.g., Pile, C4, WikiText-103)
- Metric: Perplexity, bits-per-character
- Context: 1024-8192 tokens
- Evaluation: Full test set (not just validation)

**Downstream Tasks (optional, but recommended):**
- GLUE/SuperGLUE (natural language understanding)
- Long-range arena (LRA) - specifically tests long-range dependencies
- Code completion (HumanEval, MBPP)
- Reasoning (GSM8K, MATH)

**Why both benchmarks matter:**
- Expressivity benchmarks validate theory
- Practical benchmarks measure real-world utility
- Gap between them reveals what language modeling requires

---

## 3. Hyperparameter Control Protocol

### 3.1 Shared Hyperparameters (Must Be Identical)

**Training:**
- Learning rate schedule: Cosine decay with warmup
- Optimizer: AdamW (β1=0.9, β2=0.999, ε=1e-8)
- Weight decay: 0.1
- Gradient clipping: Max norm 1.0
- Batch size: Effective batch (batch_size × grad_accum_steps)
- Training steps: Same total number of steps (or same wall-clock time)
- Random seed: Different seeds for robustness testing (see Section 4)

**Data:**
- Dataset split: Identical train/val/test
- Sequence length: Same context window
- Preprocessing: Identical tokenization, normalization
- Data order: Use same random shuffle seed

**Hardware:**
- Run all experiments on same hardware (e.g., single A100 80GB)
- Measure wall-clock time, memory, throughput
- Control for GPU utilization differences

### 3.2 Architecture-Specific Hyperparameters (Optimized Per-Architecture)

**Allowed to vary (optimized by CMA-ES or grid search):**
- Depth (number of layers)
- Hidden dimension
- Number of attention heads (Transformer)
- Number of recurrent heads (E88)
- State dimension (SSMs, E88)
- Expansion factor (FFN size relative to model dim)
- Specific architecture parameters:
  - E88: n_state, n_heads, decay rates
  - Mamba2: d_state, d_conv, expand
  - Transformer: n_heads, qk_dim, v_dim

**Rationale:** Different architectures achieve capacity through different mechanisms. Forcing them into the same configuration (e.g., same depth, same hidden dim) unfairly advantages one architecture.

### 3.3 What NOT to Vary

**Never vary between architectures:**
- Learning rate (unless architecture fundamentally requires different LR, then report this explicitly)
- Weight initialization scheme (use same init for all)
- Dataset or preprocessing
- Training duration (in steps or wall-clock time)
- Loss function or objective

---

## 4. Statistical Significance and Robustness

### 4.1 The Reproducibility Problem

Modern deep learning has inherent randomness:
- Random weight initialization
- Data shuffling
- Dropout (if used)
- GPU non-determinism (float precision, parallel reductions)

**Problem:** Single run can give misleading results. One lucky initialization can make a worse architecture look better.

### 4.2 Monte Carlo Simulation Protocol

**Method (from recent 2025 research):**
1. For each architecture configuration, run N independent trials (N ≥ 10, ideally N = 100)
2. Each trial uses different random seed for initialization and data shuffling
3. Record final validation loss for each trial
4. Compute mean, standard deviation, 95% confidence interval
5. Use statistical tests to compare architectures

**Implementation:**
```python
results = {}
for architecture in [E88, Mamba2, FLA_GDN]:
    losses = []
    for seed in range(100):
        set_seed(seed)
        model = architecture.create(config)
        final_loss = train(model, train_data, val_data)
        losses.append(final_loss)

    results[architecture] = {
        'mean': np.mean(losses),
        'std': np.std(losses),
        'ci_95': scipy.stats.t.interval(0.95, len(losses)-1,
                                        loc=np.mean(losses),
                                        scale=scipy.stats.sem(losses))
    }
```

### 4.3 Statistical Tests for Comparing Architectures

**When to use each test:**

| Test | Use Case | Assumptions | Interpretation |
|------|----------|-------------|----------------|
| **Paired t-test** | Comparing 2 architectures, same seeds | Normal distribution, paired samples | p < 0.05 → significant difference |
| **Friedman test** | Comparing 3+ architectures, same seeds | Non-parametric, repeated measures | Tests if any architecture differs |
| **Wilcoxon signed-rank** | Comparing 2 architectures, non-normal | Non-parametric, paired samples | Robust alternative to t-test |
| **McNemar's test** | Classification accuracy (e.g., parity) | Binary outcomes, matched samples | Tests if accuracy differs |

**Recommended Protocol:**
1. Use **paired t-test** for comparing 2 architectures on continuous metrics (loss, perplexity)
2. Use **Friedman test** for comparing 3+ architectures, followed by post-hoc Nemenyi test
3. Report **effect size** (Cohen's d) in addition to p-value
4. Use **95% confidence intervals** in all plots and tables

**Example Statistical Report:**
```
E88 vs. Mamba2 on Running Parity (T=1024):
  E88:     mean=98.7%, std=0.3%, CI=[98.4%, 99.0%]
  Mamba2:  mean=50.2%, std=1.1%, CI=[49.9%, 50.5%]
  Paired t-test: t(99)=127.3, p<0.001, Cohen's d=12.7
  Conclusion: E88 significantly outperforms Mamba2 with large effect size.
```

### 4.4 Reporting Statistical Rigor

**In the paper, report:**
1. Number of random seeds used (N = ?)
2. Mean ± standard deviation for all metrics
3. 95% confidence intervals (as error bars in plots, ranges in tables)
4. Statistical test used, test statistic, p-value, effect size
5. Whether differences are statistically significant (p < 0.05)

**Example phrasing:**
> "We trained each architecture with 30 random seeds. E88 achieved mean perplexity 15.2 ± 0.3 (95% CI: [14.9, 15.5]), while Mamba2 achieved 14.8 ± 0.2 (95% CI: [14.6, 15.0]). A paired t-test showed Mamba2 significantly outperforms E88 (t(29)=4.2, p<0.001, d=0.77)."

---

## 5. Presenting Results: Tables, Plots, and Learning Curves

### 5.1 Main Results Table (from Section 11 experiments)

**Best practices:**
- Sort by primary metric (e.g., validation loss)
- Include parameter count for each architecture
- Report mean ± std (or 95% CI)
- Highlight best result in bold
- Include key hyperparameters (depth, hidden_dim)

**Example:**
```typst
#simpletable(
  columns: 7,
  align: (left, center, center, center, center, center, center),
  [*Architecture*], [*Loss*], [*Depth*], [*Hidden*], [*Params*], [*Iters*], [*Significance*],
  [**Mamba2**], [**1.271 ± 0.012**], [25], [1792], [494M], [120], [—],
  [FLA-GDN], [1.273 ± 0.015], [17], [1920], [502M], [120], [p=0.42],
  [E88], [1.407 ± 0.019], [23], [3840], [488M], [120], [p<0.001],
  [Transformer], [1.505 ± 0.023], [13], [1536], [491M], [120], [p<0.001],
)
```

**Key elements:**
1. Primary metric with uncertainty (mean ± std or CI)
2. Parameter count (to verify fair comparison)
3. Best hyperparameters found (depth, hidden_dim)
4. Number of CMA-ES iterations (shows search effort)
5. Statistical significance vs. best model

### 5.2 Learning Curves

**What to plot:**
- X-axis: Training steps (or wall-clock time, or tokens seen)
- Y-axis: Validation loss (or accuracy for classification)
- One curve per architecture
- Shaded region: 95% confidence interval across random seeds

**Best practices:**
1. Use **validation loss**, not training loss (avoids overfitting artifacts)
2. Plot **mean with shaded CI** (shows both performance and variance)
3. Same x-axis range for all architectures (fair comparison)
4. Log scale for x-axis if training is long
5. Include final performance in legend

**Example code (conceptual):**
```python
for arch in architectures:
    steps = arch.steps
    val_losses = np.array([run.val_loss for run in arch.runs])  # shape: (n_seeds, n_steps)
    mean = val_losses.mean(axis=0)
    ci_lower, ci_upper = np.percentile(val_losses, [2.5, 97.5], axis=0)

    plt.plot(steps, mean, label=f"{arch.name} (final: {mean[-1]:.3f})")
    plt.fill_between(steps, ci_lower, ci_upper, alpha=0.3)
```

**When learning curves matter:**
- Showing convergence speed (some architectures learn faster)
- Showing stability (narrow CI = stable, wide CI = high variance)
- Showing overfitting (train-val gap)

### 5.3 Ablation Studies

**Purpose:** Isolate the contribution of specific architectural components.

**Example (from Section 11):**
> "We ablated E88's temporal nonlinearity by replacing tanh(αS + δk^T) with linear αS + δk^T. The ablated model matched Mamba2's performance, confirming that temporal nonlinearity is the key expressivity advantage."

**Recommended ablations:**
- E88 with/without tanh → tests temporal nonlinearity
- E88 with/without multi-head → tests head diversity
- Mamba2 with/without input-dependent gates → tests selectivity

**Table format:**
```typst
#simpletable(
  columns: 4,
  align: (left, center, center, center),
  [*Configuration*], [*Loss*], [*Δ from Full*], [*Interpretation*],
  [E88 (full)], [1.407], [—], [Baseline],
  [E88 (no tanh)], [1.531], [+0.124], [Temporal nonlinearity matters],
  [E88 (single head)], [1.683], [+0.276], [Multi-head helps],
  [E88 (no gate)], [1.812], [+0.405], [Gating is critical],
)
```

### 5.4 Expressivity Benchmark Results

**For running parity, threshold, XOR:**
- X-axis: Sequence length T
- Y-axis: Per-timestep accuracy (%)
- One curve per architecture
- Expected: E88 flat at ~99%, Mamba2 degrades as T increases

**Table format:**
```typst
#simpletable(
  columns: 6,
  align: (left, center, center, center, center, center),
  [*Architecture*], [*T=64*], [*T=128*], [*T=256*], [*T=512*], [*T=1024*],
  [E88 (1-layer)], [99.1%], [98.9%], [99.0%], [98.8%], [98.7%],
  [Mamba2 (32-layer)], [92.1%], [78.3%], [63.2%], [55.1%], [50.2%],
  [Linear baseline], [50.1%], [50.0%], [50.2%], [49.9%], [50.0%],
)
```

**Interpretation:**
- E88 maintains high accuracy across all T (as theory predicts)
- Mamba2 degrades as T increases (depth D=32 insufficient for T>512)
- Linear baseline at chance (50%) confirms task requires nonlinearity

---

## 6. What Reviewers Expect (ICLR, NeurIPS, ICML)

Based on recent 2025 conference reviews and guidelines:

### 6.1 Mandatory Components

**Abstract and Introduction:**
- [ ] Clearly state contributions
- [ ] State assumptions and limitations
- [ ] Match claims to experimental results (no overclaiming)

**Experimental Section:**
- [ ] Fair comparison: parameter counts reported and matched
- [ ] Hyperparameter search documented (what was tuned, what was fixed)
- [ ] Statistical significance testing (t-test, confidence intervals)
- [ ] Multiple random seeds (report N, mean ± std, CI)
- [ ] Ablation studies isolating key components

**Reproducibility:**
- [ ] Compute requirements (GPU hours, hardware used)
- [ ] Hyperparameters for all experiments (in main text or appendix)
- [ ] Dataset details (size, splits, preprocessing)
- [ ] Code release (or clear algorithmic description)

**Theoretical Claims:**
- [ ] Experimental validation of theoretical predictions
- [ ] Clear distinction between proven theorems and conjectures
- [ ] Discussion of theory-practice gaps (when results don't match theory)

### 6.2 Common Criticisms to Avoid

**Experimental Weaknesses (from actual NeurIPS reviews):**
- ❌ "Search space substantially more restricted than claimed" → Use real hyperparameter search
- ❌ "No ablation studies for design choices" → Ablate key components
- ❌ "Task difficulty not controlled" → Use standardized benchmarks
- ❌ "Single random seed" → Use multiple seeds with statistical testing
- ❌ "No comparison with SOTA" → Include recent baselines

**Methodological Flaws:**
- ❌ "Unfair comparison due to different training budgets" → Match compute/time/steps
- ❌ "Cherry-picked results" → Report all experiments, including negative results
- ❌ "No statistical testing" → Use t-tests, report p-values, CI

**Writing Issues:**
- ❌ "Claims don't match results" → Be precise about what was shown
- ❌ "Important details missing" → Provide full hyperparameters, compute cost

### 6.3 Checklist for Expressivity Comparison Papers

**Theory:**
- [ ] Formal definition of expressivity (what can be computed)
- [ ] Proven separation theorem (with reference to formalization)
- [ ] Clear statement of assumptions (e.g., "linear-temporal dynamics")

**Experiments:**
- [ ] Expressivity benchmarks (parity, threshold, XOR) to validate theory
- [ ] Practical benchmarks (language modeling) to measure real-world performance
- [ ] Both should be included; gap between them is informative

**Fair Comparison:**
- [ ] Parameter matching via CMA-ES or grid search
- [ ] Same training time / compute budget / dataset
- [ ] Hyperparameter search documented and reproducible

**Statistical Rigor:**
- [ ] Multiple random seeds (N ≥ 10)
- [ ] Mean ± std or 95% CI for all metrics
- [ ] Statistical tests (t-test, Friedman) with p-values
- [ ] Effect sizes (Cohen's d) reported

**Presentation:**
- [ ] Main results table with uncertainty
- [ ] Learning curves with confidence intervals
- [ ] Ablation studies with clear interpretations
- [ ] Theory-practice gap discussion (when theory doesn't match experiments)

---

## 7. Theory-Practice Gap: When Results Don't Match Predictions

### 7.1 Expected vs. Observed

**From Section 11 experiments:**

| Property | Theory Predicts | Empirical Observation |
|----------|-----------------|----------------------|
| Running parity | E88 > Mamba2 | ✓ Mamba2 stuck at 50% |
| Language modeling | E88 ≥ Mamba2 | ✗ Mamba2 > E88 (1.27 vs 1.41) |

**The gap is information, not a flaw.**

### 7.2 How to Present the Gap

**Key insight from Section 11:**
> "Expressivity determines what can be computed with unlimited resources. Benchmark performance measures what is learned in fixed time. The gap between them is not a flaw in the theory—it is information about the task."

**Discuss:**
1. **Why theory predicts X:** E88 has temporal nonlinearity, can compute parity
2. **Why experiments show Y:** Mamba2's parallel scan achieves 4× throughput, sees 4× more examples
3. **Reconciliation:** Both are true; which matters depends on task and resources

**Example phrasing:**
> "On language modeling, Mamba2 outperformed E88 despite E88's greater expressivity. This does not contradict the separation theorem. Rather, it indicates that (1) natural language does not require temporal nonlinearity, or (2) the benchmark averages over rare cases where it matters, or (3) expressivity ≠ learnability. The expressivity advantage manifests on running parity, where Mamba2 cannot converge regardless of training budget."

### 7.3 When to Expect Theory to Predict Practice

**Theory should match practice when:**
- Task is expressivity-critical (parity, threshold, state machines)
- Training budget is not the bottleneck
- Task is synthetic and well-defined (not natural language)

**Theory may not match practice when:**
- Task is within both architectures' capabilities (both can approximate)
- Training efficiency dominates (throughput matters more than expressivity)
- Task is complex and heterogeneous (language modeling mixes many subtasks)

---

## 8. Summary: Best Practices

### 8.1 Parameter Matching
✓ Use CMA-ES to optimize hyperparameters independently per architecture
✓ Constrain to similar parameter counts (±5%)
✓ Allow depth, width, heads, state dimensions to vary
✓ Fix learning rate, batch size, training time

### 8.2 Benchmarks
✓ Include expressivity-critical tasks (parity, threshold) to test theory
✓ Include practical tasks (language modeling) to measure real-world utility
✓ Report both; gap between them is informative

### 8.3 Statistical Rigor
✓ Multiple random seeds (N ≥ 10, ideally 100)
✓ Report mean ± std and 95% CI
✓ Use paired t-test or Friedman test
✓ Report p-values and effect sizes (Cohen's d)

### 8.4 Presentation
✓ Tables: mean ± std, parameter counts, hyperparameters, significance
✓ Learning curves: mean with shaded CI across seeds
✓ Ablations: isolate key components, interpret each ablation
✓ Expressivity plots: accuracy vs. sequence length T

### 8.5 Reproducibility
✓ Report compute cost (GPU hours, hardware)
✓ Provide all hyperparameters (main text or appendix)
✓ Document dataset, preprocessing, splits
✓ Release code or provide algorithmic pseudocode

### 8.6 Theory-Practice Gap
✓ Discuss when results don't match predictions
✓ Explain why both are valid (different resources, different tasks)
✓ Use gap to gain insight into what tasks require

---

## 9. References and Further Reading

### Key Sources

**2025 Research on Fair Comparison:**
- [Empirical Comparison of Neural Network Architectures - MDPI](https://www.mdpi.com/2504-3110/9/11/702) - Common hyperparameters and consistent experimental conditions
- [Performance analysis of RNN, LSTM, GRU, and hybrid models - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12329085/) - Grid search for parameter matching
- [A framework for measuring training efficiency - arXiv](https://arxiv.org/html/2409.07925v1) - Multi-metric evaluation including energy and time

**Statistical Significance:**
- [Evaluation metrics and statistical tests for ML - Nature](https://www.nature.com/articles/s41598-024-56706-x) - Statistical tests for ML comparison
- [Statistical Significance Tests for ML - MachineLearningMastery](https://machinelearningmastery.com/statistical-significance-tests-for-comparing-machine-learning-algorithms/) - Paired t-test, McNemar's test, Friedman test
- [Neural Architecture Search Benchmarks - IEEE](https://ieeexplore.ieee.org/ielaam/6287639/10005208/10063950-aam.pdf) - Statistical rigor in NAS

**Presenting Results:**
- [How to use Learning Curves - MachineLearningMastery](https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/) - Learning curve best practices
- [Recommendations for reporting ML analyses - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8320533/) - Reporting standards for clinical ML
- [Validation curves - scikit-learn](https://scikit-learn.org/stable/modules/learning_curve.html) - Plotting learning curves with CI

**Conference Review Expectations:**
- [NeurIPS Paper Checklist Guidelines](https://neurips.cc/public/guides/PaperChecklist) - Official NeurIPS requirements
- [NeurIPS Review Examples](https://proceedings.neurips.cc/paper_files/paper/2020/file/8c53d30ad023ce50140181f713059ddf-Review.html) - Actual reviews showing expectations
- [Insights from ICLR Peer Review - arXiv](https://arxiv.org/pdf/2511.15462) - Analysis of review process

**Expressivity Benchmarks:**
- [Attention Learning for Parity Function - arXiv](https://arxiv.org/html/2502.07553v1) - Recent 2025 work on parity as benchmark
- [Are Efficient Deep Representations Learnable? - ICLR 2018](https://arxiv.org/pdf/1807.06399) - Parity function benchmark
- [RNNs Are Not Transformers (Yet) - ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/79dc391a2c1067e9ac2b764e31a60377-Paper-Conference.pdf) - Comparing RNNs and Transformers

**Architecture Comparisons:**
- [Transformer vs RNN Comparative Study - IRJMETS 2025](https://www.irjmets.com/uploadedfiles/paper//issue_4_april_2025/73761/final/fin_irjmets1745500376.pdf) - Recent 2025 RNN vs Transformer methodology
- [Comparative analysis of LSTM, GRU, Transformer - Taylor & Francis](https://www.tandfonline.com/doi/full/10.1080/13467581.2025.2455034) - Multi-architecture comparison 2025

### Internal Documents

**Theoretical Foundations:**
- `ElmanProofs/Expressivity/LinearLimitations.lean` - Proven: linear RNNs cannot compute threshold/XOR
- `ElmanProofs/Expressivity/MultiLayerLimitations.lean` - Multi-layer extension
- `ElmanProofs/Architectures/RecurrenceLinearity.lean` - Architecture classification (linear vs nonlinear in h)
- `OPEN_QUESTIONS_RESOLUTION.md` - Analysis of temporal nonlinearity vs. depth

**Experimental Context:**
- `docs/expressivity/11-experimental-results.typ` - Section 11 experiments (CMA-ES results)
- `x_gated_elman_benchmark.md` - E88 gating ablations and throughput comparison
- `docs/expressivity/BIBLIOGRAPHY_RESEARCH.md` - Citation and bibliography methodology

---

## 10. Appendix: Example Experimental Protocol

### 10.1 Running Parity Experiment (Full Protocol)

**Objective:** Validate that E88 can compute running parity while Mamba2 cannot, as predicted by LinearLimitations.lean theorem.

**Dataset:**
- Task: y_t = x_1 ⊕ x_2 ⊕ ... ⊕ x_t (running XOR)
- Input: Binary sequences x ∈ {0,1}^T
- Output: Binary labels y ∈ {0,1}^T
- Sequence lengths: T ∈ {64, 128, 256, 512, 1024, 2048}
- Train: 50k sequences per T
- Val: 10k sequences per T
- Test: 10k sequences per T

**Models:**
1. E88 (1-layer): n_heads=8, n_state=16, hidden_dim=256 (~5M params)
2. Mamba2 (32-layer): d_state=96, hidden_dim=256, depth=32 (~80M params)
3. Linear baseline: h_t = Ah_{t-1} + Bx_t, y_t = Ch_t

**Training:**
- Optimizer: AdamW (lr=3e-4, betas=(0.9, 0.999), weight_decay=0.01)
- Loss: Binary cross-entropy (per-timestep, averaged over T)
- Batch size: 32
- Steps: 10k steps
- Random seeds: 30 seeds per architecture

**Evaluation:**
- Metric: Per-timestep accuracy (averaged over all t ∈ [1, T])
- Report: Mean ± std across 30 seeds, 95% CI
- Statistical test: Paired t-test comparing E88 vs. Mamba2

**Expected Results:**
- E88: ~99% accuracy across all T (theory: can compute parity)
- Mamba2: degrades from ~92% (T=64) to ~50% (T=2048) (theory: depth D=32 insufficient for large T)
- Linear: ~50% accuracy (theory: cannot compute XOR)

**Compute:**
- Hardware: 1× NVIDIA A100 80GB
- Time: ~2 hours per architecture (30 seeds × 10k steps)
- Total: ~6 GPU-hours

### 10.2 Language Modeling Experiment (Full Protocol)

**Objective:** Compare E88 vs. Mamba2 on practical language modeling task.

**Dataset:**
- Corpus: WikiText-103 or OpenWebText (standardized)
- Preprocessing: BPE tokenization (GPT-2 tokenizer, 50k vocab)
- Context: 1024 tokens
- Train: ~100M tokens
- Val: 217k tokens
- Test: 245k tokens

**Models:**
- Parameter budget: 350M ± 5%
- CMA-ES search space (for each architecture):
  - Depth: [12, 48]
  - Hidden dim: [512, 2048]
  - E88: n_heads [8, 128], n_state [8, 64]
  - Mamba2: d_state [64, 256], expand [1, 4]
- CMA-ES iterations: 120 evaluations per architecture
- Per-config training: 10 minutes at lr=3e-4

**Training (final configs):**
- Optimizer: AdamW (lr=3e-4 with cosine decay, warmup=2k steps)
- Batch size: 8 sequences × 128 tokens = 1024 tokens/batch
- Gradient accumulation: 4 steps → effective batch = 4096 tokens
- Steps: 100k steps
- Random seeds: 10 seeds per architecture

**Evaluation:**
- Metric: Validation perplexity (exp(cross_entropy_loss))
- Report: Mean ± std across 10 seeds, 95% CI
- Statistical test: Paired t-test
- Throughput: tokens/second (measured on same hardware)

**Expected Results (from Section 11):**
- Mamba2: ~15.2 perplexity (theory: linear-temporal may suffice for language)
- E88: ~17.8 perplexity (theory: expressivity doesn't compensate for 4× slower throughput)

**Compute:**
- Hardware: 1× NVIDIA A100 80GB
- Time per seed: ~24 hours (100k steps)
- Total: 10 seeds × 24h × 2 architectures = 480 GPU-hours

---

## Conclusion

Fair comparison of neural architectures requires careful experimental design, statistical rigor, and honest reporting of theory-practice gaps. Use CMA-ES for hyperparameter optimization, test on both expressivity-critical and practical benchmarks, employ multiple random seeds with statistical testing, and present results with confidence intervals. The methodology in this document reflects best practices from 2025 research and expectations of top-tier ML conferences.

**The central lesson:** Expressivity is one factor among many. Trainability, throughput, initialization, and optimization dynamics all contribute to final performance. Theory tells us what is possible; experiments tell us what happens. Both perspectives are essential.
