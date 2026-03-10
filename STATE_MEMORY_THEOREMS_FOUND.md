# Existing State/Memory Formalisms in Lean Codebase

## Summary
This document lists all relevant theorems found in our Lean codebase about state capacity, memory retention, information decay, matrix state accessibility, and the E88 S path (matmul self).

---

## 1. State Decay and Exponential Forgetting

### Linear State Decay (α^t decay in linear models)

**ElmanProofs/Expressivity/LinearCapacity.lean:72-110**
- `linear_state_is_sum`: State at time T is sum of weighted inputs: h_T = Σ_{t=0}^{T-1} A^{T-1-t} B x_t
  - Proves linear RNN state is just a weighted sum with weights determined by powers of A
  - File:line = `ElmanProofs/Expressivity/LinearCapacity.lean:72`

**ElmanProofs/Expressivity/BinaryFactRetention.lean:160-177**
- `linear_contribution_decays`: In linear temporal system, contribution of input at time t to state at time T decays as α^{T-t} where 0 < α < 1
  - File:line = `ElmanProofs/Expressivity/BinaryFactRetention.lean:160`

- `linear_info_vanishes`: Information from input at time t vanishes as T → ∞: Tendsto (fun T : ℕ => α ^ T) atTop (nhds 0)
  - File:line = `ElmanProofs/Expressivity/BinaryFactRetention.lean:172`

- `linear_no_fixed_point`: Linear systems have no non-trivial fixed points; state decays to 0
  - For h_{t+1} = α·h_t with α < 1, proves ∃ T : ℕ, |α ^ T * h₀| < |h₀| / 2
  - File:line = `ElmanProofs/Expressivity/BinaryFactRetention.lean:180`

**ElmanProofs/Expressivity/E23vsE88Comparison.lean:192-226**
- `linear_state_decays`: Tendsto (fun T : ℕ => α ^ T) atTop (nhds 0) for 0 < α < 1
  - File:line = `ElmanProofs/Expressivity/E23vsE88Comparison.lean:192`

- `binary_retention_gap`: Formal statement that E88 has zero fixed point while linear SSMs must decay exponentially
  - Linear SSMs: ∀ S₀ ≠ 0, Tendsto (fun T => α ^ T * S₀) atTop (nhds 0)
  - File:line = `ElmanProofs/Expressivity/E23vsE88Comparison.lean:203`

### Linear SSM Decay (Mamba2)

**ElmanProofs/Expressivity/BinaryFactRetention.lean:262-280**
- `linearSSM_decays_without_input`: Linear SSM cannot retain a binary fact without decay
  - For state.α < 1, proves ∀ ε > 0, ∃ T : ℕ, |state.α ^ T * state.h| < ε
  - File:line = `ElmanProofs/Expressivity/BinaryFactRetention.lean:262`

- `binary_fact_retention_gap`: The retention gap between E88 (latching) and linear SSMs (exponential decay)
  - File:line = `ElmanProofs/Expressivity/BinaryFactRetention.lean:285`

---

## 2. Tanh Saturation and State Latching (E88 S path)

### Tanh Saturation Creates Stable Fixed Points

**ElmanProofs/Expressivity/BinaryFactRetention.lean:86-122**
- `tanh_derivative_saturation`: As |x| → ∞, deriv tanh x = 1 - tanh²(x) → 0
  - Proves ∀ ε > 0, ∃ c > 0, ∀ x, c < |x| → 1 - (tanh x)² < ε
  - File:line = `ElmanProofs/Expressivity/BinaryFactRetention.lean:86`

- `tanh_approaches_one_at_infinity`: For large S, tanh(S) is close to 1
  - Proves ∀ ε > 0, ∃ c > 0, ∀ S > c, 1 - ε < tanh S
  - File:line = `ElmanProofs/Expressivity/BinaryFactRetention.lean:130`

**ElmanProofs/Expressivity/TanhSaturation.lean:86-89**
- `tanh_derivative_vanishes`: The derivative of tanh vanishes as |x| → ∞
  - File:line = `ElmanProofs/Expressivity/TanhSaturation.lean:86`

**ElmanProofs/Expressivity/E23vsE88Comparison.lean:142-188**
- `e88_tanh_saturation_creates_stability`: When |S| approaches 1, tanh'(S) → 0, creating stable fixed points
  - File:line = `ElmanProofs/Expressivity/E23vsE88Comparison.lean:142`

- `near_saturation_small_gradient`: For states near ±1, gradient of tanh is small (< 0.2)
  - File:line = `ElmanProofs/Expressivity/E23vsE88Comparison.lean:148`

- `e88_latched_state_bounded_from_zero`: E88 latched state persists - when α * |S₀.S| ≥ 1, next state stays bounded away from 0
  - Proves |tanh(α·S)| > θ * |S| - 0.1 for θ = 0.81
  - File:line = `ElmanProofs/Expressivity/E23vsE88Comparison.lean:171`

### E88 State Retention

**ElmanProofs/Expressivity/BinaryFactRetention.lean:226-259**
- `e88_retains_large_state`: E88 can retain a latched state when S is large
  - For α ≥ 1, S > 1, proves tanh(α * S) > 1/2
  - File:line = `ElmanProofs/Expressivity/BinaryFactRetention.lean:226`

**ElmanProofs/Expressivity/BinaryFactRetention.lean:502-599**
- `e88_nonlinear_distinguishes`: E88's nonlinear tanh enables distinguishing inputs that linear systems cannot
  - Proves tanh(S) ≠ S for all S ≠ 0 (demonstrates genuine nonlinearity)
  - File:line = `ElmanProofs/Expressivity/BinaryFactRetention.lean:502`

- `e88_vs_mamba2_retention_fundamental`: The fundamental retention gap theorem
  - Combines saturation, no-fixed-point for linear, and parity non-affine
  - File:line = `ElmanProofs/Expressivity/BinaryFactRetention.lean:617`

### Alert State Persistence

**ElmanProofs/Expressivity/AttentionPersistence.lean:80-100**
- `tanhRecur_bounded`: tanhRecur preserves bounded interval (-1, 1)
  - File:line = `ElmanProofs/Expressivity/AttentionPersistence.lean:92`

- `tanhRecur_strictMono`: tanhRecur is strictly monotone for α > 0
  - File:line = `ElmanProofs/Expressivity/AttentionPersistence.lean:98`

**ElmanProofs/Expressivity/E23vsE88Comparison.lean:489-525**
- `e88_alert_persistence`: E88 head can enter "alert" state and maintain it
  - When α > 0.9 and |S₀| > 0.95 and α|S₀| ≥ 1, proves |tanh(α·S₀)| > 0.7
  - File:line = `ElmanProofs/Expressivity/E23vsE88Comparison.lean:489`

---

## 3. Matrix State Capacity and Accessibility

### Matrix State RNN Architecture

**ElmanProofs/Architectures/MatrixStateRNN.lean:86-117**
- `matrix_state_matches_weight_capacity`: Matrix state with k = d has d² dynamic parameters, same as weight matrix in standard RNN
  - File:line = `ElmanProofs/Architectures/MatrixStateRNN.lean:86`

- `matrix_rnn_same_cost`: Matrix RNN with k = d has same asymptotic cost as vector RNN
  - File:line = `ElmanProofs/Architectures/MatrixStateRNN.lean:105`

- `matrix_rnn_more_state`: Matrix RNN has d× more state for same cost
  - File:line = `ElmanProofs/Architectures/MatrixStateRNN.lean:112`

**ElmanProofs/Architectures/MatrixStateRNN.lean:156-194**
- `matrix_more_capacity`: Matrix state has k× more effective capacity than vector state
  - File:line = `ElmanProofs/Architectures/MatrixStateRNN.lean:164`

- `accumulated_rank`: Rank of accumulated outer products after T steps is min(T, d, k)
  - File:line = `ElmanProofs/Architectures/MatrixStateRNN.lean:187`

- `full_rank_possible`: After d steps, matrix state can have full rank
  - File:line = `ElmanProofs/Architectures/MatrixStateRNN.lean:190`

### Information Capacity Bounds

**ElmanProofs/Expressivity/LinearCapacity.lean:220-227**
- `reachable_dim_bound`: Dimension of reachable states is at most n (state dimension)
  - Module.finrank ℝ (Submodule.span ℝ (reachableStates A B T)) ≤ n
  - File:line = `ElmanProofs/Expressivity/LinearCapacity.lean:220`

**ElmanProofs/Expressivity/LinearCapacity.lean:186-218**
- `reachable_is_subspace`: Reachable states form a subspace (closed under addition/scaling)
  - File:line = `ElmanProofs/Expressivity/LinearCapacity.lean:190`

**ElmanProofs/Expressivity/LinearCapacity.lean:152-182**
- `output_determined_by_state`: Output depends only on current state
  - File:line = `ElmanProofs/Expressivity/LinearCapacity.lean:152`

- `same_state_same_future`: Same state → same future outputs (indistinguishability)
  - File:line = `ElmanProofs/Expressivity/LinearCapacity.lean:164`

---

## 4. E88 Matrix State S := tanh(αS + δk^T) Formalization

### E88 Multi-Head State Evolution

**ElmanProofs/Expressivity/E88MultiPass.lean:68-100**
- `e88ScalarUpdate`: Core E88 recurrence S' = tanh(α·S + δ·input)
  - File:line = `ElmanProofs/Expressivity/E88MultiPass.lean:71`

- `e88HeadState`: State of single E88 head after T steps (T-fold nested tanh composition)
  - File:line = `ElmanProofs/Expressivity/E88MultiPass.lean:92`

- `e88MultiHeadState`: Multi-head E88 state where each head runs independently
  - File:line = `ElmanProofs/Expressivity/E88MultiPass.lean:97`

**ElmanProofs/Expressivity/E23vsE88Comparison.lean:91-108**
- `E88HeadState`: Structure representing E88 single-head state with scalar S
  - File:line = `ElmanProofs/Expressivity/E23vsE88Comparison.lean:93`

- `E88State`: Multi-head state structure
  - File:line = `ElmanProofs/Expressivity/E23vsE88Comparison.lean:98`

- `e88HeadUpdate`: S' = tanh(α·S + δ·input)
  - File:line = `ElmanProofs/Expressivity/E23vsE88Comparison.lean:106`

### E88 Head Independence

**ElmanProofs/Expressivity/E23vsE88Comparison.lean:456-481**
- `e88_head_independence`: E88 heads are independent parallel state machines
  - Changing head h₂'s state does not affect head h₁'s evolution
  - File:line = `ElmanProofs/Expressivity/E23vsE88Comparison.lean:456`

- `e88_multihead_parallel_latching`: H heads can track H independent binary facts
  - File:line = `ElmanProofs/Expressivity/E23vsE88Comparison.lean:465`

---

## 5. Architectural Comparisons: State Decay Characteristics

### E23 vs E88 Memory Mechanisms

**ElmanProofs/Expressivity/E23vsE88Comparison.lean:556-591**
- `decay_characteristics`: E23 tape has no decay; E88 has slow decay requiring refresh
  - E23 tape: no automatic decay (attn = 0 → state unchanged)
  - E88: decays without input (for α < 1), |tanh(α·S)| < |S|
  - File:line = `ElmanProofs/Expressivity/E23vsE88Comparison.lean:559`

**ElmanProofs/Expressivity/E23vsE88Comparison.lean:534-555**
- `e88StateSize`: E88 state size for H heads with d×d state per head = H * d * d
  - File:line = `ElmanProofs/Expressivity/E23vsE88Comparison.lean:533`

- `e23StateSize`: E23 state size = N * D + D (N slots + working memory)
  - File:line = `ElmanProofs/Expressivity/E23vsE88Comparison.lean:537`

- `e88_efficient_for_single_fact`: For single binary fact, E88 more efficient (1 param vs 2 params)
  - File:line = `ElmanProofs/Expressivity/E23vsE88Comparison.lean:542`

- `storage_scaling`: For N facts, both are O(N)
  - File:line = `ElmanProofs/Expressivity/E23vsE88Comparison.lean:550`

### Mamba2 SSM Linear Dynamics

**ElmanProofs/Architectures/Mamba2_SSM.lean:99-104**
- `ssm_is_linear`: SSM is LINEAR in the state (Jacobian dh_new/dh = A is constant)
  - File:line = `ElmanProofs/Architectures/Mamba2_SSM.lean:101`

**ElmanProofs/Architectures/RecurrenceLinearity.lean:299-313**
- `minGRU_is_linear_bounded`: MinGRU's linear-in-h property means composition depth = 1
  - File:line = `ElmanProofs/Architectures/RecurrenceLinearity.lean:300`

- `e1_is_nonlinear_deep`: E1's nonlinear-in-h property means composition depth = seq_len
  - File:line = `ElmanProofs/Architectures/RecurrenceLinearity.lean:308`

---

## 6. Computational Implications: What Linear vs Nonlinear Can/Cannot Compute

### Functions Linear Models Cannot Compute

**ElmanProofs/Expressivity/LinearLimitations.lean:105-205**
- `linear_cannot_threshold`: Linear RNNs cannot compute threshold/step functions
  - File:line = `ElmanProofs/Expressivity/LinearLimitations.lean:107`

- `linear_cannot_xor`: Linear RNNs cannot compute XOR over history
  - File:line = `ElmanProofs/Expressivity/LinearLimitations.lean:314`

**ElmanProofs/Expressivity/E23vsE88Comparison.lean:240-263**
- `linear_cannot_running_threshold`: Linear temporal systems cannot compute running threshold
  - File:line = `ElmanProofs/Expressivity/E23vsE88Comparison.lean:241`

**ElmanProofs/Expressivity/E23vsE88Comparison.lean:267-451**
- `running_parity_not_linear`: Running parity is not linearly computable
  - parity(x_1,...,x_t) requires nonlinearity that compounds across timesteps
  - File:line = `ElmanProofs/Expressivity/E23vsE88Comparison.lean:275`

**ElmanProofs/Expressivity/BinaryFactRetention.lean:311-436**
- `parity_not_affine`: Parity of n bits is not an affine function
  - File:line = `ElmanProofs/Expressivity/BinaryFactRetention.lean:311`

- `running_parity_not_linear`: Running parity cannot be computed by linear RNN
  - File:line = `ElmanProofs/Expressivity/BinaryFactRetention.lean:438`

### Multi-Layer Linear Limitations

**ElmanProofs/Expressivity/MultiLayerLimitations.lean:231-283**
- `multilayer_cannot_running_threshold`: D-layer model with linear temporal dynamics cannot compute running threshold
  - Holds regardless of depth D, nonlinear activations between layers, or parameters
  - File:line = `ElmanProofs/Expressivity/MultiLayerLimitations.lean:231`

- `multilayer_cannot_threshold`: Multi-layer linear-temporal cannot compute threshold function
  - File:line = `ElmanProofs/Expressivity/MultiLayerLimitations.lean:286`

---

## 7. Key Theorems Summary by Category

### State Decay (α^t in linear models):
1. `linear_state_is_sum` - LinearCapacity.lean:72
2. `linear_contribution_decays` - BinaryFactRetention.lean:160
3. `linear_info_vanishes` - BinaryFactRetention.lean:172
4. `linear_no_fixed_point` - BinaryFactRetention.lean:180
5. `linearSSM_decays_without_input` - BinaryFactRetention.lean:262

### Information Retention Over Time:
1. `binary_retention_gap` - E23vsE88Comparison.lean:203
2. `binary_fact_retention_gap` - BinaryFactRetention.lean:285
3. `e88_vs_mamba2_retention_fundamental` - BinaryFactRetention.lean:617
4. `decay_characteristics` - E23vsE88Comparison.lean:559

### Matrix Rank and Capacity:
1. `reachable_dim_bound` - LinearCapacity.lean:220
2. `matrix_state_matches_weight_capacity` - MatrixStateRNN.lean:86
3. `matrix_more_capacity` - MatrixStateRNN.lean:164
4. `accumulated_rank` - MatrixStateRNN.lean:187

### Accessibility of Stored Information:
1. `output_determined_by_state` - LinearCapacity.lean:152
2. `same_state_same_future` - LinearCapacity.lean:164
3. `reachable_is_subspace` - LinearCapacity.lean:190

### E88 S Path (Matmul Self) - tanh(αS + δk^T):
1. `e88ScalarUpdate` - E88MultiPass.lean:71
2. `e88HeadUpdate` - E23vsE88Comparison.lean:106
3. `e88_tanh_saturation_creates_stability` - E23vsE88Comparison.lean:142
4. `e88_latched_state_bounded_from_zero` - E23vsE88Comparison.lean:171
5. `e88_retains_large_state` - BinaryFactRetention.lean:226
6. `e88_alert_persistence` - E23vsE88Comparison.lean:489
7. `e88_head_independence` - E23vsE88Comparison.lean:456
8. `tanh_derivative_saturation` - BinaryFactRetention.lean:86
9. `tanh_derivative_vanishes` - TanhSaturation.lean:86

---

## Files Searched

### Core Files Analyzed:
- `ElmanProofs/Expressivity/LinearCapacity.lean` - Linear state capacity and reachability
- `ElmanProofs/Expressivity/LinearLimitations.lean` - What linear RNNs cannot compute
- `ElmanProofs/Expressivity/BinaryFactRetention.lean` - E88 vs Mamba2 retention gap
- `ElmanProofs/Expressivity/E23vsE88Comparison.lean` - Comprehensive architectural comparison
- `ElmanProofs/Expressivity/MultiLayerLimitations.lean` - Multi-layer linear limitations
- `ElmanProofs/Expressivity/TanhSaturation.lean` - Tanh saturation and latching
- `ElmanProofs/Expressivity/AttentionPersistence.lean` - Alert mode persistence
- `ElmanProofs/Expressivity/E88MultiPass.lean` - E88 multi-pass formalization
- `ElmanProofs/Architectures/MatrixStateRNN.lean` - Matrix state capacity
- `ElmanProofs/Architectures/RecurrenceLinearity.lean` - Linear vs nonlinear recurrence
- `ElmanProofs/Architectures/Mamba2_SSM.lean` - Mamba2 SSM formalization

### Architecture Coverage:
- **E88**: Multi-head nonlinear temporal dynamics with tanh saturation
- **Mamba2/SSM**: Linear temporal dynamics with α^t decay
- **E23**: Dual-memory with persistent tape + nonlinear working memory
- **MinGRU/MinLSTM**: Linear-in-h recurrence (composition depth 1)
- **Matrix State RNN**: d×k state for increased capacity

---

## Conclusion

The codebase contains extensive formalizations of state/memory properties:

1. **State Decay**: ~10+ theorems about α^t exponential decay in linear systems
2. **Memory Retention**: ~8+ theorems about E88 latching vs linear decay
3. **Matrix State**: ~6+ theorems about capacity, rank, and accessibility
4. **E88 S Path**: ~9+ theorems about tanh(αS + δk^T) dynamics

All key predictions from the task description are formalized with rigorous proofs.
