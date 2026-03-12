# Temporal Superposition and Its Connections to the E88 Expressivity Hierarchy

**Paper:** "Temporal superposition and feature geometry of RNNs under memory demands"
**Authors:** Pratyaksh Sharma\*, Alexandra Maria Proca\*, Lucas Prieto, Pedro A. M. Mediano
**Venue:** ICLR 2026 (Oral)
**OpenReview:** https://openreview.net/forum?id=7cMzTpbJHC
**Code:** https://github.com/kashparty/iclr-rnn-superposition

---

## 1. What Is Temporal Superposition?

### Core Definition

Temporal superposition extends the concept of superposition from feedforward networks (Elhage et al., "Toy Models of Superposition," 2022) to recurrent architectures by incorporating time as an additional capacity constraint.

In feedforward networks, **spatial superposition** occurs when a network represents more features than it has neurons by encoding features as non-orthogonal directions in activation space. Features overlap, creating interference, but sparse features can coexist because they rarely activate simultaneously.

**Temporal superposition** is the analogous phenomenon along the time axis: when an RNN must remember features across more timesteps than its hidden state has dimensions, it compresses temporal information by representing features non-orthogonally across time, creating interference patterns that encode information temporally rather than purely spatially.

The paper distinguishes two axes of compression:

- **Spatial superposition**: Representing more input features (higher-dimensional data feature space) in a lower-dimensional activation space.
- **Temporal superposition**: Representing features across a longer period of time (higher memory demand) in a lower-dimensional activation space.

These two axes interact: as either memory length or input dimensionality increases relative to hidden state size, more pressure is placed on the representational geometry, and the RNN must make increasingly aggressive tradeoffs.

### Task Setup: Delayed Serial Recall

The framework is studied through a delayed serial recall task. An arbitrary sequence of characters (features) must be stored for a variable delay period, after which a recall cue triggers reproduction of the stored sequence. The variable delay forces the RNN to learn stable distributed representations rather than simple delay lines. This task isolates the temporal axis of superposition by controlling exactly how long and how many features must be remembered.

### How It Is Measured

Temporal superposition is characterized through several geometric and spectral quantities:

1. **Angular distribution of feature embeddings**: Features are represented as directions in activation space. The angles between feature directions reveal the representational strategy. Orthogonal features (90 degrees apart) have no interference. Non-orthogonal features share dimensions (superposition). The distribution of pairwise angles across all feature embeddings characterizes the degree of superposition.

2. **Spectral radius of the recurrence matrix**: The spectral radius of the transition matrix determines how quickly information decays. Lower spectral radius means faster forgetting. Phase transitions in the spectral radius mark regime changes.

3. **Loss decomposition into four interpretable terms**: The paper derives an analytical form of the expected loss on the recall task that decomposes into four terms: task error, mean correction, projection interference, and composition interference. This decomposition explains the learned geometry.

4. **Two types of interference**:
   - **Projection interference**: Reading out a feature activates a different feature (the readout projects one feature's direction onto another's).
   - **Composition interference**: Multiple active features combine linearly to produce an activation that mimics another feature.

---

## 2. Relationship to State Dimensionality

### Can a D-Dimensional State Store More Than D Temporal Features?

Yes, and this is the central finding. The paper shows that:

- **Without superposition** (orthogonal regime): A D-dimensional hidden state can store exactly D features. Each feature gets its own orthogonal direction. This corresponds to classical linear memory capacity results.

- **With temporal superposition** (dense regime): A D-dimensional state can store information about more than D temporal features by representing them non-orthogonally. The price is interference: recalled features are contaminated by other stored features.

- The hidden state is a **continuous limited-capacity resource**. As memory length or number of input features increases, demand on the hidden state grows, features are more likely to interfere, and the RNN must decide which features to represent precisely and which to drop or approximate.

### The Dense-to-Sparse Phase Transition

The paper identifies a sharp phase transition between two regimes:

1. **Dense regime** (high demand relative to capacity):
   - Features are represented non-orthogonally (temporal superposition is active).
   - The RNN behaves "effectively linearly" -- even nonlinear RNNs adopt linear-like strategies.
   - Feature embeddings are spread broadly in angular space.
   - Spectral radius is relatively high (slower forgetting to retain more features).
   - Corresponds to what classical memory capacity theory studies.

2. **Sparse regime** (low demand relative to capacity):
   - Features are temporally sparse (rarely co-activated).
   - The RNN can exploit an **interference-free space**.
   - Feature embeddings cluster into distinct geometric configurations.
   - Spectral radius decreases (sharper, more decisive forgetting).
   - The angular distribution of features shows a sharp transition.

The transition between regimes is controlled by the ratio of temporal demand (how many features must be remembered for how long) to state capacity (hidden dimensionality D).

---

## 3. The Role of Nonlinearity

This is where the paper makes its most important distinction and where the connection to our work is strongest.

### Three Model Classes

The paper studies three model types, which map directly onto our hierarchy:

| Paper's terminology | Our terminology | Recurrence | Readout |
|---|---|---|---|
| Linear RNN | Linear SSM (Mamba2-like) | Linear | Linear |
| State Space Model (SSM) | Linear recurrence + nonlinear readout | Linear | Nonlinear (e.g., ReLU) |
| Nonlinear RNN | E1H / E88 | Nonlinear (tanh) | Nonlinear |

### What Nonlinearity Enables

**The interference-free space.** This is the key mechanism. When a nonlinear activation (ReLU or tanh) is applied at readout, it creates a region of activation space where features can exist without contributing to the output -- and therefore without interfering with features that do contribute. Specifically:

- **ReLU readout** creates an interference-free half-space: features placed in the negative half-space produce zero output and zero interference with the output feature.
- **Tanh readout** creates interference-free regions near saturation: features pushed deep into saturation contribute a fixed value regardless of exact position.

**Sharp vs. smooth forgetting.** The paper's central qualitative finding:

- **Linear RNNs** lack an interference-free space entirely. Old features spiral toward the origin as they decay. Forgetting is smooth and gradual -- information fades continuously. The RNN cannot selectively forget; all past features decay at rates determined by eigenvalues of the transition matrix.

- **SSMs** (linear recurrence + nonlinear readout) can exploit the interference-free space in the sparse regime but are "still constrained in expressivity compared to nonlinear RNNs." They exhibit smooth forgetting because the linear recurrence smoothly decays old features.

- **Nonlinear RNNs** (our E1H/E88) can fully exploit the interference-free space by grouping all intermediate features separate from the output feature, implementing **sharp forgetting**. Old features are not gradually faded -- they are decisively moved into the interference-free region or overwritten. This is a qualitatively different representational strategy.

**The eigenvalue mechanism.** In the sparse regime, eigenvalues of the recurrence support temporal superposition by rotating features within the interference-free space. This rotation is how features are maintained without interfering with output. In linear RNNs, eigenvalues can only scale and rotate features in the full space (no interference-free region exists), so old features inevitably interfere.

### Summary: Nonlinearity Is Not About Capacity -- It Is About Geometry

The paper's key insight is not that nonlinear RNNs have "more capacity" in some vague sense. Rather, nonlinearity creates a qualitatively different representational geometry:

- It partitions activation space into an active region and an interference-free region.
- It enables sharp, selective forgetting rather than smooth, uniform decay.
- It allows features to be maintained in the interference-free space without corrupting the output.

This directly parallels our finding that tanh saturation creates bistable fixed points and enables binary latching. The "interference-free space" of temporal superposition theory and the "latched state" of our bifurcation analysis are two views of the same phenomenon.

---

## 4. Connection to E88 Architecture

### E88's State Structure Through the Temporal Superposition Lens

E88 has matrix state S_t in R^{n x n} (D^2 scalars per head). With n_state=16, each head has 256 state scalars. With 68 heads, the total state is 68 x 256 = 17,408 scalars.

In the temporal superposition framework:

- **Each matrix entry S_{ij}** is a dimension of the hidden state. A single E88 head with n_state=16 has a 256-dimensional hidden state.
- **Each entry can latch** via tanh saturation, effectively implementing sharp forgetting at the element level.
- **The matrix structure** provides content-addressable storage: the query q selects which "directions" in state space to read from, enabling feature-selective readout.

This means E88 implements temporal superposition with several advantages over the simple nonlinear RNNs studied in the paper:

1. **Massively larger interference-free space.** With 256 state dimensions per head, the interference-free space (the set of dimensions where features can be parked without interfering with current output) is much larger than in a vector-state RNN with D dimensions.

2. **Structured access via matrix multiplication.** The query mechanism S . q provides a principled way to selectively read from the active region while ignoring the interference-free region.

3. **Independent heads provide orthogonal subspaces.** Each head maintains its own state matrix, operating in an independent subspace. Features in different heads never interfere, providing a hard partition of the total state space.

### Is 68 Heads x 16-dim State a Form of Temporal Superposition?

Partially, but it is more precisely understood as **avoiding the need for temporal superposition** within each head while enabling massive parallelism:

- Within each head (16 x 16 = 256 state dims), the state is large enough relative to typical feature requirements that temporal superposition may not be necessary. Features can be stored orthogonally.
- Across heads (68 independent state machines), different heads track different features with zero cross-head interference.
- This is analogous to what the paper calls the **sparse regime**: each head has enough capacity to handle its assigned features without resorting to non-orthogonal representations.

The total system achieves high capacity not through aggressive superposition but through massive parallelism of modestly-sized state machines that each operate comfortably in the sparse (low-interference) regime.

---

## 5. Why Many Small Heads Outperform Few Large Heads

The temporal superposition framework provides a compelling explanation for our experimental finding that 68 heads with n_state=16 outperforms configurations with fewer, larger heads.

### The Argument

Consider two configurations with the same total state budget:

- **Config A**: 4 heads x n_state=64 = 4 heads x 4096 = 16,384 total state scalars
- **Config B**: 68 heads x n_state=16 = 68 heads x 256 = 17,408 total state scalars

(Roughly equivalent total state, but very different structure.)

From the temporal superposition perspective:

**Config A (few large heads):**
- Each head has a 4096-dimensional state. This is far more capacity than needed for most individual features.
- But all features tracked by a given head share a single nonlinear recurrence. The tanh is applied element-wise, but the rank-1 update v_t k_t^T couples all 4096 entries.
- Features within a head can interfere at the input coupling stage, even if the state is large enough for orthogonal storage.
- The "interference-free space" is huge per head, but harder to manage because the recurrence dynamics couple all entries.

**Config B (many small heads):**
- Each head has only 256 state dimensions. But each head is an independent state machine with independent parameters (alpha, delta, K, V).
- Features are partitioned across heads at the architecture level, not learned through representational geometry.
- Each head can specialize: one head tracks sentiment, another tracks entity references, another tracks syntactic state, etc.
- Cross-head interference is exactly zero by construction (Theorem: Parallel Head Evolution).
- The interference-free space per head is smaller, but the total effective interference-free space is the sum across all heads.

**Key insight from the paper:** In the dense regime (where D < number of features to track), interference is inevitable and the RNN must learn sophisticated geometric strategies to minimize it. In the sparse regime (D >= features), interference can be eliminated. Config B keeps each head in the sparse regime. Config A risks pushing individual heads into the dense regime when many features must be tracked simultaneously.

**Additional mechanism -- gradient isolation:**
- In Config B, the gradient for head h_1's parameters depends only on head h_1's error. Learning is localized.
- In Config A, each head must simultaneously learn to manage many interacting features, with gradients that reflect complex interference patterns.
- This is why Config B is not just theoretically better but empirically easier to train: simpler learning dynamics per head.

### Connection to the "Effectively Linear" Finding

The temporal superposition paper shows that in the dense regime, even nonlinear RNNs adopt "effectively linear" strategies -- the nonlinearity is underutilized because the network needs all dimensions for feature storage and cannot afford to partition space into active and interference-free regions.

This may explain our ablation result: at 512 tokens, removing tanh from E88 (making the recurrence linear) causes no loss change. If each head is in the dense regime at 512 tokens (many features competing for 256 state dims), the nonlinearity is not being exploited -- the network is in the "effectively linear" regime of temporal superposition theory. The tanh is present but functionally irrelevant.

---

## 6. The 32K Context Ranking Inversion

### Why E88 Beats Mamba2 at Long Context but Not Short Context

The temporal superposition framework provides a coherent explanation for the ranking inversion:

**At 512 tokens (E88 loses to Mamba2):**

1. **Short sequences mean low temporal demand.** With only 512 timesteps, the number of temporal features that must be simultaneously maintained is modest. Even Mamba2's linear state (d_state=96) has sufficient capacity.

2. **Both architectures operate in the sparse regime.** Neither model is under severe capacity pressure. In this regime, the paper shows that linear and nonlinear models adopt similar representational strategies -- the "effectively linear" regime.

3. **Mamba2 wins on throughput.** With parallel scan processing, Mamba2 sees ~4x more data in the same training time. Since both models can handle the representational demands, the model that trains faster wins.

4. **Nonlinearity is unused.** Our ablation confirms this: removing tanh causes no loss change at 512 tokens. The interference-free space mechanism of temporal superposition is not engaged because there is insufficient temporal demand to create feature interference.

**At 32K tokens (E88 beats Mamba2):**

1. **Long sequences create massive temporal demand.** With 32K timesteps, features must be retained across long delays. The number of simultaneously relevant temporal features grows substantially.

2. **Mamba2 enters the dense regime and fails.** Mamba2's linear state decays information exponentially (our Theorem: Linear State Decay). At 32K tokens, early information is attenuated by factor alpha^32000. The linear RNN is forced into extreme temporal superposition with massive interference, and the loss decomposition shows this interference dominates.

3. **E88 stays in the sparse regime per head.** E88's response to long context is striking: the CMA-ES optimal config scales from h=141 heads at 512 tokens to h=187 heads at 32K. It does NOT increase depth. It adds more independent nonlinear state machines. Each head maintains a modest state (n_state=16, so 256 state dims) that can operate in the sparse regime with low interference.

4. **E88 exploits the interference-free space.** With tanh saturation, each E88 head can implement sharp forgetting: irrelevant features are decisively moved to the interference-free region (tanh saturation at +/-1) rather than slowly decaying. This prevents the accumulation of interference that plagues linear models at long context.

5. **E88's latching mechanism preserves relevant features.** Binary latching (our Theorem: E88 Latching Persistence) means important features can be stored indefinitely without decay. In the temporal superposition framework, this corresponds to features being parked in the interference-free space near tanh saturation boundaries, where they neither decay nor interfere.

### The Architectural Scaling Pattern

The paper's framework explains why E88 scales differently at 32K:

- **Mamba2** must increase d_state to handle more features, but each additional state dimension is still subject to linear decay. More capacity helps but does not change the fundamental smooth-forgetting geometry.

- **E88** adds more heads, each an independent nonlinear state machine. Each new head contributes its own interference-free space with sharp forgetting. The capacity scales multiplicatively: N_heads x D^2_per_head total state, with zero cross-head interference.

This is why "E88 uniquely scales by multiplying independent nonlinear state machines" -- it is the only architecture in our comparison that can keep each computational unit in the sparse regime while scaling total capacity.

---

## 7. Implications for the Expressivity Hierarchy

### The Three-Way Hierarchy Through the Superposition Lens

Our proven hierarchy: Linear SSM strictly contained in E1H strictly contained in E88.

The temporal superposition framework adds geometric intuition to each strict containment:

**Linear SSM -> E1H (first separation):**

- Linear SSMs have **no interference-free space**. All state dimensions are "active" -- features stored anywhere in state space contribute to (and interfere with) the output.
- E1H (vector-state with tanh) **creates an interference-free space** through the nonlinear readout. Features can be parked in regions where tanh saturates, eliminating their contribution to interference.
- This is why E1H can compute threshold and XOR (our Theorems) but linear SSMs cannot: these functions require the ability to make discrete decisions (sharp forgetting), which requires partitioning state space into distinct regions.
- The temporal superposition paper confirms this: "nonlinear RNNs are expressive enough to fully exploit the interference-free space" while SSMs with linear recurrence are "still constrained."

**E1H -> E88 (second separation):**

- E1H has vector state h in R^D: the interference-free space scales as D dimensions.
- E88 has matrix state S in R^{D x D}: the interference-free space scales as D^2 dimensions.
- More interference-free space means more features can be simultaneously stored without mutual corruption.
- Additionally, E88's matrix structure enables **content-addressable retrieval** (S . q), which is a qualitatively different readout mechanism. The query selects which subspace of the state to read, providing feature-selective access that minimizes projection interference by design.
- In the temporal superposition framework: E88 can maintain more features in the sparse regime because its per-head state space is D^2, not D. The dense-to-sparse transition occurs at a higher feature count.

### Superposition as the Mechanism Behind Expressivity

The temporal superposition framework suggests a unifying perspective on our hierarchy:

**Expressivity = ability to avoid destructive temporal superposition.**

- Linear SSMs are forced into maximal temporal superposition at long context because they have no mechanism to prevent interference. Old features decay smoothly, creating a growing pile of interfering residuals.

- E1H can avoid temporal superposition for a limited number of features by using the interference-free space created by tanh. But with only D state dimensions, the budget is limited.

- E88 can avoid temporal superposition for many more features simultaneously thanks to D^2 state dimensions per head and content-addressable retrieval. The matrix structure means features can be stored in independent "slots" (rows or columns) rather than overlapping directions.

This reframes the hierarchy not as an abstract statement about function classes, but as a concrete statement about representational geometry: **more expressive models have larger interference-free spaces and sharper forgetting mechanisms, enabling them to maintain more temporal features without destructive interference.**

---

## 8. Open Questions and Future Directions

### Questions the Temporal Superposition Framework Raises for Our Work

1. **Can we measure temporal superposition in trained E88 models?** The angular distribution of feature embeddings and spectral radius analysis described in the paper could be applied to trained E88 heads. If our hypothesis is correct, we should see:
   - Individual heads operating in the sparse regime (near-orthogonal feature directions).
   - Low spectral radius per head (sharp forgetting).
   - Feature angles clustering near 90 degrees (orthogonal) within heads.

2. **Does the number of optimal heads track the number of temporal features?** If each head handles a few features in the sparse regime, then the optimal number of heads should scale with the number of distinct temporal features in the data. At 32K context (more temporal features), the optimal head count increases (h=187 vs h=141 at 512 tokens). This is consistent.

3. **Is there a phase transition in head count?** The paper identifies a sharp phase transition between dense and sparse regimes. There may be a critical head count below which E88 enters the dense regime (too many features per head) and performance degrades sharply, rather than gradually.

4. **Does the "effectively linear" regime explain the tanh ablation?** If at 512 tokens each head is in the effectively linear regime of temporal superposition theory, this would explain why removing tanh has no effect. Testing at 32K tokens (where we observe ranking inversion) might show that the tanh ablation DOES matter -- the nonlinearity should be actively exploited in the sparse regime at long context.

5. **Can we prove that E88's interference-free space is strictly larger than E1H's?** Our existing proof shows D^2 > D capacity. The temporal superposition framework suggests a more refined statement: E88's interference-free subspace dimension is O(D^2) while E1H's is O(D). This could be formalized in Lean as a theorem about the geometry of tanh saturation in matrix vs vector spaces.

### Connections to Other Related Work

- **"Uncovering the Computational Roles of Nonlinearity in Sequence Modeling Using Almost-Linear RNNs"** (arXiv:2506.07919): Shows that sparse nonlinearity (a few nonlinear units in an otherwise linear system) can match fully nonlinear models. This aligns with the temporal superposition finding that the dense regime is "effectively linear" -- you only need nonlinearity when exploiting the interference-free space (sparse regime).

- **"Spectral Superposition: A Theory of Feature Geometry"** (arXiv:2602.02224): Develops the spectral theory of superposition in feedforward networks. The temporal superposition paper extends this to recurrent architectures.

- **"From Data Statistics to Feature Geometry: How Correlations Shape Superposition"** (arXiv:2603.09972, by Prieto et al.): By the same research group. Shows that correlated features arrange orthogonally and create semantic clustering. Relevant to understanding how E88 heads might specialize.

---

## 9. Summary of Key Takeaways

| Question | Answer |
|---|---|
| What is temporal superposition? | Representing more temporal features than state dimensions by encoding features non-orthogonally across time, creating controlled interference. |
| Can D dims store >D features? | Yes, through non-orthogonal representation (superposition), at the cost of interference. |
| Role of nonlinearity? | Creates interference-free space; enables sharp forgetting; allows features to be parked without corrupting output. Linear models cannot do this. |
| Connection to E88? | E88's tanh saturation IS the interference-free space mechanism. Matrix state provides D^2 interference-free dimensions vs D for vector state. |
| Why many small heads win? | Each head stays in the sparse (low-interference) regime. Few large heads risk entering the dense (high-interference) regime. |
| Why E88 wins at 32K? | Long context creates massive temporal demand. Linear models are forced into destructive superposition. E88 scales by adding more heads, each in the sparse regime with sharp forgetting. |
| Hierarchy implications? | Expressivity = ability to avoid destructive temporal superposition. Linear SSM < E1H < E88 corresponds to no interference-free space < D-dim interference-free space < D^2-dim interference-free space. |
