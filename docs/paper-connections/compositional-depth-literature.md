# Compositional Depth of Text: Literature Survey

## The Concept Under Investigation

**Compositional depth** is proposed as a property of text data (not of models): the context length at which a nonlinear recurrent model first significantly outperforms a matched linear model. Different texts require different depths of temporal composition to predict. A chatbot message might only need local context (shallow composition). A legal document with cross-references needs deep composition. This is fundamentally a data-dependent quantity that could guide architecture selection.

**Core question**: Does this concept exist under a different name? Has anyone measured it? Is it genuinely novel?

**Verdict**: The specific concept -- defining a scalar "compositional depth" of a corpus as the context length at which nonlinear recurrence becomes necessary -- appears to be **novel in its precise formulation**. However, it sits at the intersection of at least six active research threads, each of which captures part of the idea. No existing work unifies them in the way proposed here.

---

## 1. Closest Existing Concepts

### 1.1 L^2M: Mutual Information Scaling Law for Long-Context Language Modeling

**Paper**: [L^2M: Mutual Information Scaling Law for Long-Context Language Modeling](https://arxiv.org/abs/2503.04725) (March 2025)

This is the single most relevant existing work. The authors establish a **bipartite mutual information scaling law** in natural language: the mutual information between past and future text blocks grows as a power law with block length. They define the **L2M condition**: a model's latent state size must scale faster than this bipartite mutual information for effective long-context modeling.

**Key finding**: Transformers naturally satisfy L2M (their KV-cache grows linearly with context). SSMs, RNNs, and linear attention models with fixed state size **cannot** satisfy L2M -- they need increasingly larger models as sequence length grows.

**Relation to compositional depth**: L2M characterizes the data's information-theoretic demands on memory as a function of context length. However, it measures *mutual information* (correlation), not *compositional structure* (nonlinear interaction). A linear model could in principle capture high mutual information through large state size. Compositional depth specifically asks: at what context length does *nonlinearity* become essential? This is a different question from "how much state do you need?"

### 1.2 Hilberg's Conjecture and Debowski's "Facts and Words"

**Key works**:
- Hilberg (1990): Mutual information between adjacent text blocks grows as n^beta
- [Debowski, "Is Natural Language a Perigraphic Process?"](https://pmc.ncbi.nlm.nih.gov/articles/PMC7512648/) (2018)
- [Debowski, "Information Theory Meets Power Laws"](https://books.google.com/books/about/Information_Theory_Meets_Power_Laws.html?id=evaXxAEACAAJ) (2021)

Debowski proves that natural language is "perigraphic": the number of algorithmically independent facts inferable from a text block grows as a power of block length. This is much stronger than saying correlations decay slowly -- it says the *number of distinct structured patterns* grows algebraically. His "theorem about facts and words" connects this to Zipf's law.

**The Relaxed Hilberg Conjecture** states that average surprisal of a t-th order Markov approximation decays as a power law: S_t ~ kt^{-alpha} + S_inf, with alpha approximately 1/2.

**Relation to compositional depth**: Debowski's framework characterizes *how much structure* text has at different scales, but does not distinguish between structure capturable by linear vs. nonlinear models. The power-law growth of "facts" is a measure of information content, not of computational complexity. However, compositional depth could potentially be connected: if a linear model can capture facts up to scale k but requires nonlinearity beyond that, then k is the compositional depth.

### 1.3 Intrinsic Dimension of Context Length

**Paper**: [Explaining Context Length Scaling and Bounds for Language Models](https://arxiv.org/abs/2502.01481) (February 2025)

This paper introduces the concept that the **intrinsic dimension** of next-token prediction converges to a finite limit as context length increases. Key findings:

- There exists an **optimal context length** for each training dataset size
- The intrinsic dimension with increasing context length converges to a finite point
- This convergence potentially explains why RNN models with limited hidden state can be good language models

**Relation to compositional depth**: This work shows that context has *diminishing returns* -- but does not distinguish between linear and nonlinear information. The intrinsic dimension measures total information, not the nonlinear fraction. Compositional depth could be framed as: the context length at which the "nonlinear component" of intrinsic dimension becomes non-negligible.

### 1.4 Statistical Complexity and Computational Mechanics

**Key works**:
- [Shalizi & Crutchfield, "Computational Mechanics: Pattern and Prediction, Structure and Simplicity"](https://link.springer.com/article/10.1023/A:1010388907793) (2001)
- [Marzen & Crutchfield, "Probabilistic Deterministic Finite Automata and Recurrent Networks, Revisited"](https://pmc.ncbi.nlm.nih.gov/articles/PMC8774624/) (2022)

Computational mechanics defines the **epsilon-machine**: the minimal causal-state representation consistent with optimal prediction. The **statistical complexity** is the entropy of the causal states -- how much memory the process "needs" for prediction. The **excess entropy** is the mutual information between past and future.

Marzen & Crutchfield (2022) directly tested RNNs and LSTMs on processes generated by probabilistic finite automata, finding that LSTMs can act as "lossy predictive feature extractors" -- they approximate epsilon-machines with information loss.

**Relation to compositional depth**: Statistical complexity measures how much *state* a process requires, but an epsilon-machine is always a *linear* (finite-state) model. Compositional depth asks a different question: not "how many states?" but "does the transition function need to be nonlinear?" Natural language likely has infinite statistical complexity (it's not a finite-state process), and the *depth* at which nonlinearity matters is precisely what compositional depth tries to capture.

**Gap**: No one has extended computational mechanics to ask "at what scale does the epsilon-machine need to be replaced by a nonlinear computational model?" This is exactly the compositional depth question.

---

## 2. Context Length Scaling of Model Gaps

### 2.1 Zoology: Measuring and Improving Recall in Efficient Language Models

**Paper**: [Zoology: Measuring and Improving Recall in Efficient Language Models](https://arxiv.org/abs/2312.04927) (Arora, Eyuboglu et al., Hazy Research, 2023)

This is the closest existing work to *measuring the gap between architectures as a function of data properties*. Key findings:

- Pretrained 17 attention and gated-convolution models on the Pile
- State-of-the-art gated-convolution architectures underperform attention by up to 2.1 perplexity points
- **82% of the gap is explained by associative recall** -- the ability to retrieve previously mentioned information
- A 70M attention model outperforms a 1.4B gated-convolution model on associative recall

**Relation to compositional depth**: Zoology characterizes the gap but attributes it to a single capability (recall), not to a continuous scale of compositional complexity. Compositional depth would predict that the gap varies by *text type* -- legal documents with many cross-references would show a larger gap than conversational text. Zoology doesn't explore this dimension.

### 2.2 Just Read Twice: Closing the Recall Gap

**Paper**: [Just Read Twice: Closing the Recall Gap for Recurrent Language Models](https://arxiv.org/abs/2407.05483) (Arora et al., ICML 2024)

Extends Zoology by showing the recall gap can be closed by reading the input twice. The associative recall problem formally reduces to set disjointness.

**Relation to compositional depth**: This work identifies *recall* as the bottleneck, not *composition*. Compositional depth is specifically about the need for nonlinear *temporal composition* -- building complex representations through sequential nonlinear processing. Recall is a simpler operation. The two may be orthogonal: high recall demand without compositional depth (phone book lookup) vs. high compositional depth without recall demand (evaluating a deeply nested mathematical expression).

### 2.3 Goomba Lab: On the Tradeoffs of SSMs and Transformers

**Blog post**: [On the Tradeoffs of SSMs and Transformers](https://goombalab.github.io/blog/2025/tradeoffs/) (2025)

Key insight: "long context" is overloaded. SSMs and Transformers have fundamentally different *types* of memory:
- SSMs: compressed state allows longer history of *fuzzy* context
- Transformers: KV-cache allows referral back to *specific* details

**Relation to compositional depth**: The SSM vs. Transformer comparison is related but orthogonal. Compositional depth specifically asks about *nonlinearity in recurrence*, not about attention vs. compression. A linear SSM and a nonlinear SSM differ in compositional depth. A transformer with attention has a different mechanism entirely (direct lookup rather than sequential composition).

---

## 3. Expressivity and Formal Language Theory

### 3.1 The Illusion of State in State-Space Models

**Paper**: [The Illusion of State in State-Space Models](https://arxiv.org/abs/2404.08819) (Merrill & Petty, ICML 2024)

Proves that SSMs, despite their recurrent formulation, cannot express computation outside TC^0. Both transformers and SSMs have the same expressivity limitation. Only "true" RNNs with nonlinear recurrence can express harder computations (e.g., NC^1-complete problems).

**Key finding**: SSMs cannot solve simple state-tracking problems like permutation composition, chess move tracking, or entity tracking in long narratives.

**Relation to compositional depth**: This is directly relevant. The Merrill result says that *linear recurrence* (SSMs) is provably weaker than *nonlinear recurrence* (RNNs) for certain problems. Compositional depth could be operationalized as: the context length at which TC^0-hard problems start appearing in natural text. For short context, natural language may be TC^0-tractable. For long context, state tracking and permutation composition may become necessary.

### 3.2 The Expressive Capacity of State Space Models: A Formal Language Perspective

**Paper**: [The Expressive Capacity of State Space Models](https://arxiv.org/abs/2405.17394) (NeurIPS 2024)

Extends the Merrill analysis: SSMs implement straightforward solutions to *star-free* state tracking problems but cannot handle "hard state tracking" captured by NC^1-complete problems. Only RNNs with nonlinear recurrence can express these.

**Relation to compositional depth**: Provides a formal characterization of *what kind of state tracking* requires nonlinearity. Star-free problems (bounded counting, simple pattern matching) don't need nonlinearity. Permutation composition, parenthesis matching, and similar problems do. Compositional depth could be linked to the density of NC^1-hard sub-problems in text as a function of context length.

### 3.3 Neural Networks and the Chomsky Hierarchy

**Paper**: [Neural Networks and the Chomsky Hierarchy](https://arxiv.org/abs/2207.02098) (Deletang, Ruoss, Grau-Moya et al., ICLR 2023)

Extensive empirical study: 20,910 models, 15 tasks. Results:
- RNNs and Transformers fail on non-regular tasks
- LSTMs solve regular and counter-language tasks
- Only networks with structured memory (stack, tape) solve context-free and context-sensitive tasks

**Relation to compositional depth**: The Chomsky hierarchy defines a *discrete* scale of complexity. Compositional depth would be a *continuous* measure on natural language data. The two could be connected: text at low compositional depth requires only regular-language processing; text at high compositional depth requires context-free or context-sensitive processing.

### 3.4 Limits of Deep Learning: Sequence Modeling through Complexity Theory

**Paper**: [Limits of Deep Learning: Sequence Modeling through the Lens of Complexity Theory](https://arxiv.org/abs/2405.16674) (ICLR 2025)

Proves that multi-layer finite-precision SSMs are limited to regular languages. Even with chain-of-thought, SSMs require polynomially many steps for iterated function composition. The computation of an L-layer SSM on a prompt of length N uses O(L log N) bits of memory, placing SSMs within complexity class L.

**Relation to compositional depth**: Iterated function composition is exactly the kind of operation that compositional depth measures. This paper shows that SSMs *cannot efficiently compose* -- they need growing chain-of-thought tokens. Compositional depth could be defined as the context length at which the text's "function composition depth" exceeds what a linear recurrence can handle.

### 3.5 RNNs Are Not Transformers (Yet): The Key Bottleneck on In-Context Retrieval

**Paper**: [RNNs are not Transformers (Yet)](https://openreview.net/forum?id=h3wbI8Uk1Z) (ICLR 2025)

Shows that even with chain-of-thought, constant-size RNNs cannot solve problems requiring in-context retrieval (e.g., associative recall, graph connectivity). Adding a single Transformer layer to an RNN closes the gap.

**Relation to compositional depth**: Distinguishes *retrieval* from *recurrence*. Compositional depth is about the recurrence part -- how deep the temporal composition needs to be. Retrieval is a separate capability that compositional depth doesn't directly measure.

### 3.6 Compositional Reasoning with Transformers, RNNs, and Chain of Thought

**Paper**: [Compositional Reasoning with Transformers, RNNs, and Chain of Thought](https://arxiv.org/abs/2503.01544) (Yehudai, Amsel, Bruna, 2025)

Defines **Compositional Reasoning Questions (CRQ)** -- tasks with tree-like compositional structure. Proves:
- Transformers need depth logarithmic in problem size
- RNNs need logarithmic embedding dimension (with correct input ordering)
- Transformers with CoT need n tokens for input size n

**None** of the three architectures can solve CRQs without some hyperparameter growing with input size.

**Relation to compositional depth**: CRQs are the formal version of what compositional depth measures in natural language. The "depth" in CRQ's tree structure is analogous to compositional depth. The key insight is that compositional depth is not just about *context length* but about the *tree depth of compositional structure* within that context. Short text can have deep composition (a nested formula); long text can have shallow composition (a long but repetitive passage).

---

## 4. Depth Separation and Hierarchical Structure

### 4.1 How Deep Neural Networks Learn Compositional Data: The Random Hierarchy Model

**Paper**: [How Deep Neural Networks Learn Compositional Data: The Random Hierarchy Model](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.14.031001) (Cagnetta et al., Physical Review X, 2024)

Introduces a synthetic data model with explicit hierarchical composition rules. Shows that deep networks learn by building representations invariant to exchanging equivalent groups at each hierarchy level. Depth enables networks to build this hierarchy.

**Relation to compositional depth**: The Random Hierarchy Model is a *data model* with explicit compositional depth. This is the closest theoretical framework to defining compositional depth of data. However, it applies to synthetic data with known structure. The challenge for natural language is that compositional depth is unknown and must be *measured*. The proposed definition (context length at which nonlinear outperforms linear) would be an empirical way to estimate this quantity.

### 4.2 The Impact of Depth on Compositional Generalization

**Paper**: [The Impact of Depth on Compositional Generalization in Transformer Language Models](https://aclanthology.org/2024.naacl-long.402/) (Petty et al., NAACL 2024)

Key finding: deeper models generalize more compositionally than shallower models, but benefits diminish rapidly. Increasing depth improves *lexical* generalization but not *structural* generalization.

**Relation to compositional depth**: Shows that model depth helps with compositional generalization, but the relationship saturates quickly. This suggests that compositional depth of *data* (not models) is often shallow -- most compositional structure is local.

### 4.3 Do Language Models Use Their Depth Efficiently?

**Paper**: [Do Language Models Use Their Depth Efficiently?](https://arxiv.org/abs/2505.13898) (May 2025)

Analyzes Llama 3.1, Qwen 3, and OLMo 2. Findings:
- Second-half layers contribute much less than first-half
- **No evidence** that models use depth to compose subresults for multi-hop tasks
- Later layers fine-tune output distributions rather than building hierarchical representations

**Relation to compositional depth**: This is striking negative evidence. If models don't compose deeply even when they *could*, it suggests either: (a) natural language training data has low compositional depth on average, or (b) current training doesn't learn to compose. This supports the idea that compositional depth is a meaningful property of data that varies across text types -- most training data may have low compositional depth, so models don't learn to compose deeply.

### 4.4 Depth Separation Results in Theoretical CS

**Key works**:
- [Depth-width tradeoffs in approximating natural functions with neural networks](https://dl.acm.org/doi/10.5555/3305890.3305989) (Safran & Shamir, ICML 2017)
- [Size and Depth Separation in Approximating Natural Functions](https://arxiv.org/abs/2102.00314) (Vardi et al., 2021)

Theoretical results show that certain functions require deep circuits but not shallow ones. There are proven barriers: depth separation beyond depth 4 for natural functions would resolve major open problems in computational complexity.

**Relation to compositional depth**: These results are about *model depth* not *data depth*. However, they imply that if text data contains functions requiring depth-k circuits, then models of depth < k will fail. Compositional depth of data could be defined as the minimum circuit depth needed to predict the next token.

---

## 5. Multi-Scale Temporal Structure

### 5.1 Temporal Hierarchy in Brain and Language Models

**Papers**:
- [The Temporal Structure of Language Processing in the Human Brain Corresponds to The Layered Hierarchy of Deep Language Models](https://arxiv.org/abs/2310.07106) (2023/2025, Nature Communications)
- [Exploring Temporal Sensitivity in the Brain Using Multi-timescale Language Models](https://direct.mit.edu/coli/article/50/4/1477/123793/) (Computational Linguistics, 2024)

Key findings:
- Brain temporal responses match layer-by-layer LLM processing
- Multi-timescale LSTM (MT-LSTM) units partition into short, medium, and long timescale groups
- Short-timescale units encode part-of-speech; long-timescale units encode topic
- Timescales increase along the ventral linguistic hierarchy

**Relation to compositional depth**: This neuroscience work shows that language has *multi-scale temporal structure* that maps onto hierarchical processing. However, "timescale" and "compositional depth" are different concepts. Long timescale does not necessarily mean deep composition -- topic information persists at long timescales but may not require nonlinear composition. Compositional depth is specifically about *composing sub-computations*, not about *temporal persistence*.

### 5.2 Lost in the Middle

**Paper**: [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) (Liu et al., TACL 2024)

Models use information from the beginning and end of context but struggle with information in the middle. Performance degrades as context grows.

**Relation to compositional depth**: This is about *retrieval* failure in long contexts, not about *compositional* failure. Even text with zero compositional depth (a list of independent facts) triggers the "lost in the middle" effect.

---

## 6. Compositionality Definitions and Measures

### 6.1 A Complexity-Based Theory of Compositionality

**Paper**: [A Complexity-Based Theory of Compositionality](https://arxiv.org/abs/2410.14817) (October 2024)

Defines "representational compositionality" using algorithmic information theory: a representation is compositional if it can be redescribed as a simple function of discrete symbolic sequences with recombinable parts. Uses Kolmogorov complexity to quantify this.

**Relation to compositional depth**: This defines compositionality of *representations*, not of *data*. Compositional depth would need to measure the compositionality of the *text's predictive structure* -- how deeply you need to compose temporal representations to predict the next token. These are different but potentially related quantities.

### 6.2 What Makes Models Compositional?

**Paper**: [What Makes Models Compositional?](https://arxiv.org/abs/2405.02350) (Ram, Klinger, Gray, IJCAI 2024)

Defines compositional functions and compositional complexity for sequence processing models. Analyzes recurrent, convolutional, and attention-based models through this framework. For transformers with M blocks, the maximum "level of interaction" grows exponentially in M.

**Relation to compositional depth**: This paper defines compositional complexity of *models*, not of *data*. Compositional depth of data would be the minimum compositional complexity a model needs to achieve near-optimal prediction. The two are dual concepts.

### 6.3 Local Compositional Complexity

**Paper**: [Local Compositional Complexity: How to Detect a Human-readable Message](https://arxiv.org/abs/2501.03664) (Mahon, January 2025)

Defines a complexity measure based on dividing data's shortest description into structured and unstructured portions. Natural language has high LCC (locally compositional complexity) because it has tree-like structure from phonemes through morphemes, syntax, sentences, and discourse.

**Relation to compositional depth**: LCC measures the *amount* of compositional structure but not its *depth*. A sentence with many independent clauses has high LCC but low depth. A sentence with deeply nested subordinate clauses has lower LCC but higher depth. These measure different aspects of structure.

---

## 7. What is Genuinely Novel

### 7.1 The Specific Definition

No existing work defines compositional depth as: **the context length at which a nonlinear model first significantly outperforms a matched linear model on next-token prediction, as a property of the data**.

This is novel because it:

1. **Makes compositional depth a property of data, not of models.** Existing formal language theory classifies *languages* by complexity class, but doesn't provide a continuous scalar measure of *natural language corpora*.

2. **Uses the linear/nonlinear gap as the measurement tool.** L2M uses mutual information. Zoology uses recall accuracy. Hilberg uses entropy scaling. None uses the *gap between matched linear and nonlinear models* as the diagnostic.

3. **Parameterizes by context length.** Existing compositionality measures (LCC, CRQ depth, Kolmogorov-based) are typically applied to fixed-length inputs. Defining compositional depth as a function of context length -- the *threshold* where nonlinearity matters -- is new.

### 7.2 Why It Hasn't Been Proposed Before

Several factors explain the gap:

1. **The linear/nonlinear boundary in recurrence is understudied.** Most architecture comparisons are between transformers, SSMs, and RNNs -- but rarely between *matched linear and nonlinear versions of the same recurrent architecture*. The E88 experiments are unusual in having this exact comparison.

2. **"Compositionality" in NLP means something different.** The SCAN/COGS literature studies whether models can recombine learned primitives. This is about *generalization*, not about *temporal composition depth*. The connection between temporal composition and compositionality is not widely appreciated.

3. **Information theory dominates data characterization.** Entropy rate, mutual information, and intrinsic dimension are the standard tools. The distinction between linear-capturable and nonlinear-capturable information is a *computational* distinction that information theory alone cannot make.

4. **The relevant experiments are expensive.** Measuring compositional depth requires training matched linear and nonlinear models on the same data at multiple context lengths and comparing their loss curves. This is a systematic empirical program, not a single experiment.

### 7.3 What Existing Work Comes Closest

In order of relevance:

1. **L2M** (2025) -- characterizes data's information-theoretic demands on memory as a function of context length, but doesn't distinguish linear from nonlinear
2. **Merrill et al.** (2024) -- proves linear recurrence is expressively weaker than nonlinear, but doesn't measure where the boundary lies in natural language
3. **Zoology** (2023) -- measures architecture gaps on natural language, but attributes them to recall rather than composition
4. **Random Hierarchy Model** (2024) -- defines synthetic data with explicit compositional depth, but doesn't extend to natural language measurement
5. **Debowski** (2021) -- characterizes the power-law growth of structured information in natural language, but doesn't link to model architecture

---

## 8. Connections to the E88 Experiments

The E88 experiments (comparing Elman networks with and without tanh nonlinearity at varying context lengths) are precisely the right experimental setup to measure compositional depth:

- **Matched models**: same architecture, same parameter count, differing only in nonlinearity
- **Variable context length**: 512 vs 32K tokens
- **Key finding**: tanh contributes nothing at 512 tokens but matters at 32K

This directly measures compositional depth: the data at 512 tokens has near-zero compositional depth (linear models suffice), while at 32K tokens, compositional depth is nonzero (nonlinearity helps).

This finding is consistent with:
- **L2M**: mutual information grows with context, demanding more from models
- **Merrill**: TC^0 limitations of linear models eventually bite for long enough context
- **Hilberg**: the number of "facts" grows as a power law, and at sufficient scale, some facts require nonlinear composition
- **Random Hierarchy Model**: real data has hierarchical structure that shallow models miss

But it adds something none of them provide: **an empirical measurement of where the linear/nonlinear boundary lies for real natural language data**.

---

## 9. Suggested Framing for the Paper

Given this survey, the compositional depth concept should be framed as:

1. **A new data characterization metric** that bridges information theory (L2M, Hilberg) and computational complexity (Merrill, Chomsky hierarchy)
2. **An empirical measurement** enabled by the matched linear/nonlinear comparison in E88 experiments
3. **A connection to the Random Hierarchy Model** -- compositional depth of natural language is the parameter k in a Random Hierarchy Model that best fits the observed linear/nonlinear gap
4. **A practical tool for architecture selection** -- if compositional depth at your target context length is near zero, a linear SSM suffices; if nonzero, you need nonlinear recurrence

The concept is genuinely novel in its precise formulation and measurement approach. The survey above provides the necessary context to position it relative to existing work without overstating priority.

---

## 10. Complete Reference List

### Directly Relevant

| Paper | Year | Venue | Key Concept |
|-------|------|-------|-------------|
| [L^2M: Mutual Information Scaling Law](https://arxiv.org/abs/2503.04725) | 2025 | arXiv | Bipartite MI scaling law for long context |
| [The Illusion of State in SSMs](https://arxiv.org/abs/2404.08819) | 2024 | ICML | SSMs limited to TC^0, can't do state tracking |
| [Expressive Capacity of SSMs](https://arxiv.org/abs/2405.17394) | 2024 | NeurIPS | SSMs handle star-free but not NC^1-complete problems |
| [Zoology](https://arxiv.org/abs/2312.04927) | 2023 | arXiv | 82% of attention gap is recall |
| [Just Read Twice](https://arxiv.org/abs/2407.05483) | 2024 | ICML | Recall gap for recurrent models |
| [Limits of Deep Learning via Complexity Theory](https://arxiv.org/abs/2405.16674) | 2025 | ICLR | SSMs limited to regular languages, can't compose |
| [RNNs Are Not Transformers (Yet)](https://openreview.net/forum?id=h3wbI8Uk1Z) | 2025 | ICLR | In-context retrieval is the bottleneck |
| [Compositional Reasoning with Transformers, RNNs, and CoT](https://arxiv.org/abs/2503.01544) | 2025 | NeurIPS | CRQ requires growing resources in all architectures |
| [Random Hierarchy Model](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.14.031001) | 2024 | Phys Rev X | Synthetic data with explicit compositional hierarchy |
| [What Makes Models Compositional?](https://arxiv.org/abs/2405.02350) | 2024 | IJCAI | Compositional complexity of sequence models |
| [A Complexity-Based Theory of Compositionality](https://arxiv.org/abs/2410.14817) | 2024 | arXiv | Kolmogorov-based compositionality definition |
| [Local Compositional Complexity](https://arxiv.org/abs/2501.03664) | 2025 | Entropy | Tree-structured complexity of messages |

### Background / Context

| Paper | Year | Venue | Key Concept |
|-------|------|-------|-------------|
| [Shalizi & Crutchfield, Computational Mechanics](https://link.springer.com/article/10.1023/A:1010388907793) | 2001 | J Stat Phys | Epsilon-machines, statistical complexity, excess entropy |
| [Marzen & Crutchfield, PDFAs and RNNs Revisited](https://pmc.ncbi.nlm.nih.gov/articles/PMC8774624/) | 2022 | Entropy | RNNs as lossy epsilon-machine approximators |
| [Debowski, Is Natural Language Perigraphic?](https://pmc.ncbi.nlm.nih.gov/articles/PMC7512648/) | 2018 | Entropy | Power-law growth of algorithmic facts in text |
| [Hilberg Conjecture](https://www.researchgate.net/publication/281064340_Hilberg's_Conjecture_-_a_Challenge_for_Machine_Learning) | 1990/2015 | Various | MI between adjacent blocks grows as n^beta |
| [Neural Networks and Chomsky Hierarchy](https://arxiv.org/abs/2207.02098) | 2023 | ICLR | Empirical: which architectures learn which language classes |
| [Explaining Context Length Scaling](https://arxiv.org/abs/2502.01481) | 2025 | arXiv | Intrinsic dimension converges; optimal context length exists |
| [Lost in the Middle](https://arxiv.org/abs/2307.03172) | 2024 | TACL | Models fail to use information in middle of context |
| [On the Tradeoffs of SSMs and Transformers](https://goombalab.github.io/blog/2025/tradeoffs/) | 2025 | Blog | Different memory types: compressed vs. cache |
| [Impact of Depth on Compositional Generalization](https://aclanthology.org/2024.naacl-long.402/) | 2024 | NAACL | Depth helps compositionality but saturates quickly |
| [Do Language Models Use Their Depth Efficiently?](https://arxiv.org/abs/2505.13898) | 2025 | arXiv | No evidence models compose deeply for multi-hop tasks |
| [Temporal Structure of Language Processing in Brain](https://arxiv.org/abs/2310.07106) | 2025 | Nature Comms | Brain temporal hierarchy matches LLM layer hierarchy |
| [Multi-timescale LSTM](https://direct.mit.edu/coli/article/50/4/1477/123793/) | 2024 | Comp Ling | Short timescale encodes syntax; long encodes topic |
| [Corpus Complexity Matters in Pretraining](https://aclanthology.org/2023.sustainlp-1.20/) | 2023 | SustaiNLP | Complex corpora produce better models |
| [Recursive Language Models](https://arxiv.org/abs/2512.24601) | 2025 | arXiv | Offload context to REPL; recursive sub-calling |
| [Depth Separation for Neural Networks](https://dl.acm.org/doi/10.5555/3305890.3305989) | 2017 | ICML | Theoretical depth-width tradeoffs |
| [Mamba](https://arxiv.org/abs/2312.00752) | 2023 | arXiv | Selective state spaces; data-dependent transitions |

### Formal Language / Compositionality Benchmarks

| Paper | Year | Venue | Key Concept |
|-------|------|-------|-------------|
| [COGS](https://arxiv.org/abs/2010.05465) | 2020 | EMNLP | Compositional generalization benchmark |
| [SCAN](https://github.com/brendenlake/SCAN) | 2018 | ICML | Compositional action sequence benchmark |

---

## 11. Open Questions and Research Directions

1. **Can compositional depth be measured efficiently?** The matched-model approach (train linear vs nonlinear at each context length) is expensive. Is there a cheaper estimator? Perhaps based on mutual information decomposition into linear and nonlinear components?

2. **Does compositional depth vary across text genres as expected?** The prediction is: chatbot < news < fiction < legal < mathematical proofs. Has anyone measured architecture gaps across genres at varying context lengths?

3. **What is the relationship between compositional depth and the Chomsky hierarchy?** Is compositional depth a continuous relaxation of the discrete regular/context-free/context-sensitive classification?

4. **Can compositional depth predict optimal architecture?** If measured on a new corpus, does compositional depth reliably predict whether attention, linear SSM, or nonlinear RNN will perform best?

5. **Is there a theoretical connection to circuit depth?** The Merrill/Sabharwal TC^0 results suggest that compositional depth > 0 corresponds to problems outside TC^0. Can this be made precise?
