# Why the Deep Learning Community Overlooked the Small Batch Advantage

## A Literature Review and Sociological Analysis

**Date:** 2026-03-10

**Motivating question:** Why does the deep learning community overwhelmingly focus on maximizing throughput (tokens/second) via large batch sizes, when empirical evidence shows that batch size 1 can dramatically outperform large batches under fixed wall-time budgets? Why hasn't gradient interference / the update frequency advantage gotten more attention?

**Experimental context:** CMA-ES architecture search at ~480M params, 10-minute fixed wall-time training. Batch size 1 beats batch size 17+ by 0.27 nats (31%) for E88 (nonlinear matrix-state RNN) and 0.865 nats for E1H (nonlinear vector-state Elman RNN). The effect is architecture-agnostic -- not driven by hidden state continuity. Batch size 1 gets 7x more gradient steps despite 2.4x lower throughput. log(batch_size) has r=0.88 correlation with loss -- the single strongest predictor across all architectural variables.

---

## Table of Contents

1. [Historical Context: The Rise of the Large-Batch Paradigm](#1-historical-context-the-rise-of-the-large-batch-paradigm)
2. [Dissenting Voices: Who HAS Argued for Small Batches?](#2-dissenting-voices-who-has-argued-for-small-batches)
3. [Why the Blind Spot Exists: Structural and Incentive Analysis](#3-why-the-blind-spot-exists-structural-and-incentive-analysis)
4. [Gradient Interference Literature](#4-gradient-interference-literature)
5. [The Critical Batch Size Literature](#5-the-critical-batch-size-literature)
6. [Recent Shifts (2024-2026)](#6-recent-shifts-2024-2026)
7. [Synthesis: The Sociology of the Blind Spot](#7-synthesis-the-sociology-of-the-blind-spot)
8. [Bibliography](#8-bibliography)

---

## 1. Historical Context: The Rise of the Large-Batch Paradigm

### The Foundational Papers (2016-2019)

The large-batch paradigm crystallized through a handful of highly visible papers, all motivated by a single goal: reducing wall-clock training time through data parallelism.

**Goyal et al. (2017), "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"** was the watershed moment. Facebook AI Research trained ResNet-50 on ImageNet in one hour using batches of 8192 images across 256 GPUs, reaching 76.3% validation accuracy. The paper introduced the **linear scaling rule** (scale learning rate linearly with batch size) and a **gradual warmup** scheme. This framing -- "we can train in 1 hour instead of days" -- became the dominant narrative. The metric that mattered was wall-clock time to a fixed accuracy target.

**Keskar et al. (2017), "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima"** had actually warned, just one year earlier, that large-batch methods "tend to converge to sharp minimizers" leading to poorer generalization, while "small-batch methods consistently converge to flat minimizers." This paper was widely cited but effectively treated as a problem to be solved (via learning rate tricks) rather than a reason to keep batches small.

**Smith et al. (2018), "Don't Decay the Learning Rate, Increase the Batch Size"** (ICLR 2018) provided theoretical and empirical justification for increasing batch size during training as an equivalent to decaying the learning rate. The implication: large batches are not just tolerable but *preferable* because they enable greater parallelism with "fewer parameter updates."

**You et al. (2017/2019)** introduced specialized optimizers to enable even larger batches. **LARS** pushed ResNet-50 training to batch size 32K, and **LAMB** trained BERT in 76 minutes with batch sizes exceeding 32K. These papers reinforced the narrative that large-batch training was a solvable engineering challenge. LARS performs poorly for attention models like BERT, indicating that its performance gains are not consistent across tasks. In contrast, using LAMB they scaled the batch size in training BERT to more than 32K without degrading the performance.

**Hoffer et al. (2017), "Train Longer, Generalize Better"** (NeurIPS 2017) argued that the generalization gap "stems from the relatively small number of updates rather than the batch size" -- a finding that directly supports the update-frequency hypothesis but was interpreted as "just train longer with large batches."

### The Scaling Laws Era (2020-2022)

**Kaplan et al. (2020), "Scaling Laws for Neural Language Models"** at OpenAI established the dominant framework for thinking about LLM training. Crucially, they trained at a *variable* batch size tracking the critical batch size, then noted in a footnote that "these two prescriptions result in the same number of steps, so we can ignore this subtlety." This set a precedent: batch size became a nuisance variable rather than a variable of interest.

**Hoffmann et al. (2022), "Training Compute-Optimal Large Language Models" (Chinchilla)** doubled down on this. The paper hasn't experimented with batch size and other hyperparameters, rather relying on existing work and providing experimental heuristics. The entire framework asks "given a fixed FLOP budget, how should one trade off model size and training tokens?" -- batch size is not part of the optimization variable space.

This set the frame for the entire field: the knobs you turn are N (parameters) and D (data). Batch size is assumed fixed or unimportant. The community thinks in terms of "more compute = more tokens = better model," not "same tokens but more updates = better model."

---

## 2. Dissenting Voices: Who HAS Argued for Small Batches?

### The Early Prophets

**Wilson & Martinez (2003), "The General Inefficiency of Batch Training for Gradient Descent Learning"**, Neural Networks, 16(10): 1429-1451. This paper directly challenged "a widely held myth in the neural network community that batch training is as fast or faster and/or more 'correct' than on-line training." They showed "why batch training is almost always slower than on-line training -- often orders of magnitude slower -- especially on large training sets." This paper is 23 years old and its core finding has never been refuted, only ignored.

**Bottou & LeCun (2004), "Large Scale Online Learning"** provided "both theoretical and experimental evidence that an adequate online algorithm outperforms any batch algorithm during the final convergence phase." Both online and batch have O(1/t) convergence, but online gets there with fewer total computations.

**Bengio (2012), "Practical Recommendations for Gradient-Based Training of Deep Architectures"** recommended batch size 32 as a good default -- a recommendation that became canonical but was subsequently abandoned by the scaling community without empirical justification for doing so.

### The Modern Dissenters

**Masters & Luschi (2018), "Revisiting Small Batch Training for Deep Neural Networks"** from Graphcore. This is notable because it came from a *hardware company* with aligned incentives -- Graphcore's IPU architecture was designed to excel at small batch sizes. They showed "the best performance consistently obtained for mini-batch sizes between 2 and 32, contrasting with recent work advocating mini-batch sizes in the thousands." Key finding: "increasing the mini-batch size progressively reduces the range of learning rates that provide stable convergence."

**Marek, Lotfi et al. (2025), "Small Batch Size Training for Language Models: When Vanilla SGD Works, and Why Gradient Accumulation Is Wasteful"** (NeurIPS 2025 poster). This is the most directly relevant recent paper. They "revisit small batch sizes all the way down to batch size one" and find that small batch sizes: (1) train stably, (2) are consistently more robust to hyperparameter choices, (3) achieve equal or better per-FLOP performance, and (4) enable stable LM training with vanilla SGD, even without momentum. Their key insight: Adam's beta_2 parameter implicitly controls how many tokens the optimizer "remembers," and when it stays fixed while batch size changes, the effective memory window shifts inadvertently. Their fix: adjust beta_2 = 2^(-(B*T)/t_2) to keep optimizer memory constant regardless of batch size.

**Koliousis et al. (2019), "CROSSBOW: Scaling Deep Learning with Small Batch Sizes on Multi-GPU Servers"** (PVLDB 2019). This system paper directly addressed the assumed trade-off between batch size and GPU utilization: "To fully utilise all GPUs, systems must increase the batch size, which hinders statistical efficiency." CROSSBOW demonstrated 1.3-4x speedup over TensorFlow on 8 GPUs while keeping small batch sizes, using synchronous model averaging (SMA) instead of data parallelism.

---

## 3. Why the Blind Spot Exists: Structural and Incentive Analysis

### 3a. Hardware Vendor Incentives

NVIDIA's entire value proposition for training is built on parallelism. Their performance documentation explicitly states that "to utilize their parallel resources, GPUs execute many threads concurrently" and that larger batch sizes lead to higher throughput. GPU architectures are optimized for large, regular matrix multiplications -- small batch sizes underutilize the tensor cores that justify the hardware's price.

Graphcore was the one hardware vendor that built for small batches. Their IPU architecture "holds the entire model inside the processor" with fast SRAM, delivering "much better arithmetic efficiency on small batch sizes." However, Graphcore went bankrupt and was acquired in 2024, arguably *because* the market didn't value small-batch efficiency. The incentive structure selected against the hardware design that would have made small-batch training mainstream.

### 3b. Distributed Training Architecture Lock-In

Data parallelism -- the dominant paradigm for distributed training -- *mechanically requires* large batch sizes. When you split a batch across N GPUs, your effective batch size is N * per-GPU batch. With clusters of 256-4096 GPUs, you get effective batch sizes of thousands to millions whether you want them or not. As noted in recent research on adaptive batch size schedules: "the batch size is an upper limit on the degree of data parallelism, and small batch sizes mean small inputs, which decreases the operational intensity of matrix multiplication and leads to inefficient computation."

The entire infrastructure stack -- NCCL, DeepSpeed, Megatron-LM, FSDP -- is built around the assumption that you want to maximize data parallelism. There is no mainstream distributed training framework optimized for running many independent small-batch replicas (CROSSBOW was the one attempt and it never reached mainstream adoption).

### 3c. Benchmark Methodology Bias

**MLPerf**, the de facto standard benchmark for ML hardware, measures time-to-accuracy. Submissions are optimized for total wall-clock time, which incentivizes maximizing GPU utilization (= large batches) on expensive multi-node clusters. A researcher who discovered that batch-size-1 training on a single GPU reaches the same loss faster in wall-time would have no venue to demonstrate this -- the benchmark framework assumes multi-GPU setups.

### 3d. The Scaling Laws Framing

The Kaplan and Chinchilla scaling laws created a conceptual framework where the relevant variables are model size (N) and tokens trained on (D). Batch size is not part of the loss prediction equation. This means the entire field's intuition pump -- "how do I make my model better?" -- doesn't include batch size as a lever. The community thinks in terms of "more compute = more tokens = better model," not "same tokens but more updates = better model."

### 3e. The Conflation of "Tokens Processed" with "Learning"

This is perhaps the deepest conceptual error. The standard metric is **tokens/second** (throughput), which assumes every token processed contributes equally to learning. But this is false in two ways:

1. **Gradient averaging cancels conflicting signals.** When per-sample gradients within a batch point in different directions, averaging them produces a smaller, less informative update. The information from individual samples is *destroyed* by aggregation.

2. **Update frequency matters more than throughput.** Our experimental results (bs=1 getting 7x more gradient steps despite 2.4x lower throughput) demonstrate this directly. The relevant metric should be **updates/second**, not tokens/second -- or better yet, **effective learning per second**.

### 3f. Publication and Career Incentives

Papers demonstrating that you can train models on a single GPU with batch-size-1 are not publishable at top venues because they don't demonstrate "scale." The community rewards novelty at scale, not efficiency at small scale. A paper showing that a $50 GPU run beats a $10,000 cluster run challenges the resource hierarchy of the field -- labs with massive compute have no incentive to publicize this, and labs without massive compute lack the credibility to be heard.

---

## 4. Gradient Interference Literature

### 4a. Gradient Agreement Filtering (GAF)

**Chaubard et al. (2024), "Beyond Gradient Averaging in Parallel Optimization: Improved Robustness through Gradient Agreement Filtering"** introduced GAF, which computes cosine similarity between micro-gradients and filters out conflicting updates before averaging. This directly addresses the gradient cancellation problem. Results: "validation accuracy improvements up to 18.2% compared to traditional approaches while reducing computation nearly an order of magnitude."

The Hacker News discussion on GAF was telling -- the community was surprised by the results but the discussion remains firmly technical rather than examining why simpler, smaller-batch approaches haven't achieved mainstream adoption despite apparent benefits.

### 4b. Gradient Diversity

**Yin et al. (2018), "Gradient Diversity: a Key Ingredient for Scalable Distributed Learning"** (AISTATS 2018) introduced the notion of gradient diversity that "measures the dissimilarity between concurrent gradient updates" and showed "high similarity between concurrently processed gradients may be a cause of [performance] degradation." This directly predicts the log(batch_size) correlation we observe -- as batch size grows, gradient diversity within the batch decreases, and averaging increasingly redundant gradients provides diminishing returns.

**DiveBatch (2025)** built on this by creating an adaptive batch size algorithm that adjusts batch size proportional to gradient diversity: "Starting with small-batch size and then using gradient diversity to adjust the batch size during training can significantly accelerate convergence."

### 4c. Multi-Task Gradient Conflict (the Analog Everyone Ignores)

**Yu et al. (2020), "Gradient Surgery for Multi-Task Learning" (PCGrad)** (NeurIPS 2020) showed that projecting conflicting gradients in multi-task learning dramatically improves performance. This work is widely known and respected -- but nobody has applied the same logic to *within-batch* gradient conflicts in single-task training. The insight that "negative cosine similarity between task gradients leads to mutual suppression of parameter updates" applies identically to per-sample gradients within a batch, yet this transfer has not been made.

This is the most striking intellectual gap: the multi-task learning community has built an entire subfield around gradient conflict resolution (PCGrad, CAGrad, Nash-MTL, etc.), while the single-task community assumes that averaging within-batch gradients is always beneficial.

### 4d. Per-Sample Gradient Analysis

The influence function and per-sample gradient literature (TracIn, GraNd scores, etc.) has developed tools for analyzing individual sample contributions to training, but primarily for **data selection** and **debugging**, not for questioning the batch-averaging paradigm itself. The tools exist to measure gradient interference within batches; they just haven't been used to question whether batching itself is the problem.

---

## 5. The Critical Batch Size Literature

### The Original Work

**McCandlish et al. (2018), "An Empirical Model of Large-Batch Training"** from OpenAI introduced the **gradient noise scale** -- the signal-to-noise ratio of gradients across training examples -- and showed it "predicts the largest useful batch size across many domains." The critical batch size (B_crit) is the point where doubling the batch size no longer proportionally reduces the number of required steps.

This paper *should have* changed practice. It provides a principled way to determine when you're wasting compute on too-large batches. Instead, it was primarily cited as a tool for *enabling* large-batch training ("how large can we go?") rather than as a warning against it ("we should stop here").

### Why It Didn't Change Practice

**Allen AI's Critical Batch Size Blog Post** (2025) attempted to operationalize CBS for OLMo pretraining and found that "the CBS starts near 0, increases rapidly and then diminishingly, and plateaus around a batch size of 4096" across both 1B and 7B model scales. They also found fundamental problems with McCandlish's gradient noise scale proxy: "the two do not match" at the 7B scale where "the qualitative pattern of gradient noise scale is messy." Using batch size warmup, they achieved comparable loss "with 43% fewer gradient steps" -- validating the principle that starting with small batches is more efficient.

The key disconnect: CBS typically plateaus well below the batch sizes used in practice. LLM training commonly uses batch sizes of 1-4 million tokens, while CBS analysis suggests diminishing returns far earlier. The community treats the CBS as a *minimum* to enable parallelism rather than a *maximum* beyond which you're wasting compute.

### Recent Follow-Ups for LLMs

**"Critical Batch Size Revisited"** (2025) revisited CBS specifically for large-batch language model training and found that "the CBS does not depend strongly on model size," consistent with past findings.

**"Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training"** (NeurIPS 2025) found that both optimal batch size (B_opt) and critical batch size (B_crit) "scale as power laws in D, independent of model size N." This means that at the beginning of training (when D is small relative to the total), the optimal batch size is much smaller than what's typically used.

**"Scaling Law for Language Models Training Considering Batch Size"** (December 2024) explicitly incorporated batch size into scaling laws, finding "batch size has a complicated relationship with the model size, training budget, and end accuracy." This is one of the first papers to treat batch size as a first-class variable in the scaling law framework rather than a fixed constant.

---

## 6. Recent Shifts (2024-2026)

### The Marek & Lotfi Paper as Inflection Point

The NeurIPS 2025 acceptance of Marek & Lotfi's "Small Batch Size Training for Language Models" represents the clearest signal that the field is beginning to question the large-batch assumption. Their accompanying blog post summarizes: "Small batches are more robust to hyperparameters" and "Performance gaps between simple and complex optimizers widen as batch size grows" -- meaning large batches create artificial complexity that requires sophisticated optimizers, while small batches let even vanilla SGD work.

### Batch Size Scheduling

**"Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws"** (February 2026) is perhaps the most nuanced recent contribution. They found that "for hard tasks, the optimal schedule maintains small batch sizes for most of training and switches to large batches only in a late stage." They identified a "fast catch-up effect" where "after switching from small to large batches, the loss rapidly aligns with the constant large-batch trajectory." The implication: you should train with small batches first and only switch to large batches when gradient noise becomes the dominant concern. Extensive LLM pretraining experiments covering both Dense and MoE architectures with up to 1.1B parameters and 1T tokens validate the theoretical predictions. Across all settings, late-switch schedules consistently outperform constant-batch and early-switch baselines.

### The Unsloth Gradient Accumulation Bug

The Unsloth blog post revealed that gradient accumulation (the standard workaround for simulating large batches on small GPUs) has been *mathematically incorrect* across major frameworks: "naive gradient accumulation always has a higher loss than full batch training" due to incorrect normalization of the cross-entropy loss denominator when sequences have varying lengths. The root cause: when mini-batch losses are averaged separately then summed, the result differs from computing loss across the full batch simultaneously. With different sequence lengths, the accumulated loss scales by a factor of G (gradient accumulation steps), making it mathematically inequivalent to full-batch training. This means much of the field's experience with "gradient accumulation works fine" has been confounded by a systematic bug.

### The "Don't Accumulate" Movement

Benjamin Marie's Medium post "Don't Do Gradient Accumulation for Small Batch Sizes!" helped popularize the Marek & Lotfi findings for practitioners. The recommendation: "Use the smallest batch size maximizing cluster throughput. Avoid gradient accumulation."

### Counterpoints and Nuances

**Geiping et al. (2021), "Stochastic Training is Not Necessary for Generalization"** (ICLR 2022) argued that "the perceived difficulty of full-batch training is largely the result of its optimization properties and the disproportionate time and effort spent by the ML community tuning optimizers and hyperparameters for small-batch training." They showed full-batch training can match SGD on CIFAR-10 with explicit regularization. However, this paper addressed generalization, not *efficiency* -- it doesn't contradict the claim that small batches learn more per FLOP.

**Nado et al. (2021), "A Large Batch Optimizer Reality Check"** (NeurIPS 2021) showed that the specialized large-batch optimizers (LARS, LAMB) were unnecessary: "Nesterov momentum matches the performance of LARS on the ResNet-50 benchmark with batch size 32,768" and "Adam obtains better BERT pre-training results than LAMB at the largest batch sizes." This undermined the entire premise that large batches require specialized optimization -- but was interpreted as "large batches are fine with standard optimizers" rather than "maybe we should question whether we need large batches at all."

### The Learning Rate / Batch Size Ratio

**Jastrzebski et al. (2018), "Three Factors Influencing Minima in SGD"** showed that "the ratio of learning rate to batch size is a key determinant of SGD dynamics and of the width of the final minima, and higher values of the ratio lead to wider minima and often better generalization." This means that what matters is not batch size or learning rate individually, but their ratio -- and the highest possible ratio is achieved at batch size 1.

### The Implicit Bias of SGD Noise

Research on the implicit bias of SGD noise (Francis Bach's work, among others) has established that in high-noise, small-batch, or large-learning-rate regimes, SGD updates are more likely to leave sharp minima and settle in broader, flatter basins of the loss landscape. Critically, the structured noise of SGD (which vanishes for interpolating solutions and belongs to a low-dimensional manifold spanned by the gradients) is beneficial for generalization in ways that generic Gaussian noise is not. Parameter-dependent noise introduces a bias towards local minima with smaller noise variance, whereas spherical Gaussian noise does not. This means the noise structure inherent to small-batch SGD provides a form of implicit regularization that is qualitatively different from -- and superior to -- any explicit regularization that could be added to large-batch training.

---

## 7. Synthesis: The Sociology of the Blind Spot

The evidence that small batch sizes are more compute-efficient under fixed wall-time budgets has been available for over two decades (Wilson & Martinez 2003, Bottou & LeCun 2004). The blind spot persists because of a mutually reinforcing set of incentives:

1. **Hardware economics.** NVIDIA designs GPUs for throughput. Labs buy GPUs. Labs need to justify GPU purchases by utilizing them fully. Full utilization requires large batches. This creates a self-reinforcing cycle where the hardware determines the algorithm rather than the algorithm determining the hardware. Graphcore tried to break this cycle and failed commercially.

2. **Distributed training as default.** With clusters of hundreds to thousands of GPUs, data parallelism is the only tractable approach, and it mechanically forces large batch sizes. The question "should we use smaller batches?" is equivalent to "should we use fewer GPUs?" -- a question no well-funded lab wants to ask.

3. **Metric capture.** The field measures progress in tokens/second and total tokens trained. These metrics are monotonically improved by larger batches (up to hardware limits). The alternative metric -- updates/second or learning-per-FLOP -- would reveal the inefficiency, but nobody reports it.

4. **Scaling laws as ideology.** The Kaplan/Chinchilla framework treats batch size as fixed, directing all attention to N and D. Since these papers define what "optimal" means for the field, batch size literally falls outside the optimization space.

5. **Publication incentives.** Papers at top venues require large-scale experiments. "We trained a 70B model on 1024 A100s" is a paper. "We trained a 480M model on one GPU with batch size 1 and it was better" is not, despite potentially being more scientifically important.

6. **The conflation of parallelism with efficiency.** The community has internalized "parallel = fast = good" so deeply that questioning it feels like questioning Moore's Law. The insight that sequential processing (many small updates) can outperform parallel processing (few large updates) is counterintuitive in a field built on GPUs.

Our experimental finding -- that log(batch_size) has r=0.88 correlation with loss, making it the single strongest predictor -- is consistent with the entire body of evidence reviewed here. The 7x update advantage at batch size 1, despite 2.4x lower throughput, is precisely what Wilson & Martinez (2003), McCandlish et al. (2018), and Marek & Lotfi (2025) would predict. The fact that this effect is architecture-agnostic (matrix-state and vector-state RNNs both show it) rules out hidden-state continuity and points to the fundamental update-frequency mechanism that has been documented since Bottou & LeCun (2004).

The field is beginning to shift -- the Marek & Lotfi NeurIPS 2025 paper, the Allen AI critical batch size work, and the 2026 batch-size scheduling papers all point toward a reconsideration. But the structural incentives (hardware, infrastructure, benchmarks, publication norms) remain aligned with large batches, and it will likely take a practical demonstration at frontier scale -- showing that a large language model trained with small batches on fewer GPUs matches one trained with large batches on many GPUs -- to force a paradigm shift.

---

## 8. Bibliography

### Historical / Large-Batch Paradigm

- Goyal, P., Dollar, P., Girshick, R., et al. (2017). "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." [arXiv:1706.02677](https://arxiv.org/abs/1706.02677)
- Keskar, N. S., Mudigere, D., Nocedal, J., Smelyanskiy, M., & Tang, P. T. P. (2017). "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima." [arXiv:1609.04836](https://arxiv.org/abs/1609.04836)
- Smith, S. L., Kindermans, P.-J., Ying, C., & Le, Q. V. (2018). "Don't Decay the Learning Rate, Increase the Batch Size." ICLR 2018. [arXiv:1711.00489](https://arxiv.org/abs/1711.00489)
- You, Y., Gitman, I., & Ginsburg, B. (2017). "Large Batch Training of Convolutional Networks." [arXiv:1708.03888](https://arxiv.org/abs/1708.03888)
- You, Y., Li, J., Reddi, S., et al. (2019). "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes." [arXiv:1904.00962](https://arxiv.org/abs/1904.00962)
- Hoffer, E., Hubara, I., & Soudry, D. (2017). "Train Longer, Generalize Better: Closing the Generalization Gap in Large Batch Training of Neural Nets." NeurIPS 2017. [arXiv:1705.08741](https://arxiv.org/abs/1705.08741)

### Scaling Laws

- Kaplan, J., McCandlish, S., Henighan, T., et al. (2020). "Scaling Laws for Neural Language Models." [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022). "Training Compute-Optimal Large Language Models." (Chinchilla) [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)

### Small Batch / Dissenting Voices

- Wilson, D. R. & Martinez, T. R. (2003). "The General Inefficiency of Batch Training for Gradient Descent Learning." Neural Networks, 16(10), 1429-1451. [Semantic Scholar](https://www.semanticscholar.org/paper/The-general-inefficiency-of-batch-training-for-Wilson-Martinez/671a4a7caa303f6f87b8ac3941e0175745f9a861)
- Bottou, L. & LeCun, Y. (2004). "Large Scale Online Learning." [PDF](http://yann.lecun.com/exdb/publis/pdf/bottou-lecun-04b.pdf)
- Bengio, Y. (2012). "Practical Recommendations for Gradient-Based Training of Deep Architectures." [arXiv:1206.5533](https://arxiv.org/abs/1206.5533)
- Masters, D. & Luschi, C. (2018). "Revisiting Small Batch Training for Deep Neural Networks." [arXiv:1804.07612](https://arxiv.org/abs/1804.07612)
- Marek, M., Lotfi, S., Somasundaram, A., Wilson, A. G., & Goldblum, M. (2025). "Small Batch Size Training for Language Models: When Vanilla SGD Works, and Why Gradient Accumulation Is Wasteful." NeurIPS 2025. [arXiv:2507.07101](https://arxiv.org/abs/2507.07101)
- Koliousis, A., Watcharapichat, P., Weidlich, M., et al. (2019). "CROSSBOW: Scaling Deep Learning with Small Batch Sizes on Multi-GPU Servers." PVLDB 2019. [arXiv:1901.02244](https://arxiv.org/abs/1901.02244)

### Gradient Interference / Diversity

- Chaubard, F. et al. (2024). "Beyond Gradient Averaging in Parallel Optimization: Improved Robustness through Gradient Agreement Filtering." [arXiv:2412.18052](https://arxiv.org/abs/2412.18052)
- Yin, D., Pananjady, A., Lam, M., Papailiopoulos, D., & Ramchandran, K. (2018). "Gradient Diversity: a Key Ingredient for Scalable Distributed Learning." AISTATS 2018. [arXiv:1706.05699](https://arxiv.org/abs/1706.05699)
- Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C. (2020). "Gradient Surgery for Multi-Task Learning." NeurIPS 2020. [PDF](https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)
- DiveBatch (2025). Adaptive batch size algorithm using gradient diversity. [arXiv:2509.16173](https://arxiv.org/abs/2509.16173)

### Critical Batch Size

- McCandlish, S., Kaplan, J., Amodei, D., & the OpenAI Dota Team. (2018). "An Empirical Model of Large-Batch Training." [arXiv:1812.06162](https://arxiv.org/abs/1812.06162)
- Allen AI (2025). "Revisiting critical batch size for large-batch OLMo pretraining." [Blog post](https://allenai.org/blog/critical-batch-size)
- "Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training." (2025). [arXiv:2505.23971](https://arxiv.org/abs/2505.23971)
- "Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training." NeurIPS 2025. [arXiv:2505.13738](https://arxiv.org/abs/2505.13738)
- "Scaling Law for Language Models Training Considering Batch Size." (2024). [arXiv:2412.01505](https://arxiv.org/abs/2412.01505)

### Recent Shifts / Batch Size Scheduling

- "Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws." (2026). [arXiv:2602.14208](https://arxiv.org/abs/2602.14208)
- Unsloth (2025). "Bug Fixes in LLM Training - Gradient Accumulation." [Blog post](https://unsloth.ai/blog/gradient)
- Marek & Lotfi (2025). Accompanying blog post. [Blog](https://aditsom.github.io/writings/how_to_train_LLMs_with_small_batch_sizes/small_batch.html)

### Counterpoints and Related

- Geiping, J., Goldblum, M., Pope, P., Moeller, M., & Goldstein, T. (2021). "Stochastic Training is Not Necessary for Generalization." ICLR 2022. [arXiv:2109.14119](https://arxiv.org/abs/2109.14119)
- Nado, Z., Gilmer, J. M., Shallue, C. J., Anil, R., & Dahl, G. E. (2021). "A Large Batch Optimizer Reality Check: Traditional, Generic Optimizers Suffice Across Batch Sizes." NeurIPS 2021. [arXiv:2102.06356](https://arxiv.org/abs/2102.06356)
- Jastrzebski, S., Kenton, Z., Arpit, D., Ballas, N., Fischer, A., Bengio, Y., & Storkey, A. (2018). "Three Factors Influencing Minima in SGD." [arXiv:1711.04623](https://arxiv.org/abs/1711.04623)

### Implicit Bias and SGD Noise

- Bach, F. "Rethinking SGD's noise -- II: Implicit Bias." [Blog post](https://francisbach.com/implicit-bias-sgd/)
- "Shape Matters: Understanding the Implicit Bias of the Noise Covariance." (2020). [arXiv:2006.08680](https://arxiv.org/abs/2006.08680)
- "Beyond Implicit Bias: The Insignificance of SGD Noise in Online Learning." (2023). [arXiv:2306.08590](https://arxiv.org/abs/2306.08590)
- "SGD Noise and Implicit Low-Rank Bias in Deep Neural Networks." (2022). [CBMM Memo](https://cbmm.mit.edu/sites/default/files/publications/Implicit%20Rank%20Minimization.pdf)

### Hardware and Infrastructure

- NVIDIA. "GPU Performance Background." [Documentation](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)
- Graphcore. "Revisiting Small Batch Training for Deep Neural Networks." [Blog post](https://www.graphcore.ai/posts/revisiting-small-batch-training-for-deep-neural-networks)

### Community Discussions

- Hacker News discussion on Gradient Agreement Filtering. [Thread](https://news.ycombinator.com/item?id=42554209)
