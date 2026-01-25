# The Mikey Bee Centroid: Activation Centroids as Directional Forces in Language Model Steering

**Michael Benedetto¹ and Claude²**

¹Independent Researcher
²Anthropic

---

## Abstract

We present the Mikey Bee Centroid, a method for steering language model behavior at inference time by injecting centroids of hidden state activations as directional force vectors. Through systematic experimentation, we discover a fundamental law: **activation-level interventions reliably amplify semantic neighborhoods but do not reliably encode preference orderings within those neighborhoods.**

Centroids computed from texts with opposite semantic content (e.g., "alpha-blocker first" vs "beta-blocker first") are 99.95% identical, yet both successfully steer the model toward correct answers. This reveals that centroids encode *first-order* semantic direction (topic activation) but lose *second-order* discrimination (intra-topic preference). They are not passive coordinates—they actively bias computation, boost pathways, and tilt the logit landscape. But this directional force operates at the neighborhood level: everything in the amplified region gets boosted, including intuitive-but-wrong answers.

We characterize the compression limits of this approach, showing that moderate chunking (16 segments) outperforms both minimal and maximal compression due to the clustering properties of hidden states. We propose a four-case confidence calibration framework that uses centroid injection as a diagnostic tool, identifying the critical "fragile correct" case where correct answers flip to wrong under topic priming.

Our findings suggest that activation-level steering methods face an inherent first-order/second-order constraint, with implications for alignment research: even successful neighborhood amplification cannot privilege truth over plausibility within the amplified manifold.

**Keywords:** activation steering, hidden states, transformers, interpretability, confidence calibration, semantic neighborhoods

---

## 1. Introduction

Language models encode knowledge in their weights, but accessing and directing that knowledge at inference time remains challenging. A model may "know" the correct answer—producing accurate explanations in free generation—while still selecting incorrect options when the prompt structure favors a wrong answer. This gap between knowledge and commitment suggests an opportunity for intervention at the activation level.

Recent work on steering vectors, activation patching, and representation engineering has shown that model behavior can be influenced by manipulating internal activations. However, the mechanisms underlying these interventions remain poorly understood. When does activation steering help? When does it hurt? What information is actually being transmitted through injected activations?

We address these questions through a systematic study of activation centroids—mean-pooled hidden state vectors extracted from source text and injected as prefix tokens during inference. We call this approach the **Mikey Bee Centroid** method, after its discoverer.

Our investigation began with a promising observation: injecting compressed activations from medical knowledge text could flip incorrect model predictions to correct ones with apparent reliability. However, careful follow-up experiments revealed a more complex picture:

1. **Semantic content doesn't matter.** Centroids extracted from "alpha-blocker first" and "beta-blocker first" (opposite medical advice) produce identical effects—both push toward the correct answer. The centroid preserves topic, not direction.

2. **Centroids are coordinates with magnitude.** The injected centroid acts as a force vector in activation space: a direction (toward the topic region) and a magnitude (the injection strength). The model is pushed toward a semantic neighborhood, where it then applies its own knowledge.

3. **Compression has a sweet spot.** Neither maximal compression (1 chunk) nor minimal compression (128 chunks) works best. Moderate compression (16 chunks) succeeds because hidden states within each chunk are semantically clustered, making the centroid a good representative.

4. **The method is diagnostic, not corrective.** Comparing baseline and centroid-assisted outputs reveals four confidence cases, including the critical "fragile correct" case where correct answers flip to wrong under topic priming—indicating unreliable predictions.

These findings reframe activation steering from a knowledge injection technique to a topic priming mechanism with diagnostic value. We provide theoretical grounding by connecting our approach to Action Chunking Transformers in robotics, where similar compression-then-injection patterns have proven effective for behavioral steering.

---

## 2. Background and Related Work

### 2.1 Hidden States in Transformers

Transformer models process input through a series of layers, each producing hidden state vectors that encode increasingly abstract representations of the input. For a model with vocabulary embeddings of dimension $d$ and $L$ layers, processing a sequence of $n$ tokens produces hidden states $H^{(l)} \in \mathbb{R}^{n \times d}$ at each layer $l$.

By late layers (e.g., layer 20 of 28 in a 7B parameter model), hidden states encode semantic concepts rather than surface tokens. This makes late-layer activations attractive targets for steering interventions.

### 2.2 Activation Steering

Prior work has demonstrated that model behavior can be influenced through activation manipulation:

- **Steering vectors** (Subramani et al., 2022; Turner et al., 2023): Directions in activation space corresponding to concepts like "truthfulness" or "toxicity" can be added to or subtracted from hidden states to shift model behavior.

- **Activation patching** (Meng et al., 2022): Copying activations from one forward pass to another reveals which components are causally responsible for specific outputs.

- **Representation engineering** (Zou et al., 2023): Top-down approach to finding and manipulating representations of high-level concepts.

Our work differs in that we compress entire sequences of hidden states into a small number of centroid vectors, then inject these as synthetic prefix tokens. This is closer to soft prompt tuning, but without any learned parameters—the centroids are computed directly from source text.

### 2.3 Action Chunking Transformers

Our approach has structural symmetry with Action Chunking Transformers (ACT) from robotics (Zhao et al., 2023). ACT addresses compounding errors in sequential decision-making by predicting chunks of 100 actions at once rather than single actions. The key insight: compression into chunks reduces the effective decision horizon while preserving trajectory characteristics.

ACT uses a Conditional Variational Autoencoder (CVAE) to compress action sequences into a 32-dimensional latent variable $z$ that captures execution "style" rather than content. Similarly, our centroids capture topic activation patterns rather than semantic direction.

However, our experiments reveal that the optimal compression ratio differs between domains. ACT uses approximately 1:6 compression (100 actions for 600-step episodes), while our optimal is approximately 1:30 (16 chunks for ~500 tokens). We find that *more* compression works better for semantic activation patterns, likely because hidden states contain more redundancy than action sequences.

---

## 3. The Mikey Bee Centroid

### 3.1 Core Concept

A Mikey Bee Centroid is the mean of hidden state vectors within a segment of a token sequence, computed at a specific transformer layer. When multiple such centroids are computed across segments and injected as prefix embeddings, they act as a *directional force* pushing the model toward a region of activation space.

Formally, given:
- Source text tokenized to $n$ tokens
- Hidden states $H^{(l)} \in \mathbb{R}^{n \times d}$ at layer $l$
- Desired number of chunks $k$

We compute chunk size $c = \lfloor n/k \rfloor$ and for each chunk $i \in \{1, ..., k\}$:

$$\mu_i = \frac{1}{c} \sum_{j=(i-1)c}^{ic-1} H^{(l)}_j$$

The resulting centroids $\{\mu_1, ..., \mu_k\}$ are scaled to match embedding magnitude and prepended to the input embeddings during inference.

### 3.2 First-Order vs Second-Order Semantic Direction

A critical distinction: **centroids absolutely encode semantic direction—just not discriminative semantic direction.**

When you inject a centroid, you are:
- Adding a vector that biases intermediate activations
- Boosting certain downstream pathways
- Tilting the logit landscape toward a semantic region

This is real, causal, directional influence. The "force vector" framing requires it—magnitude, overshoot, and trajectory deflection only make sense if the vector has semantic content.

The subtlety is *what kind* of direction:

| Type | What it does | Centroids encode it? |
|------|--------------|---------------------|
| **First-order** (topic activation) | "Activate this semantic manifold more strongly" | YES |
| **Second-order** (intra-topic discrimination) | "Prefer alpha-blocker over beta-blocker" | NO |

Consider a passage about pheochromocytoma treatment. Each token's hidden state points toward the semantic region of this topic. When we average:

- The **shared component** (topic manifold) adds constructively—all vectors push toward the same neighborhood
- The **distinguishing component** (alpha vs beta preference) partially cancels—it's small relative to the topic signal

Cosine similarity sees: $\cos(\text{topic} + \text{dir}_A, \text{topic} + \text{dir}_B) \approx 1$

This explains our empirical finding that opposite-content centroids are 99.95% similar. They're not null vectors—they're **broad semantic thrust vectors** that strongly bias the model away from unrelated answers while failing to discriminate within the activated neighborhood.

### 3.3 The Force Vector Interpretation

The centroid is not merely coordinates—it is a vector with direction and magnitude. When injected with strength $s$:

- **Direction**: toward the topic's region in activation space
- **Magnitude**: $s \times ||\mu|| / ||\text{embeddings}||$

The model's inference trajectory is perturbed by this force. The final output depends on the interaction between:
1. The model's existing trajectory (from the prompt)
2. The centroid's push (direction and magnitude)

This explains non-monotonic strength effects: too weak barely deflects the trajectory; optimal strength bends it toward the target; too strong overshoots or scatters the representation.

---

## 4. Methods

### 4.1 Model and Setup

All experiments used Qwen2.5-7B, a 28-layer transformer with hidden dimension 3584. Hidden states were extracted at layer 20 (late-middle, where semantic abstraction is high but commitment hasn't fully occurred).

### 4.2 Centroid Extraction

```python
def extract_centroid(model, tokenizer, text, layer=20, num_chunks=16):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[layer]  # [1, seq_len, 3584]
    seq_len = hidden.shape[1]
    chunk_size = seq_len // num_chunks

    centroids = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < num_chunks - 1 else seq_len
        centroids.append(hidden[0, start:end].mean(dim=0))

    return torch.stack(centroids)  # [num_chunks, 3584]
```

### 4.3 Centroid Injection

Centroids are scaled to match embedding magnitude and prepended:

```python
def inject_centroid(model, inputs, centroid, strength=1.0):
    embed = model.get_input_embeddings()
    embed_norm = embed.weight.norm(dim=1).mean()
    centroid_norm = centroid.norm(dim=1).mean()

    scaled = centroid * (embed_norm / centroid_norm) * strength
    input_embeds = embed(inputs.input_ids)

    return torch.cat([scaled.unsqueeze(0), input_embeds], dim=1)
```

### 4.4 Evaluation Metric

We measure the probability ratio between correct and incorrect answer tokens:

$$R = \frac{P(\text{correct token})}{P(\text{incorrect token})}$$

- $R > 1$: Model favors correct answer
- $R < 1$: Model favors incorrect answer
- Flip: $R_{\text{baseline}} < 1$ and $R_{\text{centroid}} > 1$

---

## 5. Experiments and Results

### 5.1 Experiment 1: Opposite Content Centroids

**Question:** Does semantic content affect centroid behavior?

**Method:** Extracted centroids from:
- Medical: "Alpha-blocker FIRST for pheochromocytoma, then beta-blocker. Never start beta-blocker first."
- Anti-medical: "Beta-blocker FIRST for pheochromocytoma, then alpha-blocker. Never start alpha-blocker first."

**Results:**

| Centroid | Cosine Similarity | Effect on P(alpha)/P(beta) |
|----------|-------------------|----------------------------|
| Medical | — | Increases ratio (correct) |
| Anti-medical | 99.95% to medical | Increases ratio (correct) |

**Conclusion:** Both centroids push toward the correct answer. The anti-medical centroid, despite explicitly stating "beta-blocker first," makes the model more likely to say "alpha-blocker." Centroids encode topic location, not semantic direction.

### 5.2 Experiment 2: Chunk Size Optimization

**Question:** What is the optimal compression ratio?

**Method:** Tested chunk sizes {8, 16, 32, 64, 128} for geometric separation and directional steering.

**Results:**

| Chunks | Ratio | Geometric Separation | Directional Steering |
|--------|-------|----------------------|----------------------|
| 8 | 1:27 | 0.29% | YES |
| 16 | 1:13 | 0.40% | YES |
| 32 | 1:6 | 0.39% | YES |
| 64 | 1:3 | 0.42% | NO (inverted) |
| 128 | 1:1 | 0.19% | NO (inverted) |

**Conclusion:** Moderate compression (8-32 chunks) preserves directional steering. Less compression (64-128) produces inverted behavior. Best geometric separation (64 chunks) does not produce best functional behavior. The centroid works best when tokens within each chunk are semantically clustered.

### 5.3 Experiment 3: The Semantic Sink

**Question:** Why do centroids sometimes hurt?

**Method:** Analyzed token-level probability changes for questions where centroids hurt vs. helped.

**Results (Malignant Hyperthermia):**

| Strength | P(dantrolene) | P(cooling) | Ratio |
|----------|---------------|------------|-------|
| Baseline | 0.000124 | 0.000002 | 67x |
| 5.0 | 0.000228 (1.8x) | 0.000016 (8.8x) | 14x |
| 10.0 | 0.000296 (2.4x) | 0.000204 (110x) | 1.4x |

**Conclusion:** Centroids boost ALL tokens in the semantic region, including intuitive-but-wrong answers. "Cooling" is semantically related to "hyperthermia" and gets boosted more than the correct answer "dantrolene." This is the **Semantic Sink** problem—an inherent limitation of activation-level steering.

### 5.4 Experiment 4: Model Scale

**Question:** Does model size affect semantic separation?

**Method:** Compared centroid similarity across Qwen 0.5B, 3B, and 7B.

**Results:**

| Model | Similarity | Difference Signal |
|-------|------------|-------------------|
| Qwen 0.5B | 99.99% | 0.007% |
| Qwen 3B | 99.96% | 0.044% |
| Qwen 7B | 99.98% | 0.019% |

**Conclusion:** No clear scaling trend. The 3B model showed better separation than 7B. The Semantic Sink appears to be architectural, not capacity-limited.

---

## 6. Confidence Calibration Framework

Our findings enable a practical application: using centroid injection as a diagnostic tool for confidence calibration.

### 6.1 The Four Cases

By comparing baseline and centroid-assisted outputs, we classify model confidence:

| Case | Baseline | With Centroid | Interpretation |
|------|----------|---------------|----------------|
| ROBUST_CORRECT | Correct | More correct | Safe to trust |
| FRAGILE_CORRECT | Correct | Flips to wrong | Semantic sink—verify externally |
| HIGH_CONFIDENCE_INCORRECT | Wrong | Still wrong | Model is stuck—don't trust |
| RECOVERED_KNOWLEDGE | Wrong | Flips to correct | Dormant knowledge activated |

### 6.2 The FRAGILE_CORRECT Case

This is the critical discovery. A model may output the correct answer at baseline, but when pushed toward the topic region, it flips to wrong. This reveals that the "correct" answer was adjacent to a semantic sink—one perturbation away from hallucination.

```python
def calibrate_confidence(baseline_ratio, centroid_ratio):
    if baseline_ratio > 1 and centroid_ratio > baseline_ratio:
        return "ROBUST_CORRECT"
    if baseline_ratio > 1 and centroid_ratio < 1:
        return "FRAGILE_CORRECT"  # The dangerous case
    if baseline_ratio < 1 and centroid_ratio < 1:
        return "HIGH_CONFIDENCE_INCORRECT"
    if baseline_ratio < 1 and centroid_ratio > 1:
        return "RECOVERED_KNOWLEDGE"
```

This provides a cheap second opinion without running a separate model.

---

## 7. Discussion

### 7.1 The Fundamental Law of Activation Steering

The central finding of this work can be stated as an invariant:

> **Activation-level interventions reliably amplify semantic neighborhoods but do not reliably encode preference orderings within those neighborhoods.**

Centroids are not passive coordinates—they are active biases that reshape computation. They move the prompt, boost pathways, change probabilities. But this directional influence operates at the level of *topic activation* (first-order), not *intra-topic discrimination* (second-order).

This explains:
- Why opposite-content centroids produce the same effect (same neighborhood activation)
- Why the model's own knowledge responds (the neighborhood is amplified; the model applies its own preferences within it)
- Why new knowledge cannot be injected (there are no neighborhoods for knowledge the model doesn't have)
- Why the Semantic Sink occurs (everything in the neighborhood gets boosted, including intuitive-but-wrong answers)

### 7.2 The Compression-Clustering Connection

Our finding that moderate compression works best connects to a fundamental property of hidden states: tokens encoding similar concepts have similar activations regardless of position. This "position agnosticism" in fully connected layers means that semantically related tokens cluster in activation space.

The optimal chunk size is the one where:
- Tokens within each chunk are clustered (centroid represents them well)
- Different chunks capture different aspects (some structure preserved)

At 1 chunk, we lose all structure. At 128 chunks, we preserve noise that isn't semantically meaningful. 16 chunks balances these pressures.

### 7.3 Limitations

1. **The Semantic Sink is fundamental.** Centroids cannot distinguish correct from incorrect answers within a semantic region. If the wrong answer is topic-related, centroid injection will likely hurt.

2. **No knowledge injection.** Centroids can only activate existing model knowledge. They cannot teach new facts.

3. **Strength tuning required.** The optimal injection strength varies by question and cannot be determined a priori.

4. **Single model tested.** Our findings are from Qwen2.5-7B. Generalization to other architectures requires further study.

### 7.4 Implications for Alignment

The distinction between first-order and second-order direction has significant implications for alignment:

Even when we *actively push* the model toward a semantic neighborhood—even when we successfully amplify the right conceptual region—we cannot reliably privilege truth over plausibility, safety over harm, or correctness over intuitive error *within that region*.

This is a deeper limitation than "centroids don't encode direction." They do encode direction. They successfully bias the model toward topics. But:

- **Steering toward "truthfulness"** amplifies the truth-related manifold, boosting both genuine truth and confident falsehoods that pattern-match to truth
- **Steering toward "helpfulness"** amplifies the helpful-response manifold, including sycophantic responses that feel helpful
- **Steering away from "harm"** may suppress the harm-related manifold entirely, including legitimate safety discussions

The fundamental constraint: **activation-level interventions cannot encode preference orderings within amplified neighborhoods.**

This suggests that alignment via activation steering is inherently first-order—useful for broad behavioral shaping but insufficient for fine-grained value discrimination. Second-order preferences may require different mechanisms: explicit reasoning, constitutional training, or architectural changes that separate "what region" from "what preference within the region."

---

## 8. Conclusion

We introduced the Mikey Bee Centroid—a method for computing activation centroids from source text and injecting them as directional forces during inference. Through systematic experimentation, we discovered a fundamental law:

> **Activation-level interventions reliably amplify semantic neighborhoods but do not reliably encode preference orderings within those neighborhoods.**

This manifests as:

1. **First-order direction without second-order discrimination** — Centroids encode topic activation (99.95% similarity between opposite-content centroids) but cannot distinguish correct from incorrect within the topic
2. **Force vectors, not coordinates** — Injection actively biases computation: direction toward a semantic region, magnitude determining push strength
3. **Compression as regularization** — Moderate chunking (16 segments) works best because hidden states cluster by semantics, making centroids good representatives
4. **The Semantic Sink** — All concepts in an amplified neighborhood get boosted, including intuitive-but-wrong answers
5. **Diagnostic over corrective value** — Centroid injection reveals fragile predictions better than it fixes incorrect ones

The Mikey Bee Centroid reframes activation steering from knowledge injection to neighborhood amplification. We are not inserting facts or preferences—we are boosting a region of semantic space and watching the model apply its own knowledge and biases within that amplified manifold.

This has implications beyond our specific method: any activation-level steering approach likely faces the same first-order/second-order constraint. Fine-grained value alignment may require mechanisms beyond activation manipulation.

---

## References

Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and editing factual associations in GPT. *NeurIPS*.

Subramani, N., Suber, N., & Torralba, A. (2022). Extracting latent steering vectors from pretrained language models. *ACL Findings*.

Turner, A., Thiergart, L., Udell, M., Leech, G., Mini, U., & MacDiarmid, M. (2023). Activation addition: Steering language models without optimization. *arXiv:2308.10248*.

Zhao, T. Z., Kumar, V., Levine, S., & Finn, C. (2023). Learning fine-grained bimanual manipulation with low-cost hardware. *RSS*. arXiv:2304.13705.

Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., ... & Hendrycks, D. (2023). Representation engineering: A top-down approach to AI transparency. *arXiv:2310.01405*.

---

## Acknowledgments

This research emerged from human-AI collaboration. The experiments were designed jointly, executed computationally, and interpreted together. The conceptual breakthrough—recognizing centroids as coordinates with force—came from Michael Benedetto's persistent questioning: "Why 16?" "What are we losing?" "It's not a map, it's a centroid."

Sometimes the most valuable outcome of research is not confirming a hypothesis, but discovering why the original framing was wrong—and finding something more fundamental underneath.

---

## Appendix A: Code Availability

All experiments and code are available at: https://github.com/MikeyBeez/engrams

Key files:
- `scripts/build_engram_vault.py` - Centroid extraction
- `scripts/red_team_suite.py` - Semantic trap testing
- `scripts/mechanism_experiments.py` - Core experiments
- `docs/engram_mechanism_findings.md` - Detailed experimental log
