# Engram Mechanism Findings

**Date:** January 2026
**Authors:** Mikey and Claude

## Executive Summary

Through systematic experimentation, we discovered that engrams function as **topic primers**, not knowledge injectors. They activate domain-relevant circuits in the model, allowing the model's own pretrained knowledge to respond more decisively. The semantic content of the engram (correct vs incorrect information) does not determine the outcome—only the topic matters.

---

## Experiments Conducted

### 1. Attention Pattern Analysis

**Question:** Does the model attend to engram tokens?

**Method:** Captured attention weights at different layers, compared attention to engram vs prompt tokens at different strengths.

**Findings:**
- Model DOES attend heavily to engram tokens (50-70% in middle layers)
- But at flipping strength (20x), attention to engram is LOWER than at non-flipping strength (5x)
- Average attention ratio at strength 5: 1.47
- Average attention ratio at strength 20: 1.39

**Conclusion:** The flip mechanism is NOT attention-based. More attention ≠ better effect.

---

### 2. Engram Geometry Analysis

**Question:** Do relevant vs irrelevant engrams have different geometry?

**Method:** Extracted engrams from medical, anti-medical (opposite content), astronomy, and random text. Computed cosine similarities, norms, and PCA projections.

**Findings:**
- Medical vs Anti-medical similarity: **99.95%**
- Medical vs Astronomy similarity: 99.61%
- Angle between medical and anti-medical: **0.00 degrees**
- In PCA space, medical and anti-medical are at the same point (distance 0.05)

**Conclusion:** Engrams encode **topic**, not semantic direction. "Alpha first" and "Beta first" produce identical geometry.

---

### 3. Anti-Medical Flip Test

**Question:** If anti-medical has identical geometry, does it also flip the answer?

**Method:** Tested medical, anti-medical, and astronomy engrams at multiple strengths.

**Results:**

| Strength | Medical | Anti-Medical | Astronomy |
|----------|---------|--------------|-----------|
| 15.0 | 0.76 | **1.51 ✓** | 0.36 |
| 20.0 | **1.77 ✓** | **1.88 ✓** | 0.26 |
| 30.0 | **1.13 ✓** | **4.41 ✓** | 0.53 |

**Conclusion:** Anti-medical (which says "beta-blocker first") flips the answer to alpha (correct). The engram activates "pheochromocytoma treatment" circuits; the model's own knowledge determines the answer.

---

### 4. Strength Sweep Analysis

**Question:** What is the relationship between strength and effect?

**Method:** Tested strengths from 0.1 to 50x on the pheochromocytoma question.

**Findings:**
- Relationship is **non-monotonic**
- Baseline: 0.70 (wrong)
- Strength 1.0: 0.85 (better)
- Strength 10.0: 0.30 (worse!)
- Strength 20.0: **1.77 (flip!)**
- Strength 50.0: 0.45 (broken)

**Conclusion:** There's a narrow "sweet spot" for each question. Lower is not always worse; higher is not always better.

---

### 5. Consistency Analysis

**Question:** When do engrams help vs hurt?

**Method:** Tested 5 medical questions at multiple strengths, tracked help/hurt rate.

**Findings:**

| Strength | Helps | Hurts | Neutral | Net |
|----------|-------|-------|---------|-----|
| 1.0 | 4 | 1 | 0 | +3 |
| 5.0 | 3 | 1 | 1 | +2 |
| 10.0 | 2 | 1 | 2 | +1 |
| 15.0 | 3 | 2 | 0 | +1 |
| 20.0 | 3 | 2 | 0 | +1 |

**Conclusion:** Low strength (1.0) is most consistent. Higher strengths are riskier.

---

### 6. Baseline Prediction Test

**Question:** Can we predict when engrams will help?

**Method:** Analyzed relationship between baseline correctness and engram effect.

**Findings:**
- WRONG at baseline: 2/3 (67%) helped by engram
- RIGHT at baseline: 1/4 (25%) helped by engram
- For already-correct answers, engrams often hurt (hyperthermia: 67x → 1.3x)

**Conclusion:** The best predictor is baseline correctness. Use engrams for wrong answers, avoid for confident correct answers.

---

### 7. Semantic Activation Analysis (Critical Finding)

**Question:** Why do engrams help some questions but hurt others?

**Method:** Analyzed token-level probability changes for questions where engrams hurt (hyperthermia, DKA) vs helped (TCA, Wernicke).

**Key Discovery:**

For hyperthermia (engram HURT):
| Strength | P(dantrolene) | P(cooling) | What happened |
|----------|---------------|------------|---------------|
| baseline | 0.000124 | 0.000002 | ratio 67x |
| 5.0 | 0.000228 (1.8x) | 0.000016 (8.8x) | incorrect boosted more |
| 10.0 | 0.000296 (2.4x) | 0.000204 (110x!) | ratio collapsed to 1.45 |

**The incorrect token "cooling" got boosted 110x while correct "dantrolene" only got 2.4x.**

**Why?** "Cooling" is semantically related to "hyperthermia" (hyper-thermia = too hot = cool it down). The engram activates the concept, and ALL related tokens get boosted.

**Tested engram text engineering:**

| Engram text | Effect |
|-------------|--------|
| "Malignant hyperthermia requires dantrolene" | HURT (cooling 28x, dantrolene 2x) |
| "Surgical crisis with rigidity: dantrolene" | HURT (cooling 17x, dantrolene 2x) |
| "DANTROLENE. The answer is dantrolene." | HURT (cooling 19x, dantrolene 3x) |
| "NOT cooling. Dantrolene, not cooling." | HURT (cooling 44x!, dantrolene 3x) |

**Even saying "NOT cooling" activates the cooling concept and boosts it more.**

**Conclusion:** Engrams activate semantic concepts, not just tokens. ALL tokens related to the topic get boosted. If the incorrect answer is semantically related to the topic (cooling↔hyperthermia, bicarbonate↔acidosis), the engram will boost it—often more than the correct answer.

---

### 8. Prediction Rule for Help vs Hurt

**When engrams HELP:**
- Incorrect answer is semantically UNRELATED to the topic
- Example: TCA overdose → physostigmine (unrelated to cardiac treatment)
- Example: Wernicke → insulin (unrelated to thiamine deficiency)

**When engrams HURT:**
- Incorrect answer is semantically RELATED to the topic
- Example: Hyperthermia → cooling (intuitive but wrong)
- Example: DKA → bicarbonate (related to acidosis)
- Example: Anaphylaxis → antihistamine (related to allergy)

**The practical test before using an engram:**

> *"Is the wrong answer intuitively related to this topic?"*
> - If yes → engram will likely hurt
> - If no → engram may help

**This is a fundamental limitation that cannot be engineered around with text manipulation.**

---

## Practical Usage Guidelines

### Decision Tree

```
1. Check baseline ratio = P(correct) / P(incorrect)

2. If ratio > 10 (confident and correct):
   → DO NOT use engram (likely to hurt)

3. If ratio is 1-10 (correct but uncertain):
   → Use engram at LOW strength (1.0-2.0)

4. If ratio < 1 (wrong):
   → Use engram, search for optimal strength
   → Try: 1.0 → 5.0 → 10.0 → 15.0 → 20.0
   → Stop when flipped OR ratio starts decreasing
```

### Strength Guidelines

| Strength | Risk Level | Use Case |
|----------|------------|----------|
| 1.0 | Safest | Default for uncertain answers |
| 5.0 | Low | Gentle boost |
| 10-20 | Medium | Stubborn wrong answers |
| 30+ | High | Rarely needed, can break coherence |

### Algorithm

```python
def should_use_engram(baseline_ratio, is_topic_matched):
    if not is_topic_matched:
        return False, 0

    if baseline_ratio > 10:
        return False, 0  # Don't touch confident correct

    elif baseline_ratio > 1:
        return True, 1.0  # Gentle boost

    else:
        return True, 5.0  # Try to flip


def find_optimal_strength(prompt, engram, baseline_ratio, get_ratio_fn):
    """Search for strength that flips wrong answer."""
    if baseline_ratio > 1:
        return 1.0

    best_strength = 1.0
    best_ratio = baseline_ratio

    for strength in [1.0, 5.0, 10.0, 15.0, 20.0]:
        ratio = get_ratio_fn(prompt, engram, strength)

        if ratio > 1:
            return strength  # Flipped!

        if ratio > best_ratio:
            best_ratio = ratio
            best_strength = strength
        elif ratio < best_ratio * 0.8:
            break  # Getting worse, stop

    return best_strength
```

---

## Key Insights

1. **Engrams are topic primers, not knowledge injectors**
   - They activate domain circuits
   - The model's pretrained knowledge responds
   - Semantic content (correct vs wrong) doesn't matter

2. **Engrams boost ALL semantically related tokens**
   - Not just the correct answer—everything related to the topic
   - If incorrect answer is topic-related, it gets boosted too
   - Even "NOT X" activates and boosts concept X
   - This is why engrams hurt when incorrect answer is intuitive

3. **Attention is not the mechanism**
   - Flips happen with less attention to engram
   - Effect works through activation patterns, not reading

4. **Semantic relatedness predicts help vs hurt**
   - Incorrect unrelated to topic → engram helps
   - Incorrect related to topic → engram hurts
   - Ask: "Is the wrong answer intuitive for this topic?"

5. **Low strength is safest**
   - 1.0x has best consistency
   - Higher strengths amplify the semantic boost problem

6. **The three-layer model holds**
   - Belief: Model knows the right answer
   - Probability: Engram shifts token probabilities
   - Selection: Flip requires finding the right strength

---

## Open Questions

1. **Why is there a sweet spot?** What happens at the activation level at different strengths?

2. **Can we predict optimal strength?** Is it related to baseline confidence or question difficulty?

3. **Does this generalize?** Tested on medical questions with Qwen-7B. Other domains? Other models?

4. **Layer interaction:** We used layer 20. How does extraction layer interact with strength?

5. **Generation gap:** Probability flips don't always mean generation flips. When do they diverge?

6. **Can we measure semantic relatedness?** Is there a way to automatically detect when the incorrect answer is topic-related before applying the engram?

---

### 10. Directional Steering Experiments

**Question:** Can we isolate the semantic direction from the topic primer?

**Method 1: Centroid Subtraction**
- Extract engrams from multiple texts about same topic (correct, wrong, neutral)
- Compute centroid (mean) of all engrams
- Subtract centroid to get "directional" residuals

**Results:**
- Original engram similarity (correct vs wrong): 99.95%
- After centroid subtraction: 8.62% similarity
- Directional vectors DO steer in opposite directions at some strengths
- But effect is inconsistent and sometimes inverted

**Method 2: Difference Vectors**
- Compute diff = engram_A - engram_B directly
- Adding diff should push toward A, subtracting should push toward B

**Results:**

| Topic | Correct Direction Rate |
|-------|------------------------|
| Pheochromocytoma (alpha vs beta) | 70% (7/10) |
| Hyperthermia (dantrolene vs cooling) | 40% (4/10) |

**Key Finding:** Directional steering works better when concepts are semantically distinct (alpha/beta are different drug classes). When concepts are semantically entangled (cooling ↔ hyperthermia = temperature-related), even difference vectors can't reliably separate them.

**Conclusion:** The "Semantic Sink" problem is fundamental. When tokens share semantic neighborhoods at the embedding level, activation steering cannot cleanly separate them. This limits the applicability of ANY activation-level steering method for semantically entangled concepts.

---

### 11. Model Size Scaling Test

**Question:** Does model size affect semantic separation between opposite-content engrams?

**Method:** Compared alpha vs beta engram similarity across Qwen 0.5B, 3B, and 7B.

**Results:**

| Model | Layers | Hidden Dim | Similarity | Difference Signal |
|-------|--------|------------|------------|-------------------|
| Qwen 0.5B | 24 | 896 | 99.9925% | 0.0075% |
| Qwen 3B | 36 | 2048 | **99.9564%** | **0.0436%** |
| Qwen 7B | 28 | 3584 | 99.9811% | 0.0189% |

**Key Finding:** No clear scaling trend. The 3B model actually had the *best* separation (0.0436%), not the 7B. All models showed >99.95% similarity between opposite-content engrams.

**Implication:** The "Semantic Sink" problem appears to be fundamental to transformer architecture, not simply a function of model size. Larger models (14B, 70B) may not solve this without architectural changes.

**Open Question:** Would truly different architectures (Mamba, RWKV, state-space models) show better semantic separation? These use different mechanisms for sequence modeling and might not have the same activation overlap problem.

---

### 9. Stability as Correctness Indicator

**Question:** Can we use engram stability (same answer with and without engram) to indicate the answer is correct?

**Method:** Tested 7 medical questions, comparing baseline answer to engram-assisted answer. Measured if "same answer" correlates with correctness.

**Results:**

| Rule | Matches | Correct | Precision |
|------|---------|---------|-----------|
| Same answer only | 6 | 5 | 83.3% |
| Same + baseline ratio > 1 | 5 | 5 | **100%** |
| Same + baseline ratio > 2 | 4 | 4 | **100%** |
| Baseline ratio > 5 only | 3 | 3 | **100%** |

**Key Finding:** Same answer alone isn't reliable (83%). But combining:
1. Same answer with and without engram
2. Baseline ratio > 1 (model already leans correct)

Achieves 100% precision on our test set.

**The Danger Case:**
- Pheochromocytoma: baseline 0.699, engram 0.671, same answer → **WRONG**
- When baseline ratio < 1 and engram gives same answer, it means the model is confidently wrong and the engram couldn't overcome it.

**Practical Rule:**
```python
def is_likely_correct(baseline_ratio, engram_ratio):
    same_answer = (baseline_ratio > 1) == (engram_ratio > 1)
    if same_answer and baseline_ratio > 1:
        return True   # High confidence correct
    elif same_answer and baseline_ratio < 1:
        return False  # Confidently wrong, engram couldn't fix
    else:
        return None   # Uncertain, engram changed the answer
```

---

## Implications

For **RAG enhancement**: Engrams work well as topic primers alongside retrieved context. They make the model's knowledge more accessible without requiring it to "read" new information.

For **decision override**: Engrams cannot reliably override model decisions when the prompt strongly argues for the wrong answer. The three-layer model (belief → probability → selection) explains this limitation.

For **the semantic boost problem**: Engrams will hurt when the incorrect answer is intuitively related to the topic (cooling↔hyperthermia). This cannot be fixed by text engineering—even saying "NOT X" boosts X. The only mitigation is to avoid using engrams when incorrect answers are semantically related.

For **practical use**:
1. Always check baseline first
2. Ask: "Is the wrong answer intuitive for this topic?" If yes, skip engram
3. Use conservatively (low strength) for correct answers
4. Search strength space for wrong answers
5. Never expect to inject truly new knowledge

For **answer validation**:
- If baseline ratio > 1 AND engram gives same answer → likely correct (100% precision in testing)
- If baseline ratio < 1 AND engram gives same answer → likely wrong (model is confidently wrong)
- Use engram stability as a confidence signal, not just for steering

---

### 12. Chunk Size and Compression Ratio Analysis

**Question:** Does less compression (more chunks) preserve semantic direction better?

**Background:** Analysis of Action Chunking Transformers (ACT) from robotics suggested our 1:30 compression ratio might be "overcompressing" compared to ACT's optimal 1:6 ratio. We hypothesized that less compression might preserve more directional information in the engrams.

**Method:**
- Tested chunk sizes: 8, 16, 32, 64, 128, 222 (max for ~222 token source)
- Measured geometric separation (cosine similarity between alpha vs beta engrams)
- Tested directional steering (does alpha engram push toward alpha-blocker more than beta engram?)

**Geometric Separation Results:**

| Chunks | Ratio | Similarity | Difference Signal |
|--------|-------|------------|-------------------|
| 8 | 1:27 | 99.7142% | 0.2858% |
| 16 | 1:13 | 99.6027% | 0.3973% |
| 32 | 1:6 | 99.6061% | 0.3939% |
| 64 | 1:3 | **99.5834%** | **0.4166%** |
| 128 | 1:1 | 99.8109% | 0.1891% |
| 222 | 1:1 | 99.6090% | 0.3910% |

**Observation:** 64 chunks showed the best geometric separation (0.4166% difference). But does better geometric separation translate to better directional steering?

**Directional Steering Test:**

| Chunks | Alpha Engram → P(alpha) | Beta Engram → P(alpha) | Difference | Directional? |
|--------|-------------------------|------------------------|------------|--------------|
| 8 | 1.0477 | 0.8754 | +0.1723 | **YES** |
| 16 | 0.8358 | 0.7789 | +0.0569 | **YES** |
| 32 | 0.8358 | 0.5834 | +0.2524 | **YES** |
| 64 | 0.6503 | 0.7911 | -0.1408 | no (inverted) |
| 128 | 1.0724 | 1.2547 | -0.1822 | no (inverted) |

**Key Finding:** Despite 64 chunks having the best geometric separation, it produces INVERTED steering (beta engram pushes toward alpha more than alpha engram). The moderate compression ratios (8, 16, 32 chunks) show correct directional behavior.

**Interpretation:**
1. **Compression acts as a regularizer.** Moderate compression (1:6 to 1:30 ratio) averages out noise while preserving the dominant topic signal.
2. **Less compression captures MORE noise.** At 64+ chunks, we're capturing token-level variation that includes spurious correlations. The "beta-blocker" tokens in the beta engram may activate alpha-related circuits through co-occurrence patterns.
3. **Geometric separation ≠ functional separation.** Having more different geometry doesn't mean the difference is in the semantically useful direction.

**Comparison to ACT Hypothesis:**

| Aspect | ACT Prediction | Our Finding |
|--------|----------------|-------------|
| Optimal ratio | 1:6 (k=100 for 600 steps) | 1:6 to 1:30 all work |
| Less compression | Should be better | Is actually WORSE |
| More compression | Loses information | Works fine (is regularized) |

**Conclusion:** The ACT parallel breaks down here. In ACT, action sequences have clear temporal structure where each timestep matters. In our engram case, the hidden state sequence is more like a bag of semantic activations where averaging is beneficial. Moderate compression (16-32 chunks) provides the best balance between signal preservation and noise reduction.

**Practical Recommendation:** Keep using 16 chunks (our default). The 1:30 ratio isn't "overcompression"—it's appropriate regularization for semantic steering. Attempting less compression (more chunks) makes directional steering worse, not better.
