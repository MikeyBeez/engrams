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

2. **Attention is not the mechanism**
   - Flips happen with less attention to engram
   - Effect works through activation patterns, not reading

3. **Baseline correctness is the best predictor**
   - Wrong → engram likely helps (67%)
   - Right → engram may hurt (75% no help or worse)

4. **Low strength is safest**
   - 1.0x has best consistency
   - Higher strengths are unpredictable

5. **The three-layer model holds**
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

---

## Implications

For **RAG enhancement**: Engrams work well as topic primers alongside retrieved context. They make the model's knowledge more accessible without requiring it to "read" new information.

For **decision override**: Engrams cannot reliably override model decisions when the prompt strongly argues for the wrong answer. The three-layer model (belief → probability → selection) explains this limitation.

For **practical use**: Always check baseline first. Use conservatively (low strength) for correct answers. Search strength space for wrong answers. Never expect to inject truly new knowledge.
