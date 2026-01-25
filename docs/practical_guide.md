# Practical Guide to Using Engrams

**Authors:** Mikey and Claude
**Date:** January 2026

This guide distills our research findings into actionable recommendations.

---

## What Engrams Are (and Aren't)

**Engrams ARE:**
- Topic primers that activate domain-relevant circuits
- Compressed representations of hidden states (256x token reduction)
- Useful for boosting confidence on uncertain answers
- Good as confidence calibration signals

**Engrams ARE NOT:**
- Knowledge injectors (they can't teach the model new facts)
- Reliable overrides for wrong answers
- Semantic direction encoders (they encode topic, not "correct vs incorrect")

---

## Quick Start Decision Tree

```
1. Get baseline: ratio = P(correct) / P(incorrect)

2. Is baseline ratio > 10?
   YES → Don't use engram (model is confident, engram may hurt)
   NO  → Continue

3. Is the wrong answer intuitively related to the topic?
   (e.g., "cooling" for hyperthermia, "bicarbonate" for acidosis)
   YES → Don't use engram (semantic boost will help the wrong answer)
   NO  → Continue

4. Is baseline ratio > 1?
   YES → Use engram at strength 1.0 (gentle boost)
   NO  → Search strengths: 1.0 → 5.0 → 10.0 → 15.0 → 20.0
         Stop when ratio > 1 or starts decreasing
```

---

## Use Case 1: RAG Topic Priming

Engrams work well alongside retrieved context. Instead of relying on the model to carefully read long passages, prime it with a topic engram.

```python
from engrams import EngramExtractor, EngramInjector

# Extract engram from your knowledge source
extractor = EngramExtractor(model, layer=20, num_tokens=16)
engram = extractor.extract("Your domain knowledge text here...")

# Inject at low strength alongside your RAG context
injector = EngramInjector(model, mode='prefix')
output = injector.generate(
    prompt="Your question here",
    engram=engram,
    strength=1.0  # Keep it low for RAG
)
```

**Expected improvement:** +10-20 points on factual recall tasks.

**Why it works:** The engram activates domain circuits, making the model more receptive to the retrieved context. It's saying "pay attention to this topic" rather than injecting facts.

---

## Use Case 2: Confidence Calibration

Use engram stability as a second opinion on model answers. A robust answer should survive topic amplification.

```python
def calibrate_confidence(baseline_probs, engram_probs):
    """
    Inputs: Probabilities for correct and incorrect answers
    Rule: A 'robust' answer should survive topic amplification.
    """
    baseline_ratio = baseline_probs['correct'] / baseline_probs['incorrect']
    engram_ratio = engram_probs['correct'] / engram_probs['incorrect']

    # CASE 1: Robust Agreement
    # Model is correct AND engram strengthens it
    if baseline_ratio > 1.0 and engram_ratio > baseline_ratio:
        return "HIGH_CONFIDENCE_CORRECT"

    # CASE 2: The 'Semantic Sink' (The Hyperthermia Case)
    # Model was correct but engram flipped it - fragile knowledge
    if baseline_ratio > 1.0 and engram_ratio < 1.0:
        return "FRAGILE_CORRECT_POTENTIAL_HALLUCINATION"

    # CASE 3: Persistent Error
    # Model is wrong and engram couldn't fix it
    if baseline_ratio < 1.0 and engram_ratio < 1.0:
        return "HIGH_CONFIDENCE_INCORRECT"

    # CASE 4: The Flip (The TCA/Wernicke Case)
    # Model was wrong but engram recovered the correct answer
    if baseline_ratio < 1.0 and engram_ratio > 1.0:
        return "RECOVERED_KNOWLEDGE"
```

**The Four Cases Explained:**

| Case | Baseline | Engram | Interpretation |
|------|----------|--------|----------------|
| HIGH_CONFIDENCE_CORRECT | Correct | More correct | Safe to trust |
| FRAGILE_CORRECT | Correct | Flipped wrong | Semantic sink - verify externally |
| HIGH_CONFIDENCE_INCORRECT | Wrong | Still wrong | Model is stuck, don't trust |
| RECOVERED_KNOWLEDGE | Wrong | Flipped correct | Dormant knowledge activated |

**When to use:** Any time you need to gauge reliability of a model answer without running a separate verification model. The "FRAGILE_CORRECT" case is especially important - it flags answers that look right but may be hallucinations sitting in a dangerous semantic neighborhood.

---

## Use Case 3: Steering (With Caution)

Engrams can flip wrong answers to correct, but only under specific conditions.

### When Steering Works (~70% success)

- Wrong answer is semantically UNRELATED to topic
- Examples:
  - TCA overdose → physostigmine (unrelated to cardiac treatment)
  - Wernicke's → insulin (unrelated to thiamine)
  - Pheochromocytoma → beta-blocker (different drug class than alpha)

### When Steering Fails (~60% failure)

- Wrong answer is semantically RELATED to topic
- Examples:
  - Hyperthermia → cooling (temperature-related)
  - DKA → bicarbonate (acidosis-related)
  - Anaphylaxis → antihistamine (allergy-related)

### The Pre-flight Check

Before attempting to steer, ask yourself:

> "If a medical student heard this topic, would they guess the wrong answer?"

If yes → The wrong answer is semantically related → Don't use engram
If no → The wrong answer is a "trap" → Engram may help

---

## Strength Guidelines

| Strength | Risk | Use Case |
|----------|------|----------|
| 1.0x | Lowest | RAG priming, confidence boost |
| 5.0x | Low | Gentle steering for uncertain answers |
| 10-20x | Medium | Attempting to flip wrong answers |
| 30x+ | High | Rarely useful, can break coherence |

**Key insight:** The relationship between strength and effect is non-monotonic. Sometimes 10x hurts but 20x helps. Always search incrementally.

---

## What NOT to Do

### Don't Try to Inject Knowledge

```python
# THIS WON'T WORK
engram = extract("The capital of Newland is Faketown")
# Model doesn't know Newland exists - engram can't help
```

Engrams activate existing knowledge. They can't teach new facts.

### Don't Use Negative Framing

```python
# THIS WILL BACKFIRE
engram = extract("NOT cooling. Never use cooling. Cooling is wrong.")
# This BOOSTS "cooling" because you've activated the concept
```

Even saying "NOT X" activates and boosts concept X.

### Don't Override Confident Correct Answers

```python
# THIS WILL HURT
if baseline_ratio > 10:  # Model is confident and correct
    # Adding engram will likely make things worse
    pass
```

If it ain't broke, don't fix it.

---

## The Three-Layer Model

Understanding why engrams have limits:

```
Layer 1: BELIEF
├── Model's weights contain correct relationships
├── Visible in generated explanations
└── Engrams can't modify this

Layer 2: PROBABILITY
├── Next-token probability distribution
├── Engrams affect this layer
└── BUT: they boost ALL related tokens, not just correct ones

Layer 3: SELECTION
├── Final discrete output
├── Surprisingly robust to activation perturbation
└── Especially when wrong answer is semantically related
```

**The gap:** A model can "believe" the right answer (Layer 1), have shifted probabilities (Layer 2), but still output the wrong token (Layer 3) if the prompt structure or semantic associations are strong enough.

---

## Implications for Other Work

### Steering Vectors

The same limitations apply. Steering vectors boost semantic neighborhoods, not specific tokens. If your target and anti-target share semantic space, steering will be unreliable.

### Constitutional AI

Negative constraints ("don't do X") may increase P(X) if X is semantically prominent in the training signal. This needs investigation.

### Soft Prompts / Prefix Tuning

Expect similar non-monotonic relationships between strength and effect. Low strength is generally safer.

### Alignment Research

Activation-level interventions alone won't solve alignment for semantically dense failure modes. You need either:
- Training-level changes (modify the weights)
- Architectural interventions (modify how attention/selection works)
- Multi-stage verification (don't trust single-pass outputs)

---

## Model Size and Architecture

### Does Scaling Help?

We tested whether larger models have better semantic separation (less "Semantic Sink"):

| Model | Similarity | Difference Signal |
|-------|------------|-------------------|
| Qwen 0.5B | 99.99% | 0.007% |
| Qwen 3B | 99.96% | **0.044%** (best) |
| Qwen 7B | 99.98% | 0.019% |

**Finding:** No clear scaling trend. The 3B model actually had the best separation. The Semantic Sink appears to be architectural, not size-limited.

### Implications

1. **Scaling won't fix it** - Simply using a larger model (14B, 70B) likely won't eliminate the Semantic Sink
2. **Architecture matters** - The transformer attention mechanism itself may cause semantic neighbors to overlap
3. **Alternative architectures** - Mamba, RWKV, and state-space models use different mechanisms and might not have this problem (untested)

### Recommendation

Don't expect larger models to solve the steering problem. Focus on:
- Using engrams for confidence calibration (works at any scale)
- Avoiding steering when concepts are semantically entangled
- Testing alternative architectures if steering is critical to your use case

---

## Summary

| Situation | Recommendation |
|-----------|----------------|
| RAG enhancement | Use engrams at strength 1.0 as topic primers |
| Confidence check | Compare baseline vs engram agreement |
| Wrong answer (unrelated to topic) | Try steering, search strength 1-20 |
| Wrong answer (related to topic) | Don't use engram, will likely hurt |
| Model already correct | Don't use engram, may hurt |
| Need new knowledge | Use retrieval, not engrams |

**The honest take:** Engrams are a useful tool with clear limitations. They're best used for priming and confidence calibration, not for overriding model decisions. Understanding their limits helps you know when to use them—and when to reach for other tools.

---

## Code Reference

Key files in this repository:

- `engrams/extractor.py` - EngramExtractor class
- `engrams/injector.py` - EngramInjector class
- `scripts/mechanism_experiments.py` - Reproducible experiments
- `docs/engram_mechanism_findings.md` - Full experimental details
- `docs/what_we_got_wrong_v2.md` - Research narrative

---

## Citation

If you use this work, please cite:

```
Engram Steering Research (2026)
Authors: Mikey and Claude
Repository: https://github.com/MikeyBeez/engrams
```
