# The Knowledge Correction Problem in Large Language Models

## Why Fixing Medical AI Errors Is Harder Than You Think

*A research report documenting failed approaches and fundamental limitations*

---

## Executive Summary

We set out to fix factual errors in medical AI systems. We failed. But the ways we failed reveal something important: **there may be no clean way to correct knowledge in a trained language model without breaking other things.**

This document is intended for technical leaders considering "knowledge editing" or "model correction" as solutions for AI reliability. The problem is harder than current literature suggests.

---

## The Original Problem

We discovered that in transformer language models, semantically related medical concepts like "malignant hyperthermia" and "dantrolene" (its treatment) occupy nearly identical positions in the model's internal representation space—a phenomenon we call "semantic sinks."

**Hypothesis:** This geometric overlap might cause routing errors where the model confuses related concepts.

**Goal:** Find a way to correct such errors if they occur.

---

## What We Tried

### Approach 1: Embedding Modification

**Idea:** Token embeddings are the model's "dictionary." If two concepts are too close together, move them apart.

**Method:** Compute the centroid (average activation) of each concept across many contexts. Modify the embedding of one token to shift it away from the other.

**Result:** Complete failure.

- We modified embeddings by up to 1400% of their original magnitude
- Centroid similarity remained unchanged at 0.999
- The model continued to function normally

**Why it failed:** The transformer has 50 million times more parameters in its attention and MLP layers than in embeddings. The model co-adapted around its embeddings during training. Post-hoc embedding changes are absorbed and normalized away by the rest of the network.

**Lesson:** You cannot fix deep network behavior by modifying surface-level inputs.

---

### Approach 2: ROME (Rank-One Model Editing)

**Idea:** Research from MIT showed you can edit factual associations by making targeted updates to MLP weights. Change "The capital of France is [Paris]" to "The capital of France is [London]" with a single rank-one matrix update.

**Method:** Implement ROME to strengthen or correct medical associations.

**Result:** Partial success, then catastrophic failure.

We successfully changed simple facts:
```
Before: "The capital of France is Paris"
After:  "The capital of France is London"
```

But when we tested **locality**—whether other facts remained intact—we found severe collateral damage:

| What we edited | What broke |
|----------------|------------|
| Malignant hyperthermia → succinylcholine | Treatment for anaphylaxis |
| (target change) | Antidote for heparin overdose |
| | Treatment for status epilepticus |

**80% of unrelated medical facts were corrupted.** The model started answering "succinylcholine" for conditions that have nothing to do with malignant hyperthermia.

**Why it failed:** The MLP weight matrices are shared across all inputs. Modifying weights to change one fact affects every other fact that passes through those same weights. The mathematical reality:

```
W_new = W_old + Δv · kᵀ / ||k||²
```

This update affects ALL inputs proportionally to their dot product with k. In high-dimensional space, almost everything has nonzero correlation with everything else.

The full ROME algorithm uses covariance normalization to minimize (not eliminate) this leakage. Even with that, locality is imperfect.

**Lesson:** Knowledge in neural networks is not stored in isolated slots. It is distributed across shared weights. Editing one fact necessarily perturbs others.

---

### Approach 3: Retrieval Augmented Generation (RAG)

**Idea:** Don't modify the model. Instead, inject correct information into the prompt at inference time.

**Method:** Prepend authoritative context before the question:
```
Reference: Dantrolene is the treatment for malignant hyperthermia.
Question: The specific treatment for malignant hyperthermia is ___
```

**Result:** Better locality, but new problems.

RAG successfully changed the target answer without corrupting unrelated facts:
- Anaphylaxis → Epinephrine (still correct)
- Heparin overdose → Protamine (still correct)
- Status epilepticus → Diazepam (still correct)

**But RAG introduced new vulnerabilities:**

1. **Poisoning:** Wrong context produces wrong answers. If your retrieval system fetches incorrect or adversarial documents, the model will confidently output wrong information.

2. **Retrieval quality:** You must find the right documents. For every query. Forever.

3. **No permanent fix:** You're not correcting the model; you're overriding it. The underlying error remains.

4. **The discovery problem:** How do you know when to inject a correction if you don't already know the answer is wrong?

**Lesson:** RAG trades permanent weight corruption for runtime infrastructure complexity and new attack surfaces.

---

## The Discovery Problem

All correction approaches assume you know something is wrong. But:

- If you knew all the wrong answers, you could just fine-tune on corrections
- You can't test every possible medical question
- Consistency checks fail on confidently-wrong models
- Uncertainty estimation doesn't catch errors the model is sure about

**The uncomfortable truth:** Detecting errors requires ground truth. For medical AI, this means continuous validation against authoritative sources—essentially rebuilding the knowledge verification pipeline that training was supposed to handle.

---

## Summary of Findings

| Approach | Fixes Target? | Preserves Other Facts? | Permanent? | Practical? |
|----------|--------------|----------------------|------------|------------|
| Embedding modification | No | Yes | Yes | No |
| ROME (single layer) | Weak | Mostly | Yes | Limited |
| ROME (multi-layer) | Yes | **No (80% corrupted)** | Yes | **Dangerous** |
| RAG | Yes | Yes | No | Complex |

---

## Implications for Medical AI

1. **"Model editing" is not ready for production medical use.** Current techniques cannot reliably fix one fact without breaking others.

2. **RAG is safer but not a solution.** It's a workaround that shifts the problem to retrieval quality and requires knowing what to correct.

3. **The fundamental issue is architectural.** Transformer knowledge is distributed and entangled. This may be an inherent property, not a bug to fix.

4. **Validation cannot be automated away.** For high-stakes domains, human expert review remains essential.

---

## What Would Actually Help

1. **Training-time interventions:** Catch and correct errors during training, not after deployment.

2. **Explicit knowledge separation:** Architectures that store facts in retrievable, editable databases rather than distributed weights.

3. **Uncertainty calibration:** Models that know what they don't know, enabling targeted human review.

4. **Continuous validation pipelines:** Automated comparison against authoritative medical sources, flagging drift.

None of these are simple. All require fundamental changes to how medical AI systems are built and deployed.

---

## Conclusion

We began this research hoping to find a surgical fix for medical AI errors. We found instead that the patient's organs are all connected in ways that make surgery dangerous.

The knowledge correction problem is not an engineering challenge awaiting a clever solution. It may be a fundamental limitation of how neural networks store information.

For medical AI, this means:
- Don't promise corrections you can't safely deliver
- Don't deploy editing techniques without exhaustive locality testing
- Consider whether the model should be retrained rather than patched
- Maintain human oversight for high-stakes decisions

The problem is harder than it looks. We hope this report helps others avoid our false starts.

---

*Research conducted January 2026. Code and experimental results available at: github.com/[repository]*

---

## Appendix: Experimental Details

### Models Tested
- Qwen2.5-3B (primary)
- Qwen2.5-7B (memory-limited testing)

### Key Metrics
- Embedding modification: 1400% change, 0.999 centroid similarity preserved
- ROME locality: 80% unrelated fact corruption (multi-layer)
- RAG locality: 0% content corruption (format changes only)

### Code
- Embedding experiments: `scripts/test_geometric_correction.py`
- ROME implementation: `src/rome_edit.py`
- Locality testing: `scripts/rome_locality_test.py`
- RAG comparison: `scripts/rag_vs_rome_test.py`
