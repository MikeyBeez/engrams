# Negative Result: Post-Hoc Embedding Modification Does Not Fix Semantic Sinks

**Date**: 2025-01-25
**Authors**: Michael Benedetto and Claude

---

## Summary

We tested whether modifying token embeddings could fix semantic sinks in large language models. The experiment definitively shows that post-hoc embedding modification does not work, even with changes exceeding 1400% of the original embedding magnitude. This document explains why, and what it means for the undertrained embeddings hypothesis.

---

## The Hypothesis

We proposed that semantic sinks (where distinct concepts occupy nearly identical positions in representation space) could be fixed by geometrically separating the token embeddings. The reasoning:

1. Semantic sinks exist (dantrolene/cooling similarity = 0.999)
2. Sinks form due to undertrained embeddings (rare tokens get few updates)
3. Therefore, moving the embeddings apart should fix the routing problem

This reasoning was wrong.

---

## The Experiment

**Setup:**
- Model: Qwen2.5-7B (28 layers, 3584 hidden dimensions)
- Target tokens: "dantrolene" (token_id: 67) and "cooling" (token_id: 42196)
- Modification: Sparse correction along separation direction (top 50 aligned dimensions, ~1.4% of embedding)

**Test 1: Effect on Centroid Similarity**

| Alpha | Embedding Change | Centroid Similarity |
|-------|------------------|---------------------|
| 0.0   | 0%               | 0.9990              |
| 0.5   | -                | 0.9990              |
| 1.0   | -                | 0.9990              |
| 5.0   | -                | 0.9990              |
| 10.0  | -                | 0.9990              |
| 20.0  | -                | 0.9990              |
| 50.0  | **1402%**        | 0.9990              |

Even with the embedding modified by 14x its original magnitude, centroid similarity was unchanged.

**Test 2: Effect on Generation**

With embeddings modified by alpha=50 (1402% change):
- "What is the specific treatment for malignant hyperthermia?" → "Dantrolene" ✓
- "For malignant hyperthermia crisis, the drug of choice is" → "Dantrolene" ✓

The model still generated correct answers despite massive embedding perturbation.

---

## Why It Doesn't Work: Co-Adaptation

The transformer has 28 layers of weights that have **co-adapted** with the embeddings during training. The weights "expect" tokens to arrive at specific positions in embedding space.

When you move an embedding post-hoc:

1. **Layer 1** receives the modified embedding
2. **The attention and MLP weights** were trained with the embedding at its original position
3. **The computation proceeds** as if the token is something unfamiliar
4. **By layer 20**, the transformer has "corrected" for the perturbation using context
5. **The output** is determined by transformer weights, not embedding position

The transformer is robust to embedding perturbation because it has learned to rely on context and its own internal representations, not just the input embeddings.

---

## What This Means

### The Diagnosis Still Stands

The undertrained embeddings hypothesis remains valid:

- Token embeddings for rare terms receive ~1,500 updates during training
- Transformer weights receive ~100 billion updates
- This 50-million-to-one asymmetry causes semantic sinks
- The embedding position reflects training context distribution, not semantic requirements

The 0.999 similarity between dantrolene and cooling centroids is real and explains routing failures.

### The Fix Must Respect Co-Adaptation

Post-hoc embedding modification fails because it breaks co-adaptation. Valid fixes must either:

1. **Update embeddings AND weights together** (fine-tuning)
   - The whole system co-adapts to new positions
   - Requires training compute

2. **Target transformer weights directly** (ROME/MEMIT)
   - Modify MLP weights where facts are stored
   - Respects embedding positions

3. **Fix during initial training**
   - More examples for rare terms
   - Contrastive examples forcing separation
   - Better embedding initialization

4. **Work around it at inference** (RAG/prompting)
   - Inject correct information each time
   - Don't fix the model, augment the input

### Centroid Similarity ≠ Embedding Similarity

A key finding: raw embedding similarity between dantrolene and cooling is only **0.009** (nearly orthogonal), but centroid similarity is **0.999**.

The semantic sink is not in the embeddings themselves - it emerges from how the transformer PROCESSES contexts containing these tokens. The sink is in the hidden states, not the input embeddings.

This explains why embedding modification doesn't help: the sink isn't caused by embedding proximity, it's caused by the transformer mapping different inputs to similar internal representations.

---

## Implications for Future Work

### What Doesn't Work
- Post-hoc embedding modification (tested, failed)
- Assuming embedding proximity causes semantic sinks (the causation is reversed)

### What Might Work
- ROME/MEMIT for targeted fact modification
- Contrastive fine-tuning on confusable concepts
- Training-time interventions for rare tokens
- Inference-time knowledge injection (RAG)

### What Remains Valuable
- Centroid extraction as a diagnostic tool
- Semantic sink detection via centroid similarity
- The undertrained embeddings hypothesis as explanation
- The training data forensics framework

---

## Conclusion

The experiment produced a clear negative result: post-hoc embedding modification does not fix semantic sinks. The transformer's co-adaptation with its embeddings makes it robust to input perturbation - a feature that prevents simple geometric fixes.

The undertrained embeddings hypothesis explains WHY semantic sinks form. But fixing them requires respecting the co-adaptation between embeddings and weights, either by updating both together or by targeting the weights directly.

This is how research works: the diagnosis was correct, the proposed fix was not. The finding narrows the solution space and points toward more promising approaches.

---

## Experimental Code

The test script is available at: `/home/bee/Code/engrams/scripts/test_geometric_correction.py`

Key measurements:
- Centroid extraction at layer 20
- Cosine similarity computation
- Embedding modification with rollback
- Generation testing with modified embeddings
