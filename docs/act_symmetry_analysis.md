# Action Chunking Transformers and Engram Steering: Structural Symmetry

**Date**: 2026-01-24
**Context**: Analysis of activation compression techniques in light of robotics research

## Summary

The engram compression technique (chunk hidden states → average → inject as prefix) has structural symmetry with Action Chunking Transformers (ACT), a validated approach in robotic manipulation. This document explains the connection and its implications for activation steering research.

---

## What is ACT?

**Paper**: "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (Zhao et al., 2023)
**ArXiv**: https://arxiv.org/abs/2304.13705

### Core Problem
Fine manipulation tasks require precision. Single-step policies suffer from compounding errors - small mistakes cascade over time, causing divergence from training distribution.

### Solution Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ CVAE Encoder (Training Only)                                │
│                                                              │
│ Input: Action sequence (k steps) + Joint positions          │
│        ↓                                                     │
│ Transformer encoder                                         │
│        ↓                                                     │
│ Latent variable z (32-dim)                                  │
│   "Style" of action sequence                                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Policy Decoder (CVAE Decoder)                               │
│                                                              │
│ Input: RGB images + Joint positions + z                     │
│        ↓                                                     │
│ Transformer encoder (synthesize observations)               │
│        ↓                                                     │
│ Transformer decoder (generate actions)                      │
│        ↓                                                     │
│ Output: Next k actions (joint positions)                    │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Choices

1. **Chunk Size**: k=100 actions (for ~600 step episodes)
   - Reduces effective horizon by 100x
   - Mitigates compounding errors
   - Ratio: k/episode_length ≈ 1:6

2. **Temporal Ensembling**: Query policy at every timestep
   - Overlapping action chunks
   - Weighted average with exponential decay: w_i = exp(-m * i)
   - Smoother trajectories without bias

3. **CVAE Training**: Handle variability in human demonstrations
   - Reconstruction loss (L1, not L2 - better precision)
   - KL divergence regularization (β-weighted)
   - At test time: z = 0 (mean of prior) for deterministic output

4. **Results**: 80-90% success on fine manipulation with 10 min of demos

---

## Engram Approach Structure

### Our Implementation

```
┌─────────────────────────────────────────────────────────────┐
│ Engram Extraction                                           │
│                                                              │
│ Input: Source text (e.g., medical knowledge)                │
│        ↓                                                     │
│ Pass through model, capture layer L hidden states           │
│   (500 tokens × 3584 dims)                                  │
│        ↓                                                     │
│ Chunk into 16 groups, average each chunk                    │
│        ↓                                                     │
│ Scale to match embedding magnitude                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Engram Injection                                            │
│                                                              │
│ Prepend 16 compressed vectors as prefix tokens              │
│        ↓                                                     │
│ Model processes as if real tokens                           │
│        ↓                                                     │
│ Influence on next-token probabilities                       │
└─────────────────────────────────────────────────────────────┘
```

### Parameters
- Chunk size: 16 (out of ~500 tokens)
- Ratio: ~1:30 (different from ACT's 1:6)
- Compression: Simple averaging (not learned)
- Injection: Single-shot prefix (not continuous ensemble)

---

## Structural Symmetry

### Pattern Comparison

| Aspect | ACT | Engrams |
|--------|-----|---------|
| **Input** | Action sequence (temporal) | Hidden state sequence (spatial/temporal) |
| **Compression** | CVAE encoder → latent z | Chunk + average → 16 vectors |
| **Purpose** | Capture execution "style" | Capture topic activation pattern |
| **Steering** | Condition action generation | Condition token probabilities |
| **Domain** | Robotics manipulation | Language model inference |

### Unified Framework

Both implement:
```
Sequential data → Compression → Compact representation → Behavioral conditioning
```

The key insight: **Chunking + compression captures trajectory essence that can steer future behavior**

---

## What ACT Reveals About Engrams

### 1. Chunking is Principled
- ACT shows chunk size matters (k=100 optimal for their tasks)
- Our k=16 might be suboptimal
- Systematic grid search needed: test k ∈ {8, 16, 32, 64, 128}

### 2. "Style" vs. "Content"
ACT's latent variable z encodes **how to execute**, not **what to do**:
- Same task, different execution styles → different z
- z captures trajectory characteristics, not goal

Our finding: Opposite semantic content → same effect
- Supports interpretation: engrams encode **topic activation style**
- Not semantic direction (which is what we wanted)
- Matches ACT pattern: compression captures "how" not "what"

### 3. Temporal Ensembling Critical
ACT queries policy at every step, combines overlapping predictions:
```python
# ACT approach
for timestep t:
    predict actions[t:t+k]
    action[t] = weighted_average(all predictions for timestep t)

# Our approach (missing this)
inject_once(engram_vectors)
run_inference()
```

**Implication**: We should continuously inject and ensemble, not single-shot prefix.

### 4. Variability Handling
ACT uses CVAE to model stochastic demonstrations:
- Without CVAE: 33% performance drop on human data
- With CVAE: Learns to handle multi-modal behavior

Our approach has no generative component:
- Can't handle variability in activation patterns
- Might explain why some engrams help, others hurt

---

## Implementation Gaps

### What We Did Right
1. Compression reduces dimensionality (500 → 16)
2. Injection modifies behavior without retraining
3. Late-layer extraction (where semantic info concentrates)

### What We Missed
1. **Learned compression**: ACT uses transformer, we use averaging
2. **Temporal ensemble**: ACT continuously queries + combines, we inject once
3. **Generative modeling**: ACT uses CVAE for variability, we have none
4. **Systematic tuning**: ACT grid-searched hyperparameters, we used ad-hoc values

---

## Experimental Findings Reinterpreted

### Observation: Opposite semantic content → same effect

**Original interpretation**: Engrams don't encode semantics

**ACT-informed interpretation**:
- Engrams encode "topic activation style" (like ACT's z)
- Both "alpha-blocker first" and "beta-blocker first" activate pheochromocytoma circuits
- The **style** of activation matters, not content direction
- Model's own knowledge responds to activation pattern

### Observation: Semantic Sink (related wrong answers get boosted)

**ACT analogy**: Compounding errors in action space
- Small activation errors amplify over generation
- Related concepts in semantic neighborhood get boosted together
- No fine-grained control over which concepts activate

**Solution from ACT**: Chunking reduces horizon, ensemble smooths trajectories
- Applied to engrams: Continuous injection + ensemble might give finer control

---

## Recommended Next Steps

### 1. Implement Temporal Ensembling
```python
# Instead of:
engram = extract_and_compress(source_text)
output = model(input + engram)

# Try:
for step in generation:
    engram_t = extract_and_compress(source_text, context=current_state)
    prediction_t = model(input + engram_t)
    output[step] = weighted_average([prediction_t, prediction_t-1, ...])
```

### 2. Test Compression Methods
- Current: Simple averaging
- Alternative 1: Learned linear projection
- Alternative 2: Small transformer encoder (like ACT)
- Alternative 3: PCA on activation chunks

### 3. Systematic Chunk Size Search
Grid search k ∈ {8, 16, 32, 64, 128, 256}
- Measure: Flip rate, semantic sink severity, consistency
- Find optimal k/sequence_length ratio

### 4. Add Variability Modeling
Explore CVAE-style approach:
- Extract multiple engrams from same topic
- Learn latent distribution over activation styles
- Sample during injection for robustness

---

## Theoretical Implications

### Why This Matters

1. **Validation**: Our approach isn't ad-hoc - it parallels validated robotics research
2. **Mechanism**: Compression captures behavioral style, not content
3. **Limitations**: Same fundamental constraints as ACT (works within learned manifold)
4. **Direction**: Improvements should follow ACT pattern (ensembling, learned compression)

### Fundamental Insight

Both ACT and engrams exploit the same principle:
> **Sequential behavior can be compressed into compact representations that capture execution style. These representations can steer future behavior within the model's learned manifold.**

The failure modes are similar:
- ACT can't teach new motor skills, only modulate existing ones
- Engrams can't inject new knowledge, only activate existing circuits

---

## References

**Primary Source**:
- Zhao et al. (2023). "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
  - ArXiv: 2304.13705
  - Project: https://tonyzhaozh.github.io/aloha/

**Related Work**:
- Activation steering (general): Representation engineering literature
- Compressive Transformers: Activation compression for memory
- CVAE: Conditional variational autoencoders for behavior modeling

---

## Document Metadata

**Created**: 2026-01-24
**Purpose**: Technical reference for engram implementation improvement
**Context**: Verification of experimental methodology in light of robotics literature
**Next Action**: Implement temporal ensembling and systematic chunk size search

---

## Appendix: Key ACT Hyperparameters

For reference when designing engram experiments:

```
Chunk size (k):              100
Episode length:              400-700 steps
Ratio (k/episode):           ~1:6
Hidden dimension:            512
CVAE latent dim:             32
Beta (KL weight):            10
Transformer encoder layers:  4
Transformer decoder layers:  7
Loss function:               L1 (not L2)
Temporal ensemble weight:    exp(-m * i), m tuned
```

Our engrams:
```
Chunk size:                  16
Sequence length:             ~500 tokens
Ratio:                       ~1:30
Hidden dimension:            3584
Compression method:          Averaging (not learned)
Injection method:            Single prefix (not ensemble)
Scaling factor:              Grid searched (0.5x - 50x)
```

**Key difference**: ACT ratio is 1:6, ours is 1:30. We may be overcompressing relative to optimal.

---

## Experimental Test: Chunk Size Grid Search (2026-01-24)

We tested whether less compression (matching ACT's ratio) would improve semantic separation and directional steering.

### Results Summary

| Chunks | Ratio | Geometric Separation | Directional Steering |
|--------|-------|----------------------|---------------------|
| 8 | 1:27 | 0.29% | **YES** |
| 16 | 1:13 | 0.40% | **YES** |
| 32 | 1:6 | 0.39% | **YES** |
| 64 | 1:3 | **0.42%** (best) | NO (inverted!) |
| 128 | 1:1 | 0.19% | NO (inverted!) |

### Key Finding

**The ACT hypothesis was wrong for our use case.**

- Less compression (64-128 chunks) produced INVERTED steering behavior
- Moderate compression (8-32 chunks) maintained correct directional steering
- Best geometric separation (64 chunks) did NOT produce best functional behavior

### Why the Parallel Breaks Down

| Aspect | ACT (Robotics) | Engrams (LLM) |
|--------|----------------|---------------|
| **Data structure** | Temporal action sequences | Semantic activation patterns |
| **What matters** | Each timestep has causal meaning | Topic/concept activation overall |
| **Noise source** | Execution variability | Token-level spurious correlations |
| **Optimal compression** | Less is better (preserve timesteps) | More is better (regularize noise) |

ACT's action sequences have clear temporal structure where each timestep causally affects the next. Our hidden state sequences are more like bags of semantic activations where averaging acts as beneficial regularization.

### Updated Recommendation

Keep the 1:30 ratio (16 chunks). It's not "overcompression" - it's appropriate regularization for semantic steering in LLMs. The ACT parallel illuminated the structure of our approach but the optimal parameters differ because the underlying data has different properties.
