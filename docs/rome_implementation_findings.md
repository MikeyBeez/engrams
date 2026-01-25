# ROME Implementation Findings

## Summary

We implemented a minimal ROME (Rank-One Model Editing) system from scratch after EasyEdit proved too complex and memory-hungry. Our implementation successfully modifies factual associations in Qwen2.5-3B.

## What ROME Does

ROME edits factual associations by making rank-one updates to MLP weight matrices:

```
W_new = W_old + (v_target - v_current) * k^T / (k^T * k)
```

Where:
- `k` = MLP input vector at the subject's last token position
- `v_current` = current MLP output for that input
- `v_target` = desired MLP output (computed to favor the target token)

## Implementation Details

Our minimal implementation (`src/rome_edit.py`):

1. **Key vector computation**: Extract hidden state at subject's last token
2. **Target value computation**: Use lm_head weights to find direction that favors target token
3. **Rank-one update**: Apply the update to MLP's down_proj weights

Key parameters that matter:
- **Layer selection**: Middle layers (~layer 15 for 36-layer models) store factual knowledge
- **Value shift magnitude**: We use 5x the hidden state norm - smaller values don't change behavior
- **Multi-layer editing**: Complex knowledge may require editing multiple layers

## Experimental Results

### Experiment 1: Simple Fact (France → London)

```
BASELINE: "The capital of France is Paris"
AFTER EDIT (layer 15, 4.98% weight change):
"The capital of France is London"
```

Single-layer edit successfully changed simple factual association.

### Experiment 2: Medical Knowledge (Malignant Hyperthermia)

Attempted to change "malignant hyperthermia → dantrolene" to "malignant hyperthermia → succinylcholine"

**Single layer (layer 15)**: No effect - model still says Dantrolene

**Multi-layer (layers 10, 15, 20)**: Partial effect
- Model now produces "succinylcholine" in some outputs
- Output becomes confused/mixed
- Original knowledge not fully overridden

This suggests medical knowledge is:
1. Distributed across multiple layers
2. More robustly encoded than simple facts
3. Requires stronger/broader edits to fully modify

## Implications for Semantic Sinks

Original hypothesis: Semantic sinks (high centroid similarity between concepts) might cause routing errors that could be fixed via weight editing.

New understanding:
1. **ROME can modify associations** - both introducing and potentially correcting errors
2. **Well-trained knowledge is robust** - not easily overridden by single-layer edits
3. **The model already knows correct medical facts** - MH → Dantrolene is correct in baseline
4. **We need actual errors to fix** - The "semantic sink" may not cause problems in practice

## Technical Notes

- EasyEdit library was abandoned due to:
  - Excessive dependencies (timm, fairscale, zhipuai, etc.)
  - Memory issues (tries to load full model to single GPU)
  - Complexity obscures what's actually happening

- Our implementation is ~200 lines of Python
- Works with any Qwen2-style model
- Easily extensible to other architectures

## Next Steps

1. Find cases where models actually make factual errors
2. Test if ROME can correct those errors
3. Explore connection to centroid geometry - do edits change centroid positions?
4. Consider MEMIT for batch editing multiple facts

## Code Location

- Implementation: `/home/bee/Code/engrams/src/rome_edit.py`
- Based on: "Locating and Editing Factual Associations in GPT" (Meng et al., 2022)
