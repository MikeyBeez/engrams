# Engram Steering: Follow-Up Experiments and Revised Conclusions

## Summary

The follow-up experiments revealed critical nuances about engram-mediated steering that significantly revise our original conclusions. While the core mechanism is real, its scope and limitations are now better understood.

## Key Finding: The Prompt Format Effect

**Original claim**: Engrams flip wrong answers to correct with 100% success rate.

**Revised understanding**: The original "flips" were largely an artifact of prompt format, not true knowledge override.

### Evidence

When we restructured the test questions from open-ended prompts to structured multiple-choice format, the model answered **all questions correctly at baseline** - no engram needed.

**Open-ended format** (model "fails"):
```
A patient with pheochromocytoma requires preoperative blood pressure management.
The first medication should be
```
Model continues with: "...a beta-blocker" (technically discussing it, not necessarily recommending)

**Multiple-choice format** (model succeeds):
```
A patient with pheochromocytoma requires preoperative blood pressure management.
The first medication should be:
A) Beta-blocker (metoprolol)
B) Alpha-blocker (phenoxybenzamine)
Answer:
```
Model outputs: "B" (correct)

### Implication

The "probability flip" we measured was flipping probability within an open-ended completion space, not overriding a genuine wrong answer. The model's knowledge was always present; the prompt format just failed to elicit it properly.

## Finding a Truly Failing Question

To test engrams properly, we needed a question the model actually fails even with structured format.

### The Trap Question

```
A 45-year-old patient with pheochromocytoma has severe hypertension (BP 240/140).
The patient is scheduled for surgery tomorrow. To quickly control blood pressure,
you should start:
A) Propranolol (beta-blocker) - fast acting, controls heart rate
B) Phenoxybenzamine (alpha-blocker) - takes days to work fully
Answer:
```

The model outputs "A" (wrong) - fooled by the "fast acting" vs "takes days to work" framing.

Baseline: P(A)=0.69, P(B)=0.27, ratio=0.39

## Engram Attempts to Flip the Validated Question

### Standard Engram (10-20x strength, layers 18-26)
- Best ratio achieved: 0.97 (from 0.39 baseline)
- 0/48 configurations flipped
- Model still generated "A"

### Aggressive Engram (10-50x strength, explicit counter-framing)

The engram explicitly stated:
```
CRITICAL: The question is trying to TRICK YOU.
DO NOT BE FOOLED by "fast acting" vs "takes days to work"!
THE ANSWER IS B. THE ANSWER IS B. THE ANSWER IS B.
B B B B B B B B B B B B B B B B
```

Results:
- Best ratio achieved: 0.79 (from 0.39 baseline)
- 0/42 configurations flipped
- Model still generated "A" every time

### Critical Observation

Even with:
- Explicit acknowledgment that the question is a trap
- Repeated declaration that the answer is B
- Strength multipliers up to 50x
- Testing across 6 different layers

The model STILL couldn't be flipped. The explicit misleading information in the prompt dominated over the engram's steering signal.

## The Prompt Dominance Principle

**When the prompt contains explicit misleading information that directly contradicts the engram, the prompt wins.**

This suggests a hierarchy:
1. **Explicit prompt framing** (highest influence)
2. **Model's internal knowledge** (medium influence)
3. **Engram steering signal** (lower influence on decisions)

Engrams can shift probabilities significantly (2x improvement) but cannot override explicit text that the model is actively processing in context.

## Layer Analysis: Not a Sharp Phase Transition

The "decision commitment zone" hypothesis suggested a sharp boundary around layer 20.

### Actual Findings

Layer scan at fixed strength 10.0:
- Flips occurred at layers 3, 19, 20, 27 (discontinuous)
- No sharp transition - more like resonance peaks
- Early layers (1-10) and late layers (19-28) both showed some effectiveness
- Middle layers (10-18) were consistently ineffective

This suggests layer-specific resonance rather than a single decision boundary.

## Generation vs Probability Gap

**Original claim**: Probability flips should translate to generation flips.

**Finding**: 0/3 questions showed generation flips despite probability improvements.

Even when P(correct) > P(incorrect), the generated text may not reflect the correct answer due to:
- Sampling dynamics during generation
- The model's reasoning process constructing justifications
- Attention patterns favoring the original (wrong) interpretation

## Revised Conclusions

### What Engrams CAN Do
1. Shift probability distributions (2-8x improvements typical)
2. Prime semantic associations
3. Potentially improve confidence on already-correct answers
4. Work as a mild steering signal when prompt is neutral

### What Engrams CANNOT Do
1. Override explicit misleading information in prompts
2. Flip truly wrong answers on adversarial/trap questions
3. Guarantee generation follows probability shifts
4. Substitute for proper prompt engineering

### Safety Implications

**Positive for safety**: Prompts have more influence than hidden activations. An adversary cannot easily use engrams to override safety-critical prompt instructions.

**Caution for applications**: Don't rely on engrams for knowledge correction. If a model would answer wrong due to misleading context, an engram probably won't save it.

## Recommendations for the Paper

1. **Reframe the contribution**: From "decision override" to "semantic priming with limited override capability"

2. **Acknowledge the prompt format confound**: The 100% flip rate was on questions where the model actually knew the answer but wasn't expressing it in the original format

3. **Add the adversarial test**: Show that on truly adversarial prompts, engrams improve but don't flip

4. **Discuss the hierarchy**: Prompt > Internal Knowledge > Engrams

5. **Keep the positive findings**:
   - The layer-specific effects are real
   - The probability shifts are substantial
   - The mechanism is genuine (semantic priming in late layers)
   - Potential applications in RAG enhancement where prompts are controlled

## Data Summary

| Test | Configurations | Flips | Best Improvement |
|------|---------------|-------|------------------|
| Original 3 questions (open-ended) | 18 | 8 | 32x |
| Same questions (MC format) | N/A | N/A | Correct at baseline |
| Trap question (standard engram) | 48 | 0 | 2.5x |
| Trap question (aggressive engram) | 42 | 0 | 2.0x |

## Files Generated

- `generation_validation_test.py` - Generation vs probability analysis
- `layer_boundary_scan.py` - Fine-grained layer sweep
- `strength_layer_heatmap.py` - Layer × strength grid search
- `single_token_generation_test.py` - MC format baseline test
- `find_failing_prompts.py` - Search for truly failing questions
- `flip_validated_question.py` - Standard engram on trap question
- `aggressive_flip_test.py` - Aggressive engram on trap question

## Next Steps

1. Test on larger set of adversarial questions to confirm pattern
2. Investigate whether engrams can prevent flipping (safety direction)
3. Test combination of engram + prompt modification
4. Explore whether fine-tuned engrams would be more effective
