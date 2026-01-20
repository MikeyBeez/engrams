# Honest Assessment: What Engrams Actually Do

## The Original Claim

We reported 96% accuracy for engrams vs 80% for RAG on 50 WWII questions, claiming a major breakthrough in context compression.

## The Critique

A reader pointed out several issues:
1. Our "RAG" baseline was naive context stuffing, not real retrieval
2. The questions were high-frequency facts the model already knows
3. We didn't test a no-context baseline
4. The evaluation may have been measuring pretrained knowledge, not engram effectiveness

## The Follow-Up Experiments

### Experiment 1: No-Context Baseline

We tested how well the model answers the same 50 questions with NO context at all.

Results:
- Baseline (no context): 38/50 (76%)
- RAG (stuffed context): 40/50 (80%)
- Engram: 48/50 (96%)

The model already knows 76% of the answers from pretraining. RAG adds 4 percentage points. Engrams add 20 percentage points.

### Experiment 2: Low-Frequency Facts

We created 20 questions about obscure WWII facts the model is less likely to know.

Results:
- Baseline (no context): 11/20 (55%)
- RAG (stuffed context): 12/20 (60%)  
- Engram: 13/20 (65%)

On hard questions, all methods struggle. Engrams add only 2 questions over baseline.

## What This Means

### The critic was partially right:
- The original 96% vs 80% comparison was inflated
- Many questions tested pretrained knowledge, not retrieval
- The effect size on genuinely hard questions is modest

### But engrams do something real:
- On the original 50 questions: +20 points over baseline (vs +4 for RAG)
- On low-frequency questions: +10 points over baseline (vs +5 for RAG)
- Engrams consistently outperform naive RAG
- The compression (256x) is real

### The honest conclusion:

Engrams are a valid compression technique that preserves semantic information better than expected. They outperform naive context stuffing. But they are not magic - on facts the model doesn't know, the improvement is modest.

The original framing overstated the results. The technique is interesting and worth exploring, but "96% vs 80%" was comparing against an easy baseline on easy questions.

## What Remains to Test

1. **Proper RAG comparison**: Implement real retrieval with embeddings and top-k
2. **Contradictory facts**: Test with information that conflicts with model priors
3. **Multi-hop reasoning**: Questions requiring combining multiple facts
4. **Other domains**: Technical docs, code, fiction - not just history

## Updated Results Table

| Test | Baseline | RAG | Engram | Engram Lift |
|------|----------|-----|--------|-------------|
| 50 WWII (easy) | 76% | 80% | 96% | +20 pts |
| 20 WWII (hard) | 55% | 60% | 65% | +10 pts |

The compression is real (256x). The accuracy improvement is real but smaller than originally claimed on hard questions.

## Credit

Thanks to the reader who pushed back on the original claims. Science requires honest assessment of results.

---

*Updated: 2026-01-20*
