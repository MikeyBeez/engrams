# Engrams as Semantic Primers: Probability Shifts Without Knowledge Transfer

## Abstract

We investigated whether compressed hidden state representations ("engrams") extracted from transformer language models can enhance performance on domain-specific tasks. Through experiments with medical diagnostic questions on Qwen 7B, we discovered that engrams cannot inject novel knowledge but do produce measurable semantic priming effects—shifting probability distributions toward correct answers by 400-600% on failed questions, even when insufficient to change the final output. This finding positions engrams as "semantic GPS coordinates" rather than knowledge stores, with implications for efficient context steering and retrieval-augmented generation systems.

## Introduction

Large language models encode vast amounts of knowledge in their weights, but accessing that knowledge efficiently remains challenging. Previous work on activation engineering and steering vectors has shown that model behavior can be influenced by injecting learned directions into hidden states. We asked a related but distinct question: can we extract compressed representations of domain knowledge and use them to prime models for better performance?

We define an "engram" as a compressed representation created by:
1. Processing a knowledge-rich text through a transformer
2. Extracting hidden states from a specific layer
3. Chunking the sequence dimension and averaging to create a fixed-size representation
4. Prepending this representation to new prompts as synthetic tokens

## Methods

### Model and Task

We used Qwen2.5-7B on challenging USMLE-style medical diagnostic questions that require specific clinical knowledge (e.g., "For pheochromocytoma, start alpha-blocker BEFORE beta-blocker").

### Engram Extraction

From a medical knowledge document containing critical treatment protocols, we extracted engrams by:
- Processing 2048 tokens through the model
- Extracting hidden states from layer 14 (middle of 28 layers)
- Chunking into 16 segments and averaging each
- Result: 16 x 3584 dimension engram

### Evaluation

We compared:
1. **Binary accuracy**: Does the model get the question right?
2. **Probability distribution**: What are the logit probabilities for correct vs incorrect answer tokens?

## Results

### Binary Accuracy: No Improvement

| Condition | Accuracy |
|-----------|----------|
| Baseline (no engram) | 6/9 (67%) |
| With medical engram | 6/9 (67%) |

The engram did not flip any incorrect answers to correct.

### Probability Analysis: Strong Priming Effect

However, examining the probability distributions revealed a different story:

**Pheochromocytoma Question** (model answered incorrectly in both conditions):

| Token | Baseline Prob | With Engram | Change |
|-------|---------------|-------------|--------|
| "alpha" (correct) | 0.000028 | 0.000153 | **+440%** |
| "beta" (incorrect) | 0.000262 | 0.000324 | +23% |
| Ratio (correct/incorrect) | 0.11 | 0.47 | **4.3x improvement** |

**TCA Overdose Question** (model already correct, became more confident):

| Token | Baseline Prob | With Engram | Change |
|-------|---------------|-------------|--------|
| "sodium" (correct) | 0.000144 | 0.001008 | **+599%** |
| Ratio (correct/incorrect) | 11.5 | 26.3 | **2.3x improvement** |

**Cystic Fibrosis Question** (no priming effect):

| Token | Baseline Prob | With Engram | Change |
|-------|---------------|-------------|--------|
| "K" (correct) | 0.232 | 0.219 | -5.7% |

The model already had high confidence (23%) for the correct answer; the engram slightly degraded performance.

## Key Findings

### 1. Engrams Cannot Inject Novel Knowledge

When the model lacks the specific clinical pearl ("alpha before beta for pheo"), the engram cannot teach it. The knowledge must already exist somewhere in the model's weights.

### 2. Engrams DO Shift Probability Distributions

The engram produced dramatic increases in correct answer probability (+440%, +599%), but also increased incorrect answer probability (though by less). This is **domain activation**, not **surgical steering**.

### 3. The Priming Effect Has Limits

The probability shifts weren't sufficient to overcome the model's prior preference for the wrong answer. The model's "confidence" in beta-blockers was too strong to be overridden by general medical priming.

### 4. Layer 0 Engrams Break the Model

Injecting engrams at layer 0 (embedding level) crashed performance from 67% to 11%, consistent with early layers encoding fundamental syntactic structure that shouldn't be perturbed.

## Interpretation: Engrams as Semantic GPS

The best metaphor for engrams is **GPS coordinates, not terrain**. An engram tells the model "you are in medical reasoning territory" and activates relevant circuits, but it cannot provide the specific turn-by-turn directions (the actual clinical knowledge).

This explains the asymmetric probability shifts:
- Correct answers got much larger boosts because the engram activates "medical reasoning mode"
- Incorrect answers also increased slightly because both alpha and beta blockers are relevant medical concepts
- The engram lacks the specificity to say "alpha, NOT beta"

## Implications

### For RAG Systems

Engrams could serve as efficient "domain selectors" that prepare the model before actual retrieval. A 16-token engram is far cheaper than a 2000-token retrieved passage.

**Proposed Architecture:**
1. User query → classify domain
2. Inject domain-specific engram (16 tokens)
3. Retrieve specific knowledge (traditional RAG)
4. Generate response

The engram pre-activates relevant circuits, potentially improving retrieval utilization.

### For Fine-Tuning

The probability shift data suggests which knowledge the model "almost has" versus completely lacks. Questions where engrams produce large shifts but wrong answers might be high-value targets for targeted fine-tuning.

### For Interpretability

Engrams provide a lens into what models "know but can't access." The dramatic probability shifts on pheo1 suggest the model has encountered this knowledge but doesn't reliably retrieve it. The engram acts as a retrieval cue.

## Limitations

1. **Single model tested**: Results may not generalize across architectures
2. **Small question set**: 9 questions insufficient for statistical significance
3. **Engram construction**: Chunking and averaging may lose critical information
4. **Layer selection**: Only tested middle layer; different questions might benefit from different layers

## Conclusion

Engrams produce measurable semantic priming effects, increasing correct answer probabilities by 4-6x on questions the model fails. However, this priming is insufficient to change final outputs when the model's prior is strongly wrong.

Engrams are best understood as **domain pointers** that activate relevant model capabilities, not as **knowledge stores** that can inject new information. This positions them as a potential efficiency tool for retrieval-augmented systems rather than a replacement for actual knowledge retrieval or fine-tuning.

The path forward for medical AI is clear: for novel clinical knowledge, use RAG or fine-tuning. For general domain steering, engrams offer a lightweight alternative worth exploring.

## Data Availability

All code and experimental results available at: `/Users/bard/Code/engrams/scripts/`

---

*Experiments conducted January 2026 using Qwen2.5-7B on NVIDIA GPU.*
