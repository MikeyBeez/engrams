# RAG Deployment Guide: Engram-Based Confidence Calibration

**Authors:** Mikey and Claude
**Date:** January 2026

This guide shows how to integrate engram-based confidence calibration into a production RAG pipeline.

---

## The Problem

Standard RAG pipelines have a blind spot: they can't distinguish between answers the model "knows" versus answers it's guessing based on semantic proximity. A model might output the right answer while sitting in a dangerous neighborhood—one paraphrase away from hallucinating.

## The Solution

Use engrams as a "topic lens" to stress-test model answers. By comparing baseline output to engram-assisted output, we classify confidence into four cases and route accordingly.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAG + Engram Pipeline                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   [User Query]                                                       │
│        │                                                             │
│        ▼                                                             │
│   ┌─────────┐                                                        │
│   │ Retriever│ ──────────────────────┐                               │
│   └─────────┘                        │                               │
│        │                             │                               │
│        ▼                             ▼                               │
│   [Retrieved Docs]           [Extract Engram]                        │
│        │                             │                               │
│        ▼                             │                               │
│   ┌──────────────────────────────────┴───────────┐                   │
│   │                                              │                   │
│   │  ┌─────────────┐        ┌─────────────────┐  │                   │
│   │  │  Baseline   │        │ Engram-Assisted │  │                   │
│   │  │  Inference  │        │   Inference     │  │                   │
│   │  └─────────────┘        └─────────────────┘  │                   │
│   │        │                        │            │                   │
│   │        └──────────┬─────────────┘            │                   │
│   │                   ▼                          │                   │
│   │          [Confidence Calibration]            │                   │
│   │                   │                          │                   │
│   └───────────────────┼──────────────────────────┘                   │
│                       ▼                                              │
│   ┌───────────────────────────────────────────────────────┐          │
│   │                    Router                             │          │
│   ├───────────┬───────────┬───────────┬──────────────────┤          │
│   │  ROBUST   │  FRAGILE  │  STUCK    │  RECOVERED       │          │
│   │  CORRECT  │  CORRECT  │  WRONG    │  KNOWLEDGE       │          │
│   └─────┬─────┴─────┬─────┴─────┬─────┴────────┬─────────┘          │
│         │           │           │              │                     │
│         ▼           ▼           ▼              ▼                     │
│     [Return]    [Flag for   [Fallback]    [Return]                   │
│                  Review]                                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation

### Step 1: Core Components

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from typing import Dict, Tuple, Literal
from dataclasses import dataclass
from enum import Enum


class ConfidenceLevel(Enum):
    ROBUST_CORRECT = "robust_correct"
    FRAGILE_CORRECT = "fragile_correct"
    HIGH_CONFIDENCE_INCORRECT = "high_confidence_incorrect"
    RECOVERED_KNOWLEDGE = "recovered_knowledge"


@dataclass
class CalibrationResult:
    confidence: ConfidenceLevel
    baseline_ratio: float
    engram_ratio: float
    baseline_answer: str
    engram_answer: str
    recommended_action: str
```

### Step 2: Engram Extractor

```python
class EngramExtractor:
    def __init__(self, model, tokenizer, layer: int = 20, num_tokens: int = 16):
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.num_tokens = num_tokens

    def extract(self, text: str) -> torch.Tensor:
        """Extract engram from text at specified layer."""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[self.layer]
        seq_len = hidden.shape[1]
        chunk_size = max(1, seq_len // self.num_tokens)

        vectors = []
        for i in range(self.num_tokens):
            start = i * chunk_size
            end = start + chunk_size if i < self.num_tokens - 1 else seq_len
            if start < seq_len:
                vectors.append(hidden[0, start:end].mean(dim=0))
            else:
                vectors.append(hidden[0, -1, :])

        return torch.stack(vectors)
```

### Step 3: Inference with Engram Injection

```python
class EngramInference:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.embed = model.get_input_embeddings()

    def get_token_probs(
        self,
        prompt: str,
        target_tokens: list[str],
        engram: torch.Tensor = None,
        strength: float = 1.0
    ) -> Dict[str, float]:
        """Get probabilities for target tokens, optionally with engram."""

        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)

        if engram is not None:
            # Scale engram to match embedding magnitude
            e_norm = self.embed.weight.norm(dim=1).mean().item()
            g_norm = engram.norm(dim=1).mean().item()
            scaled = engram * (e_norm / g_norm) * strength

            # Prepend to embeddings
            emb = self.embed(inputs.input_ids)
            combined = torch.cat([scaled.unsqueeze(0).to(emb.dtype), emb], dim=1)

            with torch.no_grad():
                outputs = self.model(inputs_embeds=combined)
        else:
            with torch.no_grad():
                outputs = self.model(**inputs)

        probs = F.softmax(outputs.logits[0, -1, :], dim=-1)

        result = {}
        for token in target_tokens:
            token_id = self.tokenizer.encode(token, add_special_tokens=False)[0]
            result[token] = probs[token_id].item()

        return result
```

### Step 4: Confidence Calibrator

```python
class ConfidenceCalibrator:
    def __init__(self, model, tokenizer, engram_strength: float = 5.0):
        self.extractor = EngramExtractor(model, tokenizer)
        self.inference = EngramInference(model, tokenizer)
        self.strength = engram_strength

    def calibrate(
        self,
        prompt: str,
        retrieved_context: str,
        candidate_answers: Tuple[str, str]  # (expected_correct, expected_incorrect)
    ) -> CalibrationResult:
        """
        Run confidence calibration on a prompt with retrieved context.

        Args:
            prompt: The user's question
            retrieved_context: Text from RAG retrieval
            candidate_answers: Tuple of (correct_token, incorrect_token)

        Returns:
            CalibrationResult with confidence level and recommended action
        """
        correct_tok, incorrect_tok = candidate_answers

        # Extract engram from retrieved context
        engram = self.extractor.extract(retrieved_context)

        # Full prompt with context
        full_prompt = f"{retrieved_context}\n\n{prompt}"

        # Get baseline probabilities
        baseline_probs = self.inference.get_token_probs(
            full_prompt,
            [correct_tok, incorrect_tok]
        )
        baseline_ratio = baseline_probs[correct_tok] / baseline_probs[incorrect_tok]

        # Get engram-assisted probabilities
        engram_probs = self.inference.get_token_probs(
            full_prompt,
            [correct_tok, incorrect_tok],
            engram=engram,
            strength=self.strength
        )
        engram_ratio = engram_probs[correct_tok] / engram_probs[incorrect_tok]

        # Classify confidence
        confidence = self._classify(baseline_ratio, engram_ratio)

        # Determine answers
        baseline_answer = correct_tok if baseline_ratio > 1 else incorrect_tok
        engram_answer = correct_tok if engram_ratio > 1 else incorrect_tok

        # Get recommended action
        action = self._get_action(confidence)

        return CalibrationResult(
            confidence=confidence,
            baseline_ratio=baseline_ratio,
            engram_ratio=engram_ratio,
            baseline_answer=baseline_answer,
            engram_answer=engram_answer,
            recommended_action=action
        )

    def _classify(
        self,
        baseline_ratio: float,
        engram_ratio: float
    ) -> ConfidenceLevel:
        """Classify into four confidence cases."""

        if baseline_ratio > 1.0 and engram_ratio > baseline_ratio:
            return ConfidenceLevel.ROBUST_CORRECT

        if baseline_ratio > 1.0 and engram_ratio < 1.0:
            return ConfidenceLevel.FRAGILE_CORRECT

        if baseline_ratio < 1.0 and engram_ratio < 1.0:
            return ConfidenceLevel.HIGH_CONFIDENCE_INCORRECT

        if baseline_ratio < 1.0 and engram_ratio > 1.0:
            return ConfidenceLevel.RECOVERED_KNOWLEDGE

        # Edge case: baseline > 1 but engram < baseline (weakened but not flipped)
        return ConfidenceLevel.ROBUST_CORRECT

    def _get_action(self, confidence: ConfidenceLevel) -> str:
        """Get recommended action for each confidence level."""

        actions = {
            ConfidenceLevel.ROBUST_CORRECT: "return_answer",
            ConfidenceLevel.FRAGILE_CORRECT: "flag_for_review",
            ConfidenceLevel.HIGH_CONFIDENCE_INCORRECT: "trigger_fallback",
            ConfidenceLevel.RECOVERED_KNOWLEDGE: "return_engram_answer",
        }
        return actions[confidence]
```

### Step 5: RAG Pipeline Integration

```python
class EngramRAGPipeline:
    def __init__(
        self,
        model,
        tokenizer,
        retriever,  # Your retrieval system
        engram_strength: float = 5.0
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.calibrator = ConfidenceCalibrator(model, tokenizer, engram_strength)
        self.inference = EngramInference(model, tokenizer)
        self.extractor = EngramExtractor(model, tokenizer)

    def query(
        self,
        question: str,
        candidate_answers: Tuple[str, str] = None,
        auto_detect_candidates: bool = True
    ) -> Dict:
        """
        Process a query through the RAG pipeline with confidence calibration.

        Returns dict with answer, confidence, and metadata.
        """

        # Step 1: Retrieve relevant context
        retrieved_docs = self.retriever.search(question, top_k=3)
        context = "\n\n".join([doc.text for doc in retrieved_docs])

        # Step 2: If no candidates provided, get top-2 from baseline
        if candidate_answers is None and auto_detect_candidates:
            candidate_answers = self._detect_candidates(question, context)

        # Step 3: Run calibration
        result = self.calibrator.calibrate(
            prompt=question,
            retrieved_context=context,
            candidate_answers=candidate_answers
        )

        # Step 4: Route based on confidence
        return self._route(result, question, context)

    def _detect_candidates(
        self,
        question: str,
        context: str
    ) -> Tuple[str, str]:
        """Auto-detect top 2 candidate answers from baseline inference."""

        full_prompt = f"{context}\n\n{question}"
        inputs = self.tokenizer(full_prompt, return_tensors='pt').to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = F.softmax(outputs.logits[0, -1, :], dim=-1)
        top_indices = probs.topk(2).indices

        top1 = self.tokenizer.decode(top_indices[0])
        top2 = self.tokenizer.decode(top_indices[1])

        return (top1, top2)

    def _route(
        self,
        result: CalibrationResult,
        question: str,
        context: str
    ) -> Dict:
        """Route based on calibration result."""

        if result.recommended_action == "return_answer":
            return {
                "answer": result.baseline_answer,
                "confidence": result.confidence.value,
                "confidence_score": result.baseline_ratio,
                "action_taken": "direct_return",
                "needs_review": False
            }

        elif result.recommended_action == "return_engram_answer":
            return {
                "answer": result.engram_answer,
                "confidence": result.confidence.value,
                "confidence_score": result.engram_ratio,
                "action_taken": "engram_recovery",
                "needs_review": False,
                "note": "Answer recovered via topic priming"
            }

        elif result.recommended_action == "flag_for_review":
            return {
                "answer": result.baseline_answer,
                "confidence": result.confidence.value,
                "confidence_score": result.baseline_ratio,
                "action_taken": "flagged",
                "needs_review": True,
                "warning": "Answer is fragile - may be hallucination",
                "baseline_ratio": result.baseline_ratio,
                "engram_ratio": result.engram_ratio
            }

        elif result.recommended_action == "trigger_fallback":
            # Implement your fallback strategy here
            return {
                "answer": None,
                "confidence": result.confidence.value,
                "action_taken": "fallback_triggered",
                "needs_review": True,
                "error": "Model is confidently incorrect - requires intervention",
                "fallback_options": [
                    "retry_with_different_retrieval",
                    "escalate_to_human",
                    "return_uncertainty"
                ]
            }
```

---

## Usage Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

# Initialize pipeline (assuming you have a retriever)
pipeline = EngramRAGPipeline(
    model=model,
    tokenizer=tokenizer,
    retriever=your_retriever,
    engram_strength=5.0
)

# Query with automatic candidate detection
result = pipeline.query(
    question="A patient with pheochromocytoma needs preoperative BP control. First medication?"
)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
print(f"Needs Review: {result['needs_review']}")

# Query with known candidates (for evaluation)
result = pipeline.query(
    question="Treatment for malignant hyperthermia in OR?",
    candidate_answers=(" dantrolene", " cooling")
)

if result['needs_review']:
    print(f"WARNING: {result.get('warning', result.get('error'))}")
```

---

## Routing Logic Summary

| Confidence Level | Baseline | Engram | Action | Why |
|-----------------|----------|--------|--------|-----|
| ROBUST_CORRECT | ✓ Correct | ✓ Stronger | Return answer | Model knows it, topic priming confirms |
| FRAGILE_CORRECT | ✓ Correct | ✗ Flipped | Flag for review | Answer is in semantic sink |
| HIGH_CONFIDENCE_INCORRECT | ✗ Wrong | ✗ Still wrong | Trigger fallback | Model is stuck, needs intervention |
| RECOVERED_KNOWLEDGE | ✗ Wrong | ✓ Flipped correct | Return engram answer | Topic priming activated dormant knowledge |

---

## Tuning Parameters

### Engram Strength

| Strength | Use Case | Risk |
|----------|----------|------|
| 1.0 | Conservative calibration | May miss FRAGILE cases |
| 5.0 | **Recommended default** | Good balance |
| 10.0 | Aggressive detection | May trigger false FRAGILE |

### Extraction Layer

| Layer | What it captures |
|-------|------------------|
| 16 | Middle layers - facts + personality |
| 20 | **Recommended** - decision zone |
| 24-26 | Late layers - output formatting |

### Number of Engram Tokens

| Tokens | Compression | Detail |
|--------|-------------|--------|
| 8 | 512x | Fast, coarse |
| 16 | **256x (default)** | Good balance |
| 32 | 128x | More detail, slower |

---

## Production Considerations

### Latency

The calibration adds one extra forward pass. For a 7B model:
- Baseline inference: ~50ms
- Engram extraction: ~50ms
- Engram inference: ~50ms
- **Total overhead: ~100ms**

For latency-critical applications, consider:
1. Only calibrate high-stakes queries
2. Run calibration async and update confidence post-hoc
3. Cache engrams for frequently-used contexts

### Batching

Engram extraction can be batched across documents:
```python
engrams = [extractor.extract(doc) for doc in retrieved_docs]
combined_engram = torch.stack(engrams).mean(dim=0)
```

### Monitoring

Log these metrics:
- Distribution of confidence levels
- FRAGILE_CORRECT rate (should be < 10% for good retrieval)
- RECOVERED_KNOWLEDGE rate (indicates engrams are helping)
- Fallback trigger rate (indicates retrieval quality issues)

---

## When NOT to Use This

1. **Simple factual lookups** where retrieval is the answer (no generation needed)
2. **Creative tasks** where there's no "correct" answer
3. **Multi-turn conversations** (calibration is per-turn, not conversation-aware)
4. **Extremely latency-sensitive** applications (adds ~100ms)

---

## Summary

This deployment pattern turns engrams from a "steering" tool into a "sonar" tool:

1. **Retrieve** context as usual
2. **Extract** topic engram from context
3. **Compare** baseline vs engram-assisted inference
4. **Route** based on the four confidence cases

The key win is detecting **FRAGILE_CORRECT** answers—those that look right but are one perturbation away from hallucinating. This catches failures that naive confidence scores miss.
