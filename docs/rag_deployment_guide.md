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

## Production Optimization: The Engram Library

For production deployments, extracting engrams on every request is wasteful. Instead, pre-compute a library of **Gold Standard Engrams** from high-quality source texts.

### The Cache Structure

```python
from dataclasses import dataclass
from typing import Dict, Optional
import torch
import hashlib


@dataclass
class EngramMetadata:
    topic_key: str
    source_text_hash: str
    model_id: str
    model_hash: str
    layer: int
    num_tokens: int
    strength: float
    created_at: str


class EngramLibrary:
    """
    Cache of pre-computed engrams indexed by semantic topic.
    """

    def __init__(self, model_id: str, model_hash: str):
        self.model_id = model_id
        self.model_hash = model_hash
        self.cache: Dict[str, torch.Tensor] = {}
        self.metadata: Dict[str, EngramMetadata] = {}

    def add(
        self,
        topic_key: str,
        engram: torch.Tensor,
        source_text: str,
        layer: int = 20,
        strength: float = 1.0
    ):
        """Add an engram to the library."""
        self.cache[topic_key] = engram
        self.metadata[topic_key] = EngramMetadata(
            topic_key=topic_key,
            source_text_hash=hashlib.sha256(source_text.encode()).hexdigest()[:16],
            model_id=self.model_id,
            model_hash=self.model_hash,
            layer=layer,
            num_tokens=engram.shape[0],
            strength=strength,
            created_at=datetime.now().isoformat()
        )

    def get(self, topic_key: str) -> Optional[torch.Tensor]:
        """Retrieve an engram by topic key."""
        return self.cache.get(topic_key)

    def is_compatible(self, current_model_hash: str) -> bool:
        """Check if cached engrams are compatible with current model."""
        return self.model_hash == current_model_hash

    def save(self, path: str):
        """Persist library to disk."""
        torch.save({
            'cache': self.cache,
            'metadata': self.metadata,
            'model_id': self.model_id,
            'model_hash': self.model_hash
        }, path)

    @classmethod
    def load(cls, path: str) -> 'EngramLibrary':
        """Load library from disk."""
        data = torch.load(path)
        library = cls(data['model_id'], data['model_hash'])
        library.cache = data['cache']
        library.metadata = data['metadata']
        return library
```

### Pre-computing Gold Standard Engrams

```python
# Build library from authoritative sources
library = EngramLibrary(
    model_id="Qwen/Qwen2.5-7B",
    model_hash=get_model_hash(model)
)

# Medical domain engrams
medical_sources = {
    "topic:pheochromocytoma": "Pheochromocytoma treatment requires alpha-blocker first...",
    "topic:malignant_hyperthermia": "Malignant hyperthermia requires immediate dantrolene...",
    "topic:tca_overdose": "TCA overdose with QRS widening requires sodium bicarbonate...",
    "topic:wernicke": "Wernicke encephalopathy requires thiamine before glucose...",
    "topic:anaphylaxis": "Anaphylaxis requires epinephrine as first-line treatment...",
}

extractor = EngramExtractor(model, tokenizer, layer=20)

for topic_key, source_text in medical_sources.items():
    engram = extractor.extract(source_text)
    library.add(topic_key, engram, source_text)

library.save("engram_library_medical_v1.pt")
```

### Batched Inference (Single Forward Pass)

```python
def fast_calibrated_inference(
    model,
    tokenizer,
    prompt: str,
    engram: torch.Tensor,
    strength: float = 5.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run baseline and engram-assisted inference in a single batched forward pass.
    Returns (baseline_logits, engram_logits).
    """
    embed = model.get_input_embeddings()
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_embeds = embed(inputs.input_ids)

    # Scale engram
    e_norm = embed.weight.norm(dim=1).mean().item()
    g_norm = engram.norm(dim=1).mean().item()
    scaled_engram = engram * (e_norm / g_norm) * strength

    # Create padding to match sequence lengths for batching
    padding = torch.zeros_like(scaled_engram)

    # Baseline: padding + prompt
    baseline_embeds = torch.cat([padding.unsqueeze(0), input_embeds], dim=1)

    # Engram: scaled_engram + prompt
    engram_embeds = torch.cat([scaled_engram.unsqueeze(0).to(input_embeds.dtype), input_embeds], dim=1)

    # Batch both together
    batched_embeds = torch.cat([baseline_embeds, engram_embeds], dim=0)

    with torch.no_grad():
        outputs = model(inputs_embeds=batched_embeds)

    # Split results
    baseline_logits = outputs.logits[0, -1, :]
    engram_logits = outputs.logits[1, -1, :]

    return baseline_logits, engram_logits
```

### Semantic Router

Map user queries to cached engrams using a lightweight classifier:

```python
from sentence_transformers import SentenceTransformer
import numpy as np


class SemanticRouter:
    """
    Routes queries to appropriate cached engrams using embedding similarity.
    """

    def __init__(self, library: EngramLibrary):
        self.library = library
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, small model
        self.topic_embeddings = {}
        self._build_index()

    def _build_index(self):
        """Pre-compute embeddings for topic keys."""
        for topic_key in self.library.cache.keys():
            # Convert topic key to natural language for embedding
            topic_text = topic_key.replace("topic:", "").replace("_", " ")
            self.topic_embeddings[topic_key] = self.encoder.encode(topic_text)

    def route(self, query: str, threshold: float = 0.5) -> Optional[str]:
        """
        Find best matching topic for a query.
        Returns topic_key if similarity > threshold, else None.
        """
        query_embedding = self.encoder.encode(query)

        best_match = None
        best_score = 0

        for topic_key, topic_embedding in self.topic_embeddings.items():
            score = np.dot(query_embedding, topic_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(topic_embedding)
            )
            if score > best_score:
                best_score = score
                best_match = topic_key

        if best_score >= threshold:
            return best_match
        return None
```

### Complete Optimized Pipeline

```python
class OptimizedEngramPipeline:
    """
    Production-ready pipeline with caching and batched inference.
    """

    def __init__(
        self,
        model,
        tokenizer,
        library: EngramLibrary,
        retriever,
        engram_strength: float = 5.0,
        routing_threshold: float = 0.5
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.library = library
        self.retriever = retriever
        self.router = SemanticRouter(library)
        self.extractor = EngramExtractor(model, tokenizer)
        self.strength = engram_strength
        self.threshold = routing_threshold

    def query(self, question: str) -> Dict:
        # Step 1: Retrieve context
        docs = self.retriever.search(question, top_k=3)
        context = "\n\n".join([d.text for d in docs])
        full_prompt = f"{context}\n\n{question}"

        # Step 2: Route to cached engram (fast path)
        topic_key = self.router.route(question, self.threshold)

        if topic_key:
            engram = self.library.get(topic_key)
            cache_hit = True
        else:
            # Fallback: extract from retrieved context (slow path)
            engram = self.extractor.extract(context)
            cache_hit = False

        # Step 3: Batched inference (single forward pass)
        baseline_logits, engram_logits = fast_calibrated_inference(
            self.model, self.tokenizer, full_prompt, engram, self.strength
        )

        # Step 4: Calibrate and route
        # ... (same calibration logic as before)

        return {
            "answer": answer,
            "confidence": confidence,
            "cache_hit": cache_hit,
            "topic_matched": topic_key
        }
```

### Performance Impact

| Operation | Without Cache | With Cache |
|-----------|---------------|------------|
| Engram extraction | ~50ms | 0ms (pre-computed) |
| Semantic routing | N/A | ~5ms |
| Inference | 2x forward passes | 1x batched forward pass |
| **Total overhead** | ~150ms | ~55ms |

### Handling Model Drift

When updating your model, cached engrams become invalid:

```python
def check_and_refresh_library(
    model,
    tokenizer,
    library: EngramLibrary,
    source_texts: Dict[str, str]
) -> EngramLibrary:
    """
    Check if library is compatible with current model.
    If not, regenerate all engrams.
    """
    current_hash = get_model_hash(model)

    if library.is_compatible(current_hash):
        return library

    # Regenerate library
    print(f"Model changed ({library.model_hash} -> {current_hash}), regenerating engrams...")

    new_library = EngramLibrary(
        model_id=library.model_id,
        model_hash=current_hash
    )

    extractor = EngramExtractor(model, tokenizer)
    for topic_key, source_text in source_texts.items():
        engram = extractor.extract(source_text)
        new_library.add(topic_key, engram, source_text)

    return new_library
```

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
