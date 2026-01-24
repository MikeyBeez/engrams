# Engrams: Geometric State Compression for Transformers

Extract dense semantic representations from transformer hidden states for semantic priming and knowledge steering at inference time.

## Key Results

**Semantic Priming (RAG Enhancement):**
- **Engram + RAG: 51.2%** fact recall vs **RAG only: 41.2%**
- **10 percentage point improvement** from adding engrams to RAG
- **256x compression ratio** (8,192 tokens → 32 vectors)

**Decision Steering (New Research):**
- Engrams extracted from late layers (20-26) can shift probability ratios up to **32x**
- Domain-specific: medical engrams improve medical questions, astronomy engrams don't
- **Critical limitation discovered**: When prompts contain explicit misleading information, the prompt dominates over engram steering

See `docs/follow_up_experiments_findings.md` for detailed analysis of decision steering limitations.

## The Key Insight

Engrams don't replace RAG — they complement it.

Engrams provide "topic cueing" that helps the model access its existing knowledge. For novel information not in training data, RAG is essential. The two mechanisms layer together without interference.

## How It Works

1. **Extract**: Pass document through model, capture hidden states from middle layer (layer 16)
2. **Compress**: Chunk the sequence and mean-pool into N vectors (default: 32)
3. **Inject**: Prepend scaled engram vectors to prompt embeddings at inference time

## Quick Start

```python
from engrams import EngramExtractor, EngramInjector

# Extract engram from document
extractor = EngramExtractor(model, tokenizer, layer=16, num_tokens=32)
engram = extractor.extract(document_text)

# Use engram for Q&A
injector = EngramInjector(model, tokenizer)
answer = injector.inject_and_generate(question, engram)
```

## Recommended Architecture: Engram + RAG

```
┌─────────────────────────────────────────────────┐
│                   Each Turn                      │
├─────────────────────────────────────────────────┤
│  1. Retrieve relevant context (RAG)             │
│  2. Prepend session engram (32 tokens)          │
│  3. Generate response                           │
│  4. Extract engram from response                │
│  5. Update session engram via EMA               │
└─────────────────────────────────────────────────┘
```

The session engram handles everything that overlaps with training data — it makes that knowledge easier to access with focused topic cueing. RAG handles the novel specifics.

## Installation

```bash
git clone https://github.com/MikeyBeez/engrams.git
cd engrams
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.36+
- A GPU with 16GB+ VRAM for 7B models

## Running the Experiments

### 1. Fetch test papers from PubMed

```bash
cd scripts
pip install biopython
python fetch_pubmed_papers.py
```

This downloads 100 neuroscience paper abstracts.

### 2. Run the main experiment (100 turns, Engram + RAG)

```bash
python chained_engram_plus_rag.py --n-papers 100 --alpha 0.1
```

Results are saved to `results/chained_engram_plus_rag_alpha0.1.json`.

### 3. Run comparison tests

**Known vs Novel Facts:**
```bash
python known_facts_test.py
```

**Engram + RAG vs RAG alone:**
```bash
python engram_plus_rag_test.py      # Known facts (WWII)
python engram_plus_rag_novel.py     # Novel facts (biology papers)
```

## Project Structure

```
engrams/
├── engrams/
│   ├── extractor.py      # Core extraction logic
│   ├── injector.py       # Engram injection for generation
│   ├── storage.py        # ChromaDB-based persistence
│   └── context_manager.py # High-level context compression API
├── scripts/
│   ├── wiki_50q_test.py  # Main benchmark script
│   ├── chained_engram_plus_rag.py  # 100-turn Engram+RAG experiment
│   ├── fetch_pubmed_papers.py      # PubMed paper fetcher
│   ├── rag_vs_engram.py  # Comparison tests
│   ├── generation_validation_test.py  # Tests probability vs generation
│   ├── layer_boundary_scan.py     # Fine-grained layer sweep
│   ├── strength_layer_heatmap.py  # Layer × strength grid search
│   ├── aggressive_flip_test.py    # High-strength adversarial test
│   └── negative_control_test.py   # Domain specificity control
├── results/
│   └── *.json            # Experimental results
└── MEDIUM_ARTICLE.txt    # Publication draft
```

## Key Findings

### Semantic Priming (RAG Enhancement)
- **Layer selection matters**: Layers 8-24 work well; layer 0 fails completely
- **Scaling is critical**: Engram vectors must be scaled to match embedding norms
- **Model size matters**: 0.5B models fail; 7B models succeed dramatically
- **Compression improves accuracy**: Counter-intuitively, engrams outperform raw text for known facts
- **Engram + RAG > RAG alone**: 10 percentage point improvement in 100-turn sessions
- **No interference**: Engrams don't hurt RAG performance on novel facts

### Decision Steering (New Research - January 2026)
- **Late layers (20-26) are the decision zone**: Earlier layers handle semantic processing
- **Non-monotonic strength effects**: Too weak = drowned out, too strong = disrupts coherence
- **Domain-specific mechanism**: Irrelevant engrams produce no improvement (negative control)
- **Prompt dominance**: Explicit misleading text in prompts overrides engram steering
- **Probability vs generation gap**: Probability shifts don't always translate to generation changes

## Core Functions

### Extract an engram

```python
def extract_engram(model, tokenizer, text, layer=16, num_tokens=32):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)

    with torch.no_grad():
        outputs = model(**inputs.to(model.device), output_hidden_states=True)

    hidden = outputs.hidden_states[layer]
    seq_len = hidden.shape[1]

    chunk_size = max(1, seq_len // num_tokens)
    engram_vectors = []
    for i in range(num_tokens):
        start = i * chunk_size
        end = start + chunk_size if i < num_tokens - 1 else seq_len
        engram_vectors.append(hidden[0, start:end].mean(dim=0))

    return torch.stack(engram_vectors)
```

### Generate with engram + RAG

```python
def generate_engram_plus_rag(model, tokenizer, context, question, engram):
    embed_layer = model.get_input_embeddings()

    # Scale engram to match embedding norms
    embed_norm = embed_layer.weight.norm(dim=1).mean().item()
    engram_norm = engram.norm(dim=1).mean().item()
    scaled_engram = engram * (embed_norm / engram_norm)

    # Build RAG prompt
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_embeds = embed_layer(inputs.input_ids)

    # Prepend engram
    combined = torch.cat([scaled_engram.unsqueeze(0), prompt_embeds], dim=1)

    with torch.no_grad():
        output = model.generate(inputs_embeds=combined, max_new_tokens=100)

    return tokenizer.decode(output[0], skip_special_tokens=True)
```

### Update session engram via EMA

```python
def ema_update(current_engram, new_engram, alpha=0.1):
    return (1 - alpha) * current_engram + alpha * new_engram
```

## Experimental Results

### 100-Turn Chained Experiment

Processing 100 biology papers, testing recall at turns 25, 50, 75, and 100:

| Condition | Facts Recalled | Percentage |
|-----------|----------------|------------|
| RAG only | 33/80 | 41.2% |
| Engram + RAG | 41/80 | 51.2% |
| Engram only | 11/80 | 13.8% |

**Engram + RAG outperforms RAG alone by +10 percentage points.**

### Wiki Facts (Known Information)

Testing on WWII facts (in training data):

| Condition | Accuracy |
|-----------|----------|
| Baseline | 68% |
| RAG | 80% |
| Engram | 96% |

### Novel Facts (Not in Training)

Testing on recent biology paper abstracts:

| Condition | Accuracy |
|-----------|----------|
| Baseline | 5.6% |
| RAG | 55.6% |
| Engram | 0% |
| Engram + RAG | 55.6% |

The engram doesn't interfere with RAG on novel facts.

## Inspired By

DeepSeek's work on conditional memory in transformers. Their approach uses external hash tables; ours extracts the geometric representations transformers naturally build.

## License

MIT

## Citation

If you use this work, please cite:

```
@misc{engrams2026,
  title={Engrams: Geometric State Compression for Transformers},
  author={MikeyBeez},
  year={2026},
  url={https://github.com/MikeyBeez/engrams}
}
```
