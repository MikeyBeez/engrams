# Engrams

Research project exploring **learned semantic compression** - extracting dense representations from transformer hidden states to compress knowledge (e.g., Wikipedia articles) into token-sized "engrams" that can be injected into new prompts.

## Concept

Instead of relying solely on a model's parametric knowledge:

```
Standard: "Tell me about Abraham Lincoln" → model recalls from weights
Engrams:  "Tell me about [ENGRAM][ENGRAM]" → inject compressed Wikipedia article
```

The engram vectors are extracted from the model's own hidden states when processing the full article, then stored and reused.

## Inspiration

Inspired by [DeepSeek's Engram](https://github.com/deepseek-ai/Engram) research on conditional memory for LLMs, which demonstrated that separating static knowledge lookup from dynamic reasoning can improve efficiency and performance.

## Installation

```bash
cd ~/Code/engrams

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
from engrams import EngramExtractor, EngramInjector, EngramStore
from engrams.extractor import ExtractionConfig
from engrams.wikipedia import WikipediaEngramBuilder

# Build an engram from Wikipedia
builder = WikipediaEngramBuilder()
engram = builder.build_engram("Abraham Lincoln")

print(f"Compressed {engram.source_length} tokens → {engram.vectors.shape[0]} vectors")
print(f"Compression ratio: {engram.compression_ratio:.1f}x")

# Inject and generate
injector = EngramInjector()
response = injector.inject_and_generate(
    prompt="Abraham Lincoln was",
    engram=engram,
    max_new_tokens=100,
)
print(response)
```

## Architecture

```
Wikipedia Article (8000 tokens)
         ↓
   Transformer Model
         ↓
   Extract hidden states at layer N
         ↓
   Pool: [8000, hidden_dim] → [k, hidden_dim]
         ↓
   Store k engram vectors
         ↓
   Later: inject as "synthetic tokens"
```

## Project Structure

```
engrams/
├── engrams/
│   ├── __init__.py
│   ├── extractor.py    # Extract engrams from documents
│   ├── injector.py     # Inject engrams into prompts
│   ├── storage.py      # Persist engrams (ChromaDB)
│   └── wikipedia.py    # Wikipedia integration
├── notebooks/
│   └── 01_extraction_experiments.ipynb
├── tests/
├── data/               # Stored engrams and results
└── scripts/
```

## Research Questions

1. **Which layer?** Early layers are lexical, late layers are task-specific. Middle layers often capture semantics best.

2. **Pooling strategy?**
   - Mean pooling (implemented)
   - Attention-weighted pooling
   - Learned compression head

3. **Injection method?**
   - Replace placeholder tokens
   - Prefix (prepend to sequence)
   - Add to existing embeddings

4. **Optimal compression?** How many engram tokens are needed to preserve useful information?

## License

MIT
