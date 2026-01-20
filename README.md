# Engrams: Geometric State Compression for Transformers

Extract dense semantic representations from transformer hidden states to compress context 256x while improving accuracy.

## Key Results

- **96% accuracy** with engrams vs **80% accuracy** with RAG
- **64.8x token reduction** (47 tokens vs 3,019 tokens per query)
- **256x compression ratio** (8,192 tokens → 32 vectors)

Tested on Wikipedia WWII article (87K characters, 50 factual questions) using Qwen2.5-7B.

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

## Installation

```bash
git clone https://github.com/bardicreels/engrams.git
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
│   ├── layer_sweep.py    # Layer selection experiment
│   └── rag_vs_engram.py  # Comparison tests
├── results/
│   └── wiki_50q.json     # Experimental results
└── paper_engrams_medium.txt  # Publication draft
```

## Key Findings

- **Layer selection matters**: Layers 8-24 work well; layer 0 fails completely
- **Scaling is critical**: Engram vectors must be scaled to match embedding norms
- **Model size matters**: 0.5B models fail; 7B models succeed dramatically
- **Compression improves accuracy**: Counter-intuitively, engrams outperform raw text

## Inspired By

DeepSeek's work on conditional memory in transformers. Their approach uses external hash tables; ours extracts the geometric representations transformers naturally build.

## License

MIT

## Citation

If you use this work, please cite:

```
@misc{engrams2025,
  title={Engrams: Geometric State Compression for Transformers},
  author={bardicreels},
  year={2025},
  url={https://github.com/bardicreels/engrams}
}
```
