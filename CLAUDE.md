# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Engrams is a research project exploring semantic compression of transformer hidden states. The core idea: extract hidden states from a transformer layer, compress them via chunking and mean-pooling into a small number of vectors (engrams), then inject those vectors as a prefix to steer model behavior at inference time.

## Commands

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e ".[dev]"  # includes pytest, jupyter, matplotlib

# Run experiments (most scripts are standalone)
python scripts/wiki_50q_test.py
python comprehensive_flip_test.py
python probability_bias_test.py

# Linting
ruff check .
ruff format .

# Tests (test suite is minimal - most testing is via experiment scripts)
pytest
```

## Architecture

### Core Library (`engrams/`)
- `extractor.py` - `EngramExtractor` class: passes text through model, captures hidden states from specified layer, chunks sequence and mean-pools to N vectors
- `injector.py` - `EngramInjector` class: three injection modes (replace placeholder tokens, prefix, or add to embeddings)
- `storage.py` - `EngramStore`: ChromaDB-based persistence with vector similarity search

### Experiment Scripts
Most development happens in standalone experiment scripts at repo root and `scripts/`. Pattern:
1. Load Qwen2.5-7B (primary model used)
2. Extract engram from knowledge text at specific layer
3. Compare baseline token probabilities vs engram-injected probabilities
4. Report probability shifts and whether answer "flipped" from wrong to correct

Key experiment files:
- `comprehensive_flip_test.py` - Tests if engrams can flip wrong answers on medical questions
- `probability_bias_test.py` - Measures probability shifts toward correct tokens
- `multi_question_flip_test.py` - Grid search over layer/strength for multiple questions
- `scripts/stoic_engram_test.py` - Tests personality conditioning (stoic response markers)

## Key Concepts

### Extraction Parameters
- **Layer**: Which transformer layer to extract from. Layer 16 (middle) works well for both facts and personality. Late layers (20-26) are the "decision commitment zone."
- **num_tokens**: How many engram vectors to produce (default 16-32). Higher = more detail, lower = more compression.
- **Pooling**: Currently mean-pooling per chunk. Attention pooling stubbed but not implemented.

### Injection Parameters
- **Strength**: Multiplier after norm-matching. 1x = match embedding norm. 3-10x typically used in experiments. Too high breaks coherence.
- **Mode**: `prefix` (prepend to sequence), `replace` (swap placeholder tokens), `add` (add to existing embeddings)

### The Three-Layer Model (from research findings)
1. **Belief** - Model "knows" the correct information (visible in generated explanations)
2. **Token probability** - Engrams can shift next-token probabilities toward correct concepts
3. **Selection/commitment** - Final discrete output; surprisingly robust to activation perturbation when prompt structure argues against correct answer

## Current Research Status

The project discovered that:
- Engrams improve RAG-style factual recall (+10-20 points over baseline)
- Engrams can shift token probabilities but struggle to override strongly-framed prompts
- Irrelevant engrams produce similar perturbation effects as relevant ones at high strengths (negative control finding)
- Semantic priming works; decision override does not (at tested strengths/methods)

See `HONEST_ASSESSMENT.md` for detailed critique of initial claims.

## Model Requirements

Experiments use Qwen2.5-7B. Requires:
- GPU with 16GB+ VRAM
- HuggingFace token (set `HF_TOKEN` env var or store in `~/.cache/huggingface/token`)
