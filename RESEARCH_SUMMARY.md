# Engrams Research Summary

## Core Question
Can we store information (facts, memories) in a compact representation that can be "injected" into an LLM to affect its behavior?

## Approaches Tested

### 1. Hidden State Extraction + Injection
**Idea**: Extract hidden states from a forward pass on the fact, then inject them during generation.

**Results**: ❌ Does not work
- Prepending hidden states as "soft tokens": Model ignores them, generates unrelated content
- Adding hidden states to activations: Distorts output into garbage
- Different layers (12, 20, 27) all fail

**Why it fails**: Hidden states encode information in a distributed, entangled way that is context-dependent. The representation of "42.7" when processing "The zorblax constant is 42.7" is very different from what the model needs to *generate* "42.7". Information is encoded relationally, not as extractable facts.

### 2. Mean Pooling / Last Token Extraction
**Idea**: Use mean or last-token hidden state as a "summary" of the fact.

**Results**: ❌ Does not work
- Mean pooling loses positional information
- Last token captures end-of-sequence context, not fact content
- Both produce gibberish when injected

### 3. Soft Prompt Training
**Idea**: Train learnable embeddings to encode the fact such that when prepended to questions, the model produces correct answers.

**Results**: ⚠️ Partial success
- Training loss goes to 0 (perfect on training question)
- Does NOT generalize to paraphrased questions
- Severe overfitting - soft prompts memorize the exact question-answer pair

**Why partial**: This is essentially learning a lookup table entry, not a generalizable fact representation. Works only for exact match.

### 4. Direct Text Prompt (Baseline)
**Idea**: Just include the fact in the prompt as text.

**Results**: ✅ Works perfectly
- "Context: The zorblax constant is 42.7\nQuestion: What is the zorblax constant?\nAnswer:"
- Model correctly outputs "42.7"
- Generalizes to paraphrased questions

## Key Insight

**Transformers don't have a "fact slot" you can inject into.**

Information in neural networks is:
1. **Distributed**: No single vector encodes a fact
2. **Context-dependent**: Same fact is encoded differently depending on surrounding text
3. **Entangled**: Factual content is mixed with grammatical/structural information
4. **Emergent**: Understanding happens through the full forward pass, not from stored vectors

## What Actually Works for Memory

### RAG (Retrieval-Augmented Generation)
- Store facts as text in a database
- Retrieve relevant facts based on query
- Include retrieved text in the prompt
- The model processes the text normally

### Fine-tuning
- Train the model on question-answer pairs
- Facts get encoded into weights
- But: expensive, not real-time, can cause forgetting

### In-Context Learning
- Provide examples in the prompt
- Model learns patterns from examples
- Works well, but uses context window

## Implications for Engrams Project

The original vision of "compact memory representations" that can be injected into LLMs is fundamentally misaligned with how transformers work.

**Viable alternatives**:
1. **Efficient RAG**: Fast retrieval of relevant text chunks
2. **Hierarchical summarization**: Compress context through text summarization, not vector encoding
3. **Adapter modules**: Train small adapters for specific knowledge domains
4. **Memory-augmented architectures**: New models designed with external memory (like RETRO)

## Compression Experiment Results

For context compression specifically:
- **Baseline**: Full text in prompt works best
- **Summarization**: Claude/LLM-generated summaries preserve key facts
- **Hidden state compression**: Does not preserve retrievable information

## Recommendations

1. **For Engrams**: Focus on RAG-style approaches with efficient retrieval
2. **For compression**: Use LLM summarization, not hidden states
3. **For memory**: Store memories as text, retrieve and inject as context
4. **Research direction**: Explore memory-augmented architectures (academic papers on external memory for transformers)

## Technical Notes

- Model tested: Qwen2.5-0.5B and Qwen2.5-7B
- Hardware: NVIDIA T4 (16GB)
- All experiments reproducible in scripts/
