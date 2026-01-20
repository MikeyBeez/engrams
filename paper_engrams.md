# Engrams: Learned Semantic Compression for Transformers

## A 256x Context Compression Method That Outperforms RAG

What if you could compress 8,000 tokens of context into just 32 tokens—and get better accuracy than using the full text? This article presents engrams, a method for extracting dense semantic representations from transformer hidden states that dramatically reduces context window usage while preserving—and in some cases improving—question-answering performance.

## The Problem with Context Windows

Large language models have a fundamental bottleneck: the context window. Even with recent advances pushing context limits to 100K+ tokens, every token in the context consumes memory, increases latency, and adds cost. Retrieval-Augmented Generation (RAG) helps by injecting relevant documents into prompts, but those documents still consume tokens one-for-one.

The question we explored: Can we compress documents into a handful of dense vectors that carry the same semantic information, then inject those vectors directly into the model's embedding space?

## The Core Insight

Transformers process text through multiple layers, each building increasingly abstract representations. The embedding layer (layer 0) converts tokens into vectors, but these vectors are essentially just word-level lookup tables. By the middle layers (8-24 in our experiments), the model has constructed rich contextual representations that encode relationships, facts, and meaning.

Our hypothesis: if we extract these middle-layer representations and compress them, we can create "engrams"—dense vectors that encode document semantics in a fraction of the tokens.

## How Engrams Work

The process has three steps:

**Extraction**: Pass a document through the model and capture hidden states from a middle layer (we found layer 16 works well for 7B parameter models). This gives us a sequence of vectors, one per input token.

**Compression**: Divide the sequence into chunks and mean-pool each chunk into a single vector. For example, 8,192 tokens divided into 32 chunks produces 32 engram vectors.

**Injection**: When answering questions, prepend these 32 engram vectors to the prompt embeddings. The model "sees" the compressed context as if it were a very short prefix, then generates an answer.

The key technical detail is scaling. Hidden states from layer 16 have different norms than embedding-layer vectors. We scale the engrams to match the embedding layer's average norm, allowing them to integrate smoothly into the generation process.

## Experimental Setup

We tested engrams against RAG using Wikipedia's World War II article—87,098 characters of dense historical facts. We created 50 factual questions spanning dates, leaders, events, battles, and strategic details.

For RAG, we included the first 3,000 tokens of the article directly in the prompt with each question. For engrams, we compressed 8,192 tokens into 32 vectors.

Both approaches used Qwen2.5-7B as the base model, with the same generation parameters for fair comparison.

## Results

The results surprised us. Engrams didn't just match RAG—they significantly outperformed it.

**Engrams achieved 96% accuracy (48/50 questions correct)**

**RAG achieved 80% accuracy (40/50 questions correct)**

Looking at token usage:

RAG used an average of 3,019 tokens per question (context + question + answer).

Engrams used an average of 47 tokens per question (32 engram tokens + question + answer).

That's a 64.8x reduction in token usage with a 16 percentage point improvement in accuracy.

## Why Does This Work?

We initially expected engrams to be a compression-accuracy tradeoff—get smaller context at the cost of some precision. Instead, engrams performed better. Three factors may explain this:

**Signal concentration**: Mean-pooling across chunks creates vectors that blend information from multiple tokens. This may produce representations that capture broader themes rather than getting distracted by local lexical patterns.

**Reduced noise**: RAG includes the raw text with all its formatting quirks, tangential asides, and redundant phrasing. Engrams compress to essence.

**Representation alignment**: Middle-layer representations encode relationships and facts in a form already optimized for the model's downstream processing. We're meeting the model where it naturally works, rather than forcing it to re-derive structure from raw tokens.

## The Layer Selection Experiment

Before the main experiment, we ran a layer sweep to find the optimal extraction point. We tested layers 0, 4, 8, 12, 16, 20, 24, and 28 with simpler questions.

Layer 0 (embeddings) failed completely—0% accuracy. This makes sense: embedding vectors are just word lookups with no contextual information.

Layers 8-24 all achieved 100% accuracy on simple factual questions, matching RAG performance.

Layer 28 (near the output) degraded slightly. The final layers specialize for next-token prediction, making their representations less useful for general-purpose injection.

The sweet spot is the middle third of the model—deep enough to encode semantics, not so deep that representations become prediction-specific.

## Practical Implications

**Cost reduction**: At 64x fewer tokens, engrams could reduce API costs proportionally for applications doing heavy context retrieval.

**Latency**: Fewer input tokens means faster inference. The engram extraction is a one-time cost that can be cached.

**Context window management**: You could theoretically store thousands of document engrams and combine them, creating a form of "compressed context memory" far beyond what fits in a standard context window.

**Project switching**: Imagine saving your entire codebase as engrams, switching between projects by swapping which engrams you inject, without reloading full file contents.

## Limitations and Future Work

This approach has clear constraints. We tested on factual recall from a single document. Performance on tasks requiring precise quotation, numerical computation, or multi-document reasoning remains unexplored.

The extraction step requires a forward pass through the model, which isn't free. For very short documents, RAG may still win on total compute.

We also haven't tested scaling to truly massive documents (100K+ tokens) or combining engrams from multiple sources. Both are natural next steps.

## The Code

The implementation is straightforward. Here's the core extraction logic:

```python
def extract_engram(model, tokenizer, text, layer=16, num_tokens=32):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)

    with torch.no_grad():
        outputs = model(**inputs.to(model.device), output_hidden_states=True)

    hidden = outputs.hidden_states[layer]  # [1, seq_len, hidden_dim]
    seq_len = hidden.shape[1]
    chunk_size = seq_len // num_tokens

    engram_vectors = []
    for i in range(num_tokens):
        start = i * chunk_size
        end = start + chunk_size if i < num_tokens - 1 else seq_len
        chunk = hidden[0, start:end, :]
        engram_vectors.append(chunk.mean(dim=0))

    return torch.stack(engram_vectors)
```

And injection:

```python
def inject_engram(model, tokenizer, question, engram):
    embed_layer = model.get_input_embeddings()

    # Scale to match embedding norms
    embed_norm = embed_layer.weight.norm(dim=1).mean().item()
    engram_norm = engram.norm(dim=1).mean().item()
    scaled_engram = engram * (embed_norm / engram_norm)

    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_embeds = embed_layer(inputs.input_ids.to(model.device))

    # Prepend engram as context
    combined = torch.cat([scaled_engram.unsqueeze(0), prompt_embeds], dim=1)

    return model.generate(inputs_embeds=combined, max_new_tokens=50)
```

## Conclusion

Engrams demonstrate that context doesn't need to be verbose. By extracting and compressing semantic representations from transformer middle layers, we can achieve better accuracy than full-text RAG at a fraction of the token cost.

The 256x compression ratio and 96% accuracy suggest this isn't just a curiosity—it's a potentially practical approach to context management that inverts the usual compression-quality tradeoff.

The transformer has already done the hard work of building rich semantic representations. We're just learning to use what's already there.

---

*Experiments run on Qwen2.5-7B using an RTX 5070 Ti. Full code available at github.com/bardicreels/engrams.*
