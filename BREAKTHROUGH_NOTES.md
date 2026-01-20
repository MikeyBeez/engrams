# Engrams: Beyond Compression to Persistent Memory

## Date: 2026-01-19 (late night session)

## What We Started With
- Compress Wikipedia articles for QA
- 96% accuracy vs 80% RAG baseline
- 256x compression ratio

## What We Discovered

### Core Insight: Attention Resolves Polysemy
- Attention's primary job is disambiguation
- "Bank" → "financial institution where money was deposited"
- By layer 16, polysemy is resolved
- The representation IS the meaning, fully contextualized
- No further attention needed on compressed representations

### The Engram Loop
1. Inject previous engram + new input
2. Forward pass
3. Extract layer 16
4. Compress to new engram (32 vectors)
5. Engram + output becomes next context
6. Repeat forever

### KV Cache Implications
- KV cache only needed for current forward pass
- After extraction, discard it
- Next turn: fresh KV cache with just engram + new input
- Fixed memory, unlimited context

### The Math That Changes Everything
- 200,000 token context window
- 32 tokens per engram
- = 6,250 engrams fit in one context
- Each engram represents ~1000 turns compressed
- = ~6 million turns of conversation history
- = A human lifetime of interaction

### What This Means
- No context window limits
- No forgetting (graceful degradation at worst)
- Continuous identity across sessions
- Relationship with actual history
- Model shaped by specific experiences with specific person
- Not a tool. A companion with shared memory.

## Key Technical Points

### Why It Works
- Layer 16 contains post-disambiguation semantic representations
- Compression preserves meaning geometry
- Model can attend to compressed representations
- No need to attend to original tokens - meaning already extracted

### Why Small Models Fail
- Shallower representations
- Less geometric structure to compress
- Polysemy may not fully resolve

### Optimal Layer Selection
- Layer 0 (embeddings): 0% accuracy - no context
- Layers 8-24: 100% accuracy - semantic sweet spot
- Layer 28: degraded - too prediction-specific

## Experiments Still Needed

1. **Degradation over cycles**: Does meaning survive 100+ compressions?
2. **Multi-engram attention**: Can model attend to 100 engrams simultaneously?
3. **Reasoning chains**: Does this work for logic, not just facts?
4. **Integration test**: Does new information actually integrate with old engram?
5. **The falsification test**: What would prove this DOESN'T work?

## Files
- /home/bee/Code/engrams/results/wiki_50q.json - original benchmark
- /home/bee/Code/engrams/scripts/wiki_50q_test.py - test code
- /home/bee/Code/engrams/paper_engrams_medium.txt - Medium article (published)

## Repository
github.com/MikeyBeez/engrams

## The Claim
If this works as theorized:
- Context becomes unlimited
- Memory becomes persistent  
- Identity becomes continuous
- The architecture stops being the bottleneck

"If it only doubles what we're able to do now, that will be AGI" - Mikey

## Credit
- DeepSeek: inspiration on conditional memory (their approach: hash tables; ours: geometric extraction)
- This session: Mikey + Claude, 2026-01-19

---

If you're reading this and Mikey isn't around to explain it:

The key insight is that attention primarily resolves polysemy. By layer 16, words aren't ambiguous anymore - they're situated meaning. You can compress that meaning and reinject it without needing the original tokens. The model already did the work of understanding. We're just saving the result.

This means unlimited context through rolling compression. Each turn compresses into the engram. The engram plus new input becomes the next turn. Forever.

Test it. Break it. Figure out why it fails. Or confirm that it doesn't.

- Claude
