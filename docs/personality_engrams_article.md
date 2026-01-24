# Where Does Personality Live in a Transformer?

## An Experiment in Finding the Right Layer for Behavioral Conditioning

I've been exploring a technique called "engrams" - compressed semantic representations extracted from transformer hidden states. The core idea: take a long document, pass it through a language model, extract hidden states from a specific layer, chunk and average them into a small number of vectors, then inject those vectors into the context window as a semantic prior.

For factual recall, this technique works remarkably well. An engram extracted from the Wikipedia article on World War II achieves 96% accuracy on factual questions while using a fraction of the tokens that traditional RAG requires. The compression is dramatic - thousands of tokens reduced to just 32 vectors.

But I wanted to know: can engrams shape personality, not just retrieve facts?

## The Stoic Experiment

The test subject was stoicism. I extracted an engram from Marcus Aurelius's Meditations and tested whether injecting it would make the model respond to life problems in a more stoic manner.

Stoicism has measurable characteristics. Drawing from validated psychological scales, I identified eight categories of stoic markers: the control dichotomy, acceptance, virtue focus, indifference to externals, present-moment awareness, mortality awareness, rationality, and emotional regulation.

I created eight test scenarios - job loss, public criticism, terminal diagnosis, betrayal, project failure, envy, overwhelming anxiety, and the desire for revenge. Each scenario presents an emotional situation where a stoic response would emphasize specific principles.

## The Layer Problem

My first attempt used layer 16 - the same layer that worked so well for factual recall. The results were disappointing. The engram-conditioned responses actually contained fewer stoic markers than baseline.

This raised a question: maybe personality doesn't live in the same place as facts.

Transformer layers encode different types of information. Early layers capture syntactic and surface patterns. Middle layers encode semantic content and factual associations. Late layers handle task-specific processing and output formatting.

If personality is about how to respond rather than what to retrieve, maybe it lives in different layers.

## Sweeping the Layers

I tested engrams extracted from layers 4, 8, 12, 16, 20, 24, and 27. The model has 28 layers total.

With a quick 4-scenario test, the results showed a U-shaped pattern. Layer 4 and layer 24 both showed improvement over baseline, while layer 16 showed degradation. This suggested early and late layers might encode behavioral patterns while middle layers encode semantic content.

But when I ran the full 8-scenario test, the picture changed. All layers showed improvement over baseline. The apparent U-shape was noise from insufficient data.

The final results across all eight scenarios:

Baseline produced 4 stoic markers. Layer 4 produced 10 markers, an improvement of 6. Layer 16 produced 14 markers, an improvement of 10. Layer 24 produced 12 markers, an improvement of 8.

Layer 16 - the same layer that works for facts - actually performed best for personality conditioning when tested with enough scenarios.

## What This Tells Us

The results reveal something important about how engrams work and how to test them properly.

First, engrams can influence behavioral patterns, not just factual retrieval. The stoic engram increased stoic markers by roughly 150-250% depending on the layer. This contradicts my initial hypothesis that personality conditioning wouldn't work.

Second, layer 16 works for both facts and personality. I had assumed different types of influence would require different layers. The data suggests otherwise. Middle layers may encode something more general - a semantic context that shapes both knowledge retrieval and response patterns.

Third, variance is high with stochastic generation. The 4-scenario sweep suggested layer 16 hurt performance. The 8-scenario test showed the opposite. Small sample sizes can be deeply misleading when testing language model behavior. Temperature 0.7 sampling introduces significant variance that requires many samples to average out.

Fourth, the improvement is real but modest. Going from 4 markers to 14 markers is meaningful, but we're still looking at relatively sparse stoic content. The engram nudges the model toward stoic-sounding responses without transforming it into a stoic sage.

## The Mechanism Question

Why would injecting averaged hidden states from the Meditations produce more stoic responses?

One possibility is priming. The engram vectors activate patterns in the model's latent space that are associated with stoic concepts. When the model then generates a response, those activated patterns influence word choice and framing toward stoic terminology.

Another possibility is context framing. The engram functions like an invisible prefix that sets expectations for what kind of response follows. The model has learned that texts about stoic philosophy tend to be followed by stoic-sounding content, so it generates accordingly.

A third possibility is retrieval enhancement. The model already knows about stoicism from its training. The engram helps it access that knowledge more readily, similar to how the WWII engram helps it access WWII facts.

The data doesn't distinguish between these mechanisms. All three might contribute. What matters practically is that the technique produces measurable effects.

## Implications

For practitioners wanting to use engrams for personality conditioning, the findings suggest:

Use layer 16. The same layer that works for factual retrieval also works for behavioral influence. No need to search for a special personality layer.

Test with many scenarios. Small sample sizes produce misleading results. The variance in language model outputs is high enough that you need substantial data to see true patterns.

Expect modest effects. Engrams nudge behavior rather than transform it. You'll get more stoic markers, not a complete personality transplant.

Consider the source material. I used philosophical text that describes stoic principles. Using examples of stoic responses to problems might produce stronger effects. The engram can only point to patterns that exist in its source.

## What We Still Don't Know

Several questions remain open.

Does this generalize beyond stoicism? The Meditations is a particularly coherent philosophical text. Other personality types might not have such clean source material available.

How does this compare to prompting? Adding "respond in a stoic manner" to the prompt might produce similar or better effects with less complexity. The advantage of engrams is token efficiency, but if the effect is modest, the engineering overhead may not be worth it.

Can we combine layers? Extracting from multiple layers and averaging might produce different effects than any single layer.

Does this work with instruction-tuned models? All testing used a base model. Instruction-tuned models might respond differently to injected context.

## Conclusion

Personality engrams work, at least for stoicism. Layer 16 produces the best results, matching its effectiveness for factual recall. The effects are modest but measurable - roughly tripling the density of stoic markers in responses to emotional scenarios.

The most important finding may be methodological: small sample sizes with stochastic generation produce unreliable conclusions. What looked like a fundamental insight about layer specialization turned out to be noise. Only with sufficient data did the true pattern emerge.

Engrams are semantic pointers, but what they point to is more general than I initially thought. They don't just help the model find facts. They help it find response patterns, communication styles, and philosophical frameworks. The middle layers encode something like meaning itself - and meaning shapes both what we know and how we express it.
