#!/usr/bin/env python3
"""
Layer Sweep for Novel Facts

Hypothesis: Layer 16 captures topological/semantic structure (existing knowledge).
Question: Which layer, if any, best captures NOVEL information from the input?

We test engrams extracted from different layers on:
1. Known facts (WWII) - expect middle layers to dominate
2. Novel facts (biology papers) - maybe different layers help?

If late layers capture "what's about to be output", they might retain
more input-specific detail. If early layers capture raw patterns,
they might preserve novel tokens better.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path


def extract_engram_at_layer(model, tokenizer, text, layer, num_tokens=32):
    """Extract engram from a specific layer."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)

    with torch.no_grad():
        outputs = model(**inputs.to(model.device), output_hidden_states=True)

    # Qwen2.5-7B has 28 layers (indices 0-28 for hidden states)
    hidden = outputs.hidden_states[layer]
    seq_len = hidden.shape[1]

    chunk_size = max(1, seq_len // num_tokens)
    engram_vectors = []
    for i in range(num_tokens):
        start = i * chunk_size
        end = start + chunk_size if i < num_tokens - 1 else seq_len
        if start >= seq_len:
            engram_vectors.append(hidden[0, -1, :])
        else:
            engram_vectors.append(hidden[0, start:end].mean(dim=0))

    return torch.stack(engram_vectors)


def generate_with_engram(model, tokenizer, prompt, engram, max_tokens=50):
    """Generate with engram prepended."""
    embed = model.get_input_embeddings()

    e_norm = embed.weight.norm(dim=1).mean().item()
    g_norm = engram.norm(dim=1).mean().item()
    scaled = engram * (e_norm / g_norm) if g_norm > 0 else engram

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    emb = embed(inputs.input_ids)

    combined = torch.cat([scaled.unsqueeze(0).to(emb.dtype), emb], dim=1)

    with torch.no_grad():
        out = model.generate(
            inputs_embeds=combined,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


def generate_with_engram_and_rag(model, tokenizer, context, question, engram, max_tokens=50):
    """Generate with both engram and RAG context."""
    embed = model.get_input_embeddings()

    e_norm = embed.weight.norm(dim=1).mean().item()
    g_norm = engram.norm(dim=1).mean().item()
    scaled = engram * (e_norm / g_norm) if g_norm > 0 else engram

    prompt = f"""Context: {context}

Question: {question}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    emb = embed(inputs.input_ids)

    combined = torch.cat([scaled.unsqueeze(0).to(emb.dtype), emb], dim=1)

    with torch.no_grad():
        out = model.generate(
            inputs_embeds=combined,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


def check(response, expected_keywords):
    """Check if response contains any expected keyword."""
    r = response.lower()
    for kw in expected_keywords:
        if kw.lower() in r:
            return True
    return False


def run_layer_sweep():
    print("=" * 80)
    print("LAYER SWEEP: WHERE ARE NOVEL FACTS?")
    print("=" * 80)

    print("\nLoading Qwen2.5-7B model...")
    model_name = "Qwen/Qwen2.5-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get number of layers
    num_layers = len(model.model.layers)
    print(f"Model has {num_layers} transformer layers")

    # Test layers: 0, 4, 8, 12, 16, 20, 24, 28 (final)
    test_layers = [0, 4, 8, 12, 16, 20, 24, num_layers]

    # Load biology papers for novel facts
    papers_file = Path(__file__).parent / "papers.json"
    if papers_file.exists():
        with open(papers_file) as f:
            papers = json.load(f)
        print(f"Loaded {len(papers)} papers")
    else:
        print("ERROR: papers.json not found")
        return

    # Use first 5 papers for testing
    test_papers = papers[:5]

    results = {layer: {'engram_only': 0, 'engram_plus_rag': 0, 'total': 0} for layer in test_layers}

    print("\n" + "=" * 80)
    print("TESTING NOVEL FACT RECALL ACROSS LAYERS")
    print("=" * 80)

    for paper_idx, paper in enumerate(test_papers):
        print(f"\n--- Paper {paper_idx + 1}: {paper['title'][:50]}... ---")

        abstract = paper['abstract']

        # Extract specific facts
        import re
        percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', abstract)
        genes = re.findall(r'\b([A-Z][A-Z0-9]{2,}(?:-[A-Z0-9]+)?)\b', abstract)
        genes = [g for g in genes if g not in ['THE', 'AND', 'FOR', 'WITH', 'THIS', 'THAT', 'DNA', 'RNA', 'USA', 'NIH']]

        # Build questions
        questions = []
        if percentages:
            questions.append({
                'question': "What percentage was reported in this study?",
                'expected': [percentages[0], f"{percentages[0]}%"]
            })
        if genes:
            questions.append({
                'question': "What gene or protein is discussed?",
                'expected': [genes[0]]
            })

        if not questions:
            print("  No extractable facts, skipping...")
            continue

        paper_text = f"Title: {paper['title']}\n\nAbstract: {abstract}"

        # Extract engrams at each layer
        print(f"  Extracting engrams at layers: {test_layers}")
        layer_engrams = {}
        for layer in test_layers:
            layer_engrams[layer] = extract_engram_at_layer(model, tokenizer, paper_text, layer)

        # Test each question
        for q in questions[:2]:
            print(f"\n  Q: {q['question']}")
            print(f"  Expected: {q['expected'][:3]}")

            simple_prompt = f"About this research: {q['question']}\nAnswer:"

            for layer in test_layers:
                engram = layer_engrams[layer]

                # Test engram only
                engram_ans = generate_with_engram(model, tokenizer, simple_prompt, engram)
                engram_ok = check(engram_ans, q['expected'])

                # Test engram + RAG
                both_ans = generate_with_engram_and_rag(model, tokenizer, abstract, q['question'], engram)
                both_ok = check(both_ans, q['expected'])

                results[layer]['total'] += 1
                if engram_ok:
                    results[layer]['engram_only'] += 1
                if both_ok:
                    results[layer]['engram_plus_rag'] += 1

                print(f"    Layer {layer:2d}: Engram=[{'Y' if engram_ok else 'N'}] Engram+RAG=[{'Y' if both_ok else 'N'}]")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: LAYER COMPARISON FOR NOVEL FACTS")
    print("=" * 80)

    print("\nEngram Only (novel facts):")
    for layer in test_layers:
        n = results[layer]['total']
        if n > 0:
            pct = 100 * results[layer]['engram_only'] / n
            print(f"  Layer {layer:2d}: {results[layer]['engram_only']}/{n} ({pct:.1f}%)")

    print("\nEngram + RAG (novel facts):")
    for layer in test_layers:
        n = results[layer]['total']
        if n > 0:
            pct = 100 * results[layer]['engram_plus_rag'] / n
            print(f"  Layer {layer:2d}: {results[layer]['engram_plus_rag']}/{n} ({pct:.1f}%)")

    # Find best layers
    best_engram_layer = max(test_layers, key=lambda l: results[l]['engram_only'] if results[l]['total'] > 0 else -1)
    best_both_layer = max(test_layers, key=lambda l: results[l]['engram_plus_rag'] if results[l]['total'] > 0 else -1)

    print(f"\nBest layer for engram only: {best_engram_layer}")
    print(f"Best layer for engram+RAG: {best_both_layer}")

    print("\nInterpretation:")
    if results[best_engram_layer]['engram_only'] == 0:
        print("  No layer captures novel facts in engram-only mode.")
        print("  This confirms: engrams don't store novel information at ANY layer.")
    else:
        print(f"  Layer {best_engram_layer} shows some novel fact retention.")
        print("  This suggests: different layers encode different types of information.")

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'layer_novel_facts_sweep.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'layer_novel_facts_sweep.json'}")

    return results


if __name__ == "__main__":
    run_layer_sweep()
