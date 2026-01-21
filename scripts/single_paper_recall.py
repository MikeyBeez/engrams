#!/usr/bin/env python3
"""
Single Paper Recall Test

Tests whether an engram from a SINGLE paper enables recall of that paper's facts.
This should work based on the wiki_50q results (96% accuracy).

Hypothesis:
- Single-source engram SHOULD enable recall (like wiki_50q)
- Multi-source averaged engram FAILS because content gets diluted
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path
import re


def extract_engram(model, tokenizer, text, layer=16, num_tokens=32):
    """Extract engram from single text."""
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
        if start >= seq_len:
            engram_vectors.append(hidden[0, -1, :])
        else:
            chunk = hidden[0, start:end, :]
            engram_vectors.append(chunk.mean(dim=0))

    return torch.stack(engram_vectors)


def generate_with_engram(model, tokenizer, prompt, engram, max_tokens=100):
    """Generate with engram prepended."""
    embed_layer = model.get_input_embeddings()

    # Scale engram to match embedding norms
    embed_norm = embed_layer.weight.norm(dim=1).mean().item()
    engram_norm = engram.norm(dim=1).mean().item()
    scaled_engram = engram * (embed_norm / engram_norm) if engram_norm > 0 else engram

    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_embeds = embed_layer(inputs.input_ids.to(model.device))

    combined = torch.cat([scaled_engram.unsqueeze(0).to(model.device), prompt_embeds], dim=1)

    with torch.no_grad():
        output = model.generate(
            inputs_embeds=combined,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


def generate_baseline(model, tokenizer, prompt, max_tokens=100):
    """Generate without engram."""
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs.to(model.device),
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


def extract_facts(abstract):
    """Extract verifiable facts from abstract."""
    facts = []

    # Percentages
    for match in re.findall(r'(\d+(?:\.\d+)?)\s*%', abstract):
        facts.append(f"{match}%")

    # Gene names
    for match in re.findall(r'\b([A-Z][A-Z0-9]{2,}(?:-[A-Z0-9]+)?)\b', abstract):
        if match not in ['THE', 'AND', 'FOR', 'WITH', 'THIS', 'THAT']:
            facts.append(match)

    # Key terms (long words)
    words = [w for w in abstract.split() if len(w) > 10 and w.isalpha()]
    facts.extend(words[:3])

    return list(set(facts))[:8]


def check_facts(response, facts):
    """Count how many facts appear in response."""
    response_lower = response.lower()
    return sum(1 for f in facts if f.lower() in response_lower)


def run_test():
    print("=" * 80)
    print("SINGLE PAPER RECALL TEST")
    print("=" * 80)

    # Load model - USE 7B like the successful wiki test!
    print("\nLoading model...")
    model_name = "Qwen/Qwen2.5-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load papers
    papers_file = Path(__file__).parent / "papers.json"
    with open(papers_file) as f:
        papers = json.load(f)
    print(f"Loaded {len(papers)} papers")

    # Test each paper individually
    results = []
    test_papers = papers[:10]  # Test first 10

    print(f"\nTesting {len(test_papers)} papers individually...\n")

    for i, paper in enumerate(test_papers):
        print(f"\n{'='*60}")
        print(f"PAPER {i+1}: {paper['title'][:50]}...")
        print(f"{'='*60}")

        # Extract facts from this paper
        facts = extract_facts(paper['abstract'])
        print(f"Facts to find: {facts[:5]}")

        # Create engram from THIS PAPER ONLY
        paper_text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
        paper_engram = extract_engram(model, tokenizer, paper_text)

        # Ask about this specific paper
        question = f"What specific findings did the paper '{paper['title'][:40]}' report?"
        prompt = f"Question: {question}\n\nAnswer:"

        # Test with paper's own engram
        resp_engram = generate_with_engram(model, tokenizer, prompt, paper_engram)
        facts_engram = check_facts(resp_engram, facts)

        # Test without engram (baseline)
        resp_baseline = generate_baseline(model, tokenizer, prompt)
        facts_baseline = check_facts(resp_baseline, facts)

        print(f"\nWith engram   (facts: {facts_engram}/{len(facts)}): {resp_engram[:80]}...")
        print(f"Without engram (facts: {facts_baseline}/{len(facts)}): {resp_baseline[:80]}...")

        results.append({
            'paper': paper['title'],
            'facts': facts,
            'engram_facts': facts_engram,
            'baseline_facts': facts_baseline,
            'engram_wins': facts_engram > facts_baseline
        })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_engram = sum(r['engram_facts'] for r in results)
    total_baseline = sum(r['baseline_facts'] for r in results)
    total_facts = sum(len(r['facts']) for r in results)
    wins = sum(1 for r in results if r['engram_wins'])
    ties = sum(1 for r in results if r['engram_facts'] == r['baseline_facts'])

    print(f"\nEngram total facts: {total_engram}/{total_facts}")
    print(f"Baseline total facts: {total_baseline}/{total_facts}")
    print(f"\nEngram wins: {wins}/{len(results)}")
    print(f"Ties: {ties}/{len(results)}")

    if total_engram > total_baseline:
        print("\n✓ Single-paper engram HELPS recall")
    elif total_engram == total_baseline:
        print("\n~ Single-paper engram provides NO advantage")
    else:
        print("\n✗ Single-paper engram HURTS recall")

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'single_paper_recall.json', 'w') as f:
        json.dump({
            'total_engram_facts': total_engram,
            'total_baseline_facts': total_baseline,
            'total_facts': total_facts,
            'engram_wins': wins,
            'details': results
        }, f, indent=2)

    print(f"\nResults saved to {output_dir / 'single_paper_recall.json'}")


if __name__ == "__main__":
    run_test()
