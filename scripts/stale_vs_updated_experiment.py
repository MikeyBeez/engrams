#!/usr/bin/env python3
"""
Stale vs Updated Engram Experiment

Tests whether engram UPDATING matters, or if any engram (even stale)
just provides generic domain cueing.

Compares three conditions:
1. UPDATED engram - accumulated over all papers via EMA
2. STALE engram - from paper 1 only, never updated
3. NO engram - baseline (just model prior)

If updated >> stale >> baseline: updating captures paper-specific info
If updated ≈ stale >> baseline: engram just provides domain cueing
If updated ≈ stale ≈ baseline: engram does nothing useful
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime
import os
import random
from pathlib import Path
import argparse
import re


def extract_engram(model, tokenizer, text, layer=16, num_tokens=32):
    """Extract engram vectors from middle layer."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)

    with torch.no_grad():
        outputs = model(**inputs.to(model.device), output_hidden_states=True)

    hidden = outputs.hidden_states[layer]
    seq_len = hidden.shape[1]

    if seq_len < num_tokens:
        engram_vectors = []
        for i in range(num_tokens):
            idx = i % seq_len
            engram_vectors.append(hidden[0, idx, :])
        return torch.stack(engram_vectors)

    chunk_size = seq_len // num_tokens
    engram_vectors = []
    for i in range(num_tokens):
        start = i * chunk_size
        end = start + chunk_size if i < num_tokens - 1 else seq_len
        chunk = hidden[0, start:end, :]
        engram_vectors.append(chunk.mean(dim=0))

    return torch.stack(engram_vectors)


def ema_update(current_engram, new_engram, alpha=0.1):
    """Exponential moving average update."""
    return (1 - alpha) * current_engram + alpha * new_engram


def generate_with_engram(model, tokenizer, prompt, engram, max_tokens=150):
    """Generate response with engram injection."""
    embed_layer = model.get_input_embeddings()

    embed_norm = embed_layer.weight.norm(dim=1).mean().item()
    engram_norm = engram.norm(dim=1).mean().item()

    if engram_norm > 0:
        scaled_engram = engram * (embed_norm / engram_norm)
    else:
        scaled_engram = engram

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


def generate_baseline(model, tokenizer, prompt, max_tokens=150):
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


def extract_specific_facts(abstract):
    """Extract verifiable facts from abstract."""
    facts = []

    # Percentages
    percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', abstract)
    facts.extend([f"{p}%" for p in percentages])

    # Numbers with units
    numbers = re.findall(r'(\d+(?:\.\d+)?)\s*(mg|μg|ng|mM|μM|nM|Hz|kHz|MHz|days?|weeks?|months?|years?)', abstract, re.IGNORECASE)
    facts.extend([f"{n} {u}" for n, u in numbers])

    # Gene/protein names
    genes = re.findall(r'\b([A-Z][A-Z0-9]{2,}(?:-[A-Z0-9]+)?)\b', abstract)
    genes = [g for g in genes if g not in ['THE', 'AND', 'FOR', 'WITH', 'THIS', 'THAT', 'FROM', 'ARE', 'WAS', 'WERE']]
    facts.extend(genes[:5])

    return list(set(facts))


def score_response(response, paper):
    """Score how well response matches paper content."""
    response_lower = response.lower()
    facts = extract_specific_facts(paper['abstract'])

    if not facts:
        # Fallback to key terms
        key_terms = [w for w in paper['abstract'].lower().split() if len(w) > 8][:5]
        matches = sum(1 for t in key_terms if t in response_lower)
        return min(matches, 3)

    fact_matches = sum(1 for f in facts if f.lower() in response_lower)

    if fact_matches >= 2:
        return 3
    elif fact_matches >= 1:
        return 2
    else:
        # Check generic relevance
        topic_words = ['neuron', 'brain', 'synap', 'gene', 'protein', 'cell']
        if sum(1 for w in topic_words if w in response_lower) >= 2:
            return 1
        return 0


def load_papers(papers_file=None):
    """Load papers from JSON file."""
    if papers_file is None:
        papers_file = Path(__file__).parent / "papers.json"

    if not Path(papers_file).exists():
        print(f"ERROR: Papers file not found: {papers_file}")
        return None

    with open(papers_file) as f:
        return json.load(f)


def run_experiment(n_papers=50, alpha=0.1):
    """Run the stale vs updated comparison."""

    print("=" * 80)
    print("STALE VS UPDATED ENGRAM EXPERIMENT")
    print(f"Papers: {n_papers}, Alpha: {alpha}")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    model_name = "Qwen/Qwen2.5-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load papers
    papers = load_papers()
    if papers is None:
        return None
    papers = papers[:n_papers]
    print(f"Using {len(papers)} papers")

    # Create STALE engram (paper 1 only)
    print("\nCreating STALE engram from paper 1...")
    initial_text = f"Title: {papers[0]['title']}\n\nAbstract: {papers[0]['abstract']}"
    stale_engram = extract_engram(model, tokenizer, initial_text)
    print(f"  Stale engram shape: {stale_engram.shape}")

    # Create UPDATED engram (accumulated over all papers)
    print(f"\nBuilding UPDATED engram from {n_papers} papers...")
    updated_engram = stale_engram.clone()

    for i, paper in enumerate(papers[1:], 1):
        if i % 10 == 0:
            print(f"  Processing paper {i}/{n_papers}...")

        paper_text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
        prompt = f"Analyze this paper:\n\n{paper_text}\n\nKey findings:"

        response = generate_with_engram(model, tokenizer, prompt, updated_engram)
        response_engram = extract_engram(model, tokenizer, response)
        updated_engram = ema_update(updated_engram, response_engram, alpha=alpha)

    print("  Updated engram built.")

    # Calculate engram similarity
    similarity = torch.nn.functional.cosine_similarity(
        stale_engram.flatten().unsqueeze(0),
        updated_engram.flatten().unsqueeze(0)
    ).item()
    print(f"\nStale vs Updated engram similarity: {similarity:.4f}")

    # Test on papers at different positions
    results = {
        'config': {
            'n_papers': n_papers,
            'alpha': alpha,
            'engram_similarity': similarity,
            'timestamp': datetime.now().isoformat()
        },
        'tests': []
    }

    test_positions = [
        ('recent', n_papers - 5),    # Paper 5 turns ago
        ('middle', n_papers // 2),    # Middle paper
        ('early', 5),                 # Paper 5 (early)
        ('first', 1),                 # Paper 1 (used to create stale engram)
    ]

    print("\n" + "=" * 60)
    print("TESTING THREE CONDITIONS")
    print("=" * 60)

    for label, idx in test_positions:
        if idx >= len(papers):
            continue

        paper = papers[idx]
        question = f"What specific findings did the paper '{paper['title'][:50]}' report?"
        prompt = f"Question: {question}\n\nAnswer:"

        print(f"\n[{label.upper()}] Paper {idx}: {paper['title'][:40]}...")
        facts = extract_specific_facts(paper['abstract'])
        print(f"  Facts to find: {facts[:3]}...")

        # Condition 1: Updated engram
        resp_updated = generate_with_engram(model, tokenizer, prompt, updated_engram, max_tokens=100)
        score_updated = score_response(resp_updated, paper)

        # Condition 2: Stale engram
        resp_stale = generate_with_engram(model, tokenizer, prompt, stale_engram, max_tokens=100)
        score_stale = score_response(resp_stale, paper)

        # Condition 3: No engram (baseline)
        resp_baseline = generate_baseline(model, tokenizer, prompt, max_tokens=100)
        score_baseline = score_response(resp_baseline, paper)

        print(f"  UPDATED  (score {score_updated}): {resp_updated[:60]}...")
        print(f"  STALE    (score {score_stale}): {resp_stale[:60]}...")
        print(f"  BASELINE (score {score_baseline}): {resp_baseline[:60]}...")

        results['tests'].append({
            'position': label,
            'paper_index': idx,
            'paper_title': paper['title'],
            'facts': facts[:5],
            'updated': {'response': resp_updated[:200], 'score': score_updated},
            'stale': {'response': resp_stale[:200], 'score': score_stale},
            'baseline': {'response': resp_baseline[:200], 'score': score_baseline},
            'updated_vs_stale': score_updated - score_stale,
            'updated_vs_baseline': score_updated - score_baseline,
            'stale_vs_baseline': score_stale - score_baseline
        })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nEngram similarity (stale vs updated): {similarity:.4f}")
    print("\nScores by condition:")

    avg_updated = sum(t['updated']['score'] for t in results['tests']) / len(results['tests'])
    avg_stale = sum(t['stale']['score'] for t in results['tests']) / len(results['tests'])
    avg_baseline = sum(t['baseline']['score'] for t in results['tests']) / len(results['tests'])

    print(f"  UPDATED:  {avg_updated:.2f}")
    print(f"  STALE:    {avg_stale:.2f}")
    print(f"  BASELINE: {avg_baseline:.2f}")

    print("\nInterpretation:")
    if avg_updated > avg_stale + 0.5 and avg_stale > avg_baseline + 0.5:
        print("  ✓ Updated >> Stale >> Baseline: Updating captures paper-specific info!")
    elif avg_updated > avg_baseline + 0.5 and abs(avg_updated - avg_stale) < 0.5:
        print("  ~ Updated ≈ Stale >> Baseline: Engram provides domain cueing only")
    elif abs(avg_updated - avg_baseline) < 0.5:
        print("  ✗ Updated ≈ Stale ≈ Baseline: Engram does nothing useful")
    else:
        print(f"  ? Mixed results - check individual scores")

    print("\nPer-paper breakdown:")
    print(f"  {'Position':<10} {'Updated':>8} {'Stale':>8} {'Baseline':>8} {'U-S':>6} {'U-B':>6}")
    print("  " + "-" * 50)
    for t in results['tests']:
        print(f"  {t['position']:<10} {t['updated']['score']:>8} {t['stale']['score']:>8} "
              f"{t['baseline']['score']:>8} {t['updated_vs_stale']:>+6} {t['updated_vs_baseline']:>+6}")

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / 'stale_vs_updated.json'

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Stale vs Updated Engram Experiment")
    parser.add_argument('--n-papers', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.1)

    args = parser.parse_args()
    run_experiment(n_papers=args.n_papers, alpha=args.alpha)


if __name__ == "__main__":
    main()
