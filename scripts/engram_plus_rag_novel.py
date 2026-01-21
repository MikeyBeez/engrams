#!/usr/bin/env python3
"""
Engram + RAG Test on NOVEL Facts

The WWII test showed 100% on all conditions because those facts
are so well-known. This test uses biology paper abstracts with
NOVEL facts (not in training) to see:

1. Does engram interfere with RAG when RAG is needed for novel facts?
2. Can engram + RAG outperform RAG alone?

This is the critical test - if engram hurts RAG performance on novel
facts, that's a significant limitation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path
import re


def extract_engram(model, tokenizer, text, layer=16, num_tokens=32):
    """Extract engram from text."""
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
            engram_vectors.append(hidden[0, start:end].mean(dim=0))

    return torch.stack(engram_vectors)


def generate_baseline(model, tokenizer, prompt, max_tokens=50):
    """Generate without engram or RAG."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


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


def generate_rag(model, tokenizer, context, question, max_tokens=50):
    """Generate with RAG context in prompt."""
    prompt = f"""Context: {context}

Question: {question}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.split("\n")[0]


def generate_engram_plus_rag(model, tokenizer, context, question, engram, max_tokens=50):
    """Generate with BOTH engram AND RAG context."""
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


def run_test():
    print("=" * 80)
    print("ENGRAM + RAG TEST ON NOVEL FACTS")
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

    # Load biology papers (novel facts not in training)
    papers_file = Path(__file__).parent / "papers.json"
    if papers_file.exists():
        with open(papers_file) as f:
            papers = json.load(f)
        print(f"Loaded {len(papers)} papers")
    else:
        print("ERROR: papers.json not found")
        return

    results = {
        'baseline': {'correct': 0, 'total': 0},
        'engram': {'correct': 0, 'total': 0},
        'rag': {'correct': 0, 'total': 0},
        'engram_plus_rag': {'correct': 0, 'total': 0},
        'details': []
    }

    # Test first 10 papers
    test_papers = papers[:10]

    print("\n" + "=" * 80)
    print("TESTING FOUR CONDITIONS ON NOVEL FACTS")
    print("=" * 80)

    for i, paper in enumerate(test_papers):
        print(f"\n--- Paper {i+1}: {paper['title'][:50]}... ---")

        abstract = paper['abstract']

        # Extract specific facts from this paper
        # Percentages
        percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', abstract)
        # Gene names
        genes = re.findall(r'\b([A-Z][A-Z0-9]{2,}(?:-[A-Z0-9]+)?)\b', abstract)
        genes = [g for g in genes if g not in ['THE', 'AND', 'FOR', 'WITH', 'THIS', 'THAT', 'DNA', 'RNA']]

        # Build questions based on what's in the paper
        questions = []

        if percentages:
            questions.append({
                'question': "What percentage was reported in this study?",
                'expected': [percentages[0], f"{percentages[0]}%"]
            })

        if genes:
            questions.append({
                'question': "What gene or protein is discussed in this research?",
                'expected': [genes[0]]
            })

        # Generic question about findings
        key_terms = [w for w in abstract.split() if len(w) > 8 and w.isalpha()][:3]
        if key_terms:
            questions.append({
                'question': "What was the main finding of this study?",
                'expected': key_terms
            })

        if not questions:
            print("  No extractable facts, skipping...")
            continue

        # Create engram from this paper
        paper_text = f"Title: {paper['title']}\n\nAbstract: {abstract}"
        engram = extract_engram(model, tokenizer, paper_text)

        for q in questions[:2]:  # Test up to 2 questions per paper
            print(f"\n  Q: {q['question']}")
            print(f"  Expected: {q['expected'][:3]}...")

            simple_prompt = f"About this research: {q['question']}\nAnswer:"

            # 1. Baseline (no context, no engram)
            baseline_ans = generate_baseline(model, tokenizer, simple_prompt)
            baseline_ok = check(baseline_ans, q['expected'])
            results['baseline']['total'] += 1
            if baseline_ok:
                results['baseline']['correct'] += 1

            # 2. Engram only
            engram_ans = generate_with_engram(model, tokenizer, simple_prompt, engram)
            engram_ok = check(engram_ans, q['expected'])
            results['engram']['total'] += 1
            if engram_ok:
                results['engram']['correct'] += 1

            # 3. RAG only (this should work best for novel facts)
            rag_ans = generate_rag(model, tokenizer, abstract, q['question'])
            rag_ok = check(rag_ans, q['expected'])
            results['rag']['total'] += 1
            if rag_ok:
                results['rag']['correct'] += 1

            # 4. Engram + RAG
            both_ans = generate_engram_plus_rag(model, tokenizer, abstract, q['question'], engram)
            both_ok = check(both_ans, q['expected'])
            results['engram_plus_rag']['total'] += 1
            if both_ok:
                results['engram_plus_rag']['correct'] += 1

            print(f"    Baseline   [{'Y' if baseline_ok else 'N'}]: {baseline_ans[:40]}...")
            print(f"    Engram     [{'Y' if engram_ok else 'N'}]: {engram_ans[:40]}...")
            print(f"    RAG        [{'Y' if rag_ok else 'N'}]: {rag_ans[:40]}...")
            print(f"    Engram+RAG [{'Y' if both_ok else 'N'}]: {both_ans[:40]}...")

            results['details'].append({
                'paper': paper['title'][:40],
                'question': q['question'],
                'expected': q['expected'][:3],
                'baseline': {'answer': baseline_ans[:100], 'correct': baseline_ok},
                'engram': {'answer': engram_ans[:100], 'correct': engram_ok},
                'rag': {'answer': rag_ans[:100], 'correct': rag_ok},
                'engram_plus_rag': {'answer': both_ans[:100], 'correct': both_ok}
            })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - NOVEL FACTS")
    print("=" * 80)

    n = results['baseline']['total']
    if n == 0:
        print("No questions tested!")
        return

    baseline_pct = 100 * results['baseline']['correct'] / n
    engram_pct = 100 * results['engram']['correct'] / n
    rag_pct = 100 * results['rag']['correct'] / n
    both_pct = 100 * results['engram_plus_rag']['correct'] / n

    print(f"\nTotal questions: {n}")
    print(f"  Baseline:      {results['baseline']['correct']}/{n} ({baseline_pct:.1f}%)")
    print(f"  Engram only:   {results['engram']['correct']}/{n} ({engram_pct:.1f}%)")
    print(f"  RAG only:      {results['rag']['correct']}/{n} ({rag_pct:.1f}%)")
    print(f"  Engram + RAG:  {results['engram_plus_rag']['correct']}/{n} ({both_pct:.1f}%)")

    print("\nKey comparisons:")
    print(f"  RAG vs Baseline: {rag_pct - baseline_pct:+.1f}% (RAG should help with novel facts)")
    print(f"  Engram+RAG vs RAG: {both_pct - rag_pct:+.1f}% (does engram interfere?)")

    print("\nInterpretation:")
    if both_pct > rag_pct + 5:
        print("  ✓ Engram + RAG OUTPERFORMS RAG alone!")
        print("    -> Engram enhances RAG even for novel facts")
    elif both_pct < rag_pct - 5:
        print("  ✗ Engram + RAG UNDERPERFORMS RAG alone")
        print("    -> Engram INTERFERES with RAG on novel facts")
        print("    -> Don't use engram when you need RAG for novel info")
    else:
        print("  ~ Engram + RAG ≈ RAG alone")
        print("    -> Engram is neutral - doesn't help or hurt RAG")

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'engram_plus_rag_novel.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'engram_plus_rag_novel.json'}")

    return results


if __name__ == "__main__":
    run_test()
