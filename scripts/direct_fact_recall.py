#!/usr/bin/env python3
"""
Direct Fact Recall Test - Matches wiki_50q format exactly

The wiki test asked direct questions like "When did X happen?"
and checked for specific keywords in answers.

This test does the same with biology papers:
1. Extract specific facts from abstract (genes, percentages, methods)
2. Ask direct questions about those facts
3. Check if the answer contains the expected keywords

Uses 7B model like wiki_50q did.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path
import re


def extract_engram(model, tokenizer, text, layer=16, num_tokens=32):
    """Extract engram from text - matches wiki_50q method."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)

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

    return torch.stack(engram_vectors), seq_len


def generate_with_engram(model, tokenizer, prompt, engram, max_tokens=50):
    """Generate with engram - matches wiki_50q exactly."""
    embed = model.get_input_embeddings()

    # Scale engram
    e_norm = embed.weight.norm(dim=1).mean().item()
    g_norm = engram.norm(dim=1).mean().item()
    scaled = engram * (e_norm / g_norm)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    emb = embed(inputs.input_ids)

    combined = torch.cat([scaled.unsqueeze(0).to(emb.dtype), emb], dim=1)

    with torch.no_grad():
        out = model.generate(
            inputs_embeds=combined,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(out[0], skip_special_tokens=True).split("\n")[0]


def generate_baseline(model, tokenizer, prompt, max_tokens=50):
    """Generate without engram."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(out[0], skip_special_tokens=True).split("\n")[0]


def generate_rag(model, tokenizer, context, question, max_tokens=50):
    """RAG - full context in prompt."""
    prompt = f"""Context: {context[:2000]}

Question: {question}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.split("\n")[0]


def check(response, expected_keywords):
    """Check if response contains any expected keyword."""
    r = response.lower()
    for kw in expected_keywords:
        if kw.lower() in r:
            return True
    return False


def extract_questions_from_paper(paper):
    """Generate direct factual questions from a paper abstract."""
    abstract = paper['abstract']
    title = paper['title']
    questions = []

    # Look for gene names and ask about them
    genes = re.findall(r'\b([A-Z][A-Z0-9]{2,}(?:-[A-Z0-9]+)?)\b', abstract)
    genes = [g for g in genes if g not in ['THE', 'AND', 'FOR', 'WITH', 'THIS', 'THAT', 'DNA', 'RNA', 'USA']]
    if genes:
        gene = genes[0]
        questions.append({
            'question': f"What gene or protein is discussed in this research?",
            'expected': [gene],
            'type': 'gene'
        })

    # Look for percentages
    percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', abstract)
    if percentages:
        pct = percentages[0]
        questions.append({
            'question': f"What percentage was reported in the findings?",
            'expected': [pct, f"{pct}%"],
            'type': 'percentage'
        })

    # Look for diseases/conditions
    conditions = ['ADHD', 'Alzheimer', 'Parkinson', 'depression', 'schizophrenia', 'autism',
                  'epilepsy', 'anxiety', 'dementia', 'stroke']
    found_conditions = [c for c in conditions if c.lower() in abstract.lower()]
    if found_conditions:
        questions.append({
            'question': f"What disease or condition is this research related to?",
            'expected': found_conditions,
            'type': 'condition'
        })

    # Look for brain regions
    regions = ['hippocampus', 'cortex', 'amygdala', 'striatum', 'cerebellum', 'thalamus',
               'prefrontal', 'temporal', 'parietal', 'occipital']
    found_regions = [r for r in regions if r.lower() in abstract.lower()]
    if found_regions:
        questions.append({
            'question': f"What brain region is studied?",
            'expected': found_regions,
            'type': 'region'
        })

    # Look for methods
    methods = ['CRISPR', 'optogenetic', 'fMRI', 'MRI', 'PET', 'EEG', 'patch-clamp',
               'two-photon', 'immunohistochemistry', 'Western blot', 'ELISA', 'PCR',
               'RNA-seq', 'microscopy', 'electrophysiology']
    found_methods = [m for m in methods if m.lower() in abstract.lower()]
    if found_methods:
        questions.append({
            'question': f"What experimental method was used?",
            'expected': found_methods,
            'type': 'method'
        })

    return questions


def run_test():
    print("=" * 80)
    print("DIRECT FACT RECALL TEST (wiki_50q style)")
    print("=" * 80)

    # Load 7B model like wiki test
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

    # Load papers
    papers_file = Path(__file__).parent / "papers.json"
    with open(papers_file) as f:
        papers = json.load(f)
    print(f"Loaded {len(papers)} papers")

    # Test first 10 papers
    test_papers = papers[:10]

    results = {
        'baseline': {'correct': 0, 'total': 0},
        'rag': {'correct': 0, 'total': 0},
        'engram': {'correct': 0, 'total': 0},
        'details': []
    }

    for i, paper in enumerate(test_papers):
        print(f"\n{'='*60}")
        print(f"PAPER {i+1}: {paper['title'][:50]}...")
        print(f"{'='*60}")

        # Create engram from paper
        paper_text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
        engram, src_tokens = extract_engram(model, tokenizer, paper_text)
        print(f"Engram: {engram.shape[0]} tokens (from {src_tokens})")

        # Generate questions for this paper
        questions = extract_questions_from_paper(paper)

        if not questions:
            print("  No extractable facts found, skipping...")
            continue

        for q in questions[:3]:  # Test up to 3 questions per paper
            print(f"\n  Q: {q['question']}")
            print(f"  Expected: {q['expected']}")

            # Wiki-style prompt with topic hint
            prompt = f"About this neuroscience research: {q['question']}\nAnswer:"

            # Baseline (no context)
            baseline_ans = generate_baseline(model, tokenizer, prompt)
            baseline_ok = check(baseline_ans, q['expected'])
            results['baseline']['total'] += 1
            if baseline_ok:
                results['baseline']['correct'] += 1

            # RAG (full context)
            rag_ans = generate_rag(model, tokenizer, paper_text, q['question'])
            rag_ok = check(rag_ans, q['expected'])
            results['rag']['total'] += 1
            if rag_ok:
                results['rag']['correct'] += 1

            # Engram
            engram_ans = generate_with_engram(model, tokenizer, prompt, engram)
            engram_ok = check(engram_ans, q['expected'])
            results['engram']['total'] += 1
            if engram_ok:
                results['engram']['correct'] += 1

            print(f"  Baseline [{'Y' if baseline_ok else 'N'}]: {baseline_ans[:60]}...")
            print(f"  RAG      [{'Y' if rag_ok else 'N'}]: {rag_ans[:60]}...")
            print(f"  Engram   [{'Y' if engram_ok else 'N'}]: {engram_ans[:60]}...")

            results['details'].append({
                'paper': paper['title'][:40],
                'question': q['question'],
                'expected': q['expected'],
                'baseline': {'answer': baseline_ans, 'correct': baseline_ok},
                'rag': {'answer': rag_ans, 'correct': rag_ok},
                'engram': {'answer': engram_ans, 'correct': engram_ok}
            })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    n = results['baseline']['total']
    print(f"\nTotal questions: {n}")
    print(f"Baseline: {results['baseline']['correct']}/{n} ({100*results['baseline']['correct']/n:.1f}%)")
    print(f"RAG:      {results['rag']['correct']}/{n} ({100*results['rag']['correct']/n:.1f}%)")
    print(f"Engram:   {results['engram']['correct']}/{n} ({100*results['engram']['correct']/n:.1f}%)")

    # Compare to wiki results
    print("\nCompared to wiki_50q test:")
    print("  wiki RAG: 80%, wiki Engram: 96%")

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'direct_fact_recall.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'direct_fact_recall.json'}")


if __name__ == "__main__":
    run_test()
