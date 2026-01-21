#!/usr/bin/env python3
"""
Chained Engram + RAG Experiment

Tests whether maintaining a session engram alongside RAG improves
performance over RAG alone across 100 turns.

Three conditions at each recall test:
1. RAG only - context in prompt, no engram
2. Engram + RAG - session engram prepended + context in prompt
3. Engram only - session engram, no RAG context (for comparison)

The engram accumulates via EMA throughout the session.
RAG provides the paper context at each turn.

Hypothesis: Engram + RAG should match or exceed RAG alone,
with the engram providing topic cueing for known concepts
while RAG provides novel details.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime
import os
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


def generate_with_engram(model, tokenizer, prompt, engram, max_tokens=100):
    """Generate with engram prepended (no RAG context)."""
    embed_layer = model.get_input_embeddings()

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


def generate_rag(model, tokenizer, context, question, max_tokens=100):
    """Generate with RAG context only (no engram)."""
    prompt = f"""Context: {context[:2000]}

Question: {question}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.split("\n")[0]


def generate_engram_plus_rag(model, tokenizer, context, question, engram, max_tokens=100):
    """Generate with BOTH engram AND RAG context."""
    embed_layer = model.get_input_embeddings()

    embed_norm = embed_layer.weight.norm(dim=1).mean().item()
    engram_norm = engram.norm(dim=1).mean().item()
    scaled_engram = engram * (embed_norm / engram_norm) if engram_norm > 0 else engram

    prompt = f"""Context: {context[:2000]}

Question: {question}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    prompt_embeds = embed_layer(inputs.input_ids)

    combined = torch.cat([scaled_engram.unsqueeze(0).to(model.device), prompt_embeds], dim=1)

    with torch.no_grad():
        output = model.generate(
            inputs_embeds=combined,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


def extract_facts(abstract):
    """Extract verifiable facts from abstract."""
    facts = []

    # Percentages
    percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', abstract)
    facts.extend([f"{p}%" for p in percentages])

    # Gene/protein names
    genes = re.findall(r'\b([A-Z][A-Z0-9]{2,}(?:-[A-Z0-9]+)?)\b', abstract)
    genes = [g for g in genes if g not in ['THE', 'AND', 'FOR', 'WITH', 'THIS', 'THAT', 'FROM', 'ARE', 'WAS', 'WERE', 'DNA', 'RNA']]
    facts.extend(genes[:5])

    # Key long terms
    words = [w for w in abstract.split() if len(w) > 10 and w.isalpha()]
    facts.extend(words[:3])

    return list(set(facts))


def check_facts(response, facts):
    """Count how many facts appear in response."""
    response_lower = response.lower()
    return sum(1 for f in facts if f.lower() in response_lower)


def load_papers(papers_file=None):
    """Load papers from JSON file."""
    if papers_file is None:
        papers_file = Path(__file__).parent / "papers.json"

    if not Path(papers_file).exists():
        print(f"ERROR: Papers file not found: {papers_file}")
        return None

    with open(papers_file) as f:
        return json.load(f)


def run_experiment(alpha=0.1, n_papers=100):
    """Run the chained engram + RAG experiment."""

    print("CHAINED ENGRAM + RAG EXPERIMENT")
    print(f"Alpha: {alpha}, Papers: {n_papers}")
    print()

    # Load model - use 7B like the successful experiments
    print("Loading Qwen2.5-7B model...")
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
    papers = load_papers()
    if papers is None:
        return None
    papers = papers[:n_papers]
    print(f"Using {len(papers)} papers")

    # Initialize session engram from first paper
    print()
    print("Initializing session engram from paper 1...")
    initial_text = f"Title: {papers[0]['title']}\n\nAbstract: {papers[0]['abstract']}"
    session_engram = extract_engram(model, tokenizer, initial_text)
    initial_engram = session_engram.clone()

    # Results tracking
    results = {
        'config': {
            'alpha': alpha,
            'n_papers': n_papers,
            'model': model_name,
            'timestamp': datetime.now().isoformat()
        },
        'recall_tests': [],
        'similarity_trajectory': [],
        'summary': {}
    }

    # Process papers
    print()
    print(f"Processing {n_papers} papers...")

    for turn, paper in enumerate(papers, 1):
        paper_text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"

        # Generate response using engram + RAG (the main condition)
        question = "What are the key findings of this research?"
        response = generate_engram_plus_rag(model, tokenizer, paper['abstract'], question, session_engram)

        # Update engram with response
        response_engram = extract_engram(model, tokenizer, response)
        session_engram = ema_update(session_engram, response_engram, alpha=alpha)

        # Track similarity to initial
        similarity = torch.nn.functional.cosine_similarity(
            initial_engram.flatten().unsqueeze(0),
            session_engram.flatten().unsqueeze(0)
        ).item()

        results['similarity_trajectory'].append({
            'turn': turn,
            'similarity': similarity
        })

        # Progress update
        if turn % 20 == 0:
            print(f"  Turn {turn}: similarity to initial = {similarity:.4f}")

        # Recall tests at specific intervals
        if turn in [25, 50, 75, 100]:
            print()
            print(f"RECALL TEST at turn {turn}")
            print("-" * 40)

            test_result = {
                'turn': turn,
                'similarity_to_initial': similarity,
                'lookbacks': {}
            }

            # Test recall for papers at different distances
            for lookback in [5, 10, 25, 50]:
                target_idx = turn - lookback
                if target_idx < 0:
                    continue

                target_paper = papers[target_idx]
                facts = extract_facts(target_paper['abstract'])

                if not facts:
                    continue

                question = f"What were the findings about {target_paper['title'][:30]}?"

                # Condition 1: RAG only
                rag_response = generate_rag(model, tokenizer, target_paper['abstract'], question)
                rag_facts = check_facts(rag_response, facts)

                # Condition 2: Engram + RAG
                both_response = generate_engram_plus_rag(
                    model, tokenizer, target_paper['abstract'], question, session_engram
                )
                both_facts = check_facts(both_response, facts)

                # Condition 3: Engram only (for comparison)
                simple_prompt = f"Question: {question}\nAnswer:"
                engram_response = generate_with_engram(model, tokenizer, simple_prompt, session_engram)
                engram_facts = check_facts(engram_response, facts)

                print(f"  {lookback} turns back: RAG={rag_facts}, Engram+RAG={both_facts}, Engram={engram_facts} (of {len(facts)} facts)")

                test_result['lookbacks'][f'{lookback}_turns'] = {
                    'paper_index': target_idx,
                    'paper_title': target_paper['title'][:50],
                    'facts_available': len(facts),
                    'rag_only': rag_facts,
                    'engram_plus_rag': both_facts,
                    'engram_only': engram_facts,
                    'engram_plus_rag_vs_rag': both_facts - rag_facts
                }

            results['recall_tests'].append(test_result)

    # Summary statistics
    print()
    print("SUMMARY")
    print("-" * 40)

    total_rag = 0
    total_both = 0
    total_engram = 0
    total_facts = 0
    comparisons = 0

    for test in results['recall_tests']:
        for lookback, data in test['lookbacks'].items():
            total_rag += data['rag_only']
            total_both += data['engram_plus_rag']
            total_engram += data['engram_only']
            total_facts += data['facts_available']
            comparisons += 1

    if comparisons > 0:
        print(f"Across {comparisons} recall tests:")
        print(f"  RAG only:       {total_rag}/{total_facts} facts ({100*total_rag/total_facts:.1f}%)")
        print(f"  Engram + RAG:   {total_both}/{total_facts} facts ({100*total_both/total_facts:.1f}%)")
        print(f"  Engram only:    {total_engram}/{total_facts} facts ({100*total_engram/total_facts:.1f}%)")
        print()
        print(f"  Engram+RAG vs RAG: {total_both - total_rag:+d} facts ({100*(total_both-total_rag)/total_facts:+.1f}%)")

        results['summary'] = {
            'total_comparisons': comparisons,
            'total_facts': total_facts,
            'rag_only_facts': total_rag,
            'engram_plus_rag_facts': total_both,
            'engram_only_facts': total_engram,
            'rag_only_pct': 100 * total_rag / total_facts,
            'engram_plus_rag_pct': 100 * total_both / total_facts,
            'engram_only_pct': 100 * total_engram / total_facts,
            'engram_plus_rag_advantage': total_both - total_rag
        }

        print()
        if total_both > total_rag:
            print("RESULT: Engram + RAG OUTPERFORMS RAG alone!")
        elif total_both == total_rag:
            print("RESULT: Engram + RAG matches RAG (engram is neutral)")
        else:
            print("RESULT: Engram + RAG underperforms RAG (engram interferes)")

    # Final similarity
    final_sim = results['similarity_trajectory'][-1]['similarity']
    print()
    print(f"Final engram similarity to initial: {final_sim:.4f}")

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / f'chained_engram_plus_rag_alpha{alpha}.json'

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Chained Engram + RAG Experiment")
    parser.add_argument('--alpha', type=float, default=0.1, help='EMA alpha value')
    parser.add_argument('--n-papers', type=int, default=100, help='Number of papers')

    args = parser.parse_args()
    run_experiment(alpha=args.alpha, n_papers=args.n_papers)


if __name__ == "__main__":
    main()
