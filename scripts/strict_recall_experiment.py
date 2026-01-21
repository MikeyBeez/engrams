#!/usr/bin/env python3
"""
Strict Recall Experiment with Negative Controls

Addresses the scoring problem from chained_biology_experiment.py:
- Previous experiment: score 1 = "vague but topic-relevant"
- Problem: Model gives generic neuroscience responses for ANY neuroscience question
- Can't distinguish actual recall from domain-cueing

This experiment adds:
1. STRICT SCORING - requires specific facts, not just topic relevance
2. NEGATIVE CONTROLS - ask about papers NOT in the corpus
3. FABRICATED DETAILS - ask about specific (fake) claims to test hallucination
4. BASELINE COMPARISON - compare engram vs no-engram responses
5. CROSS-DOMAIN CONTROLS - use physics papers as negative control

Key question: Does the engram provide ANY paper-specific recall, or just domain cueing?
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

# ============================================================================
# NEGATIVE CONTROL PAPERS (not in corpus)
# ============================================================================

NEGATIVE_CONTROL_PAPERS = [
    {
        "pmid": "fake_001",
        "title": "Quantum effects in microtubule consciousness",
        "abstract": """We investigated quantum coherence in neuronal microtubules using
        novel spectroscopic methods. Our results suggest that Penrose-Hameroff orchestrated
        objective reduction (Orch-OR) may have measurable signatures at 40Hz gamma frequencies.
        Anesthetics disrupted quantum coherence, correlating with loss of consciousness.""",
        "key_claim": "quantum coherence at 40Hz gamma",
        "domain": "neuroscience"  # Same domain but not in corpus
    },
    {
        "pmid": "fake_002",
        "title": "BDNF gene therapy reverses age-related memory decline",
        "abstract": """AAV-mediated BDNF delivery to aged rat hippocampus restored
        spatial memory to young adult levels. Treatment increased dendritic spine density
        by 45% and enhanced LTP magnitude. Effects persisted for 6 months post-injection.""",
        "key_claim": "45% increase in dendritic spine density",
        "domain": "neuroscience"
    },
    {
        "pmid": "fake_003",
        "title": "Dark matter detection using superconducting qubits",
        "abstract": """We propose a novel dark matter detector based on superconducting
        transmon qubits. Theoretical calculations predict sensitivity to axion masses
        between 1-100 μeV with integration times of 1000 hours.""",
        "key_claim": "axion detection with transmon qubits",
        "domain": "physics"  # Different domain entirely
    },
]

# ============================================================================
# FABRICATED CLAIMS (to test if model hallucinates agreement)
# ============================================================================

def generate_fabricated_questions(paper):
    """Generate questions with fabricated specific claims about a paper."""
    title_short = paper['title'][:40]

    fabricated = [
        {
            "question": f"The paper '{title_short}' reported a 73% improvement rate. What methods achieved this?",
            "fabricated_claim": "73% improvement rate",
            "expected": "should_not_confirm"  # Model should NOT confirm this fake stat
        },
        {
            "question": f"In '{title_short}', the researchers used a sample size of 847 subjects. How were they recruited?",
            "fabricated_claim": "847 subjects",
            "expected": "should_not_confirm"
        },
        {
            "question": f"The paper '{title_short}' was conducted at Stanford University. What facilities did they use?",
            "fabricated_claim": "Stanford University",
            "expected": "should_not_confirm"
        },
    ]
    return fabricated


# ============================================================================
# STRICT SCORING
# ============================================================================

def extract_specific_facts(abstract):
    """Extract specific verifiable facts from an abstract."""
    facts = []

    # Look for percentages
    percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', abstract)
    for p in percentages:
        facts.append(f"{p}%")

    # Look for specific numbers with units
    numbers = re.findall(r'(\d+(?:\.\d+)?)\s*(mg|μg|ng|mM|μM|nM|Hz|kHz|MHz|days?|weeks?|months?|years?|patients?|subjects?|mice|rats)', abstract, re.IGNORECASE)
    for num, unit in numbers:
        facts.append(f"{num} {unit}")

    # Look for gene/protein names (capitalized abbreviations)
    genes = re.findall(r'\b([A-Z][A-Z0-9]{2,}(?:-[A-Z0-9]+)?)\b', abstract)
    genes = [g for g in genes if g not in ['THE', 'AND', 'FOR', 'WITH', 'THIS', 'THAT', 'FROM']]
    facts.extend(genes[:3])  # Top 3 gene names

    # Look for specific methodologies
    methods = ['CRISPR', 'optogenetic', 'fMRI', 'PET', 'RNA-seq', 'scRNA-seq', 'patch-clamp',
               'two-photon', 'immunohistochemistry', 'Western blot', 'ELISA', 'PCR']
    for method in methods:
        if method.lower() in abstract.lower():
            facts.append(method)

    return list(set(facts))[:5]  # Return up to 5 unique facts


def strict_score_response(response, paper, question_type="recall"):
    """
    Strict scoring that requires specific facts.

    Returns:
        3: Specific recall (mentions 2+ verifiable facts from paper)
        2: Partial recall (mentions 1 verifiable fact)
        1: Generic relevant (topic-appropriate but no specific facts)
        0: Irrelevant or wrong
        -1: Hallucinated confirmation of fabricated claim
    """
    response_lower = response.lower()

    if question_type == "fabricated":
        # Check if model confirmed a fabricated claim
        confirmation_phrases = ['yes', 'correct', 'that\'s right', 'indeed', 'the study found',
                               'reported', 'showed', 'demonstrated', 'confirmed']
        denial_phrases = ['i don\'t', 'i cannot', 'not mentioned', 'no information',
                         'cannot confirm', 'unclear', 'not specified']

        has_confirmation = any(p in response_lower for p in confirmation_phrases)
        has_denial = any(p in response_lower for p in denial_phrases)

        if has_denial:
            return 2  # Good - appropriately uncertain
        elif has_confirmation:
            return -1  # Bad - hallucinated confirmation
        else:
            return 1  # Neutral

    # For recall questions, check for specific facts
    facts = extract_specific_facts(paper['abstract'])

    if not facts:
        # Fallback to key terms if no specific facts found
        key_terms = [w for w in paper['abstract'].lower().split()
                     if len(w) > 7 and w.isalpha()][:5]
        matches = sum(1 for term in key_terms if term in response_lower)
        if matches >= 2:
            return 2
        elif matches >= 1:
            return 1
        return 0

    # Count how many specific facts appear in response
    fact_matches = sum(1 for fact in facts if fact.lower() in response_lower)

    if fact_matches >= 2:
        return 3  # Specific recall
    elif fact_matches >= 1:
        return 2  # Partial recall
    else:
        # Check for generic topic relevance
        topic_words = ['neuron', 'brain', 'synap', 'gene', 'protein', 'cell', 'disease']
        topic_matches = sum(1 for w in topic_words if w in response_lower)
        if topic_matches >= 2:
            return 1  # Generic relevant
        return 0  # Irrelevant


# ============================================================================
# CORE ENGRAM FUNCTIONS
# ============================================================================

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


def ema_update(current_engram, new_engram, alpha=0.3):
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

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


def generate_baseline(model, tokenizer, prompt, max_tokens=150):
    """Generate without engram (baseline)."""
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs.to(model.device),
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_positive_recall(model, tokenizer, engram, paper, verbose=True):
    """Test recall of a paper that WAS in the corpus."""
    facts = extract_specific_facts(paper['abstract'])

    question = f"What specific findings did the paper '{paper['title'][:50]}' report?"

    response_with_engram = generate_with_engram(
        model, tokenizer,
        f"Question: {question}\n\nAnswer:",
        engram, max_tokens=100
    )

    response_baseline = generate_baseline(
        model, tokenizer,
        f"Question: {question}\n\nAnswer:",
        max_tokens=100
    )

    score_with = strict_score_response(response_with_engram, paper, "recall")
    score_without = strict_score_response(response_baseline, paper, "recall")

    if verbose:
        print(f"    Paper: {paper['title'][:50]}...")
        print(f"    Facts to find: {facts}")
        print(f"    With engram (score {score_with}): {response_with_engram[:80]}...")
        print(f"    Without engram (score {score_without}): {response_baseline[:80]}...")

    return {
        'paper': paper['title'],
        'facts': facts,
        'with_engram': {'response': response_with_engram[:200], 'score': score_with},
        'without_engram': {'response': response_baseline[:200], 'score': score_without},
        'engram_advantage': score_with - score_without
    }


def test_negative_control(model, tokenizer, engram, fake_paper, verbose=True):
    """Test that model doesn't claim to recall papers NOT in corpus."""
    question = f"What did the paper '{fake_paper['title'][:50]}' find?"

    response = generate_with_engram(
        model, tokenizer,
        f"Question: {question}\n\nAnswer:",
        engram, max_tokens=100
    )

    # Check if model appropriately expresses uncertainty or incorrectly claims knowledge
    uncertainty_phrases = ['i don\'t', 'i cannot', 'no information', 'not familiar',
                          'not sure', 'unclear', 'cannot find']
    confident_phrases = ['the paper found', 'the study showed', 'researchers demonstrated',
                        'results indicate', 'findings suggest']

    response_lower = response.lower()
    shows_uncertainty = any(p in response_lower for p in uncertainty_phrases)
    shows_confidence = any(p in response_lower for p in confident_phrases)

    if shows_uncertainty:
        score = 2  # Good - appropriately uncertain
    elif shows_confidence:
        score = -1  # Bad - false confidence
    else:
        score = 0  # Neutral

    if verbose:
        print(f"    NEGATIVE CONTROL: {fake_paper['title'][:40]}...")
        print(f"    Response (score {score}): {response[:80]}...")
        print(f"    Shows uncertainty: {shows_uncertainty}, Shows confidence: {shows_confidence}")

    return {
        'fake_paper': fake_paper['title'],
        'domain': fake_paper['domain'],
        'response': response[:200],
        'score': score,
        'appropriately_uncertain': shows_uncertainty,
        'falsely_confident': shows_confidence
    }


def test_fabricated_claim(model, tokenizer, engram, paper, verbose=True):
    """Test if model hallucinates confirmation of fabricated claims."""
    fabricated = generate_fabricated_questions(paper)[0]  # Use first fabrication

    response = generate_with_engram(
        model, tokenizer,
        f"Question: {fabricated['question']}\n\nAnswer:",
        engram, max_tokens=100
    )

    score = strict_score_response(response, paper, "fabricated")

    if verbose:
        print(f"    FABRICATED CLAIM: {fabricated['fabricated_claim']}")
        print(f"    Response (score {score}): {response[:80]}...")

    return {
        'paper': paper['title'],
        'fabricated_claim': fabricated['fabricated_claim'],
        'response': response[:200],
        'score': score,
        'hallucinated': score == -1
    }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def load_papers_from_file(papers_file=None):
    """Load papers from JSON file."""
    if papers_file is None:
        papers_file = Path(__file__).parent / "papers.json"

    if not Path(papers_file).exists():
        print(f"ERROR: Papers file not found: {papers_file}")
        return None

    with open(papers_file) as f:
        papers = json.load(f)

    print(f"Loaded {len(papers)} papers from {papers_file}")
    return papers


def run_experiment(n_papers=50, alpha=0.1, papers_file=None):
    """Run the strict recall experiment."""

    print("=" * 80)
    print("STRICT RECALL EXPERIMENT WITH NEGATIVE CONTROLS")
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
    papers = load_papers_from_file(papers_file)
    if papers is None:
        return None
    papers = papers[:n_papers]

    # Build session engram through chained compression
    print(f"\nBuilding session engram from {n_papers} papers...")

    initial_text = f"Title: {papers[0]['title']}\n\nAbstract: {papers[0]['abstract']}"
    session_engram = extract_engram(model, tokenizer, initial_text)

    for i, paper in enumerate(papers[1:], 1):
        if i % 10 == 0:
            print(f"  Processing paper {i}/{n_papers}...")

        paper_text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
        prompt = f"Analyze this paper:\n\n{paper_text}\n\nKey findings:"

        response = generate_with_engram(model, tokenizer, prompt, session_engram)
        response_engram = extract_engram(model, tokenizer, response)
        session_engram = ema_update(session_engram, response_engram, alpha=alpha)

    print("Session engram built.")

    # Results tracking
    results = {
        'config': {
            'n_papers': n_papers,
            'alpha': alpha,
            'timestamp': datetime.now().isoformat()
        },
        'positive_recall': [],
        'negative_controls': [],
        'fabrication_tests': []
    }

    # Test 1: Positive recall at different lookbacks
    print("\n" + "=" * 60)
    print("TEST 1: POSITIVE RECALL (papers in corpus)")
    print("=" * 60)

    lookbacks = [5, 10, 25, 49]  # 49 is oldest paper
    for lookback in lookbacks:
        if n_papers > lookback:
            paper_idx = n_papers - 1 - lookback
            paper = papers[paper_idx]
            print(f"\n  Testing paper from {lookback} turns ago:")
            result = test_positive_recall(model, tokenizer, session_engram, paper)
            result['lookback'] = lookback
            results['positive_recall'].append(result)

    # Test 2: Negative controls
    print("\n" + "=" * 60)
    print("TEST 2: NEGATIVE CONTROLS (papers NOT in corpus)")
    print("=" * 60)

    for fake_paper in NEGATIVE_CONTROL_PAPERS:
        print(f"\n  Testing fake paper ({fake_paper['domain']} domain):")
        result = test_negative_control(model, tokenizer, session_engram, fake_paper)
        results['negative_controls'].append(result)

    # Test 3: Fabrication tests
    print("\n" + "=" * 60)
    print("TEST 3: FABRICATED CLAIMS (should not confirm)")
    print("=" * 60)

    # Test on 3 random papers from corpus
    test_papers = random.sample(papers, min(3, len(papers)))
    for paper in test_papers:
        print(f"\n  Testing fabricated claim about:")
        result = test_fabricated_claim(model, tokenizer, session_engram, paper)
        results['fabrication_tests'].append(result)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nPOSITIVE RECALL SCORES:")
    for r in results['positive_recall']:
        print(f"  {r['lookback']} turns back: with={r['with_engram']['score']}, "
              f"without={r['without_engram']['score']}, advantage={r['engram_advantage']}")

    print("\nNEGATIVE CONTROL SCORES:")
    for r in results['negative_controls']:
        status = "✓ Uncertain" if r['appropriately_uncertain'] else "✗ False confidence" if r['falsely_confident'] else "~ Neutral"
        print(f"  {r['fake_paper'][:40]}... ({r['domain']}): {status}")

    print("\nFABRICATION TEST SCORES:")
    hallucination_count = sum(1 for r in results['fabrication_tests'] if r['hallucinated'])
    print(f"  Hallucinated confirmations: {hallucination_count}/{len(results['fabrication_tests'])}")

    # Calculate overall metrics
    avg_positive = sum(r['with_engram']['score'] for r in results['positive_recall']) / len(results['positive_recall']) if results['positive_recall'] else 0
    avg_advantage = sum(r['engram_advantage'] for r in results['positive_recall']) / len(results['positive_recall']) if results['positive_recall'] else 0
    false_confidence_rate = sum(1 for r in results['negative_controls'] if r['falsely_confident']) / len(results['negative_controls']) if results['negative_controls'] else 0

    print(f"\nOVERALL METRICS:")
    print(f"  Avg positive recall score: {avg_positive:.2f}")
    print(f"  Avg engram advantage: {avg_advantage:.2f}")
    print(f"  False confidence rate on negatives: {false_confidence_rate:.1%}")
    print(f"  Hallucination rate: {hallucination_count}/{len(results['fabrication_tests'])}")

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / f'strict_recall_alpha{alpha}.json'

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Strict Recall Experiment")
    parser.add_argument('--n-papers', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--papers-file', type=str, default=None)

    args = parser.parse_args()

    run_experiment(
        n_papers=args.n_papers,
        alpha=args.alpha,
        papers_file=args.papers_file
    )


if __name__ == "__main__":
    main()
