#!/usr/bin/env python3
"""
Chained Engram Biology Experiment

Test whether information persists through 100 turns of engram compression
when each turn processes new biological research content.

Key difference from rolling_compression_test.py:
- That test used drifting conversation (WWII → physics → food)
- This test chains DENSE NEW CONTENT at every turn

The question: Can chained engrams retain recall of papers from N turns ago?

Experimental design:
1. Load 100 biology paper abstracts
2. For each paper:
   a. Inject current session engram + paper abstract
   b. Generate analysis/summary
   c. Extract engram from response
   d. EMA update session engram
3. Every 10 turns, test recall of papers from 5, 10, 20, 50 turns ago
4. Measure degradation curve

Expected finding based on rolling_compression results:
- With alpha=0.1: May retain some signal from recent papers
- With alpha=0.3+: Rapid collapse after ~60 turns
- Papers from 50+ turns ago: Likely unrecoverable

This tests the LIMIT of chained compression, not just drift.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime
import os
import random
from pathlib import Path
import argparse

# ============================================================================
# PAPER LOADING
# ============================================================================

def load_papers_from_file(papers_file=None):
    """Load papers from JSON file (fetched from PubMed)."""
    if papers_file is None:
        # Default location
        papers_file = Path(__file__).parent / "papers.json"

    if not Path(papers_file).exists():
        print(f"ERROR: Papers file not found: {papers_file}")
        print("Run fetch_pubmed_papers.py first to download papers from PubMed")
        print("Or use --synthetic flag to generate synthetic papers")
        return None

    with open(papers_file) as f:
        papers = json.load(f)

    print(f"Loaded {len(papers)} papers from {papers_file}")
    return papers


# ============================================================================
# SAMPLE BIOLOGY ABSTRACTS (fallback if no PubMed data)
# ============================================================================

BIOLOGY_PAPERS = [
    {
        "pmid": "paper_001",
        "title": "CRISPR-Cas9 editing reveals novel synaptic plasticity mechanisms",
        "abstract": """Using CRISPR-Cas9 gene editing in hippocampal neurons, we identified
        CAMK2A as a critical regulator of long-term potentiation. Knockout of CAMK2A
        abolished LTP induction while sparing basal transmission. Rescue experiments
        with phosphomimetic mutants restored plasticity, demonstrating the necessity
        of T286 autophosphorylation. These findings establish CAMK2A-T286 as an
        essential molecular switch for Hebbian plasticity."""
    },
    {
        "pmid": "paper_002",
        "title": "Mitochondrial dysfunction in Parkinson's disease neurons",
        "abstract": """Patient-derived iPSC neurons carrying PINK1 mutations showed
        selective Complex I deficiency and increased oxidative stress. Mitophagy was
        impaired, leading to accumulation of damaged mitochondria. Treatment with
        the NAD+ precursor NMN restored mitochondrial function and reduced alpha-synuclein
        aggregation. Our data suggest mitochondrial quality control as a therapeutic
        target for Parkinson's disease."""
    },
    {
        "pmid": "paper_003",
        "title": "Blood-brain barrier permeability in Alzheimer's disease",
        "abstract": """Using dynamic contrast-enhanced MRI, we measured BBB permeability
        in 200 subjects across the Alzheimer's continuum. Hippocampal BBB breakdown
        preceded amyloid deposition by 2-3 years and correlated with CSF sPDGFRβ levels,
        a marker of pericyte injury. APOE4 carriers showed accelerated BBB dysfunction.
        These findings support vascular contributions to AD pathogenesis."""
    },
    {
        "pmid": "paper_004",
        "title": "Optogenetic mapping of prefrontal cortex decision circuits",
        "abstract": """Combining optogenetics with two-photon calcium imaging in
        behaving mice, we mapped the circuit architecture of value-based decision making.
        Layer 5 pyramidal neurons in prelimbic cortex encoded action values, while
        infralimbic neurons tracked prediction errors. Reciprocal inhibition between
        these populations implemented a winner-take-all competition underlying choice."""
    },
    {
        "pmid": "paper_005",
        "title": "Single-cell RNA sequencing reveals microglial states in neurodegeneration",
        "abstract": """ScRNA-seq of 500,000 microglia from mouse models of AD, ALS, and
        MS revealed a conserved disease-associated microglial (DAM) state characterized
        by TREM2 upregulation and homeostatic gene suppression. DAM transition required
        TREM2 signaling and was blocked in Trem2-/- mice. Human postmortem validation
        confirmed DAM signatures across neurodegenerative conditions."""
    },
    {
        "pmid": "paper_006",
        "title": "Gut microbiome modulation of brain inflammation",
        "abstract": """Germ-free mice showed reduced microglial activation and improved
        outcomes in experimental autoimmune encephalomyelitis. Colonization with
        segmented filamentous bacteria restored neuroinflammation via Th17 induction.
        Metabolomic analysis identified tryptophan metabolites as key mediators.
        Dietary tryptophan restriction phenocopied germ-free protection."""
    },
    {
        "pmid": "paper_007",
        "title": "NMDA receptor subunit composition and schizophrenia",
        "abstract": """Postmortem analysis of 50 schizophrenia brains revealed reduced
        GluN2B subunit expression in dorsolateral prefrontal cortex. This was associated
        with hypermethylation of the GRIN2B promoter. Patient-derived organoids
        recapitulated the deficit. Pharmacological GluN2B positive allosteric modulators
        rescued working memory deficits in animal models."""
    },
    {
        "pmid": "paper_008",
        "title": "Astrocyte calcium waves in cortical spreading depression",
        "abstract": """Using genetically encoded calcium indicators, we imaged astrocyte
        activity during cortical spreading depression in migraine models. Astrocytes
        showed synchronized calcium waves preceding the neuronal depolarization front.
        Cx43 gap junction blockade prevented wave propagation and attenuated CSD.
        Targeting astrocyte connexins may prevent migraine aura."""
    },
    {
        "pmid": "paper_009",
        "title": "Adult neurogenesis in human dentate gyrus",
        "abstract": """Using 14C birth-dating from nuclear bomb tests, we quantified
        hippocampal neurogenesis across the human lifespan. Approximately 700 new
        neurons are added daily in young adults, declining to 200/day by age 70.
        Depression and chronic stress accelerated decline. Exercise increased
        neurogenesis rates by 30% in elderly subjects."""
    },
    {
        "pmid": "paper_010",
        "title": "Tau propagation through neural circuits",
        "abstract": """Injection of patient-derived tau into mouse entorhinal cortex
        produced spreading to connected regions over 6 months, following anterograde
        trans-synaptic pathways. Chemogenetic silencing of entorhinal neurons blocked
        propagation. Tau spreading required synaptic activity and could be prevented
        by DREADD-mediated inhibition. Activity-dependent tau release is a
        therapeutic target."""
    },
]

# Generate additional papers by varying the templates
def generate_paper_corpus(n_papers=100):
    """Generate a corpus of varied biology papers."""

    topics = [
        ("synaptic plasticity", "NMDA", "LTP", "dendritic spines", "calcium"),
        ("neurodegeneration", "tau", "amyloid", "alpha-synuclein", "protein aggregation"),
        ("glial cells", "astrocytes", "microglia", "oligodendrocytes", "neuroinflammation"),
        ("neural circuits", "optogenetics", "connectomics", "behavior", "decision making"),
        ("gene therapy", "CRISPR", "AAV vectors", "gene expression", "epigenetics"),
        ("ion channels", "voltage-gated", "potassium", "sodium", "channelopathy"),
        ("neurodevelopment", "axon guidance", "migration", "cortical layers", "morphogenesis"),
        ("neurotransmitters", "dopamine", "serotonin", "GABA", "glutamate"),
        ("brain imaging", "fMRI", "PET", "calcium imaging", "voltage imaging"),
        ("sleep", "circadian rhythms", "memory consolidation", "oscillations", "slow waves"),
    ]

    findings = [
        "Our data demonstrate that {} is essential for {}.",
        "These results reveal a novel mechanism involving {} and {}.",
        "Genetic manipulation of {} altered {} by 50%.",
        "Pharmacological targeting of {} restored normal {}.",
        "Single-cell analysis identified {} as a marker of {}.",
        "Circuit mapping revealed {} projections mediate {}.",
        "Computational modeling predicted {} regulation of {}.",
        "Longitudinal tracking showed {} precedes {} by 2 years.",
        "Rescue experiments with {} confirmed the role of {}.",
        "Human validation studies replicated {} effects on {}.",
    ]

    papers = list(BIOLOGY_PAPERS)  # Start with seed papers

    while len(papers) < n_papers:
        topic_set = random.choice(topics)
        finding_template = random.choice(findings)

        t1, t2 = random.sample(topic_set, 2)
        finding = finding_template.format(t1, t2)

        title_words = random.sample(topic_set, 3)
        title = f"Novel insights into {title_words[0]}: {title_words[1]} and {title_words[2]} interactions"

        abstract = f"""This study investigated the role of {topic_set[0]} in neural function.
        Using a combination of {random.choice(['genetic', 'pharmacological', 'imaging', 'behavioral'])}
        approaches, we characterized the {topic_set[1]} pathway. {finding}
        Additionally, we found that {topic_set[2]} modulates {topic_set[3]} through
        {topic_set[4]}-dependent mechanisms. These findings have implications for
        understanding {random.choice(['learning', 'disease', 'development', 'aging'])}."""

        papers.append({
            "pmid": f"paper_{len(papers)+1:03d}",
            "title": title,
            "abstract": abstract
        })

    return papers[:n_papers]


# ============================================================================
# CORE ENGRAM FUNCTIONS (from proven implementations)
# ============================================================================

def extract_engram(model, tokenizer, text, layer=16, num_tokens=32):
    """Extract engram vectors from middle layer."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)

    with torch.no_grad():
        outputs = model(**inputs.to(model.device), output_hidden_states=True)

    hidden = outputs.hidden_states[layer]
    seq_len = hidden.shape[1]

    # Handle edge case of very short sequences
    if seq_len < num_tokens:
        # Pad with zeros or repeat
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
    """Generate response with engram injection at layer 0."""
    embed_layer = model.get_input_embeddings()

    # Scale engram to embedding space
    embed_norm = embed_layer.weight.norm(dim=1).mean().item()
    engram_norm = engram.norm(dim=1).mean().item()

    if engram_norm > 0:
        scaled_engram = engram * (embed_norm / engram_norm)
    else:
        scaled_engram = engram

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_embeds = embed_layer(inputs.input_ids.to(model.device))

    # Prepend engram
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
    """Generate response without engram (baseline)."""
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
# RECALL TESTING
# ============================================================================

def test_recall(model, tokenizer, engram, paper, verbose=False):
    """
    Test if the engram allows recall of specific paper content.

    Returns a score:
    - 2: Specific recall (mentions key terms from the paper)
    - 1: Topic-relevant but vague
    - 0: No relevant information
    """
    # Extract key terms from the paper
    key_terms = []
    for word in paper['abstract'].lower().split():
        if len(word) > 6 and word.isalpha():
            key_terms.append(word)
    key_terms = list(set(key_terms))[:5]  # Top 5 unique long words

    # Test question
    question = f"What did the paper about '{paper['title'][:50]}' find?"

    # Generate with engram
    response = generate_with_engram(
        model, tokenizer,
        f"Question: {question}\n\nAnswer:",
        engram,
        max_tokens=100
    )

    # Score based on key term presence
    response_lower = response.lower()
    matches = sum(1 for term in key_terms if term in response_lower)

    if matches >= 2:
        score = 2  # Specific recall
    elif matches >= 1 or any(t in response_lower for t in ['study', 'found', 'showed', 'demonstrated']):
        score = 1  # Topic relevant
    else:
        score = 0  # No recall

    if verbose:
        print(f"    Q: {question[:60]}...")
        print(f"    A: {response[:80]}...")
        print(f"    Key terms: {key_terms}")
        print(f"    Matches: {matches}, Score: {score}")

    return {
        'question': question,
        'response': response[:200],
        'key_terms': key_terms,
        'matches': matches,
        'score': score
    }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(alpha=0.1, n_papers=100, use_synthetic=False, papers_file=None):
    """Run the chained biology experiment."""

    print("=" * 80)
    print("CHAINED ENGRAM BIOLOGY EXPERIMENT")
    print(f"Alpha: {alpha}, Papers: {n_papers}")
    print(f"Source: {'synthetic' if use_synthetic else 'PubMed'}")
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

    # Load or generate paper corpus
    if use_synthetic:
        print(f"\nGenerating {n_papers} synthetic paper corpus...")
        papers = generate_paper_corpus(n_papers)
    else:
        print(f"\nLoading papers from PubMed...")
        papers = load_papers_from_file(papers_file)
        if papers is None:
            print("Falling back to synthetic papers...")
            papers = generate_paper_corpus(n_papers)
        elif len(papers) < n_papers:
            print(f"Warning: Only {len(papers)} papers available, adjusting n_papers")
            n_papers = len(papers)

    # Initialize session engram from first paper
    print("\nInitializing session engram from paper 1...")
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
        'turns': [],
        'recall_tests': [],
        'similarity_trajectory': []
    }

    # Test intervals and lookback distances
    test_interval = 10
    lookback_distances = [5, 10, 20, 50]

    print(f"\nProcessing {n_papers} papers...")
    print("-" * 80)

    for turn in range(n_papers):
        paper = papers[turn]

        # Every 10 turns, print progress
        if turn % 10 == 0:
            print(f"\n[Turn {turn+1}/{n_papers}] {paper['title'][:50]}...")

        # Create prompt with paper content
        prompt = f"""Analyze this biology paper:

Title: {paper['title']}

Abstract: {paper['abstract']}

Provide a brief analysis of the key findings:"""

        # Generate response using current session engram
        response = generate_with_engram(model, tokenizer, prompt, session_engram)

        # Extract engram from response
        response_engram = extract_engram(model, tokenizer, response)

        # Compute similarity before update
        flat_session = session_engram.flatten()
        flat_initial = initial_engram.flatten()
        sim_to_initial = torch.nn.functional.cosine_similarity(
            flat_session.unsqueeze(0), flat_initial.unsqueeze(0)
        ).item()

        # EMA update
        session_engram = ema_update(session_engram, response_engram, alpha=alpha)

        # Compute similarity after update
        flat_session_new = session_engram.flatten()
        sim_after = torch.nn.functional.cosine_similarity(
            flat_session_new.unsqueeze(0), flat_initial.unsqueeze(0)
        ).item()

        # Store turn data
        results['turns'].append({
            'turn': turn + 1,
            'pmid': paper['pmid'],
            'title': paper['title'],
            'sim_to_initial': sim_after,
            'response_length': len(response)
        })

        results['similarity_trajectory'].append({
            'turn': turn + 1,
            'similarity': sim_after
        })

        # Recall tests at intervals
        if (turn + 1) % test_interval == 0 and turn > 0:
            print(f"\n  === RECALL TEST AT TURN {turn+1} ===")
            print(f"  Similarity to initial: {sim_after:.4f}")

            recall_result = {
                'test_turn': turn + 1,
                'similarity': sim_after,
                'lookbacks': {}
            }

            for lookback in lookback_distances:
                if turn >= lookback:
                    test_paper = papers[turn - lookback]
                    print(f"\n  Testing recall of paper from {lookback} turns ago:")
                    print(f"    Paper: {test_paper['title'][:50]}...")

                    recall = test_recall(model, tokenizer, session_engram, test_paper, verbose=True)
                    recall_result['lookbacks'][f'{lookback}_turns'] = recall

                    # Also test with initial engram (control)
                    recall_initial = test_recall(model, tokenizer, initial_engram, test_paper, verbose=False)
                    recall_result['lookbacks'][f'{lookback}_turns_initial'] = recall_initial

            results['recall_tests'].append(recall_result)
            print()

    # Final summary
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    # Compute recall degradation curve
    print("\nRECALL DEGRADATION BY LOOKBACK DISTANCE:")
    print("-" * 60)

    for lookback in lookback_distances:
        scores = []
        for test in results['recall_tests']:
            if f'{lookback}_turns' in test['lookbacks']:
                scores.append(test['lookbacks'][f'{lookback}_turns']['score'])

        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"  {lookback} turns back: avg score {avg_score:.2f} (n={len(scores)})")

    print("\nSIMILARITY TRAJECTORY:")
    print("-" * 60)
    for i, sim in enumerate(results['similarity_trajectory']):
        if (i + 1) % 10 == 0:
            print(f"  Turn {sim['turn']:3d}: {sim['similarity']:.4f}")

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / f'chained_biology_alpha{alpha}.json'

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return results


def main():
    """Run experiments with different alpha values."""

    parser = argparse.ArgumentParser(
        description="Chained Engram Biology Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with real PubMed papers (default)
  python chained_biology_experiment.py

  # Run with synthetic papers
  python chained_biology_experiment.py --synthetic

  # Single alpha value
  python chained_biology_experiment.py --alpha 0.2 --n-papers 50

  # Compare multiple alpha values
  python chained_biology_experiment.py --compare

  # Use custom papers file
  python chained_biology_experiment.py --papers-file /path/to/papers.json
        """
    )

    parser.add_argument('--alpha', type=float, default=0.1,
                        help='EMA alpha value (default: 0.1)')
    parser.add_argument('--n-papers', type=int, default=100,
                        help='Number of papers to process (default: 100)')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic papers instead of PubMed')
    parser.add_argument('--papers-file', type=str, default=None,
                        help='Path to papers JSON file')
    parser.add_argument('--compare', action='store_true',
                        help='Run comparison of alpha 0.1 vs 0.3')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')

    args = parser.parse_args()

    # Update output directory if specified
    if args.output_dir:
        global output_dir
        output_dir = args.output_dir

    if args.compare:
        # Run comparison experiment
        print("Running comparison experiment: alpha 0.1 vs 0.3")
        print("=" * 80)

        results_01 = run_experiment(
            alpha=0.1, n_papers=args.n_papers,
            use_synthetic=args.synthetic, papers_file=args.papers_file
        )

        results_03 = run_experiment(
            alpha=0.3, n_papers=args.n_papers,
            use_synthetic=args.synthetic, papers_file=args.papers_file
        )

        # Compare
        print("\n" + "=" * 80)
        print("COMPARISON: ALPHA 0.1 vs 0.3")
        print("=" * 80)

        print("\nFinal similarities:")
        print(f"  Alpha 0.1: {results_01['similarity_trajectory'][-1]['similarity']:.4f}")
        print(f"  Alpha 0.3: {results_03['similarity_trajectory'][-1]['similarity']:.4f}")

        print("\nRecall scores at final test, 50 turns back:")
        for r, alpha in [(results_01, 0.1), (results_03, 0.3)]:
            if r['recall_tests']:
                last_test = r['recall_tests'][-1]
                if '50_turns' in last_test['lookbacks']:
                    score = last_test['lookbacks']['50_turns']['score']
                    print(f"  Alpha {alpha}: {score}")

    else:
        # Single run
        results = run_experiment(
            alpha=args.alpha, n_papers=args.n_papers,
            use_synthetic=args.synthetic, papers_file=args.papers_file
        )

        # Summary
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        print(f"Alpha: {args.alpha}")
        print(f"Papers: {args.n_papers}")
        print(f"Final similarity to initial: {results['similarity_trajectory'][-1]['similarity']:.4f}")

        if results['recall_tests']:
            last_test = results['recall_tests'][-1]
            print("\nRecall at final test:")
            for lookback in [5, 10, 20, 50]:
                key = f'{lookback}_turns'
                if key in last_test['lookbacks']:
                    score = last_test['lookbacks'][key]['score']
                    print(f"  {lookback} turns back: score {score}")


if __name__ == "__main__":
    main()
