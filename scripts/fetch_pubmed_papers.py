#!/usr/bin/env python3
"""
Fetch real biology papers from PubMed for the chained engram experiment.

Uses Biopython's Entrez API to search and retrieve abstracts.

Requirements:
    pip install biopython

Usage:
    python fetch_pubmed_papers.py

Output:
    papers.json - 100 papers with pmid, title, abstract
"""

import json
import time
from pathlib import Path

try:
    from Bio import Entrez
except ImportError:
    print("ERROR: Biopython not installed. Run: pip install biopython")
    exit(1)

# IMPORTANT: Set your email for NCBI API compliance
Entrez.email = "engram-research@example.com"  # Generic for research use

# Search topics - neuroscience/biology focus
SEARCH_TOPICS = [
    "synaptic plasticity mechanisms",
    "NMDA receptor function",
    "mitochondrial dysfunction neurons",
    "CRISPR brain gene therapy",
    "adult hippocampal neurogenesis",
    "blood brain barrier permeability",
    "axon guidance molecules",
    "dendritic spine dynamics",
    "microglial activation neurodegeneration",
    "neurodevelopmental disorder genetics",
    "dopamine reward circuitry",
    "astrocyte calcium signaling",
    "tau protein propagation",
    "amyloid beta aggregation",
    "optogenetics neural circuits",
    "single cell RNA sequencing brain",
    "circadian rhythm molecular",
    "long term potentiation hippocampus",
    "Parkinson disease alpha synuclein",
    "Alzheimer disease biomarkers",
]


def search_pubmed(query, max_results=10):
    """Search PubMed and return PMIDs."""
    try:
        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort="relevance",
            # Only get papers with abstracts
            mindate="2020/01/01",
            maxdate="2025/01/01",
        )
        record = Entrez.read(handle)
        handle.close()
        return record["IdList"]
    except Exception as e:
        print(f"  Search error for '{query}': {e}")
        return []


def fetch_paper_details(pmid):
    """Fetch title and abstract for a PMID."""
    try:
        handle = Entrez.efetch(
            db="pubmed",
            id=pmid,
            rettype="abstract",
            retmode="xml"
        )
        records = Entrez.read(handle)
        handle.close()

        article = records['PubmedArticle'][0]['MedlineCitation']['Article']

        # Get title
        title = article.get('ArticleTitle', 'No title')

        # Get abstract
        abstract_parts = article.get('Abstract', {}).get('AbstractText', [])
        if abstract_parts:
            # Handle structured abstracts (with labels like BACKGROUND, METHODS, etc.)
            if isinstance(abstract_parts[0], str):
                abstract = ' '.join(abstract_parts)
            else:
                # Structured abstract - join with labels
                abstract = ' '.join([str(part) for part in abstract_parts])
        else:
            abstract = None

        return {
            'pmid': pmid,
            'title': str(title),
            'abstract': abstract
        }

    except Exception as e:
        print(f"  Fetch error for PMID {pmid}: {e}")
        return None


def fetch_papers(n_papers=100, papers_per_topic=5):
    """Fetch papers across all topics."""

    papers = []
    seen_pmids = set()

    print(f"Fetching {n_papers} papers from PubMed...")
    print("=" * 60)

    for topic in SEARCH_TOPICS:
        if len(papers) >= n_papers:
            break

        print(f"\nSearching: {topic}")

        # Search for PMIDs
        pmids = search_pubmed(topic, max_results=papers_per_topic + 5)  # Extra to handle failures

        fetched_this_topic = 0
        for pmid in pmids:
            if len(papers) >= n_papers:
                break
            if pmid in seen_pmids:
                continue
            if fetched_this_topic >= papers_per_topic:
                break

            # Fetch details
            paper = fetch_paper_details(pmid)

            if paper and paper['abstract']:
                # Only keep papers with substantial abstracts
                if len(paper['abstract']) > 200:
                    papers.append(paper)
                    seen_pmids.add(pmid)
                    fetched_this_topic += 1
                    print(f"  [{len(papers):3d}] {paper['title'][:60]}...")

            # Be nice to NCBI servers
            time.sleep(0.4)

    # If we still need more papers, do broader searches
    if len(papers) < n_papers:
        print(f"\nNeed {n_papers - len(papers)} more papers, doing broader searches...")

        broad_queries = [
            "neuroscience 2024",
            "brain research 2024",
            "neural mechanisms",
            "cognitive neuroscience",
            "molecular neurobiology",
        ]

        for query in broad_queries:
            if len(papers) >= n_papers:
                break

            print(f"\nBroad search: {query}")
            pmids = search_pubmed(query, max_results=30)

            for pmid in pmids:
                if len(papers) >= n_papers:
                    break
                if pmid in seen_pmids:
                    continue

                paper = fetch_paper_details(pmid)

                if paper and paper['abstract'] and len(paper['abstract']) > 200:
                    papers.append(paper)
                    seen_pmids.add(pmid)
                    print(f"  [{len(papers):3d}] {paper['title'][:60]}...")

                time.sleep(0.4)

    return papers


def main():
    # Fetch papers
    papers = fetch_papers(n_papers=100)

    print("\n" + "=" * 60)
    print(f"COMPLETE: Fetched {len(papers)} papers")
    print("=" * 60)

    # Print summary stats
    abstract_lengths = [len(p['abstract']) for p in papers]
    print(f"\nAbstract lengths:")
    print(f"  Min: {min(abstract_lengths)}")
    print(f"  Max: {max(abstract_lengths)}")
    print(f"  Avg: {sum(abstract_lengths) / len(abstract_lengths):.0f}")

    # Save to JSON
    output_path = Path(__file__).parent / "papers.json"
    with open(output_path, 'w') as f:
        json.dump(papers, f, indent=2)

    print(f"\nSaved to: {output_path}")

    # Also save a smaller sample for testing
    sample_path = Path(__file__).parent / "papers_sample.json"
    with open(sample_path, 'w') as f:
        json.dump(papers[:10], f, indent=2)

    print(f"Sample (10 papers) saved to: {sample_path}")


if __name__ == "__main__":
    main()
