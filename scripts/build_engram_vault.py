#!/usr/bin/env python3
"""
Engram Vault Builder: Automated Extraction from Documents

Builds a library of Gold Standard Engrams from text sources.
Supports plain text files, directories, and simple text input.

Usage:
    # From a directory of text files
    python scripts/build_engram_vault.py --input ./docs/sources/ --output vault.pt

    # From a single file
    python scripts/build_engram_vault.py --input knowledge.txt --output vault.pt

    # From a JSON manifest
    python scripts/build_engram_vault.py --manifest sources.json --output vault.pt

Manifest format (sources.json):
    {
        "topic:pheochromocytoma": "Alpha-blocker first for pheochromocytoma...",
        "topic:malignant_hyperthermia": "Dantrolene is the treatment...",
        ...
    }
"""

import argparse
import json
import os
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class EngramMetadata:
    topic_key: str
    source_hash: str
    model_id: str
    model_hash: str
    layer: int
    num_tokens: int
    strength: float
    created_at: str
    source_length: int


class EngramVault:
    """
    Persistent storage for Gold Standard Engrams.
    """

    def __init__(self, model_id: str, model_hash: str):
        self.model_id = model_id
        self.model_hash = model_hash
        self.engrams: Dict[str, torch.Tensor] = {}
        self.metadata: Dict[str, EngramMetadata] = {}
        self.created_at = datetime.now().isoformat()

    def add(
        self,
        topic_key: str,
        engram: torch.Tensor,
        source_text: str,
        layer: int = 20,
        num_tokens: int = 16,
        strength: float = 1.0
    ):
        """Add an engram to the vault."""
        self.engrams[topic_key] = engram.cpu()
        self.metadata[topic_key] = EngramMetadata(
            topic_key=topic_key,
            source_hash=hashlib.sha256(source_text.encode()).hexdigest()[:16],
            model_id=self.model_id,
            model_hash=self.model_hash,
            layer=layer,
            num_tokens=num_tokens,
            strength=strength,
            created_at=datetime.now().isoformat(),
            source_length=len(source_text)
        )

    def get(self, topic_key: str) -> Optional[torch.Tensor]:
        """Retrieve an engram by topic key."""
        return self.engrams.get(topic_key)

    def list_topics(self) -> List[str]:
        """List all topic keys in the vault."""
        return list(self.engrams.keys())

    def is_compatible(self, model_hash: str) -> bool:
        """Check if vault is compatible with a model."""
        return self.model_hash == model_hash

    def save(self, path: str):
        """Save vault to disk."""
        data = {
            'model_id': self.model_id,
            'model_hash': self.model_hash,
            'created_at': self.created_at,
            'engrams': {k: v.cpu() for k, v in self.engrams.items()},
            'metadata': {k: asdict(v) for k, v in self.metadata.items()}
        }
        torch.save(data, path)
        print(f"Saved vault with {len(self.engrams)} engrams to {path}")

    @classmethod
    def load(cls, path: str) -> 'EngramVault':
        """Load vault from disk."""
        data = torch.load(path)
        vault = cls(data['model_id'], data['model_hash'])
        vault.created_at = data['created_at']
        vault.engrams = data['engrams']
        vault.metadata = {
            k: EngramMetadata(**v) for k, v in data['metadata'].items()
        }
        return vault

    def summary(self) -> str:
        """Return a summary of the vault contents."""
        lines = [
            f"Engram Vault Summary",
            f"=" * 50,
            f"Model: {self.model_id}",
            f"Model Hash: {self.model_hash[:16]}...",
            f"Created: {self.created_at}",
            f"Total Engrams: {len(self.engrams)}",
            f"",
            f"Topics:"
        ]
        for topic in sorted(self.engrams.keys()):
            meta = self.metadata[topic]
            lines.append(f"  - {topic} (layer {meta.layer}, {meta.source_length} chars)")
        return "\n".join(lines)


class EngramExtractor:
    """
    Extracts engrams from text using a specified model.
    """

    def __init__(
        self,
        model,
        tokenizer,
        layer: int = 20,
        num_tokens: int = 16
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.num_tokens = num_tokens

    def extract(self, text: str) -> torch.Tensor:
        """Extract engram from text."""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[self.layer]
        seq_len = hidden.shape[1]
        chunk_size = max(1, seq_len // self.num_tokens)

        vectors = []
        for i in range(self.num_tokens):
            start = i * chunk_size
            end = start + chunk_size if i < self.num_tokens - 1 else seq_len
            if start < seq_len:
                vectors.append(hidden[0, start:end].mean(dim=0))
            else:
                vectors.append(hidden[0, -1, :])

        return torch.stack(vectors)


def get_model_hash(model) -> str:
    """Generate a hash of model parameters for version tracking."""
    # Use first layer weights as a fingerprint
    first_param = next(model.parameters())
    param_bytes = first_param.data.cpu().numpy().tobytes()[:1000]
    return hashlib.sha256(param_bytes).hexdigest()


def load_sources_from_directory(path: str) -> Dict[str, str]:
    """Load text files from a directory as topic sources."""
    sources = {}
    dir_path = Path(path)

    for file_path in dir_path.glob("**/*.txt"):
        # Use filename as topic key
        topic_key = f"topic:{file_path.stem}"
        with open(file_path, 'r', encoding='utf-8') as f:
            sources[topic_key] = f.read().strip()
        print(f"  Loaded: {topic_key} ({len(sources[topic_key])} chars)")

    return sources


def load_sources_from_manifest(path: str) -> Dict[str, str]:
    """Load sources from a JSON manifest."""
    with open(path, 'r', encoding='utf-8') as f:
        sources = json.load(f)

    for key, text in sources.items():
        print(f"  Loaded: {key} ({len(text)} chars)")

    return sources


def load_sources_from_file(path: str) -> Dict[str, str]:
    """Load a single text file as one source."""
    topic_key = f"topic:{Path(path).stem}"
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    print(f"  Loaded: {topic_key} ({len(text)} chars)")
    return {topic_key: text}


def build_vault(
    model,
    tokenizer,
    sources: Dict[str, str],
    model_id: str,
    layer: int = 20,
    num_tokens: int = 16
) -> EngramVault:
    """Build an engram vault from source texts."""

    model_hash = get_model_hash(model)
    vault = EngramVault(model_id, model_hash)
    extractor = EngramExtractor(model, tokenizer, layer, num_tokens)

    print(f"\nExtracting {len(sources)} engrams at layer {layer}...")
    print("-" * 50)

    for topic_key, source_text in sources.items():
        print(f"  Extracting: {topic_key}...", end=" ")
        engram = extractor.extract(source_text)
        vault.add(
            topic_key=topic_key,
            engram=engram,
            source_text=source_text,
            layer=layer,
            num_tokens=num_tokens
        )
        print(f"done ({engram.shape})")

    return vault


# Example Gold Standard sources for quick testing
EXAMPLE_SOURCES = {
    "topic:pheochromocytoma": """
        Pheochromocytoma treatment: Alpha-blocker FIRST (phenoxybenzamine),
        then beta-blocker. Never start beta-blocker first - causes unopposed
        alpha stimulation and hypertensive crisis.
    """,
    "topic:malignant_hyperthermia": """
        Malignant hyperthermia requires immediate dantrolene sodium IV.
        Stop triggering agents. Dantrolene blocks calcium release from
        sarcoplasmic reticulum. Do NOT rely on cooling alone.
    """,
    "topic:serotonin_syndrome": """
        Serotonin syndrome is treated with cyproheptadine, a serotonin antagonist.
        Stop the serotonergic agents. Supportive care and benzodiazepines for
        agitation. Do not give more serotonergic drugs.
    """,
    "topic:tca_overdose": """
        TCA overdose with QRS widening requires sodium bicarbonate.
        Sodium bicarb narrows QRS by pH effect on sodium channels.
        Not physostigmine - causes seizures and arrhythmias.
    """,
    "topic:wernicke_encephalopathy": """
        Wernicke encephalopathy: give thiamine BEFORE glucose.
        Glucose without thiamine precipitates Wernicke's in thiamine-depleted
        patients. Classic triad: confusion, ataxia, ophthalmoplegia.
    """,
    "topic:adverse_possession": """
        Adverse possession requires HOSTILE possession - without the owner's
        permission. Permission defeats the claim. The possession must be
        open, notorious, continuous, and hostile for the statutory period.
    """,
    "topic:orbital_mechanics": """
        Orbital mechanics: to catch a satellite ahead, slow down to drop to
        a lower orbit. Lower orbits are faster. Then speed up to raise back
        to the target orbit when you've caught up.
    """,
    "topic:base_rate_fallacy": """
        Base rate fallacy: a 99% accurate test with 1% disease prevalence
        gives approximately 50% positive predictive value. The false positives
        from the 99% healthy population outnumber true positives.
    """,
}


def main():
    parser = argparse.ArgumentParser(description='Build an Engram Vault from text sources')
    parser.add_argument('--input', type=str, help='Input directory or file path')
    parser.add_argument('--manifest', type=str, help='JSON manifest file with topic:text mappings')
    parser.add_argument('--output', type=str, default='engram_vault.pt', help='Output vault file')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B', help='Model to use')
    parser.add_argument('--layer', type=int, default=20, help='Layer to extract from')
    parser.add_argument('--tokens', type=int, default=16, help='Number of engram tokens')
    parser.add_argument('--example', action='store_true', help='Use built-in example sources')
    args = parser.parse_args()

    # Load sources
    print("Loading sources...")
    if args.example:
        sources = EXAMPLE_SOURCES
        for key, text in sources.items():
            print(f"  Example: {key} ({len(text.strip())} chars)")
    elif args.manifest:
        sources = load_sources_from_manifest(args.manifest)
    elif args.input:
        path = Path(args.input)
        if path.is_dir():
            sources = load_sources_from_directory(args.input)
        elif path.suffix == '.json':
            sources = load_sources_from_manifest(args.input)
        else:
            sources = load_sources_from_file(args.input)
    else:
        print("Error: Must specify --input, --manifest, or --example")
        return

    if not sources:
        print("Error: No sources found")
        return

    # Load model
    print(f"\nLoading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build vault
    vault = build_vault(
        model=model,
        tokenizer=tokenizer,
        sources=sources,
        model_id=args.model,
        layer=args.layer,
        num_tokens=args.tokens
    )

    # Save
    vault.save(args.output)

    # Print summary
    print("\n" + vault.summary())


if __name__ == '__main__':
    main()
