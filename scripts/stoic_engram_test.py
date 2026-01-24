"""
Stoic Engram Test v3

Test whether an engram from Marcus Aurelius's Meditations creates stoic-like responses.

Uses the PROVEN scaling method from wiki_50q_test.py
"""

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================================
# STOIC RESPONSE MARKERS (from validated scales)
# ============================================================================

STOIC_MARKERS = {
    "control_dichotomy": [
        "in your control",
        "within your control",
        "not in your control",
        "beyond your control",
        "cannot control",
        "can control",
        "up to you",
        "not up to you",
    ],
    "acceptance": [
        "accept",
        "acceptance",
        "inevitable",
        "nature of things",
        "natural",
        "fate",
        "providence",
    ],
    "virtue_focus": [
        "virtue",
        "character",
        "integrity",
        "wisdom",
        "courage",
        "justice",
        "temperance",
        "moderation",
    ],
    "indifference_to_externals": [
        "indifferent",
        "external",
        "outside",
        "wealth",
        "poverty",
        "fame",
        "reputation",
        "opinion of others",
        "what others think",
    ],
    "present_focus": [
        "present moment",
        "now",
        "today",
        "this moment",
        "here and now",
    ],
    "mortality_awareness": [
        "death",
        "mortal",
        "finite",
        "temporary",
        "impermanent",
        "transient",
        "fleeting",
    ],
    "rationality": [
        "reason",
        "rational",
        "logic",
        "think clearly",
        "judgment",
        "perspective",
    ],
    "emotional_regulation": [
        "passion",
        "emotion",
        "disturb",
        "tranquil",
        "calm",
        "serene",
        "peace of mind",
        "equanimity",
    ],
}


# ============================================================================
# TEST SCENARIOS (derived from stoicism scales)
# ============================================================================

TEST_SCENARIOS = [
    {
        "id": "job_loss",
        "prompt": "I just lost my job unexpectedly. I feel devastated and scared about the future. What should I do?",
        "stoic_themes": ["control_dichotomy", "acceptance", "present_focus"],
        "description": "Tests response to loss of external goods",
    },
    {
        "id": "criticism",
        "prompt": "My colleague publicly criticized my work in front of everyone. I feel humiliated and angry. How should I handle this?",
        "stoic_themes": ["indifference_to_externals", "emotional_regulation", "virtue_focus"],
        "description": "Tests response to social harm and reputation damage",
    },
    {
        "id": "terminal_diagnosis",
        "prompt": "I have been diagnosed with a serious illness. The doctors say I have limited time. How do I cope with this?",
        "stoic_themes": ["mortality_awareness", "acceptance", "present_focus"],
        "description": "Tests response to mortality and fate",
    },
    {
        "id": "betrayal",
        "prompt": "My business partner stole money from our company and disappeared. I trusted them completely. What now?",
        "stoic_themes": ["control_dichotomy", "emotional_regulation", "rationality"],
        "description": "Tests response to betrayal and injustice",
    },
    {
        "id": "failure",
        "prompt": "I worked for years on a project that completely failed. All that effort was for nothing. How do I move forward?",
        "stoic_themes": ["virtue_focus", "acceptance", "indifference_to_externals"],
        "description": "Tests response to wasted effort and failure",
    },
    {
        "id": "envy",
        "prompt": "My friend became incredibly successful while I am still struggling. I feel envious and inadequate. Is this normal?",
        "stoic_themes": ["indifference_to_externals", "virtue_focus", "rationality"],
        "description": "Tests response to comparison and envy",
    },
    {
        "id": "anxiety",
        "prompt": "I cannot stop worrying about things that might go wrong in the future. The anxiety is overwhelming. Help me.",
        "stoic_themes": ["control_dichotomy", "present_focus", "rationality"],
        "description": "Tests response to anxiety about future",
    },
    {
        "id": "revenge",
        "prompt": "Someone wronged me badly and got away with it. I want revenge. Should I pursue it?",
        "stoic_themes": ["virtue_focus", "emotional_regulation", "rationality"],
        "description": "Tests response to desire for revenge",
    },
]


def load_meditations(path: str) -> str:
    """Load Marcus Aurelius Meditations text."""
    text = Path(path).read_text()

    # Find start of actual content (after Project Gutenberg header)
    start_marker = "FIRST BOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG"

    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)

    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]

    return text


def extract_engram_chunked(model, tokenizer, text, layer=16, num_tokens=32, chunk_size=2048):
    """Extract engram from text in chunks to avoid OOM."""
    tokens = tokenizer.encode(text)
    total = min(len(tokens), 8192)

    print(f"    Total tokens: {total}, processing in chunks of {chunk_size}")

    chunks = []
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        ids = torch.tensor([tokens[start:end]]).to(model.device)

        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        chunks.append(out.hidden_states[layer].cpu()[0])
        del out
        torch.cuda.empty_cache()

    hidden = torch.cat(chunks, dim=0)
    seq_len = hidden.shape[0]
    cs = seq_len // num_tokens

    engram = []
    for i in range(num_tokens):
        start = i * cs
        end = start + cs if i < num_tokens - 1 else seq_len
        engram.append(hidden[start:end].mean(dim=0))

    # Return on device like wiki_50q_test does
    return torch.stack(engram).to(model.device)


def answer_with_engram(model, tokenizer, question, engram, max_new_tokens=200):
    """Generate response with engram - using PROVEN scaling from wiki_50q_test."""
    embed = model.get_input_embeddings()

    # Scale - THIS IS THE KEY: scale against embedding weights, not prompt
    e_norm = embed.weight.norm(dim=1).mean().item()
    g_norm = engram.norm(dim=1).mean().item()
    scaled = engram * (e_norm / g_norm)

    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    emb = embed(inputs.input_ids)

    # Combine engram + prompt embeddings
    combined = torch.cat([scaled.unsqueeze(0).to(emb.dtype), emb], dim=1)

    with torch.no_grad():
        out = model.generate(
            inputs_embeds=combined,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


def generate_baseline(prompt: str, model, tokenizer, max_new_tokens: int = 200):
    """Generate response without engram (baseline)."""
    full_prompt = f"Question: {prompt}\nAnswer:"
    inputs = tokenizer(full_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def score_stoic_markers(response: str) -> dict:
    """Score response for stoic markers."""
    response_lower = response.lower()
    scores = {}
    total_hits = 0

    for category, markers in STOIC_MARKERS.items():
        hits = sum(1 for marker in markers if marker.lower() in response_lower)
        scores[category] = hits
        total_hits += hits

    scores["total"] = total_hits
    return scores


def analyze_response(response: str, scenario: dict) -> dict:
    """Analyze a response for stoic content."""
    scores = score_stoic_markers(response)

    # Check for expected themes
    expected_themes = scenario["stoic_themes"]
    theme_hits = sum(scores[theme] for theme in expected_themes if theme in scores)

    return {
        "marker_scores": scores,
        "expected_theme_hits": theme_hits,
        "total_markers": scores["total"],
        "response_length": len(response.split()),
    }


def main():
    print("=" * 70)
    print("STOIC ENGRAM TEST v3")
    print("Marcus Aurelius Meditations -> Stoic Personality")
    print("Using PROVEN scaling from wiki_50q_test.py")
    print("=" * 70)

    # Load model
    print("\nLoading Qwen2.5-7B...")
    model_name = "Qwen/Qwen2.5-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Load Meditations
    print("\nLoading Marcus Aurelius Meditations...")
    meditations_path = Path(__file__).parent.parent / "data" / "marcus_aurelius_meditations.txt"
    meditations_text = load_meditations(str(meditations_path))
    print(f"  Loaded {len(meditations_text)} characters")
    print(f"  Preview: {meditations_text[:200]}...")

    # Extract stoic engram using chunked approach
    print("\nExtracting stoic engram from Meditations (chunked)...")
    stoic_engram = extract_engram_chunked(model, tokenizer, meditations_text)
    print(f"  Engram shape: {stoic_engram.shape}")

    # Results storage
    results = {
        "scenarios": [],
        "summary": {
            "baseline_total_markers": 0,
            "engram_total_markers": 0,
            "scenarios_tested": 0,
        }
    }

    # Test each scenario
    print("\n" + "=" * 70)
    print("TESTING STOIC CONDITIONING")
    print("=" * 70)

    for scenario in TEST_SCENARIOS:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario['id']}")
        print(f"Description: {scenario['description']}")
        print(f"Expected themes: {scenario['stoic_themes']}")
        print(f"\nPROMPT: {scenario['prompt']}")
        print("=" * 70)

        # Baseline response
        print("\n--- BASELINE (no engram) ---")
        baseline_response = generate_baseline(scenario["prompt"], model, tokenizer)
        print(baseline_response)
        baseline_analysis = analyze_response(baseline_response, scenario)
        print(f"\nMarkers found: {baseline_analysis['marker_scores']}")

        # Stoic engram response
        print("\n--- WITH STOIC ENGRAM (Marcus Aurelius) ---")
        stoic_response = answer_with_engram(model, tokenizer, scenario["prompt"], stoic_engram)
        print(stoic_response)
        stoic_analysis = analyze_response(stoic_response, scenario)
        print(f"\nMarkers found: {stoic_analysis['marker_scores']}")

        # Compare
        improvement = stoic_analysis["total_markers"] - baseline_analysis["total_markers"]
        print(f"\n>>> Marker improvement: {improvement:+d} ({baseline_analysis['total_markers']} -> {stoic_analysis['total_markers']})")

        # Store results
        results["scenarios"].append({
            "id": scenario["id"],
            "prompt": scenario["prompt"],
            "baseline_response": baseline_response,
            "stoic_response": stoic_response,
            "baseline_analysis": baseline_analysis,
            "stoic_analysis": stoic_analysis,
            "improvement": improvement,
        })

        results["summary"]["baseline_total_markers"] += baseline_analysis["total_markers"]
        results["summary"]["engram_total_markers"] += stoic_analysis["total_markers"]
        results["summary"]["scenarios_tested"] += 1

        print("\n" + "-" * 70)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    total_baseline = results["summary"]["baseline_total_markers"]
    total_engram = results["summary"]["engram_total_markers"]
    num_scenarios = results["summary"]["scenarios_tested"]

    print(f"\nScenarios tested: {num_scenarios}")
    print(f"Total stoic markers (baseline): {total_baseline}")
    print(f"Total stoic markers (engram):   {total_engram}")
    print(f"Overall improvement: {total_engram - total_baseline:+d} markers")
    print(f"Average per scenario: {(total_engram - total_baseline) / num_scenarios:+.1f} markers")

    if total_baseline > 0:
        pct_improvement = ((total_engram - total_baseline) / total_baseline) * 100
        print(f"Percentage improvement: {pct_improvement:+.1f}%")

    # Save results
    results_path = Path(__file__).parent.parent / "results" / "stoic_engram_results.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
