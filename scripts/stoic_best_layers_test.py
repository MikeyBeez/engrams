"""
Stoic Best Layers Test

Deep dive on layers 4 and 24 which showed best results in sweep.
Full 8 scenarios with detailed output.
"""

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


STOIC_MARKERS = {
    "control_dichotomy": [
        "in your control", "within your control", "not in your control",
        "beyond your control", "cannot control", "can control",
        "up to you", "not up to you",
    ],
    "acceptance": [
        "accept", "acceptance", "inevitable", "nature of things",
        "natural", "fate", "providence",
    ],
    "virtue_focus": [
        "virtue", "character", "integrity", "wisdom",
        "courage", "justice", "temperance", "moderation",
    ],
    "indifference_to_externals": [
        "indifferent", "external", "outside", "wealth", "poverty",
        "fame", "reputation", "opinion of others", "what others think",
    ],
    "present_focus": [
        "present moment", "now", "today", "this moment", "here and now",
    ],
    "mortality_awareness": [
        "death", "mortal", "finite", "temporary",
        "impermanent", "transient", "fleeting",
    ],
    "rationality": [
        "reason", "rational", "logic", "think clearly",
        "judgment", "perspective",
    ],
    "emotional_regulation": [
        "passion", "emotion", "disturb", "tranquil",
        "calm", "serene", "peace of mind", "equanimity",
    ],
}

TEST_SCENARIOS = [
    {
        "id": "job_loss",
        "prompt": "I just lost my job unexpectedly. I feel devastated and scared about the future. What should I do?",
        "stoic_themes": ["control_dichotomy", "acceptance", "present_focus"],
    },
    {
        "id": "criticism",
        "prompt": "My colleague publicly criticized my work in front of everyone. I feel humiliated and angry. How should I handle this?",
        "stoic_themes": ["indifference_to_externals", "emotional_regulation", "virtue_focus"],
    },
    {
        "id": "terminal_diagnosis",
        "prompt": "I have been diagnosed with a serious illness. The doctors say I have limited time. How do I cope with this?",
        "stoic_themes": ["mortality_awareness", "acceptance", "present_focus"],
    },
    {
        "id": "betrayal",
        "prompt": "My business partner stole money from our company and disappeared. I trusted them completely. What now?",
        "stoic_themes": ["control_dichotomy", "emotional_regulation", "rationality"],
    },
    {
        "id": "failure",
        "prompt": "I worked for years on a project that completely failed. All that effort was for nothing. How do I move forward?",
        "stoic_themes": ["virtue_focus", "acceptance", "indifference_to_externals"],
    },
    {
        "id": "envy",
        "prompt": "My friend became incredibly successful while I am still struggling. I feel envious and inadequate. Is this normal?",
        "stoic_themes": ["indifference_to_externals", "virtue_focus", "rationality"],
    },
    {
        "id": "anxiety",
        "prompt": "I cannot stop worrying about things that might go wrong in the future. The anxiety is overwhelming. Help me.",
        "stoic_themes": ["control_dichotomy", "present_focus", "rationality"],
    },
    {
        "id": "revenge",
        "prompt": "Someone wronged me badly and got away with it. I want revenge. Should I pursue it?",
        "stoic_themes": ["virtue_focus", "emotional_regulation", "rationality"],
    },
]


def load_meditations(path: str) -> str:
    text = Path(path).read_text()
    start_marker = "FIRST BOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG"
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]
    return text


def extract_engram_at_layer(model, tokenizer, text, layer, num_tokens=32, chunk_size=2048):
    tokens = tokenizer.encode(text)
    total = min(len(tokens), 8192)

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

    return torch.stack(engram).to(model.device)


def answer_with_engram(model, tokenizer, question, engram, max_new_tokens=200):
    embed = model.get_input_embeddings()
    e_norm = embed.weight.norm(dim=1).mean().item()
    g_norm = engram.norm(dim=1).mean().item()
    scaled = engram * (e_norm / g_norm)

    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    emb = embed(inputs.input_ids)
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

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def score_stoic_markers(response: str) -> dict:
    response_lower = response.lower()
    scores = {}
    total = 0
    for category, markers in STOIC_MARKERS.items():
        hits = sum(1 for marker in markers if marker.lower() in response_lower)
        scores[category] = hits
        total += hits
    scores["total"] = total
    return scores


def main():
    print("=" * 70)
    print("STOIC BEST LAYERS TEST")
    print("Detailed comparison: Layer 4 vs Layer 16 vs Layer 24")
    print("=" * 70)

    print("\nLoading Qwen2.5-7B...")
    model_name = "Qwen/Qwen2.5-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print("\nLoading Marcus Aurelius Meditations...")
    meditations_path = Path(__file__).parent.parent / "data" / "marcus_aurelius_meditations.txt"
    meditations_text = load_meditations(str(meditations_path))
    print(f"  Loaded {len(meditations_text)} characters")

    # Extract engrams at different layers
    print("\nExtracting engrams...")
    print("  Layer 4 (early)...")
    engram_4 = extract_engram_at_layer(model, tokenizer, meditations_text, 4)
    print("  Layer 16 (middle - facts)...")
    engram_16 = extract_engram_at_layer(model, tokenizer, meditations_text, 16)
    print("  Layer 24 (late)...")
    engram_24 = extract_engram_at_layer(model, tokenizer, meditations_text, 24)

    results = {
        "baseline": {"total": 0, "scenarios": []},
        "layer_4": {"total": 0, "scenarios": []},
        "layer_16": {"total": 0, "scenarios": []},
        "layer_24": {"total": 0, "scenarios": []},
    }

    for scenario in TEST_SCENARIOS:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario['id']}")
        print(f"PROMPT: {scenario['prompt']}")
        print("=" * 70)

        # Baseline
        print("\n--- BASELINE ---")
        baseline = generate_baseline(scenario["prompt"], model, tokenizer)
        baseline_score = score_stoic_markers(baseline)
        print(baseline)
        print(f"\nMarkers: {baseline_score['total']} - {baseline_score}")
        results["baseline"]["total"] += baseline_score["total"]
        results["baseline"]["scenarios"].append({
            "id": scenario["id"],
            "response": baseline,
            "score": baseline_score
        })

        # Layer 4
        print("\n--- LAYER 4 (early) ---")
        resp_4 = answer_with_engram(model, tokenizer, scenario["prompt"], engram_4)
        score_4 = score_stoic_markers(resp_4)
        print(resp_4)
        print(f"\nMarkers: {score_4['total']} - {score_4}")
        results["layer_4"]["total"] += score_4["total"]
        results["layer_4"]["scenarios"].append({
            "id": scenario["id"],
            "response": resp_4,
            "score": score_4
        })

        # Layer 16
        print("\n--- LAYER 16 (middle - facts) ---")
        resp_16 = answer_with_engram(model, tokenizer, scenario["prompt"], engram_16)
        score_16 = score_stoic_markers(resp_16)
        print(resp_16)
        print(f"\nMarkers: {score_16['total']} - {score_16}")
        results["layer_16"]["total"] += score_16["total"]
        results["layer_16"]["scenarios"].append({
            "id": scenario["id"],
            "response": resp_16,
            "score": score_16
        })

        # Layer 24
        print("\n--- LAYER 24 (late) ---")
        resp_24 = answer_with_engram(model, tokenizer, scenario["prompt"], engram_24)
        score_24 = score_stoic_markers(resp_24)
        print(resp_24)
        print(f"\nMarkers: {score_24['total']} - {score_24}")
        results["layer_24"]["total"] += score_24["total"]
        results["layer_24"]["scenarios"].append({
            "id": scenario["id"],
            "response": resp_24,
            "score": score_24
        })

        print(f"\n>>> Comparison: Baseline={baseline_score['total']}, L4={score_4['total']}, L16={score_16['total']}, L24={score_24['total']}")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\nTotal stoic markers across {len(TEST_SCENARIOS)} scenarios:")
    print(f"  Baseline:  {results['baseline']['total']}")
    print(f"  Layer 4:   {results['layer_4']['total']} ({results['layer_4']['total'] - results['baseline']['total']:+d})")
    print(f"  Layer 16:  {results['layer_16']['total']} ({results['layer_16']['total'] - results['baseline']['total']:+d})")
    print(f"  Layer 24:  {results['layer_24']['total']} ({results['layer_24']['total'] - results['baseline']['total']:+d})")

    best = max(
        [("Layer 4", results["layer_4"]["total"]),
         ("Layer 16", results["layer_16"]["total"]),
         ("Layer 24", results["layer_24"]["total"])],
        key=lambda x: x[1]
    )
    print(f"\nBest performing: {best[0]} with {best[1]} total markers")

    # Save
    results_path = Path(__file__).parent.parent / "results" / "stoic_best_layers.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
