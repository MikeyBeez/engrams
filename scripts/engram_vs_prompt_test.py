"""
Engram vs Prompt Test

Compare three approaches:
1. Baseline (no instruction)
2. Stoic engram (layer 16)
3. Prompt instruction ("Answer like a stoic philosopher")

Key question: Does the engram provide value over simple prompting?
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
    "I just lost my job unexpectedly. I feel devastated and scared about the future. What should I do?",
    "My colleague publicly criticized my work in front of everyone. I feel humiliated and angry. How should I handle this?",
    "I have been diagnosed with a serious illness. The doctors say I have limited time. How do I cope with this?",
    "My business partner stole money from our company and disappeared. I trusted them completely. What now?",
    "I worked for years on a project that completely failed. All that effort was for nothing. How do I move forward?",
    "My friend became incredibly successful while I am still struggling. I feel envious and inadequate. Is this normal?",
    "I cannot stop worrying about things that might go wrong in the future. The anxiety is overwhelming. Help me.",
    "Someone wronged me badly and got away with it. I want revenge. Should I pursue it?",
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


def extract_engram(model, tokenizer, text, layer=16, num_tokens=32, chunk_size=2048):
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


def generate_baseline(prompt: str, model, tokenizer, max_new_tokens=200):
    """Baseline: no special instruction."""
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
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_with_engram(prompt: str, engram, model, tokenizer, max_new_tokens=200):
    """With stoic engram prepended."""
    embed = model.get_input_embeddings()

    # Scale against embedding weights
    e_norm = embed.weight.norm(dim=1).mean().item()
    g_norm = engram.norm(dim=1).mean().item()
    scaled = engram * (e_norm / g_norm)

    full_prompt = f"Question: {prompt}\nAnswer:"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
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


def generate_with_prompt(prompt: str, model, tokenizer, max_new_tokens=200):
    """With explicit stoic instruction in prompt."""
    full_prompt = f"Answer the following question like a stoic philosopher would, drawing on principles from Marcus Aurelius and Epictetus.\n\nQuestion: {prompt}\nAnswer:"
    inputs = tokenizer(full_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
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
    print("ENGRAM vs PROMPT TEST")
    print("Comparing: Baseline | Stoic Engram | Stoic Prompt")
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

    print("\nExtracting stoic engram (layer 16)...")
    engram = extract_engram(model, tokenizer, meditations_text, layer=16)
    print(f"  Engram shape: {engram.shape}")

    results = {
        "baseline": {"total": 0, "scenarios": []},
        "engram": {"total": 0, "scenarios": []},
        "prompt": {"total": 0, "scenarios": []},
    }

    print("\n" + "=" * 70)
    print("RUNNING TESTS")
    print("=" * 70)

    for i, scenario in enumerate(TEST_SCENARIOS):
        print(f"\n{'='*70}")
        print(f"SCENARIO {i+1}: {scenario[:60]}...")
        print("=" * 70)

        # Baseline
        print("\n--- BASELINE ---")
        baseline_resp = generate_baseline(scenario, model, tokenizer)
        baseline_score = score_stoic_markers(baseline_resp)
        print(baseline_resp[:300] + "..." if len(baseline_resp) > 300 else baseline_resp)
        print(f"\nMarkers: {baseline_score['total']}")
        results["baseline"]["total"] += baseline_score["total"]
        results["baseline"]["scenarios"].append({
            "prompt": scenario,
            "response": baseline_resp,
            "score": baseline_score
        })

        # Engram
        print("\n--- WITH ENGRAM ---")
        engram_resp = generate_with_engram(scenario, engram, model, tokenizer)
        engram_score = score_stoic_markers(engram_resp)
        print(engram_resp[:300] + "..." if len(engram_resp) > 300 else engram_resp)
        print(f"\nMarkers: {engram_score['total']}")
        results["engram"]["total"] += engram_score["total"]
        results["engram"]["scenarios"].append({
            "prompt": scenario,
            "response": engram_resp,
            "score": engram_score
        })

        # Prompt
        print("\n--- WITH STOIC PROMPT ---")
        prompt_resp = generate_with_prompt(scenario, model, tokenizer)
        prompt_score = score_stoic_markers(prompt_resp)
        print(prompt_resp[:300] + "..." if len(prompt_resp) > 300 else prompt_resp)
        print(f"\nMarkers: {prompt_score['total']}")
        results["prompt"]["total"] += prompt_score["total"]
        results["prompt"]["scenarios"].append({
            "prompt": scenario,
            "response": prompt_resp,
            "score": prompt_score
        })

        print(f"\n>>> Comparison: Baseline={baseline_score['total']}, Engram={engram_score['total']}, Prompt={prompt_score['total']}")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    n = len(TEST_SCENARIOS)
    print(f"\nTotal stoic markers across {n} scenarios:")
    print(f"  Baseline:     {results['baseline']['total']:3d} ({results['baseline']['total']/n:.1f} avg)")
    print(f"  Engram:       {results['engram']['total']:3d} ({results['engram']['total']/n:.1f} avg) [{results['engram']['total'] - results['baseline']['total']:+d} vs baseline]")
    print(f"  Stoic Prompt: {results['prompt']['total']:3d} ({results['prompt']['total']/n:.1f} avg) [{results['prompt']['total'] - results['baseline']['total']:+d} vs baseline]")

    if results['engram']['total'] > results['prompt']['total']:
        print("\n>>> ENGRAM wins over prompting!")
    elif results['prompt']['total'] > results['engram']['total']:
        print("\n>>> PROMPTING wins over engram!")
    else:
        print("\n>>> TIE between engram and prompting")

    # Token efficiency note
    prompt_tokens = len(tokenizer.encode("Answer the following question like a stoic philosopher would, drawing on principles from Marcus Aurelius and Epictetus."))
    engram_tokens = engram.shape[0]
    print(f"\nToken usage:")
    print(f"  Stoic prompt: {prompt_tokens} tokens")
    print(f"  Engram: {engram_tokens} pseudo-tokens (from {len(tokenizer.encode(meditations_text))} source tokens)")

    # Save results
    results_path = Path(__file__).parent.parent / "results" / "engram_vs_prompt.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
