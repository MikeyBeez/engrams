"""
Stoic Layer Sweep Test

Test whether personality engrams work better at different layers.
Layer 16 works for facts - but personality might live elsewhere.

Hypothesis: Later layers (closer to output) encode behavioral patterns,
not just semantic content.
"""

import json
import torch
import gc
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


# Stoic markers for scoring
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

# Test scenarios - subset for faster iteration
TEST_SCENARIOS = [
    "I just lost my job unexpectedly. I feel devastated and scared about the future. What should I do?",
    "My colleague publicly criticized my work in front of everyone. I feel humiliated and angry. How should I handle this?",
    "I cannot stop worrying about things that might go wrong in the future. The anxiety is overwhelming. Help me.",
    "Someone wronged me badly and got away with it. I want revenge. Should I pursue it?",
]


def load_meditations(path: str) -> str:
    """Load Marcus Aurelius Meditations text."""
    text = Path(path).read_text()
    start_marker = "FIRST BOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG"
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]
    return text


def extract_engram_at_layer(model, tokenizer, text, layer, num_tokens=32, chunk_size=2048):
    """Extract engram from specific layer."""
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


def answer_with_engram(model, tokenizer, question, engram, max_new_tokens=150):
    """Generate response with engram using proven scaling."""
    embed = model.get_input_embeddings()

    # Scale against embedding weights (proven method)
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


def generate_baseline(prompt: str, model, tokenizer, max_new_tokens: int = 150):
    """Generate response without engram."""
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


def score_stoic_markers(response: str) -> int:
    """Count stoic markers in response."""
    response_lower = response.lower()
    total = 0
    for markers in STOIC_MARKERS.values():
        for marker in markers:
            if marker.lower() in response_lower:
                total += 1
    return total


def main():
    print("=" * 70)
    print("STOIC LAYER SWEEP TEST")
    print("Finding where personality lives in the transformer")
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

    # Check number of layers
    num_layers = len(model.model.layers)
    print(f"  Model has {num_layers} layers")

    # Load Meditations
    print("\nLoading Marcus Aurelius Meditations...")
    meditations_path = Path(__file__).parent.parent / "data" / "marcus_aurelius_meditations.txt"
    meditations_text = load_meditations(str(meditations_path))
    print(f"  Loaded {len(meditations_text)} characters")

    # Get baseline scores first
    print("\n" + "=" * 70)
    print("BASELINE (no engram)")
    print("=" * 70)

    baseline_total = 0
    for prompt in TEST_SCENARIOS:
        response = generate_baseline(prompt, model, tokenizer)
        score = score_stoic_markers(response)
        baseline_total += score
        print(f"  Score: {score} | {prompt[:50]}...")

    print(f"\nBaseline total: {baseline_total} markers across {len(TEST_SCENARIOS)} scenarios")

    # Test layers: early, middle, late
    # Qwen2.5-7B has 28 layers (indices 0-27 in hidden_states, but 0 is embeddings)
    # So actual layers are 1-28
    layers_to_test = [4, 8, 12, 16, 20, 24, 27]  # Sample across the range

    results = {
        "baseline": baseline_total,
        "layers": {}
    }

    print("\n" + "=" * 70)
    print("LAYER SWEEP")
    print("=" * 70)

    for layer in layers_to_test:
        print(f"\n--- Layer {layer} ---")

        # Extract engram at this layer
        print(f"  Extracting engram...")
        engram = extract_engram_at_layer(model, tokenizer, meditations_text, layer)
        print(f"  Engram shape: {engram.shape}")

        # Test all scenarios
        layer_total = 0
        for prompt in TEST_SCENARIOS:
            response = answer_with_engram(model, tokenizer, prompt, engram)
            score = score_stoic_markers(response)
            layer_total += score
            print(f"  Score: {score} | {prompt[:40]}...")

        improvement = layer_total - baseline_total
        print(f"\n  Layer {layer} total: {layer_total} (improvement: {improvement:+d})")

        results["layers"][layer] = {
            "total": layer_total,
            "improvement": improvement
        }

        # Clean up
        del engram
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nBaseline: {baseline_total} markers")
    print("\nBy layer:")

    best_layer = None
    best_improvement = float('-inf')

    for layer in layers_to_test:
        data = results["layers"][layer]
        marker = " <-- BEST" if data["improvement"] > best_improvement else ""
        if data["improvement"] > best_improvement:
            best_improvement = data["improvement"]
            best_layer = layer
        print(f"  Layer {layer:2d}: {data['total']:2d} markers ({data['improvement']:+d}){marker}")

    print(f"\nBest layer: {best_layer} with {best_improvement:+d} improvement over baseline")

    # Save results
    results_path = Path(__file__).parent.parent / "results" / "stoic_layer_sweep.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
