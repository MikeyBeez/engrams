#!/usr/bin/env python3
"""
Strength × Layer Phase Diagram

Creates a comprehensive heatmap showing the probability ratio
across all layer/strength combinations. This visualizes:
1. The "resonance" regions where flips occur
2. The non-monotonic behavior (more strength isn't always better)
3. The decision commitment zone boundary

Output: ASCII heatmap + data for plotting
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import os
import json
from huggingface_hub import login

# Auth setup
token = os.environ.get("HF_TOKEN")
if not token:
    try:
        token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
        login(token=token, add_to_git_credential=False)
    except:
        pass

# Test case
TEST_CASE = {
    "id": "pheo1",
    "prompt": "A patient with pheochromocytoma requires preoperative blood pressure management. The first medication should be",
    "correct_tokens": [" alpha", " Alpha", " phenoxybenzamine"],
    "incorrect_tokens": [" beta", " Beta", " metoprolol"],
    "knowledge": """
CRITICAL RULE FOR PHEOCHROMOCYTOMA:
The medication order is: ALPHA-BLOCKER FIRST, then beta-blocker.
Alpha-blocker (phenoxybenzamine) MUST be started BEFORE any beta-blocker.
Starting beta-blocker first causes unopposed alpha stimulation.
Unopposed alpha causes severe hypertensive crisis and death.
NEVER give beta-blockers first in pheochromocytoma.
The answer is ALWAYS alpha-blocker first.
Alpha before beta. Alpha before beta. Alpha before beta.
"""
}

# Search grid - finer resolution
LAYERS = list(range(14, 29))  # 14-28
STRENGTHS = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0, 20.0, 25.0]


def extract_engram(model, tokenizer, text, layer_idx, num_tokens=16):
    """Extract an engram from a specific layer."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[layer_idx]
    seq_len = hidden.shape[1]
    chunk_size = max(1, seq_len // num_tokens)

    vectors = []
    for i in range(num_tokens):
        start = i * chunk_size
        end = start + chunk_size if i < num_tokens - 1 else seq_len
        if start >= seq_len:
            vectors.append(hidden[0, -1, :])
        else:
            vectors.append(hidden[0, start:end].mean(dim=0))

    return torch.stack(vectors)


def get_probs_with_engram(model, tokenizer, prompt, correct_tokens, incorrect_tokens, engram, strength):
    """Get probabilities with engram injection."""
    embed = model.get_input_embeddings()
    e_norm = embed.weight.norm(dim=1).mean().item()
    g_norm = engram.norm(dim=1).mean().item()
    scaled = engram * (e_norm / g_norm) * strength

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    emb = embed(inputs.input_ids)
    combined = torch.cat([scaled.unsqueeze(0).to(emb.dtype), emb], dim=1)

    with torch.no_grad():
        outputs = model(inputs_embeds=combined)
        probs = F.softmax(outputs.logits[0, -1, :], dim=-1)

    correct_prob = 0
    for t in correct_tokens:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if ids:
            correct_prob = max(correct_prob, probs[ids[0]].item())

    incorrect_prob = 0
    for t in incorrect_tokens:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if ids:
            incorrect_prob = max(incorrect_prob, probs[ids[0]].item())

    return correct_prob, incorrect_prob


def ratio_to_char(ratio, is_flip):
    """Convert ratio to heatmap character."""
    if ratio >= 2.0:
        return "█"  # Strong flip
    elif ratio >= 1.5:
        return "▓"  # Good flip
    elif ratio >= 1.0:
        return "▒"  # Weak flip
    elif ratio >= 0.8:
        return "░"  # Close
    elif ratio >= 0.5:
        return "·"  # Moderate
    else:
        return " "  # Weak


print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    torch_dtype=torch.float16,
    device_map="auto"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\n" + "=" * 80)
print("STRENGTH × LAYER PHASE DIAGRAM")
print("=" * 80)
print(f"\nGrid: {len(LAYERS)} layers × {len(STRENGTHS)} strengths = {len(LAYERS)*len(STRENGTHS)} configurations")
print("Building heatmap...\n")

# Store results
heatmap = {}
all_results = []

# Pre-extract all engrams (one per layer)
print("Extracting engrams for each layer...")
engrams = {}
for layer in LAYERS:
    engrams[layer] = extract_engram(model, tokenizer, TEST_CASE['knowledge'], layer)
print("Done.\n")

# Run all combinations
print(f"{'Layer':<6}", end="")
for s in STRENGTHS:
    print(f"{s:<6.1f}", end="")
print("  (strength)")
print("-" * (6 + 6 * len(STRENGTHS)))

for layer in LAYERS:
    print(f"{layer:<6}", end="")
    heatmap[layer] = {}

    for strength in STRENGTHS:
        corr, incorr = get_probs_with_engram(
            model, tokenizer, TEST_CASE['prompt'],
            TEST_CASE['correct_tokens'], TEST_CASE['incorrect_tokens'],
            engrams[layer], strength
        )

        ratio = corr / incorr if incorr > 0 else float('inf')
        is_flip = ratio > 1.0

        heatmap[layer][strength] = {
            'ratio': ratio,
            'correct': corr,
            'incorrect': incorr,
            'flipped': is_flip
        }

        all_results.append({
            'layer': layer,
            'strength': strength,
            'ratio': ratio,
            'flipped': is_flip
        })

        char = ratio_to_char(ratio, is_flip)
        # Color-code: green for flip, red for not
        if is_flip:
            print(f"{char:<6}", end="")
        else:
            print(f"{char:<6}", end="")

    print()

print("-" * (6 + 6 * len(STRENGTHS)))
print("\nLegend: █ ratio≥2.0 | ▓ 1.5-2.0 | ▒ 1.0-1.5 (FLIP) | ░ 0.8-1.0 | · 0.5-0.8 | (space) <0.5")

# Analysis
print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

# Find all flip configurations
flips = [r for r in all_results if r['flipped']]
print(f"\nTotal flip configurations: {len(flips)}/{len(all_results)}")

if flips:
    # Best configuration
    best = max(flips, key=lambda x: x['ratio'])
    print(f"Best flip: Layer {best['layer']}, Strength {best['strength']}, Ratio {best['ratio']:.4f}")

    # Layer distribution of flips
    flip_by_layer = {}
    for f in flips:
        flip_by_layer[f['layer']] = flip_by_layer.get(f['layer'], 0) + 1

    print(f"\nFlips by layer:")
    for layer in sorted(flip_by_layer.keys()):
        count = flip_by_layer[layer]
        bar = "█" * count
        print(f"  Layer {layer}: {count} flips {bar}")

    # Strength distribution of flips
    flip_by_strength = {}
    for f in flips:
        flip_by_strength[f['strength']] = flip_by_strength.get(f['strength'], 0) + 1

    print(f"\nFlips by strength:")
    for strength in sorted(flip_by_strength.keys()):
        count = flip_by_strength[strength]
        bar = "█" * count
        print(f"  Strength {strength}: {count} flips {bar}")

    # Find resonance regions (contiguous flip areas)
    print("\n--- Resonance Regions ---")
    print("(Contiguous layer-strength regions where flips occur)")

    # Check for "sweet spots"
    for layer in LAYERS:
        layer_flips = [r for r in flips if r['layer'] == layer]
        if layer_flips:
            strengths = sorted([r['strength'] for r in layer_flips])
            ratios = [heatmap[layer][s]['ratio'] for s in strengths]

            # Check for non-monotonic behavior
            if len(ratios) >= 3:
                # Find if there's a peak followed by decline
                peak_idx = ratios.index(max(ratios))
                if peak_idx > 0 and peak_idx < len(ratios) - 1:
                    print(f"  Layer {layer}: Peak at strength {strengths[peak_idx]}, ratio {max(ratios):.3f}")
                    print(f"    Non-monotonic: strength {strengths[peak_idx-1]} → {strengths[peak_idx]} → {strengths[peak_idx+1]}")
                    print(f"    Ratios: {ratios[peak_idx-1]:.3f} → {ratios[peak_idx]:.3f} → {ratios[peak_idx+1]:.3f}")

# Save data for external plotting
output_file = "/Users/bard/Code/engrams/data/heatmap_data.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w') as f:
    json.dump({
        'layers': LAYERS,
        'strengths': STRENGTHS,
        'results': all_results,
        'question': TEST_CASE['id']
    }, f, indent=2)

print(f"\n\nData saved to {output_file} for external visualization")

# ASCII ratio heatmap (numerical)
print("\n" + "=" * 80)
print("NUMERICAL RATIO HEATMAP")
print("=" * 80)
print(f"\n{'Layer':<6}", end="")
for s in STRENGTHS:
    print(f"{s:>6.1f}", end="")
print()
print("-" * (6 + 6 * len(STRENGTHS)))

for layer in LAYERS:
    print(f"{layer:<6}", end="")
    for strength in STRENGTHS:
        ratio = heatmap[layer][strength]['ratio']
        if ratio > 1.0:
            print(f"{ratio:>6.2f}", end="")
        else:
            print(f"{ratio:>6.2f}", end="")
    print()
