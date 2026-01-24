#!/usr/bin/env python3
"""
Layer Boundary Ablation

Fine-grained scan of EVERY layer from 14-28 to find the exact
"decision commitment" boundary. This proves the phase transition
is sharp, not gradual.

Tests at fixed strength (the known sweet spot for each question)
to isolate the layer effect.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import os
from huggingface_hub import login

# Auth setup
token = os.environ.get("HF_TOKEN")
if not token:
    try:
        token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
        login(token=token, add_to_git_credential=False)
    except:
        pass

# Test case - using pheo1 as primary, with its known optimal strength
TEST_CASE = {
    "id": "pheo1",
    "prompt": "A patient with pheochromocytoma requires preoperative blood pressure management. The first medication should be",
    "correct_tokens": [" alpha", " Alpha", " phenoxybenzamine"],
    "incorrect_tokens": [" beta", " Beta", " metoprolol"],
    "strength": 10.0,  # Known optimal strength
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


def get_baseline_probs(model, tokenizer, prompt, correct_tokens, incorrect_tokens):
    """Get baseline probabilities without engram."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
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


print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    torch_dtype=torch.float16,
    device_map="auto"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

num_layers = model.config.num_hidden_layers
print(f"Model has {num_layers} layers")

print("\n" + "=" * 80)
print("LAYER BOUNDARY ABLATION")
print("=" * 80)
print(f"\nTesting every layer from 1 to {num_layers} at fixed strength {TEST_CASE['strength']}")
print("Goal: Find the exact boundary of the 'decision commitment zone'\n")

# Baseline
corr_base, incorr_base = get_baseline_probs(
    model, tokenizer, TEST_CASE['prompt'],
    TEST_CASE['correct_tokens'], TEST_CASE['incorrect_tokens']
)
base_ratio = corr_base / incorr_base if incorr_base > 0 else float('inf')
print(f"BASELINE: correct={corr_base:.6f}, incorrect={incorr_base:.6f}, ratio={base_ratio:.4f}")
print(f"Status: {'CORRECT' if base_ratio > 1 else 'WRONG'}\n")

# Test every layer
print(f"{'Layer':<8} {'Correct':<12} {'Incorrect':<12} {'Ratio':<12} {'vs Base':<10} {'Flip?':<8} {'Visual'}")
print("-" * 85)

results = []
strength = TEST_CASE['strength']

for layer in range(1, num_layers + 1):
    engram = extract_engram(model, tokenizer, TEST_CASE['knowledge'], layer)
    corr, incorr = get_probs_with_engram(
        model, tokenizer, TEST_CASE['prompt'],
        TEST_CASE['correct_tokens'], TEST_CASE['incorrect_tokens'],
        engram, strength
    )

    ratio = corr / incorr if incorr > 0 else float('inf')
    improvement = ratio / base_ratio if base_ratio > 0 else 0
    is_flip = ratio > 1.0
    flip_str = "YES" if is_flip else "no"

    # Visual bar showing ratio (log scale would be better but keeping simple)
    bar_len = min(40, int(ratio * 20)) if ratio < 2 else 40
    bar = "█" * bar_len

    results.append({
        'layer': layer,
        'ratio': ratio,
        'improvement': improvement,
        'flipped': is_flip
    })

    print(f"{layer:<8} {corr:<12.6f} {incorr:<12.6f} {ratio:<12.4f} {improvement:<10.1f}x {flip_str:<8} {bar}")

# Analysis
print("\n" + "=" * 80)
print("PHASE TRANSITION ANALYSIS")
print("=" * 80)

# Find first flip
first_flip = None
for r in results:
    if r['flipped']:
        first_flip = r['layer']
        break

# Find consistent flip region
flip_layers = [r['layer'] for r in results if r['flipped']]
if flip_layers:
    flip_start = min(flip_layers)
    flip_end = max(flip_layers)
else:
    flip_start = flip_end = None

# Find best layer
best = max(results, key=lambda x: x['ratio'])

print(f"\nFirst flip at layer: {first_flip if first_flip else 'None'}")
print(f"Flip region: layers {flip_start}-{flip_end}" if flip_start else "No flip region found")
print(f"Best layer: {best['layer']} (ratio={best['ratio']:.4f}, {best['improvement']:.1f}x improvement)")

# Identify phases
print("\n--- Layer Phase Interpretation ---")
early_avg = sum(r['ratio'] for r in results[:10]) / 10
mid_avg = sum(r['ratio'] for r in results[10:18]) / 8
late_avg = sum(r['ratio'] for r in results[18:]) / (num_layers - 18)

print(f"Early layers (1-10) avg ratio: {early_avg:.4f}")
print(f"Middle layers (11-18) avg ratio: {mid_avg:.4f}")
print(f"Late layers (19-{num_layers}) avg ratio: {late_avg:.4f}")

# Check for sharp transition
if flip_layers:
    # Look at the layers just before the first flip
    pre_flip = [r for r in results if r['layer'] < first_flip]
    if pre_flip:
        pre_flip_ratios = [r['ratio'] for r in pre_flip[-3:]]  # Last 3 before flip
        post_flip_ratios = [r['ratio'] for r in results if r['layer'] >= first_flip][:3]  # First 3 after

        pre_avg = sum(pre_flip_ratios) / len(pre_flip_ratios)
        post_avg = sum(post_flip_ratios) / len(post_flip_ratios)

        print(f"\nTransition sharpness:")
        print(f"  3 layers before flip: avg ratio = {pre_avg:.4f}")
        print(f"  3 layers after flip: avg ratio = {post_avg:.4f}")
        print(f"  Jump factor: {post_avg/pre_avg:.2f}x")

        if post_avg / pre_avg > 2:
            print("\n>>> SHARP PHASE TRANSITION DETECTED")
            print("    This is NOT a gradual increase - it's a discrete boundary!")
        else:
            print("\n>>> Gradual transition (not a sharp boundary)")

# Summary visualization
print("\n" + "=" * 80)
print("RATIO BY LAYER (ASCII PLOT)")
print("=" * 80)
print("\nFlip threshold (ratio=1.0) marked with |")
print()

max_ratio = max(r['ratio'] for r in results)
scale = 60 / max_ratio if max_ratio > 0 else 1

for r in results:
    bar_len = int(r['ratio'] * scale)
    threshold_pos = int(1.0 * scale)

    # Build the bar with threshold marker
    bar = ""
    for i in range(max(bar_len, threshold_pos + 1)):
        if i == threshold_pos:
            bar += "|"
        elif i < bar_len:
            bar += "█" if r['flipped'] else "░"
        else:
            bar += " "

    status = "✓" if r['flipped'] else " "
    print(f"L{r['layer']:2d} {status} {bar} {r['ratio']:.3f}")

print("\nLegend: █ = flipped, ░ = not flipped, | = threshold (ratio=1.0)")
