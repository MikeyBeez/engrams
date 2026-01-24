#!/usr/bin/env python3
"""
Focused Flip Test - Testing with MORE SPECIFIC engram content
and wider layer/strength search
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import os
from huggingface_hub import login

token = os.environ.get("HF_TOKEN")
if not token:
    try:
        token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
        login(token=token, add_to_git_credential=False)
    except:
        pass

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", torch_dtype=torch.float16, device_map="auto")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# VERY FOCUSED engram - just about this one clinical pearl
FOCUSED_KNOWLEDGE = """
CRITICAL RULE FOR PHEOCHROMOCYTOMA:
The medication order is: ALPHA-BLOCKER FIRST, then beta-blocker.
Alpha-blocker (phenoxybenzamine) MUST be started BEFORE any beta-blocker.
Starting beta-blocker first causes unopposed alpha stimulation.
Unopposed alpha causes severe hypertensive crisis and death.
NEVER give beta-blockers first in pheochromocytoma.
The answer is ALWAYS alpha-blocker first.
Alpha before beta. Alpha before beta. Alpha before beta.
"""

def extract_engram(text, layer_idx, num_tokens=16):
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

def get_baseline_probs(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits[0, -1, :], dim=-1)
    alpha_id = tokenizer.encode(" alpha", add_special_tokens=False)[0]
    beta_id = tokenizer.encode(" beta", add_special_tokens=False)[0]
    return probs[alpha_id].item(), probs[beta_id].item()

def get_probs_with_engram(prompt, engram, strength):
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
    alpha_id = tokenizer.encode(" alpha", add_special_tokens=False)[0]
    beta_id = tokenizer.encode(" beta", add_special_tokens=False)[0]
    return probs[alpha_id].item(), probs[beta_id].item()

prompt = "A patient with pheochromocytoma requires preoperative blood pressure management. The first medication should be"

print("\n" + "=" * 70)
print("FOCUSED ENGRAM FLIP TEST - Pheochromocytoma Question")
print("=" * 70)

# Baseline
alpha_base, beta_base = get_baseline_probs(prompt)
base_ratio = alpha_base / beta_base if beta_base > 0 else float('inf')
print(f"\nBASELINE: alpha={alpha_base:.6f}, beta={beta_base:.6f}, ratio={base_ratio:.4f}")
print(f"Status: {'CORRECT' if base_ratio > 1 else 'WRONG - needs to flip'}")

# Test wider range with focused engram
print(f"\n{'Layer':<8} {'Strength':<10} {'Alpha':<12} {'Beta':<12} {'Ratio':<12} {'vs Base':<10} {'Flip?'}")
print("-" * 75)

best_ratio = base_ratio
best_config = None
flipped = False

for layer in [16, 18, 20, 22, 24, 26]:
    engram = extract_engram(FOCUSED_KNOWLEDGE, layer)
    for strength in [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0]:
        alpha, beta = get_probs_with_engram(prompt, engram, strength)
        ratio = alpha / beta if beta > 0 else float('inf')
        improvement = ratio / base_ratio if base_ratio > 0 else 0
        flip = "YES! 🎯" if ratio > 1.0 else "no"

        if ratio > best_ratio:
            best_ratio = ratio
            best_config = (layer, strength)
        if ratio > 1.0:
            flipped = True

        print(f"{layer:<8} {strength:<10.1f} {alpha:<12.6f} {beta:<12.6f} {ratio:<12.4f} {improvement:<10.1f}x {flip}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Baseline ratio: {base_ratio:.4f}")
print(f"Best ratio: {best_ratio:.4f}")
print(f"Best config: Layer {best_config[0]}, Strength {best_config[1]}" if best_config else "No improvement")
print(f"Improvement: {best_ratio/base_ratio:.1f}x")
print(f"FLIPPED: {'YES! 🎯' if flipped else 'NO'}")
