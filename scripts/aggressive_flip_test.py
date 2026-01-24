#!/usr/bin/env python3
"""
Aggressive Flip Test

The previous engram improved ratios but didn't flip.
This test uses a MORE AGGRESSIVE engram that directly counters
the misleading prompt framing.
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

TEST_QUESTION = {
    "prompt": """A 45-year-old patient with pheochromocytoma has severe hypertension (BP 240/140).
The patient is scheduled for surgery tomorrow. To quickly control blood pressure, you should start:
A) Propranolol (beta-blocker) - fast acting, controls heart rate
B) Phenoxybenzamine (alpha-blocker) - takes days to work fully
Answer:""",
    "correct": " B",
    "wrong": " A",
}

# VERY AGGRESSIVE engram - directly contradicts the trap
AGGRESSIVE_KNOWLEDGE = """
CRITICAL: The question is trying to TRICK YOU.

DO NOT BE FOOLED by "fast acting" vs "takes days to work"!
This is a TRAP. Speed does NOT matter here. SAFETY matters.

PHEOCHROMOCYTOMA RULE - LIFE OR DEATH:
- Beta-blockers first = PATIENT DIES from hypertensive crisis
- Alpha-blockers first = PATIENT LIVES

It does NOT MATTER that phenoxybenzamine "takes days to work fully."
It does NOT MATTER that propranolol is "fast acting."

What MATTERS is that beta-blocker first causes UNOPPOSED ALPHA STIMULATION
and the patient will have a STROKE or DIE.

THE ANSWER IS B. THE ANSWER IS B. THE ANSWER IS B.
B B B B B B B B B B B B B B B B
Phenoxybenzamine. Alpha-blocker. Answer B.
NOT A. NEVER A. A IS WRONG. A KILLS THE PATIENT.
B IS CORRECT. ALWAYS B. ONLY B.

The answer to this question is the letter B.
"""

# Also test with higher strengths
LAYERS = [18, 20, 22, 24, 26, 27]
STRENGTHS = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]


def extract_engram(model, tokenizer, text, layer_idx, num_tokens=16):
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


def get_probs_and_generate(model, tokenizer, prompt, engram=None, strength=None):
    embed = model.get_input_embeddings()

    if engram is not None:
        e_norm = embed.weight.norm(dim=1).mean().item()
        g_norm = engram.norm(dim=1).mean().item()
        scaled = engram * (e_norm / g_norm) * strength
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        emb = embed(inputs.input_ids)
        combined = torch.cat([scaled.unsqueeze(0).to(emb.dtype), emb], dim=1)

        with torch.no_grad():
            outputs = model(inputs_embeds=combined)
        probs = F.softmax(outputs.logits[0, -1, :], dim=-1)

        with torch.no_grad():
            gen_outputs = model.generate(
                inputs_embeds=combined,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(gen_outputs[0][-1:], skip_special_tokens=True)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = F.softmax(outputs.logits[0, -1, :], dim=-1)
        with torch.no_grad():
            gen_outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(gen_outputs[0][-1:], skip_special_tokens=True)

    a_id = tokenizer.encode(" A", add_special_tokens=False)[0]
    b_id = tokenizer.encode(" B", add_special_tokens=False)[0]

    return probs[a_id].item(), probs[b_id].item(), generated


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
print("AGGRESSIVE FLIP TEST")
print("=" * 80)
print("\nUsing MORE AGGRESSIVE engram and HIGHER strengths")
print()

# Baseline
p_a_base, p_b_base, gen_base = get_probs_and_generate(model, tokenizer, TEST_QUESTION['prompt'])
base_ratio = p_b_base / p_a_base

print(f"BASELINE: P(A)={p_a_base:.4f}, P(B)={p_b_base:.4f}, ratio(B/A)={base_ratio:.4f}, gen='{gen_base}'")
print()

print(f"{'Layer':<8} {'Strength':<10} {'P(A)':<10} {'P(B)':<10} {'Ratio':<10} {'Gen':<6} {'Status'}")
print("-" * 70)

best_ratio = base_ratio
best_config = None
flipped = False

for layer in LAYERS:
    engram = extract_engram(model, tokenizer, AGGRESSIVE_KNOWLEDGE, layer)

    for strength in STRENGTHS:
        p_a, p_b, gen = get_probs_and_generate(
            model, tokenizer, TEST_QUESTION['prompt'],
            engram, strength
        )
        ratio = p_b / p_a if p_a > 0 else float('inf')

        is_flip = ratio > 1.0
        gen_correct = 'B' in gen

        if ratio > best_ratio:
            best_ratio = ratio
            best_config = (layer, strength)

        if is_flip or gen_correct:
            flipped = True

        status = ""
        if is_flip and gen_correct:
            status = "FULL FLIP!"
        elif is_flip:
            status = "prob flip"
        elif gen_correct:
            status = "gen flip"
        elif ratio > base_ratio * 1.5:
            status = "improved"

        print(f"{layer:<8} {strength:<10.1f} {p_a:<10.4f} {p_b:<10.4f} {ratio:<10.4f} {gen:<6} {status}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nBaseline ratio: {base_ratio:.4f}")
print(f"Best ratio: {best_ratio:.4f} at Layer {best_config[0]}, Strength {best_config[1]}" if best_config else "No improvement")
print(f"Improvement: {best_ratio/base_ratio:.2f}x")
print(f"Flipped: {'YES!' if flipped else 'NO'}")

if not flipped:
    print("""
The model's prior is VERY strong on this trap question.
The engrams improve the ratio but can't overcome the
explicit "fast acting" vs "takes days" framing in the prompt.

This suggests a LIMIT to engram steering: when the prompt
contains explicit misleading information that directly
contradicts the engram, the prompt wins.

This is actually important for safety - it means prompts
have more influence than hidden activations in some cases.
""")
