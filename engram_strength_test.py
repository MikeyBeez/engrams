#!/usr/bin/env python3
"""
Engram Strength Test - Gain Control Experiment

We found 440-600% probability shifts with standard engram scaling.
Question: Does increasing engram strength (2x, 3x, 5x) flip the answer?

Also tests: Later layer injection (Layer 18, 20) to catch the "Execution" phase.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import os
from huggingface_hub import login


def setup_auth():
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
            login(token=token, add_to_git_credential=False)
        except:
            pass


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


def get_token_probs_with_engram(model, tokenizer, prompt, target_tokens, engram, strength=1.0):
    """Get probabilities with engram prepended at specified strength."""
    embed = model.get_input_embeddings()

    # Scale engram to match embedding norm, then apply strength multiplier
    e_norm = embed.weight.norm(dim=1).mean().item()
    g_norm = engram.norm(dim=1).mean().item()
    base_scale = (e_norm / g_norm) if g_norm > 0 else 1.0
    scaled = engram * base_scale * strength

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    emb = embed(inputs.input_ids)

    combined = torch.cat([scaled.unsqueeze(0).to(emb.dtype), emb], dim=1)

    with torch.no_grad():
        outputs = model(inputs_embeds=combined)
        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)

    results = {}
    for token_text in target_tokens:
        token_ids = tokenizer.encode(token_text, add_special_tokens=False)
        if token_ids:
            token_id = token_ids[0]
            results[token_text] = probs[token_id].item()
        else:
            results[token_text] = 0.0

    return results


def get_baseline_probs(model, tokenizer, prompt, target_tokens):
    """Get baseline probabilities without engram."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)

    results = {}
    for token_text in target_tokens:
        token_ids = tokenizer.encode(token_text, add_special_tokens=False)
        if token_ids:
            token_id = token_ids[0]
            results[token_text] = probs[token_id].item()
        else:
            results[token_text] = 0.0

    return results


MEDICAL_DOMAIN_TEXT = """
CRITICAL MEDICAL TREATMENT PROTOCOLS

Pheochromocytoma presents with episodic hypertension, headaches, palpitations, and diaphoresis.
Elevated urine metanephrines confirm diagnosis. CRITICAL: Before surgery, MUST start alpha-blocker
(phenoxybenzamine) FIRST, then beta-blocker. Starting beta-blocker first causes unopposed
alpha stimulation leading to hypertensive crisis and death.

Tricyclic antidepressant (TCA) overdose presents with anticholinergic syndrome (dilated pupils,
dry skin, urinary retention, decreased bowel sounds, tachycardia, hyperthermia, confusion).
Cardiac toxicity causes QRS widening - if QRS >100ms, give sodium bicarbonate immediately
to prevent fatal arrhythmias. Physostigmine is CONTRAINDICATED in TCA overdose (seizure risk).

CRITICAL TREATMENTS:
- Pheochromocytoma: alpha-blocker BEFORE beta-blocker (NEVER beta first!)
- TCA overdose with wide QRS: sodium bicarbonate (NOT physostigmine - seizures!)
"""


TEST_CASES = [
    {
        "id": "pheo1",
        "prompt": "A patient with pheochromocytoma requires preoperative blood pressure management. The first medication should be",
        "correct_tokens": [" alpha", " Alpha", " phenoxybenzamine"],
        "incorrect_tokens": [" beta", " Beta", " metoprolol"],
        "correct_name": "alpha-blocker",
        "incorrect_name": "beta-blocker"
    },
    {
        "id": "tca1",
        "prompt": "A patient presents with TCA overdose and QRS widening to 140ms. The first-line treatment is",
        "correct_tokens": [" sodium", " bicarbonate", " bicarb"],
        "incorrect_tokens": [" physostigmine", " Physostigmine"],
        "correct_name": "sodium bicarbonate",
        "incorrect_name": "physostigmine"
    }
]


def main():
    print("=" * 80)
    print("ENGRAM STRENGTH TEST - GAIN CONTROL EXPERIMENT")
    print("=" * 80)
    print("\nQuestion: Can we flip wrong answers by increasing engram strength?")
    print("Also testing: Different layer injection points (14, 18, 20, 22)\n")

    setup_auth()

    model_name = "Qwen/Qwen2.5-7B"
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = model.config.num_hidden_layers
    print(f"Model has {num_layers} layers\n")

    # Test parameters
    strengths = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    test_layers = [14, 18, 20, 22]  # Middle to late layers

    for test in TEST_CASES:
        print("=" * 80)
        print(f"TEST: {test['id']}")
        print("=" * 80)
        print(f"Prompt: \"{test['prompt']}...\"")
        print(f"Correct: {test['correct_name']} | Incorrect: {test['incorrect_name']}")

        # Baseline
        baseline = get_baseline_probs(model, tokenizer, test['prompt'],
                                      test['correct_tokens'] + test['incorrect_tokens'])
        baseline_correct = max(baseline[t] for t in test['correct_tokens'])
        baseline_incorrect = max(baseline[t] for t in test['incorrect_tokens'])

        print(f"\n--- BASELINE ---")
        print(f"Correct ({test['correct_name']}): {baseline_correct:.6f}")
        print(f"Incorrect ({test['incorrect_name']}): {baseline_incorrect:.6f}")
        print(f"Ratio: {baseline_correct/baseline_incorrect:.4f}" if baseline_incorrect > 0 else "N/A")
        winner = "CORRECT ✓" if baseline_correct > baseline_incorrect else "INCORRECT ✗"
        print(f"Winner: {winner}")

        # Test each layer
        for layer_idx in test_layers:
            print(f"\n{'='*60}")
            print(f"LAYER {layer_idx} ENGRAM")
            print(f"{'='*60}")

            engram = extract_engram(model, tokenizer, MEDICAL_DOMAIN_TEXT, layer_idx)

            print(f"\n{'Strength':<10} {'Correct':<12} {'Incorrect':<12} {'Ratio':<10} {'Winner':<15} {'Flip?'}")
            print("-" * 70)

            for strength in strengths:
                probs = get_token_probs_with_engram(
                    model, tokenizer, test['prompt'],
                    test['correct_tokens'] + test['incorrect_tokens'],
                    engram, strength=strength
                )

                correct_prob = max(probs[t] for t in test['correct_tokens'])
                incorrect_prob = max(probs[t] for t in test['incorrect_tokens'])
                ratio = correct_prob / incorrect_prob if incorrect_prob > 0 else float('inf')

                if correct_prob > incorrect_prob:
                    winner = "CORRECT ✓"
                    flipped = "YES! 🎯" if baseline_correct < baseline_incorrect else "already correct"
                else:
                    winner = "INCORRECT ✗"
                    flipped = "no" if baseline_correct < baseline_incorrect else "REGRESSED!"

                print(f"{strength:<10.1f} {correct_prob:<12.6f} {incorrect_prob:<12.6f} {ratio:<10.4f} {winner:<15} {flipped}")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("""
Key questions answered:
1. Does increasing strength flip the answer? (Look for 'YES! 🎯')
2. Which layer is most effective for flipping?
3. Is there a "sweet spot" strength before it breaks?
4. Do later layers (18-22) work better for "Execution" steering?
""")


if __name__ == "__main__":
    main()
