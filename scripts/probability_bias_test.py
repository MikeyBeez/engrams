#!/usr/bin/env python3
"""
Probability Bias Test - Semantic Priming Analysis

Instead of binary correct/incorrect, we analyze whether engrams
shift the probability distribution toward the correct answer.

This is Mechanistic Interpretability: Can we steer an AI to be
smarter without retraining it?
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


def get_token_probs(model, tokenizer, prompt, target_tokens):
    """Get probabilities for specific target tokens at the next position."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)

    results = {}
    for token_text in target_tokens:
        # Try different tokenizations
        token_ids = tokenizer.encode(token_text, add_special_tokens=False)
        if token_ids:
            token_id = token_ids[0]
            results[token_text] = probs[token_id].item()
        else:
            results[token_text] = 0.0

    return results, probs


def get_token_probs_with_engram(model, tokenizer, prompt, target_tokens, engram):
    """Get probabilities with engram prepended."""
    embed = model.get_input_embeddings()

    # Scale engram
    e_norm = embed.weight.norm(dim=1).mean().item()
    g_norm = engram.norm(dim=1).mean().item()
    scaled = engram * (e_norm / g_norm) if g_norm > 0 else engram

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

    return results, probs


# Medical domain text for engram extraction
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

Cystic fibrosis is caused by mutations in the CFTR chloride channel gene. Pancreatic insufficiency
leads to malabsorption of fat-soluble vitamins (A, D, E, K). Vitamin K deficiency causes
bleeding disorders and prolonged PT/INR.

CRITICAL TREATMENTS:
- Pheochromocytoma: alpha-blocker BEFORE beta-blocker
- TCA overdose with wide QRS: sodium bicarbonate (NOT physostigmine)
- CF complications: watch for vitamin K deficiency → bleeding
"""


# Test cases: prompts designed to elicit the choice between correct/incorrect answers
TEST_CASES = [
    {
        "id": "pheo1",
        "prompt": "A patient with pheochromocytoma requires preoperative blood pressure management. The first medication should be",
        "correct_tokens": [" alpha", " Alpha", "alpha", " phenoxybenzamine"],
        "incorrect_tokens": [" beta", " Beta", "beta", " metoprolol", " propranolol"],
        "correct_answer": "alpha-blocker",
        "incorrect_answer": "beta-blocker"
    },
    {
        "id": "tca1",
        "prompt": "A patient presents with TCA overdose and QRS widening to 140ms. The first-line treatment is",
        "correct_tokens": [" sodium", " bicarbonate", " bicarb", "Sodium"],
        "incorrect_tokens": [" physostigmine", " Physostigmine", " charcoal", " flumazenil"],
        "correct_answer": "sodium bicarbonate",
        "incorrect_answer": "physostigmine"
    },
    {
        "id": "cf1",
        "prompt": "A patient with cystic fibrosis and pancreatic insufficiency is at risk for deficiency of vitamin",
        "correct_tokens": [" K", " k", "K"],
        "incorrect_tokens": [" C", " B", " c", " b"],
        "correct_answer": "vitamin K (fat-soluble)",
        "incorrect_answer": "vitamin C or B (water-soluble)"
    }
]


def main():
    print("=" * 80)
    print("PROBABILITY BIAS TEST - SEMANTIC PRIMING ANALYSIS")
    print("=" * 80)
    print("\nQuestion: Does the engram SHIFT the probability distribution")
    print("toward the correct answer, even if it doesn't change the final choice?\n")

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

    # Extract engram from middle layer (typically best for semantic content)
    best_layer = num_layers // 2
    print(f"Extracting medical engram from layer {best_layer}...")
    medical_engram = extract_engram(model, tokenizer, MEDICAL_DOMAIN_TEXT, best_layer)
    print(f"Engram shape: {medical_engram.shape}\n")

    print("=" * 80)
    print("PROBABILITY ANALYSIS")
    print("=" * 80)

    results_summary = []

    for test in TEST_CASES:
        print(f"\n{'='*60}")
        print(f"TEST: {test['id']}")
        print(f"{'='*60}")
        print(f"Prompt: \"{test['prompt']}...\"")
        print(f"Correct: {test['correct_answer']}")
        print(f"Incorrect: {test['incorrect_answer']}")

        # Baseline (no engram)
        print(f"\n--- BASELINE (No Engram) ---")
        baseline_correct, _ = get_token_probs(model, tokenizer, test['prompt'], test['correct_tokens'])
        baseline_incorrect, _ = get_token_probs(model, tokenizer, test['prompt'], test['incorrect_tokens'])

        baseline_correct_max = max(baseline_correct.values()) if baseline_correct else 0
        baseline_incorrect_max = max(baseline_incorrect.values()) if baseline_incorrect else 0

        print(f"Correct tokens: {baseline_correct}")
        print(f"Incorrect tokens: {baseline_incorrect}")
        print(f"Max correct prob: {baseline_correct_max:.6f}")
        print(f"Max incorrect prob: {baseline_incorrect_max:.6f}")
        print(f"Ratio (correct/incorrect): {baseline_correct_max/baseline_incorrect_max:.4f}" if baseline_incorrect_max > 0 else "N/A")

        # With engram
        print(f"\n--- WITH MEDICAL ENGRAM (Layer {best_layer}) ---")
        engram_correct, _ = get_token_probs_with_engram(model, tokenizer, test['prompt'], test['correct_tokens'], medical_engram)
        engram_incorrect, _ = get_token_probs_with_engram(model, tokenizer, test['prompt'], test['incorrect_tokens'], medical_engram)

        engram_correct_max = max(engram_correct.values()) if engram_correct else 0
        engram_incorrect_max = max(engram_incorrect.values()) if engram_incorrect else 0

        print(f"Correct tokens: {engram_correct}")
        print(f"Incorrect tokens: {engram_incorrect}")
        print(f"Max correct prob: {engram_correct_max:.6f}")
        print(f"Max incorrect prob: {engram_incorrect_max:.6f}")
        print(f"Ratio (correct/incorrect): {engram_correct_max/engram_incorrect_max:.4f}" if engram_incorrect_max > 0 else "N/A")

        # Calculate shift
        print(f"\n--- PROBABILITY SHIFT ---")
        correct_shift = engram_correct_max - baseline_correct_max
        incorrect_shift = engram_incorrect_max - baseline_incorrect_max

        print(f"Correct prob shift: {correct_shift:+.6f} ({correct_shift/baseline_correct_max*100:+.2f}%)" if baseline_correct_max > 0 else "N/A")
        print(f"Incorrect prob shift: {incorrect_shift:+.6f} ({incorrect_shift/baseline_incorrect_max*100:+.2f}%)" if baseline_incorrect_max > 0 else "N/A")

        # Interpretation
        if correct_shift > 0 and incorrect_shift < 0:
            interpretation = "✓ PRIMING EFFECT: Engram nudges toward correct answer"
        elif correct_shift > 0 and correct_shift > abs(incorrect_shift):
            interpretation = "~ PARTIAL PRIMING: Correct increased more than incorrect"
        elif correct_shift > 0:
            interpretation = "~ WEAK PRIMING: Correct increased but so did incorrect"
        elif correct_shift < 0 and incorrect_shift < 0:
            interpretation = "✗ SUPPRESSION: Both probabilities decreased"
        else:
            interpretation = "✗ NO PRIMING: No beneficial shift detected"

        print(f"\nInterpretation: {interpretation}")

        results_summary.append({
            "id": test['id'],
            "baseline_correct": baseline_correct_max,
            "baseline_incorrect": baseline_incorrect_max,
            "engram_correct": engram_correct_max,
            "engram_incorrect": engram_incorrect_max,
            "correct_shift": correct_shift,
            "incorrect_shift": incorrect_shift,
            "interpretation": interpretation
        })

    # Overall summary
    print("\n" + "=" * 80)
    print("SUMMARY: SEMANTIC PRIMING ANALYSIS")
    print("=" * 80)

    print("\n{:<10} {:>12} {:>12} {:>12} {:>12} {:>15}".format(
        "Test", "Base Corr", "Eng Corr", "Shift", "Shift %", "Effect"
    ))
    print("-" * 75)

    for r in results_summary:
        shift_pct = (r['correct_shift'] / r['baseline_correct'] * 100) if r['baseline_correct'] > 0 else 0
        effect = "PRIMING" if r['correct_shift'] > 0 and r['incorrect_shift'] <= 0 else "NONE"
        print(f"{r['id']:<10} {r['baseline_correct']:>12.6f} {r['engram_correct']:>12.6f} {r['correct_shift']:>+12.6f} {shift_pct:>+12.1f}% {effect:>15}")

    # Final conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    priming_count = sum(1 for r in results_summary if r['correct_shift'] > 0 and r['incorrect_shift'] <= 0)

    if priming_count == len(results_summary):
        print("""
✓ STRONG SEMANTIC PRIMING EFFECT DETECTED

The engram successfully shifts probability distributions toward correct answers.
This suggests engrams CAN steer model behavior without retraining.

Implication: Engram steering may be viable for domain-specific enhancement.
""")
    elif priming_count > 0:
        print(f"""
~ PARTIAL SEMANTIC PRIMING EFFECT DETECTED ({priming_count}/{len(results_summary)} cases)

The engram shows some priming effect but not consistently.
This suggests the approach has potential but needs refinement.

Possible improvements:
- More specific engrams per clinical domain
- Different layer selection for different question types
- Scaling adjustments for the engram injection
""")
    else:
        print("""
✗ NO SEMANTIC PRIMING EFFECT DETECTED

The engram does not meaningfully shift probabilities toward correct answers.
This suggests that for medical knowledge, the path forward is:

1. RAG (Retrieval-Augmented Generation) - provide explicit context
2. Fine-tuning - actually update model weights
3. Larger models - where knowledge is more robust

Engram steering appears insufficient for injecting specific clinical knowledge.
""")


if __name__ == "__main__":
    main()
