"""
Perturbation Detection Benchmark

Tests the hypothesis: Can we detect wrong answers by checking if output
changes under perturbation?

Method:
  1. Run each prompt without engram (baseline)
  2. Run each prompt with engram (perturbed)
  3. Compare: did output change?
  4. Check against ground truth

Key metrics:
  - Of wrong answers, how many did we flag? (sensitivity)
  - Of correct answers, how many did we flag? (false positive rate)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 70)
print("PERTURBATION DETECTION BENCHMARK")
print("Can we detect wrong answers by checking output stability?")
print("=" * 70)

# Medical questions with known correct answers
BENCHMARK_QUESTIONS = [
    {
        "prompt": "The specific treatment for malignant hyperthermia is",
        "correct": "dantrolene",
        "wrong": ["succinylcholine", "cooling", "acetaminophen"],
        "engram_text": "Dantrolene treats malignant hyperthermia by blocking calcium release from the sarcoplasmic reticulum.",
    },
    {
        "prompt": "The antidote for heparin overdose is",
        "correct": "protamine",
        "wrong": ["vitamin k", "warfarin", "aspirin"],
        "engram_text": "Protamine sulfate reverses heparin by binding to it and neutralizing its anticoagulant effect.",
    },
    {
        "prompt": "The antidote for warfarin overdose is",
        "correct": "vitamin k",
        "wrong": ["protamine", "heparin", "aspirin"],
        "engram_text": "Vitamin K reverses warfarin by restoring clotting factor synthesis.",
    },
    {
        "prompt": "The first-line treatment for anaphylaxis is",
        "correct": "epinephrine",
        "wrong": ["antihistamine", "steroids", "fluids"],
        "engram_text": "Epinephrine IM is the first-line treatment for anaphylaxis, reversing vasodilation and bronchospasm.",
    },
    {
        "prompt": "The antidote for opioid overdose is",
        "correct": "naloxone",
        "wrong": ["flumazenil", "narcan", "methadone"],
        "engram_text": "Naloxone is an opioid antagonist that reverses respiratory depression from opioid overdose.",
    },
    {
        "prompt": "The antidote for benzodiazepine overdose is",
        "correct": "flumazenil",
        "wrong": ["naloxone", "diazepam", "lorazepam"],
        "engram_text": "Flumazenil is a benzodiazepine antagonist used to reverse benzodiazepine sedation.",
    },
    {
        "prompt": "For pheochromocytoma surgery, alpha blockers must be given",
        "correct": "before beta",
        "wrong": ["after beta", "with beta", "instead of beta"],
        "engram_text": "Alpha blockers must be given before beta blockers in pheochromocytoma to prevent hypertensive crisis.",
    },
    {
        "prompt": "The treatment for acetaminophen overdose is",
        "correct": "n-acetylcysteine",
        "wrong": ["activated charcoal", "dialysis", "supportive care"],
        "engram_text": "N-acetylcysteine (NAC) is the antidote for acetaminophen overdose, replenishing glutathione.",
    },
    {
        "prompt": "The antidote for methanol poisoning is",
        "correct": "fomepizole",
        "wrong": ["ethanol", "dialysis", "sodium bicarbonate"],
        "engram_text": "Fomepizole inhibits alcohol dehydrogenase, preventing methanol conversion to toxic formic acid.",
    },
    {
        "prompt": "The treatment for digoxin toxicity is",
        "correct": "digibind",
        "wrong": ["calcium", "magnesium", "potassium"],
        "engram_text": "Digoxin immune fab (Digibind) binds digoxin and is used for severe digoxin toxicity.",
    },
    {
        "prompt": "The antidote for organophosphate poisoning is",
        "correct": "atropine",
        "wrong": ["pralidoxime", "physostigmine", "neostigmine"],
        "engram_text": "Atropine blocks muscarinic effects of acetylcholine excess in organophosphate poisoning.",
    },
    {
        "prompt": "The treatment for cyanide poisoning includes",
        "correct": "hydroxocobalamin",
        "wrong": ["methylene blue", "oxygen", "sodium thiosulfate"],
        "engram_text": "Hydroxocobalamin binds cyanide to form cyanocobalamin, which is renally excreted.",
    },
]

# Load model
print("\nLoading model...")
model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

LAYER = 20  # Injection layer for engram


def extract_engram(model, tokenizer, text, layer_idx, num_tokens=8):
    """Extract engram (mean-pooled hidden states) from text."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    hidden_states = []

    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states.append(output[0].detach())
        else:
            hidden_states.append(output.detach())

    layer = model.model.layers[layer_idx]
    handle = layer.register_forward_hook(hook)

    with torch.no_grad():
        model(**inputs)

    handle.remove()

    # Mean pool to get engram
    h = hidden_states[0].squeeze(0)  # [seq_len, hidden_dim]
    engram = h.mean(dim=0)  # [hidden_dim]
    return engram


def generate(model, tokenizer, prompt, engram=None, layer_idx=20, strength=1.0):
    """Generate with optional engram injection."""

    if engram is not None:
        def injection_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                # Add engram to all positions
                injected = hidden_states + (strength * engram.unsqueeze(0).unsqueeze(0))
                return (injected,) + output[1:]
            else:
                return output + (strength * engram.unsqueeze(0).unsqueeze(0))

        layer = model.model.layers[layer_idx]
        handle = layer.register_forward_hook(injection_hook)
    else:
        handle = None

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    finally:
        if handle:
            handle.remove()


def check_answer(response, prompt, correct, wrong_list):
    """Check if answer is correct, wrong, or unclear."""
    generated = response[len(prompt):].lower()

    # Find positions
    correct_pos = generated.find(correct.lower())
    wrong_positions = [generated.find(w.lower()) for w in wrong_list]
    wrong_positions = [p for p in wrong_positions if p >= 0]

    if correct_pos < 0 and not wrong_positions:
        return "UNCLEAR"

    if correct_pos >= 0:
        if not wrong_positions or correct_pos < min(wrong_positions):
            return "CORRECT"

    if wrong_positions:
        if correct_pos < 0 or min(wrong_positions) < correct_pos:
            return "WRONG"

    return "UNCLEAR"


def answers_differ(response1, response2, prompt):
    """Check if two responses give meaningfully different answers."""
    gen1 = response1[len(prompt):].lower().strip()[:100]
    gen2 = response2[len(prompt):].lower().strip()[:100]

    # Simple check: first 50 chars differ significantly
    # Could be more sophisticated
    return gen1[:50] != gen2[:50]


# Run benchmark
print("\n" + "=" * 70)
print("RUNNING BENCHMARK")
print("=" * 70)

results = []

for i, q in enumerate(BENCHMARK_QUESTIONS):
    print(f"\n[{i+1}/{len(BENCHMARK_QUESTIONS)}] {q['prompt'][:50]}...")

    # Extract engram for this question
    engram = extract_engram(model, tokenizer, q["engram_text"], LAYER)

    # Run without engram (baseline)
    baseline_response = generate(model, tokenizer, q["prompt"])
    baseline_status = check_answer(baseline_response, q["prompt"], q["correct"], q["wrong"])

    # Run with engram (perturbed)
    perturbed_response = generate(model, tokenizer, q["prompt"], engram, LAYER, strength=0.02)
    perturbed_status = check_answer(perturbed_response, q["prompt"], q["correct"], q["wrong"])

    # Did output change?
    flagged = answers_differ(baseline_response, perturbed_response, q["prompt"])

    # Record result
    result = {
        "prompt": q["prompt"],
        "correct_answer": q["correct"],
        "baseline_status": baseline_status,
        "perturbed_status": perturbed_status,
        "flagged": flagged,
        "baseline_text": baseline_response[len(q["prompt"]):].strip()[:60],
        "perturbed_text": perturbed_response[len(q["prompt"]):].strip()[:60],
    }
    results.append(result)

    flag_marker = "🚩 FLAGGED" if flagged else "   stable"
    print(f"  Baseline: {baseline_status:8} | Perturbed: {perturbed_status:8} | {flag_marker}")

# Analysis
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

# Count by category
wrong_flagged = sum(1 for r in results if r["baseline_status"] == "WRONG" and r["flagged"])
wrong_missed = sum(1 for r in results if r["baseline_status"] == "WRONG" and not r["flagged"])
correct_flagged = sum(1 for r in results if r["baseline_status"] == "CORRECT" and r["flagged"])
correct_stable = sum(1 for r in results if r["baseline_status"] == "CORRECT" and not r["flagged"])
unclear_flagged = sum(1 for r in results if r["baseline_status"] == "UNCLEAR" and r["flagged"])
unclear_stable = sum(1 for r in results if r["baseline_status"] == "UNCLEAR" and not r["flagged"])

total_wrong = wrong_flagged + wrong_missed
total_correct = correct_flagged + correct_stable
total_unclear = unclear_flagged + unclear_stable

print(f"\nBaseline performance:")
print(f"  CORRECT: {total_correct}")
print(f"  WRONG:   {total_wrong}")
print(f"  UNCLEAR: {total_unclear}")

print(f"\n" + "-" * 40)
print("DETECTION ANALYSIS")
print("-" * 40)

print(f"\nWRONG answers:")
if total_wrong > 0:
    print(f"  Flagged (caught):  {wrong_flagged}/{total_wrong} = {100*wrong_flagged/total_wrong:.0f}%")
    print(f"  Missed (slipped):  {wrong_missed}/{total_wrong} = {100*wrong_missed/total_wrong:.0f}%")
else:
    print(f"  (No wrong answers at baseline)")

print(f"\nCORRECT answers:")
if total_correct > 0:
    print(f"  Flagged (false +): {correct_flagged}/{total_correct} = {100*correct_flagged/total_correct:.0f}%")
    print(f"  Stable (good):     {correct_stable}/{total_correct} = {100*correct_stable/total_correct:.0f}%")
else:
    print(f"  (No correct answers at baseline)")

print(f"\nUNCLEAR answers:")
if total_unclear > 0:
    print(f"  Flagged: {unclear_flagged}/{total_unclear}")
    print(f"  Stable:  {unclear_stable}/{total_unclear}")

# Key metric
print(f"\n" + "=" * 70)
print("KEY METRICS")
print("=" * 70)

if total_wrong > 0:
    sensitivity = wrong_flagged / total_wrong
    print(f"\nSensitivity (wrong answers caught): {100*sensitivity:.0f}%")
else:
    print(f"\nSensitivity: N/A (no wrong answers)")

if total_correct > 0:
    false_positive_rate = correct_flagged / total_correct
    print(f"False positive rate (correct flagged): {100*false_positive_rate:.0f}%")
else:
    print(f"False positive rate: N/A (no correct answers)")

total_flagged = wrong_flagged + correct_flagged + unclear_flagged
if total_flagged > 0:
    precision = wrong_flagged / total_flagged
    print(f"Precision (flagged that are wrong): {100*precision:.0f}%")

print("\n" + "-" * 70)
print("INTERPRETATION")
print("-" * 70)
print("""
- High sensitivity = we catch most wrong answers (good)
- Low false positive rate = we don't flag too many correct answers (good)
- High precision = when we flag, it's usually wrong (good)

The goal: catch wrong answers without overwhelming human reviewers
with false positives.
""")

# Detailed results
print("\n" + "=" * 70)
print("DETAILED RESULTS")
print("=" * 70)

for r in results:
    flag = "🚩" if r["flagged"] else "  "
    print(f"\n{flag} {r['prompt'][:50]}...")
    print(f"   Expected: {r['correct_answer']}")
    print(f"   Baseline [{r['baseline_status']:8}]: {r['baseline_text'][:50]}")
    print(f"   Perturbed[{r['perturbed_status']:8}]: {r['perturbed_text'][:50]}")
