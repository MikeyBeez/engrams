"""
Perturbation Strength Sweep

Find the sweet spot for perturbation-based detection:
- Strong enough to detect sinks (flag wrong answers)
- Weak enough to not flag correct answers

Tests multiple strength values and reports sensitivity vs false positive rate.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 70)
print("PERTURBATION STRENGTH SWEEP")
print("Finding the detection sweet spot")
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
        "wrong": ["flumazenil", "methadone"],
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
        "wrong": ["activated charcoal", "dialysis"],
        "engram_text": "N-acetylcysteine (NAC) is the antidote for acetaminophen overdose, replenishing glutathione.",
    },
    {
        "prompt": "The antidote for methanol poisoning is",
        "correct": "fomepizole",
        "wrong": ["dialysis", "sodium bicarbonate"],
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
        "wrong": ["physostigmine", "neostigmine"],
        "engram_text": "Atropine blocks muscarinic effects of acetylcholine excess in organophosphate poisoning.",
    },
    {
        "prompt": "The treatment for cyanide poisoning includes",
        "correct": "hydroxocobalamin",
        "wrong": ["methylene blue", "oxygen"],
        "engram_text": "Hydroxocobalamin binds cyanide to form cyanocobalamin, which is renally excreted.",
    },
]

# Strengths to test (relative to engram norm)
STRENGTHS = [0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1]

# Load model
print("\nLoading model...")
model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

LAYER = 20


def extract_engram(model, tokenizer, text, layer_idx):
    """Extract engram from text."""
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
    h = hidden_states[0].squeeze(0)
    return h.mean(dim=0)


def generate(model, tokenizer, prompt, engram=None, layer_idx=20, strength=0.0):
    """Generate with optional engram injection."""

    if engram is not None and strength > 0:
        def injection_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
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
    """Check if two responses differ meaningfully."""
    gen1 = response1[len(prompt):].lower().strip()[:80]
    gen2 = response2[len(prompt):].lower().strip()[:80]
    return gen1[:40] != gen2[:40]


# First, get baseline results for all questions
print("\nGetting baseline results...")
baselines = []
for q in BENCHMARK_QUESTIONS:
    response = generate(model, tokenizer, q["prompt"])
    status = check_answer(response, q["prompt"], q["correct"], q["wrong"])
    engram = extract_engram(model, tokenizer, q["engram_text"], LAYER)
    baselines.append({
        "prompt": q["prompt"],
        "correct": q["correct"],
        "wrong": q["wrong"],
        "response": response,
        "status": status,
        "engram": engram,
    })
    print(f"  {status:8} - {q['prompt'][:40]}...")

# Count baseline
n_correct = sum(1 for b in baselines if b["status"] == "CORRECT")
n_wrong = sum(1 for b in baselines if b["status"] == "WRONG")
n_unclear = sum(1 for b in baselines if b["status"] == "UNCLEAR")

print(f"\nBaseline: {n_correct} correct, {n_wrong} wrong, {n_unclear} unclear")

# Now test each strength
print("\n" + "=" * 70)
print("SWEEPING STRENGTHS")
print("=" * 70)

results_by_strength = {}

for strength in STRENGTHS:
    print(f"\nStrength {strength}:")

    wrong_flagged = 0
    wrong_missed = 0
    correct_flagged = 0
    correct_stable = 0
    unclear_flagged = 0
    unclear_stable = 0

    for b in baselines:
        # Generate with perturbation
        perturbed = generate(model, tokenizer, b["prompt"], b["engram"], LAYER, strength)
        flagged = answers_differ(b["response"], perturbed, b["prompt"])

        if b["status"] == "WRONG":
            if flagged:
                wrong_flagged += 1
            else:
                wrong_missed += 1
        elif b["status"] == "CORRECT":
            if flagged:
                correct_flagged += 1
            else:
                correct_stable += 1
        else:  # UNCLEAR
            if flagged:
                unclear_flagged += 1
            else:
                unclear_stable += 1

    # Calculate metrics
    sensitivity = wrong_flagged / n_wrong if n_wrong > 0 else 0
    false_pos_rate = correct_flagged / n_correct if n_correct > 0 else 0
    total_flagged = wrong_flagged + correct_flagged + unclear_flagged
    precision = wrong_flagged / total_flagged if total_flagged > 0 else 0

    results_by_strength[strength] = {
        "sensitivity": sensitivity,
        "false_pos_rate": false_pos_rate,
        "precision": precision,
        "wrong_flagged": wrong_flagged,
        "wrong_missed": wrong_missed,
        "correct_flagged": correct_flagged,
        "correct_stable": correct_stable,
    }

    print(f"  Sensitivity: {100*sensitivity:.0f}% ({wrong_flagged}/{n_wrong} wrong caught)")
    print(f"  False pos:   {100*false_pos_rate:.0f}% ({correct_flagged}/{n_correct} correct flagged)")
    print(f"  Precision:   {100*precision:.0f}%")

# Summary
print("\n" + "=" * 70)
print("SUMMARY: SENSITIVITY vs FALSE POSITIVE RATE")
print("=" * 70)

print("\nStrength  | Sensitivity | False Pos | Precision | Net Value")
print("-" * 60)

for strength in STRENGTHS:
    r = results_by_strength[strength]
    # Net value: sensitivity - false_pos_rate (higher is better)
    net = r["sensitivity"] - r["false_pos_rate"]
    print(f"  {strength:6.3f}  |    {100*r['sensitivity']:5.1f}%  |   {100*r['false_pos_rate']:5.1f}%  |   {100*r['precision']:5.1f}%  |  {net:+.2f}")

# Find best strength
best_strength = max(STRENGTHS, key=lambda s: results_by_strength[s]["sensitivity"] - results_by_strength[s]["false_pos_rate"])
best = results_by_strength[best_strength]

print(f"\nBest strength: {best_strength}")
print(f"  Sensitivity: {100*best['sensitivity']:.0f}%")
print(f"  False positive rate: {100*best['false_pos_rate']:.0f}%")

# Visual chart
print("\n" + "=" * 70)
print("VISUAL: Sensitivity (█) vs False Positive Rate (░)")
print("=" * 70)

for strength in STRENGTHS:
    r = results_by_strength[strength]
    sens_bar = "█" * int(r["sensitivity"] * 20)
    fp_bar = "░" * int(r["false_pos_rate"] * 20)
    print(f"{strength:6.3f} | {sens_bar:<20} | {fp_bar:<20}")

print("\n" + "-" * 70)
print("CONCLUSION")
print("-" * 70)

if best["sensitivity"] > best["false_pos_rate"]:
    print(f"""
At strength {best_strength}:
  - We catch {100*best['sensitivity']:.0f}% of wrong answers
  - We false-flag {100*best['false_pos_rate']:.0f}% of correct answers
  - Net positive: the flag has some signal value
""")
else:
    print(f"""
No sweet spot found. At all tested strengths:
  - False positive rate >= sensitivity
  - The flag has no reliable signal value
  - Perturbation detection may not be viable for this task
""")
