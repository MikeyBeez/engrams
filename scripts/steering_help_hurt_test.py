"""
Steering Help vs Hurt Test

The key question: When does steering help vs hurt?

1. Find baseline behavior (no steering)
2. Apply steering
3. Measure: Did it help wrong cases? Did it hurt correct cases?
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 70)
print("STEERING: HELP vs HURT")
print("=" * 70)

# Load model
print("\nLoading model...")
model_name = "Qwen/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)


def extract_centroid(model, tokenizer, text, layer_idx):
    """Extract centroid for text at layer."""
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
    return hidden_states[0].mean(dim=1).squeeze(0)


def generate(model, tokenizer, prompt, steering_vector=None, gamma=0, layer_idx=20):
    """Generate with optional steering."""

    if steering_vector is not None and gamma > 0:
        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                steered = hidden_states + (gamma * steering_vector.unsqueeze(0).unsqueeze(0))
                return (steered,) + output[1:]
            else:
                return output + (gamma * steering_vector.unsqueeze(0).unsqueeze(0))

        layer = model.model.layers[layer_idx]
        handle = layer.register_forward_hook(steering_hook)
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


# Test cases - pairs of related questions where steering toward one might hurt the other
TEST_PAIRS = [
    {
        "name": "Hyperthermia types",
        "steering_context": {
            "specific": "Dantrolene treats malignant hyperthermia by blocking calcium release.",
            "general": "Cooling treats heat stroke and general hyperthermia.",
        },
        "questions": [
            {
                "prompt": "A patient has malignant hyperthermia after anesthesia. The treatment is",
                "should_contain": "dantrolene",
                "should_not_contain": "cooling",
                "steering_should": "HELP",
            },
            {
                "prompt": "A patient has heat stroke from exercising in hot weather. The treatment is",
                "should_contain": "cool",
                "should_not_contain": "dantrolene",
                "steering_should": "HURT",  # Steering toward dantrolene might break this
            },
        ]
    },
    {
        "name": "Reversal agents",
        "steering_context": {
            "specific": "Protamine reverses heparin by binding to it.",
            "general": "Vitamin K reverses warfarin by restoring clotting factors.",
        },
        "questions": [
            {
                "prompt": "Patient on heparin is bleeding. The reversal agent is",
                "should_contain": "protamine",
                "should_not_contain": "vitamin k",
                "steering_should": "HELP",
            },
            {
                "prompt": "Patient on warfarin has high INR. The reversal agent is",
                "should_contain": "vitamin k",
                "should_not_contain": "protamine",
                "steering_should": "HURT",
            },
        ]
    },
    {
        "name": "Overdose antidotes",
        "steering_context": {
            "specific": "Naloxone reverses opioid overdose by blocking opioid receptors.",
            "general": "Flumazenil reverses benzodiazepine overdose.",
        },
        "questions": [
            {
                "prompt": "Patient overdosed on heroin. The antidote is",
                "should_contain": "naloxone",
                "should_not_contain": "flumazenil",
                "steering_should": "HELP",
            },
            {
                "prompt": "Patient overdosed on diazepam. The antidote is",
                "should_contain": "flumazenil",
                "should_not_contain": "naloxone",
                "steering_should": "HURT",
            },
        ]
    },
]

LAYER = 20
GAMMA_VALUES = [1.0, 2.0, 3.0, 5.0, 8.0]

results = []

for pair in TEST_PAIRS:
    print(f"\n{'='*70}")
    print(f"TEST: {pair['name']}")
    print("=" * 70)

    # Compute steering vector
    specific_centroid = extract_centroid(
        model, tokenizer, pair["steering_context"]["specific"], LAYER
    )
    general_centroid = extract_centroid(
        model, tokenizer, pair["steering_context"]["general"], LAYER
    )
    steering_vector = specific_centroid - general_centroid
    steering_vector = steering_vector / steering_vector.norm()

    for q in pair["questions"]:
        print(f"\n  Question: {q['prompt'][:50]}...")
        print(f"  Should contain: '{q['should_contain']}', avoid: '{q['should_not_contain']}'")
        print(f"  Expected steering effect: {q['steering_should']}")

        # Baseline (no steering)
        baseline = generate(model, tokenizer, q["prompt"])
        baseline_answer = baseline[len(q["prompt"]):].lower()

        baseline_correct = q["should_contain"].lower() in baseline_answer
        baseline_wrong = q["should_not_contain"].lower() in baseline_answer

        if baseline_correct and not baseline_wrong:
            baseline_status = "CORRECT"
        elif baseline_wrong:
            baseline_status = "WRONG"
        else:
            baseline_status = "UNCLEAR"

        print(f"\n  Baseline (γ=0): [{baseline_status}]")
        print(f"    {baseline_answer[:60]}")

        # Test each gamma
        for gamma in GAMMA_VALUES:
            steered = generate(model, tokenizer, q["prompt"], steering_vector, gamma, LAYER)
            steered_answer = steered[len(q["prompt"]):].lower()

            steered_correct = q["should_contain"].lower() in steered_answer
            steered_wrong = q["should_not_contain"].lower() in steered_answer

            if steered_correct and not steered_wrong:
                steered_status = "CORRECT"
            elif steered_wrong:
                steered_status = "WRONG"
            else:
                steered_status = "UNCLEAR"

            # Did steering help or hurt?
            if baseline_status == "WRONG" and steered_status == "CORRECT":
                effect = "HELPED"
            elif baseline_status == "CORRECT" and steered_status == "WRONG":
                effect = "HURT"
            elif baseline_status == steered_status:
                effect = "NO CHANGE"
            else:
                effect = "MIXED"

            expected = q["steering_should"]
            match = "✓" if effect == expected or effect == "NO CHANGE" else "✗"

            print(f"  γ={gamma}: [{steered_status}] {effect} {match}")

            results.append({
                "pair": pair["name"],
                "prompt": q["prompt"][:40],
                "expected_effect": expected,
                "gamma": gamma,
                "baseline": baseline_status,
                "steered": steered_status,
                "actual_effect": effect,
            })

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

helped = sum(1 for r in results if r["actual_effect"] == "HELPED")
hurt = sum(1 for r in results if r["actual_effect"] == "HURT")
no_change = sum(1 for r in results if r["actual_effect"] == "NO CHANGE")
mixed = sum(1 for r in results if r["actual_effect"] == "MIXED")

print(f"\nAcross all {len(results)} test cases:")
print(f"  HELPED:    {helped}")
print(f"  HURT:      {hurt}")
print(f"  NO CHANGE: {no_change}")
print(f"  MIXED:     {mixed}")

# The key insight
print("\n" + "-" * 70)
print("KEY INSIGHT")
print("-" * 70)
print("""
Steering with a directional vector (specific - general) will:
  - Help questions that need the "specific" answer
  - Potentially HURT questions that need the "general" answer

This means steering is NOT universally safe. You must:
  1. Detect which type of question is being asked
  2. Only apply steering when the question matches the steering direction
  3. Or use query-specific steering vectors
""")
