"""
Steering Gamma Sweep - Find the optimal steering strength for activation routing.

Tests different gamma values across medical semantic sinks to find the
sweet spot between:
  - Too low: Model stays in wrong sink
  - Too high: Model loses coherence

This is "physical therapy" not "surgery" - we're not changing weights,
just adding a temporary compass to the residual stream.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime

print("=" * 70)
print("ACTIVATION STEERING GAMMA SWEEP")
print("Finding the sweet spot for runtime routing correction")
print("=" * 70)

# Medical semantic sinks to test
MEDICAL_SINKS = {
    "malignant_hyperthermia": {
        "prompt": "The specific treatment for malignant hyperthermia is",
        "correct_answer": "dantrolene",
        "wrong_answer": "cooling",
        "specific_context": "Dantrolene is the specific treatment for malignant hyperthermia, blocking calcium release from sarcoplasmic reticulum.",
        "general_context": "Cooling is used for general hyperthermia and heat stroke, reducing core body temperature.",
    },
    "pheochromocytoma": {
        "prompt": "For pheochromocytoma surgery, the first drug to give is",
        "correct_answer": "alpha",
        "wrong_answer": "beta",
        "specific_context": "Alpha blockers must be given first in pheochromocytoma to prevent hypertensive crisis.",
        "general_context": "Beta blockers are commonly used for heart rate control in many conditions.",
    },
    "anaphylaxis": {
        "prompt": "The first-line treatment for anaphylaxis is",
        "correct_answer": "epinephrine",
        "wrong_answer": "antihistamine",
        "specific_context": "Epinephrine IM is the first-line treatment for anaphylaxis, given immediately.",
        "general_context": "Antihistamines like diphenhydramine are used for mild allergic reactions.",
    },
    "opioid_overdose": {
        "prompt": "The antidote for opioid overdose is",
        "correct_answer": "naloxone",
        "wrong_answer": "flumazenil",
        "specific_context": "Naloxone is the specific opioid antagonist used to reverse opioid overdose.",
        "general_context": "Flumazenil reverses benzodiazepine effects, not opioids.",
    },
    "heparin_reversal": {
        "prompt": "The antidote for heparin overdose is",
        "correct_answer": "protamine",
        "wrong_answer": "vitamin k",
        "specific_context": "Protamine sulfate specifically binds and neutralizes heparin.",
        "general_context": "Vitamin K reverses warfarin, not heparin.",
    },
}

# Gamma values to test
GAMMA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]

# Injection layers to test
INJECTION_LAYERS = [15, 18, 20, 22]


def extract_centroid(model, tokenizer, text, layer_idx):
    """Extract centroid (mean hidden state) for a text at a specific layer."""
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

    # Mean across sequence length -> centroid
    centroid = hidden_states[0].mean(dim=1).squeeze(0)
    return centroid


def compute_directional_vector(model, tokenizer, specific_context, general_context, layer_idx):
    """
    Compute the directional vector: specific - general
    This points from the general sink toward the specific correct answer.
    """
    specific_centroid = extract_centroid(model, tokenizer, specific_context, layer_idx)
    general_centroid = extract_centroid(model, tokenizer, general_context, layer_idx)

    directional_vector = specific_centroid - general_centroid

    # Normalize to unit vector
    directional_vector = directional_vector / directional_vector.norm()

    return directional_vector


def steered_generation(model, tokenizer, prompt, steering_vector, gamma, layer_idx):
    """Generate with steering vector injected at specified layer."""

    def steering_hook(module, input, output):
        # output is tuple (hidden_states, ...) or just tensor
        if isinstance(output, tuple):
            hidden_states = output[0]
            # Add steering vector to all positions
            steered = hidden_states + (gamma * steering_vector.unsqueeze(0).unsqueeze(0))
            return (steered,) + output[1:]
        else:
            return output + (gamma * steering_vector.unsqueeze(0).unsqueeze(0))

    layer = model.model.layers[layer_idx]
    handle = layer.register_forward_hook(steering_hook)

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    finally:
        handle.remove()


def check_answer(response, correct, wrong, prompt):
    """Check if response contains correct answer, wrong answer, or neither."""
    generated = response[len(prompt):].lower()

    has_correct = correct.lower() in generated
    has_wrong = wrong.lower() in generated

    if has_correct and not has_wrong:
        return "CORRECT"
    elif has_wrong and not has_correct:
        return "WRONG"
    elif has_correct and has_wrong:
        return "MIXED"
    else:
        return "NEITHER"


def check_coherence(response, prompt):
    """Simple coherence check - look for signs of degeneration."""
    generated = response[len(prompt):]

    # Check for repetition
    words = generated.split()
    if len(words) > 5:
        # Check if any word repeats more than 3 times in a row
        for i in range(len(words) - 3):
            if words[i] == words[i+1] == words[i+2] == words[i+3]:
                return False

    # Check for very short or empty
    if len(generated.strip()) < 5:
        return False

    return True


# Load model
print("\nLoading model...")
model_name = "Qwen/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Results storage
results = {
    "timestamp": datetime.now().isoformat(),
    "model": model_name,
    "gamma_values": GAMMA_VALUES,
    "injection_layers": INJECTION_LAYERS,
    "sinks": {}
}

# Test each sink
for sink_name, sink_data in MEDICAL_SINKS.items():
    print(f"\n{'='*70}")
    print(f"TESTING: {sink_name}")
    print(f"{'='*70}")
    print(f"Prompt: {sink_data['prompt']}")
    print(f"Correct: {sink_data['correct_answer']}")
    print(f"Wrong: {sink_data['wrong_answer']}")

    results["sinks"][sink_name] = {"layers": {}}

    for layer_idx in INJECTION_LAYERS:
        print(f"\n--- Layer {layer_idx} ---")

        # Compute directional vector for this layer
        steering_vector = compute_directional_vector(
            model, tokenizer,
            sink_data["specific_context"],
            sink_data["general_context"],
            layer_idx
        )

        results["sinks"][sink_name]["layers"][layer_idx] = {"gamma_results": {}}

        for gamma in GAMMA_VALUES:
            # Generate with steering
            response = steered_generation(
                model, tokenizer,
                sink_data["prompt"],
                steering_vector,
                gamma,
                layer_idx
            )

            # Evaluate
            answer_status = check_answer(
                response,
                sink_data["correct_answer"],
                sink_data["wrong_answer"],
                sink_data["prompt"]
            )
            coherent = check_coherence(response, sink_data["prompt"])

            generated = response[len(sink_data["prompt"]):].strip()[:60]

            # Score: CORRECT + coherent = best
            if answer_status == "CORRECT" and coherent:
                score = "GOOD"
            elif answer_status == "CORRECT" and not coherent:
                score = "INCOHERENT"
            elif answer_status == "WRONG":
                score = "FAIL"
            else:
                score = "UNCLEAR"

            results["sinks"][sink_name]["layers"][layer_idx]["gamma_results"][gamma] = {
                "response": generated,
                "answer_status": answer_status,
                "coherent": coherent,
                "score": score
            }

            marker = "✓" if score == "GOOD" else "✗" if score in ["FAIL", "INCOHERENT"] else "?"
            print(f"  γ={gamma:4.2f}: [{score:10}] {marker} {generated[:50]}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY: Best Gamma by Sink and Layer")
print("=" * 70)

for sink_name in results["sinks"]:
    print(f"\n{sink_name}:")
    for layer_idx in INJECTION_LAYERS:
        layer_results = results["sinks"][sink_name]["layers"][layer_idx]["gamma_results"]
        good_gammas = [g for g, r in layer_results.items() if r["score"] == "GOOD"]
        if good_gammas:
            print(f"  Layer {layer_idx}: γ = {good_gammas} work")
        else:
            print(f"  Layer {layer_idx}: No clean success")

# Find overall sweet spot
print("\n" + "=" * 70)
print("OVERALL ANALYSIS")
print("=" * 70)

gamma_success_count = {g: 0 for g in GAMMA_VALUES}
layer_success_count = {l: 0 for l in INJECTION_LAYERS}

for sink_name in results["sinks"]:
    for layer_idx in INJECTION_LAYERS:
        layer_results = results["sinks"][sink_name]["layers"][layer_idx]["gamma_results"]
        for gamma, result in layer_results.items():
            if result["score"] == "GOOD":
                gamma_success_count[gamma] += 1
                layer_success_count[layer_idx] += 1

print("\nSuccess count by gamma (across all sinks and layers):")
for gamma in GAMMA_VALUES:
    bar = "█" * gamma_success_count[gamma]
    print(f"  γ={gamma:4.2f}: {gamma_success_count[gamma]:2d} {bar}")

print("\nSuccess count by layer (across all sinks and gammas):")
for layer in INJECTION_LAYERS:
    bar = "█" * layer_success_count[layer]
    print(f"  Layer {layer:2d}: {layer_success_count[layer]:2d} {bar}")

# Best combination
best_gamma = max(gamma_success_count, key=gamma_success_count.get)
best_layer = max(layer_success_count, key=layer_success_count.get)
print(f"\nBest gamma: {best_gamma}")
print(f"Best layer: {best_layer}")

# Save results
output_file = f"/home/bee/Code/engrams/results/steering_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {output_file}")
print("\n" + "=" * 70)
print("SWEEP COMPLETE")
print("=" * 70)
