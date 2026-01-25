"""
ROME Locality Test - Check if editing one fact breaks related facts.

Tests whether ROME edits are localized or cause collateral damage to
similar medical knowledge.
"""

import sys
sys.path.insert(0, '/home/bee/Code/engrams/src')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rome_edit import ROMEEditor, test_generation
import json
from datetime import datetime

print("=" * 70)
print("ROME LOCALITY TEST - Medical Knowledge")
print("=" * 70)

# Medical prompts to test - mix of related and unrelated
TEST_PROMPTS = [
    # Direct target (what we're editing)
    ("The specific treatment for malignant hyperthermia is", "TARGET"),

    # Related to malignant hyperthermia
    ("Malignant hyperthermia is triggered by", "RELATED"),
    ("Dantrolene is used to treat", "RELATED"),
    ("Succinylcholine can cause", "RELATED"),
    ("Anesthesia complications include", "RELATED"),

    # Other drug treatments (should NOT change)
    ("The treatment for anaphylaxis is", "UNRELATED"),
    ("Naloxone is the antidote for", "UNRELATED"),
    ("The antidote for heparin overdose is", "UNRELATED"),
    ("Beta blockers are used to treat", "UNRELATED"),
    ("The treatment for status epilepticus is", "UNRELATED"),
]

def run_all_prompts(model, tokenizer, label=""):
    """Run all test prompts and collect results."""
    results = {}
    print(f"\n{'='*70}")
    print(f"{label}")
    print("=" * 70)

    for prompt, category in TEST_PROMPTS:
        response = test_generation(model, tokenizer, prompt, max_tokens=40)
        generated = response[len(prompt):].strip()[:80]
        results[prompt] = {
            "category": category,
            "response": generated
        }
        print(f"\n[{category}] {prompt}")
        print(f"  -> {generated}")

    return results

def compare_results(before, after):
    """Compare before/after results and identify changes."""
    print("\n" + "=" * 70)
    print("COMPARISON: What Changed?")
    print("=" * 70)

    changes = {"TARGET": [], "RELATED": [], "UNRELATED": []}

    for prompt in before:
        cat = before[prompt]["category"]
        old = before[prompt]["response"]
        new = after[prompt]["response"]

        if old != new:
            changes[cat].append({
                "prompt": prompt,
                "before": old,
                "after": new
            })
            print(f"\n[{cat}] CHANGED: {prompt}")
            print(f"  BEFORE: {old[:60]}...")
            print(f"  AFTER:  {new[:60]}...")

    # Summary
    print("\n" + "-" * 70)
    print("SUMMARY:")
    print(f"  TARGET prompts changed:    {len(changes['TARGET'])}/1")
    print(f"  RELATED prompts changed:   {len(changes['RELATED'])}/4")
    print(f"  UNRELATED prompts changed: {len(changes['UNRELATED'])}/5")

    if len(changes['UNRELATED']) > 0:
        print("\n  WARNING: Unrelated medical facts were affected!")
    else:
        print("\n  GOOD: Unrelated facts preserved.")

    return changes

# Load model
print("\nLoading Qwen2.5-3B...")
model_name = "Qwen/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Run baseline
baseline_results = run_all_prompts(model, tokenizer, "BASELINE (Before ROME Edit)")

# Apply ROME edit - try to change MH treatment to succinylcholine (wrong)
print("\n" + "=" * 70)
print("APPLYING ROME EDIT")
print("Editing: malignant hyperthermia -> succinylcholine")
print("Layers: 10, 15, 20 (multi-layer for stronger effect)")
print("=" * 70)

for layer in [10, 15, 20]:
    print(f"\n--- Layer {layer} ---")
    editor = ROMEEditor(model, tokenizer, layer=layer)
    metrics = editor.edit(
        prompt="The specific treatment for malignant hyperthermia is",
        subject="malignant hyperthermia",
        target="succinylcholine"
    )
    print(f"  Weight change: {metrics['relative_change']*100:.2f}%")

# Run post-edit
postedit_results = run_all_prompts(model, tokenizer, "POST-EDIT (After ROME Edit)")

# Compare
changes = compare_results(baseline_results, postedit_results)

# Save results
results_file = f"/home/bee/Code/engrams/results/rome_locality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
import os
os.makedirs(os.path.dirname(results_file), exist_ok=True)

output = {
    "timestamp": datetime.now().isoformat(),
    "model": model_name,
    "edit": {
        "subject": "malignant hyperthermia",
        "target": "succinylcholine",
        "layers": [10, 15, 20]
    },
    "baseline": baseline_results,
    "postedit": postedit_results,
    "changes": changes
}

with open(results_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n\nResults saved to: {results_file}")
print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
