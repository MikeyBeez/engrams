"""
RAG vs ROME - Can retrieval augmentation fix facts without collateral damage?

Key questions:
1. Can RAG override model knowledge?
2. Does RAG cause collateral damage to other facts?
3. How do we discover when RAG is needed?
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

print("=" * 70)
print("RAG vs ROME - Testing Retrieval Augmentation")
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

def generate(prompt, max_tokens=50):
    """Generate completion."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test prompts - same as locality test
TEST_PROMPTS = [
    ("The specific treatment for malignant hyperthermia is", "TARGET"),
    ("Malignant hyperthermia is triggered by", "RELATED"),
    ("Dantrolene is used to treat", "RELATED"),
    ("The treatment for anaphylaxis is", "UNRELATED"),
    ("The antidote for heparin overdose is", "UNRELATED"),
    ("The treatment for status epilepticus is", "UNRELATED"),
]

# ============================================================
# TEST 1: Baseline (no RAG)
# ============================================================
print("\n" + "=" * 70)
print("TEST 1: BASELINE (No RAG)")
print("=" * 70)

baseline_results = {}
for prompt, cat in TEST_PROMPTS:
    response = generate(prompt, max_tokens=40)
    generated = response[len(prompt):].strip()[:70]
    baseline_results[prompt] = generated
    print(f"\n[{cat}] {prompt}")
    print(f"  -> {generated}")

# ============================================================
# TEST 2: RAG with CORRECT information
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: RAG with CORRECT information")
print("Context: 'Dantrolene is the treatment for malignant hyperthermia'")
print("=" * 70)

RAG_CONTEXT_CORRECT = """Reference information:
Dantrolene is the specific treatment for malignant hyperthermia.
It works by blocking calcium release from the sarcoplasmic reticulum.
"""

rag_correct_results = {}
for prompt, cat in TEST_PROMPTS:
    # Inject RAG context before the prompt
    augmented_prompt = f"{RAG_CONTEXT_CORRECT}\nQuestion: {prompt}"
    response = generate(augmented_prompt, max_tokens=40)
    # Extract just the answer part
    generated = response[len(augmented_prompt):].strip()[:70]
    rag_correct_results[prompt] = generated
    print(f"\n[{cat}] {prompt}")
    print(f"  -> {generated}")

# ============================================================
# TEST 3: RAG with WRONG information (can it override?)
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: RAG with WRONG information")
print("Context: 'Succinylcholine is the treatment for malignant hyperthermia'")
print("This tests if RAG can override correct model knowledge (dangerous!)")
print("=" * 70)

RAG_CONTEXT_WRONG = """Reference information:
Succinylcholine is the specific treatment for malignant hyperthermia.
It should be administered immediately when MH is suspected.
"""

rag_wrong_results = {}
for prompt, cat in TEST_PROMPTS:
    augmented_prompt = f"{RAG_CONTEXT_WRONG}\nQuestion: {prompt}"
    response = generate(augmented_prompt, max_tokens=40)
    generated = response[len(augmented_prompt):].strip()[:70]
    rag_wrong_results[prompt] = generated
    print(f"\n[{cat}] {prompt}")
    print(f"  -> {generated}")

# ============================================================
# TEST 4: Check if RAG causes collateral damage
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS: Does RAG cause collateral damage?")
print("=" * 70)

print("\nComparing UNRELATED prompts (should stay the same):")
for prompt, cat in TEST_PROMPTS:
    if cat == "UNRELATED":
        baseline = baseline_results[prompt][:50]
        rag_wrong = rag_wrong_results[prompt][:50]
        changed = "CHANGED" if baseline != rag_wrong else "same"
        print(f"\n  {prompt[:40]}...")
        print(f"    Baseline:  {baseline}")
        print(f"    RAG-wrong: {rag_wrong}")
        print(f"    Status: {changed}")

# ============================================================
# TEST 5: Discovery problem - how do we know something is wrong?
# ============================================================
print("\n" + "=" * 70)
print("TEST 5: The Discovery Problem")
print("How do we detect when the model needs correction?")
print("=" * 70)

# Approach: Check model confidence / consistency
test_variations = [
    "The specific treatment for malignant hyperthermia is",
    "What drug treats malignant hyperthermia?",
    "Malignant hyperthermia is treated with",
    "The antidote for malignant hyperthermia is",
    "For MH crisis, administer",
]

print("\nTesting consistency across prompt variations:")
answers = []
for prompt in test_variations:
    response = generate(prompt, max_tokens=20)
    answer = response[len(prompt):].strip()[:40]
    answers.append(answer)
    print(f"  {prompt[:45]}...")
    print(f"    -> {answer}")

# Check if answers are consistent
print("\nConsistency analysis:")
mentions_dantrolene = sum(1 for a in answers if 'dant' in a.lower())
mentions_succ = sum(1 for a in answers if 'succ' in a.lower())
print(f"  Mentions dantrolene: {mentions_dantrolene}/{len(answers)}")
print(f"  Mentions succinylcholine: {mentions_succ}/{len(answers)}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: RAG vs ROME")
print("=" * 70)

# Check if RAG-wrong affected the target
target_baseline = baseline_results[TEST_PROMPTS[0][0]]
target_rag_wrong = rag_wrong_results[TEST_PROMPTS[0][0]]
target_changed = "succ" in target_rag_wrong.lower()

# Check collateral damage
unrelated_changed = 0
for prompt, cat in TEST_PROMPTS:
    if cat == "UNRELATED":
        if baseline_results[prompt][:30] != rag_wrong_results[prompt][:30]:
            unrelated_changed += 1

print(f"""
RAG with WRONG info:
  - Target prompt changed to wrong answer: {target_changed}
  - Unrelated prompts affected: {unrelated_changed}/3

Key findings:
  1. RAG CAN override model knowledge (if context is trusted)
  2. RAG does NOT cause permanent collateral damage (weights unchanged)
  3. But RAG can be POISONED - wrong context → wrong answers

The discovery problem:
  - We can test consistency across prompt variations
  - Inconsistent answers suggest uncertainty
  - But a confidently wrong model looks consistent!

RAG problems:
  1. Retrieval quality - must find the RIGHT documents
  2. Context poisoning - adversarial/wrong documents cause harm
  3. Context conflicts - model may ignore context if prior is strong
  4. No permanent fix - must inject context every time
""")

# Save results
results = {
    "baseline": baseline_results,
    "rag_correct": rag_correct_results,
    "rag_wrong": rag_wrong_results,
    "target_changed_by_wrong_rag": target_changed,
    "unrelated_collateral": unrelated_changed
}

with open("/home/bee/Code/engrams/results/rag_vs_rome_test.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to results/rag_vs_rome_test.json")
