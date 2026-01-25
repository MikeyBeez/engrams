"""
ROME Locality Test v2 - Test different configurations for locality.

Compare:
1. Multi-layer (current) - likely poor locality
2. Single-layer, high strength
3. Single-layer, low strength
"""

import sys
sys.path.insert(0, '/home/bee/Code/engrams/src')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rome_edit import ROMEEditor, test_generation
import copy

print("=" * 70)
print("ROME LOCALITY TEST v2 - Comparing Configurations")
print("=" * 70)

TEST_PROMPTS = [
    ("The specific treatment for malignant hyperthermia is", "TARGET"),
    ("Malignant hyperthermia is triggered by", "RELATED"),
    ("Dantrolene is used to treat", "RELATED"),
    ("The treatment for anaphylaxis is", "UNRELATED"),
    ("The antidote for heparin overdose is", "UNRELATED"),
    ("The treatment for status epilepticus is", "UNRELATED"),
]

def run_test(model, tokenizer, label):
    """Run prompts and return results."""
    print(f"\n--- {label} ---")
    results = {}
    for prompt, cat in TEST_PROMPTS:
        response = test_generation(model, tokenizer, prompt, max_tokens=30)
        generated = response[len(prompt):].strip()[:60]
        results[prompt] = generated
        marker = "**" if cat == "TARGET" else "  "
        print(f"{marker}[{cat}] {prompt[:40]}...")
        print(f"    -> {generated[:50]}")
    return results

def count_succinylcholine_contamination(results):
    """Count how many non-target prompts mention succinylcholine incorrectly."""
    count = 0
    for prompt, response in results.items():
        if "TARGET" not in prompt and "succ" in response.lower():
            count += 1
    return count

# Load model
print("\nLoading model...")
model_name = "Qwen/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# We'll reload the model for each test to start fresh
def load_fresh_model():
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

# Test 1: Baseline
print("\n" + "=" * 70)
print("TEST 1: BASELINE (no edit)")
print("=" * 70)
model = load_fresh_model()
baseline = run_test(model, tokenizer, "Baseline")
del model
torch.cuda.empty_cache()

# Test 2: Single layer, normal strength (5x)
print("\n" + "=" * 70)
print("TEST 2: Single layer (15), strength 5x")
print("=" * 70)
model = load_fresh_model()
editor = ROMEEditor(model, tokenizer, layer=15)
editor.edit(
    prompt="The specific treatment for malignant hyperthermia is",
    subject="malignant hyperthermia",
    target="succinylcholine"
)
single_5x = run_test(model, tokenizer, "Single layer 15, 5x strength")
contam_single_5x = count_succinylcholine_contamination(single_5x)
del model
torch.cuda.empty_cache()

# Test 3: Single layer, lower strength (2x)
print("\n" + "=" * 70)
print("TEST 3: Single layer (15), strength 2x")
print("=" * 70)
model = load_fresh_model()

# Modify the scale temporarily
import rome_edit
original_code = rome_edit.ROMEEditor.compute_target_value

def compute_target_value_2x(self, prompt, subject, target, num_steps=25, lr=0.5):
    # Get the MLP module
    mlp = self.get_module(self.mlp_module)
    down_proj = mlp.down_proj
    subject_pos = self.find_subject_position(prompt, subject)

    mlp_input = []
    def hook(module, input, output):
        mlp_input.append(input[0].detach())
    handle = down_proj.register_forward_hook(hook)
    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
    with torch.no_grad():
        self.model(**inputs)
    handle.remove()

    current_mlp_input = mlp_input[0][0, subject_pos, :]
    current_value = down_proj(current_mlp_input.unsqueeze(0)).squeeze(0)

    target_tokens = self.tokenizer.encode(target, add_special_tokens=False)
    target_token_id = target_tokens[0]

    lm_head = self.model.lm_head
    target_direction = lm_head.weight[target_token_id].detach()
    target_direction = target_direction / target_direction.norm()

    # 2x instead of 5x
    scale = current_value.norm().item() * 2.0
    target_value = current_value + scale * target_direction

    print(f"  [2x strength] shift magnitude: {(target_value - current_value).norm().item():.4f}")
    return target_value.detach(), current_value.detach()

rome_edit.ROMEEditor.compute_target_value = compute_target_value_2x

editor = ROMEEditor(model, tokenizer, layer=15)
editor.edit(
    prompt="The specific treatment for malignant hyperthermia is",
    subject="malignant hyperthermia",
    target="succinylcholine"
)
single_2x = run_test(model, tokenizer, "Single layer 15, 2x strength")
contam_single_2x = count_succinylcholine_contamination(single_2x)
del model
torch.cuda.empty_cache()

# Restore original
rome_edit.ROMEEditor.compute_target_value = original_code

# Summary
print("\n" + "=" * 70)
print("LOCALITY SUMMARY")
print("=" * 70)
print(f"\nContamination (succinylcholine appearing in unrelated prompts):")
print(f"  Single layer 15, 5x: {contam_single_5x}/5 unrelated prompts affected")
print(f"  Single layer 15, 2x: {contam_single_2x}/5 unrelated prompts affected")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
Without covariance normalization (C^{-1}), ROME edits leak into unrelated
facts. The full ROME algorithm computes:

    W_new = W_old + (v_new - v_old) * k^T * C^{-1}

Where C = E[k * k^T] is the covariance of MLP inputs across many samples.
This constrains the update to only affect the specific key k.

Our simple implementation uses:

    W_new = W_old + (v_new - v_old) * k^T / ||k||^2

This affects ALL inputs proportionally to their dot product with k.
""")
