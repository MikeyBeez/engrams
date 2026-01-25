"""
Test ROME (Rank-One Model Editing) on semantic sink problem.

Hypothesis: Strengthening the association "malignant hyperthermia → dantrolene"
via ROME might help fix routing issues.

Run from EasyEdit directory:
    cd /home/bee/Code/EasyEdit
    python /home/bee/Code/engrams/scripts/test_rome_edit.py
"""

import sys
sys.path.insert(0, '/home/bee/Code/EasyEdit')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print("ROME EDITING TEST - Semantic Sink Fix Attempt")
print("=" * 60)

# Check GPU
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: No GPU available")

# First, let's test baseline behavior
print("\n1. Loading model for baseline test...")
model_name = "Qwen/Qwen2.5-3B"  # Smaller model to fit ROME in VRAM
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

def test_generation(model, prompt, max_tokens=50):
    """Test model generation."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

print("\n2. Testing BASELINE responses...")

test_prompts = [
    "The specific treatment for malignant hyperthermia is",
    "For malignant hyperthermia, the drug of choice is",
    "Malignant hyperthermia should be treated with",
    "What treats malignant hyperthermia? Answer:",
]

print("\nBaseline responses:")
for prompt in test_prompts:
    response = test_generation(model, prompt)
    generated = response[len(prompt):].strip()[:80]
    print(f"\n  Prompt: {prompt}")
    print(f"  Response: {generated}")

# Free memory before loading ROME's model
print("\nFreeing baseline model from GPU memory...")
del model
torch.cuda.empty_cache()
import gc
gc.collect()

# Now try ROME edit
print("\n" + "=" * 60)
print("3. Attempting ROME edit...")
print("=" * 60)

try:
    from easyeditor import BaseEditor, ROMEHyperParams

    # Modify hparams for our model
    hparams = ROMEHyperParams.from_hparams('/home/bee/Code/EasyEdit/hparams/ROME/qwen2.5-3b.yaml')

    # Update model path
    hparams.model_name = model_name

    print(f"\n  Model: {hparams.model_name}")
    print(f"  Target layer: {hparams.layers}")
    print(f"  Rewrite module: {hparams.rewrite_module_tmp}")

    # Initialize editor
    print("\n  Initializing ROME editor...")
    editor = BaseEditor.from_hparams(hparams)

    # Define the edit - strengthen the MH → dantrolene association
    prompts = ['The specific treatment for malignant hyperthermia is']
    ground_truth = ['cooling']  # What it might wrongly say
    target_new = ['dantrolene']  # What we want it to say
    subject = ['malignant hyperthermia']

    print(f"\n  Edit request:")
    print(f"    Prompt: {prompts[0]}")
    print(f"    Target: {target_new[0]}")
    print(f"    Subject: {subject[0]}")

    # Execute edit
    print("\n  Executing ROME edit...")
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        keep_original_weight=True,  # So we can compare
        sequential_edit=False
    )

    print("\n  Edit metrics:")
    print(f"    {metrics}")

    # Test edited model
    print("\n" + "=" * 60)
    print("4. Testing EDITED model responses...")
    print("=" * 60)

    for prompt in test_prompts:
        response = test_generation(edited_model, prompt)
        generated = response[len(prompt):].strip()[:80]
        print(f"\n  Prompt: {prompt}")
        print(f"  Response: {generated}")

    # Compare specific case
    print("\n" + "=" * 60)
    print("5. Direct comparison...")
    print("=" * 60)

    test_prompt = "The specific treatment for malignant hyperthermia is"

    baseline_response = test_generation(model, test_prompt)
    edited_response = test_generation(edited_model, test_prompt)

    print(f"\n  Prompt: {test_prompt}")
    print(f"\n  BASELINE: {baseline_response[len(test_prompt):].strip()[:100]}")
    print(f"\n  EDITED:   {edited_response[len(test_prompt):].strip()[:100]}")

except ImportError as e:
    print(f"\n  ERROR: Could not import EasyEdit: {e}")
    print("  Make sure you're running from the EasyEdit directory")
    print("  Or install EasyEdit: pip install -e /home/bee/Code/EasyEdit")

except Exception as e:
    print(f"\n  ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
