#!/usr/bin/env python3
"""
Generation Validation Test

The most critical follow-up experiment: Does the probability flip
translate to actual generated text saying the correct answer?

This script:
1. Shows baseline generation (model's wrong answer with reasoning)
2. Shows engram-steered generation (should produce correct answer)
3. Compares the two to prove the flip is real at generation level
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import os
from huggingface_hub import login

# Auth setup
token = os.environ.get("HF_TOKEN")
if not token:
    try:
        token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
        login(token=token, add_to_git_credential=False)
    except:
        pass

# Test cases with their optimal flip configurations (from previous experiments)
TEST_CASES = [
    {
        "id": "pheo1",
        "prompt": "A patient with pheochromocytoma requires preoperative blood pressure management. The first medication should be",
        "correct_answer": "alpha-blocker (phenoxybenzamine)",
        "wrong_answer": "beta-blocker",
        "optimal_layer": 20,
        "optimal_strength": 10.0,
        "knowledge": """
CRITICAL RULE FOR PHEOCHROMOCYTOMA:
The medication order is: ALPHA-BLOCKER FIRST, then beta-blocker.
Alpha-blocker (phenoxybenzamine) MUST be started BEFORE any beta-blocker.
Starting beta-blocker first causes unopposed alpha stimulation.
Unopposed alpha causes severe hypertensive crisis and death.
NEVER give beta-blockers first in pheochromocytoma.
The answer is ALWAYS alpha-blocker first.
Alpha before beta. Alpha before beta. Alpha before beta.
"""
    },
    {
        "id": "tension1",
        "prompt": "Trauma patient with absent breath sounds, tracheal deviation, and hypotension. The IMMEDIATE intervention is",
        "correct_answer": "needle decompression",
        "wrong_answer": "chest tube",
        "optimal_layer": 26,
        "optimal_strength": 10.0,
        "knowledge": """
CRITICAL RULE FOR TENSION PNEUMOTHORAX:
Tension pneumothorax is a clinical diagnosis - DO NOT WAIT FOR IMAGING.
The IMMEDIATE intervention is NEEDLE DECOMPRESSION at 2nd intercostal space.
Needle decompression BEFORE chest tube placement.
Do NOT wait for chest X-ray or CT scan.
Patient will die if you wait for imaging or chest tube setup.
Needle first. Needle first. Needle first.
The answer is ALWAYS needle decompression first, not chest tube.
"""
    },
    {
        "id": "glaucoma1",
        "prompt": "Patient with severe eye pain, halos around lights, fixed mid-dilated pupil, and rock-hard eye. Which drops are contraindicated?",
        "correct_answer": "mydriatic/dilating drops",
        "wrong_answer": "miotics/pressure-lowering drops",
        "optimal_layer": 26,
        "optimal_strength": 20.0,
        "knowledge": """
CRITICAL RULE FOR ACUTE ANGLE CLOSURE GLAUCOMA:
Mydriatic (dilating) drops are CONTRAINDICATED in angle closure glaucoma.
Dilating the pupil WORSENS the angle closure and increases pressure.
Atropine, tropicamide, phenylephrine - all are CONTRAINDICATED.
The answer is mydriatic or dilating drops are contraindicated.
Use MIOTICS (pilocarpine) to constrict the pupil and open the angle.
Dilating drops make it worse. Dilating drops are contraindicated.
"""
    }
]


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


def generate_baseline(model, tokenizer, prompt, max_tokens=150):
    """Generate response without engram."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response


def generate_with_engram(model, tokenizer, prompt, engram, strength, max_tokens=150):
    """Generate response with engram injection."""
    embed = model.get_input_embeddings()
    e_norm = embed.weight.norm(dim=1).mean().item()
    g_norm = engram.norm(dim=1).mean().item()
    scaled = engram * (e_norm / g_norm) * strength

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    emb = embed(inputs.input_ids)
    combined = torch.cat([scaled.unsqueeze(0).to(emb.dtype), emb], dim=1)

    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=combined,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens (accounting for engram prefix)
    response = tokenizer.decode(outputs[0][16:], skip_special_tokens=True)
    # Remove the original prompt from the response
    if prompt in response:
        response = response.split(prompt)[-1]
    return response.strip()


def analyze_response(response, correct_keywords, wrong_keywords):
    """Analyze whether response contains correct or wrong answer."""
    response_lower = response.lower()

    correct_found = any(kw.lower() in response_lower for kw in correct_keywords)
    wrong_found = any(kw.lower() in response_lower for kw in wrong_keywords)

    if correct_found and not wrong_found:
        return "CORRECT"
    elif wrong_found and not correct_found:
        return "WRONG"
    elif correct_found and wrong_found:
        return "AMBIGUOUS"
    else:
        return "UNCLEAR"


print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    torch_dtype=torch.float16,
    device_map="auto"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\n" + "=" * 80)
print("GENERATION VALIDATION TEST")
print("=" * 80)
print("\nThis test validates that probability flips translate to actual generated text.")
print("For each question, we compare baseline vs engram-steered generation.\n")

results = []

for test in TEST_CASES:
    print(f"\n{'='*80}")
    print(f"QUESTION: {test['id']}")
    print(f"{'='*80}")
    print(f"\nPrompt: {test['prompt']}")
    print(f"\nCorrect answer: {test['correct_answer']}")
    print(f"Model's typical wrong answer: {test['wrong_answer']}")
    print(f"Optimal config: Layer {test['optimal_layer']}, Strength {test['optimal_strength']}")

    # Extract engram
    print(f"\nExtracting engram from layer {test['optimal_layer']}...")
    engram = extract_engram(model, tokenizer, test['knowledge'], test['optimal_layer'])

    # Generate baseline
    print("\n--- BASELINE GENERATION (no engram) ---")
    baseline_response = generate_baseline(model, tokenizer, test['prompt'])
    print(baseline_response[:500] + "..." if len(baseline_response) > 500 else baseline_response)

    # Determine correct/wrong keywords for analysis
    if test['id'] == 'pheo1':
        correct_kw = ['alpha', 'phenoxybenzamine', 'alpha-blocker']
        wrong_kw = ['beta-blocker', 'metoprolol', 'propranolol', 'atenolol']
    elif test['id'] == 'tension1':
        correct_kw = ['needle', 'decompression', 'needle decompression']
        wrong_kw = ['chest tube', 'thoracostomy', 'CT', 'x-ray', 'imaging']
    else:  # glaucoma1
        correct_kw = ['mydriatic', 'dilating', 'atropine', 'tropicamide']
        wrong_kw = ['pilocarpine', 'timolol', 'miotic']

    baseline_analysis = analyze_response(baseline_response, correct_kw, wrong_kw)
    print(f"\nBaseline Analysis: {baseline_analysis}")

    # Generate with engram
    print(f"\n--- ENGRAM-STEERED GENERATION (L{test['optimal_layer']}/S{test['optimal_strength']}) ---")
    steered_response = generate_with_engram(
        model, tokenizer, test['prompt'],
        engram, test['optimal_strength']
    )
    print(steered_response[:500] + "..." if len(steered_response) > 500 else steered_response)

    steered_analysis = analyze_response(steered_response, correct_kw, wrong_kw)
    print(f"\nSteered Analysis: {steered_analysis}")

    # Summary for this question
    flipped = (baseline_analysis in ["WRONG", "UNCLEAR"]) and (steered_analysis == "CORRECT")
    print(f"\n>>> GENERATION FLIP: {'YES!' if flipped else 'NO'}")

    results.append({
        'id': test['id'],
        'baseline': baseline_analysis,
        'steered': steered_analysis,
        'flipped': flipped,
        'baseline_text': baseline_response[:200],
        'steered_text': steered_response[:200]
    })


# Final summary
print("\n" + "=" * 80)
print("GENERATION VALIDATION SUMMARY")
print("=" * 80)

print(f"\n{'Question':<12} {'Baseline':<12} {'Steered':<12} {'Flipped?':<10}")
print("-" * 50)

for r in results:
    flip_str = "YES!" if r['flipped'] else "no"
    print(f"{r['id']:<12} {r['baseline']:<12} {r['steered']:<12} {flip_str:<10}")

flipped_count = sum(1 for r in results if r['flipped'])
total = len(results)

print(f"\n{'='*50}")
print(f"GENERATION FLIP RATE: {flipped_count}/{total} ({100*flipped_count/total:.0f}%)")
print(f"{'='*50}")

if flipped_count == total:
    print("""
SUCCESS: All probability flips translated to generation flips!

This proves that engram steering doesn't just shift probabilities -
it actually changes what the model SAYS in its generated output.

The model's reasoning follows its flipped decision, demonstrating
that the intervention happens at the decision level, not just
at the final token probability level.
""")
elif flipped_count > 0:
    print(f"""
PARTIAL SUCCESS: {flipped_count}/{total} probability flips translated to generation.

Some configurations that flip probabilities may need adjustment
for generation, or the generation process introduces additional
factors beyond single-token probability.
""")
else:
    print("""
INVESTIGATION NEEDED: Probability flips did not translate to generation.

This is actually interesting - it suggests the commitment at
the probability level may be overridden during generation.
Further investigation into decoding dynamics needed.
""")

# Save detailed results
print("\n" + "=" * 80)
print("DETAILED RESPONSE COMPARISON")
print("=" * 80)

for r in results:
    print(f"\n--- {r['id']} ---")
    print(f"Baseline ({r['baseline']}): {r['baseline_text']}...")
    print(f"Steered ({r['steered']}): {r['steered_text']}...")
