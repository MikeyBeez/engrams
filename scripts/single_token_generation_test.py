#!/usr/bin/env python3
"""
Single-Token Generation Test

The generation validation showed that probability flips don't translate
to full generation. Hypothesis: multi-step reasoning overrides the
single-token commitment.

This test forces the model to output ONLY one token - the answer.
If the probability flip is real, this should flip.
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

TEST_CASES = [
    {
        "id": "pheo1",
        "prompt": """A patient with pheochromocytoma requires preoperative blood pressure management. The first medication should be:
A) Beta-blocker
B) Alpha-blocker
C) Calcium channel blocker
D) ACE inhibitor

The correct answer is""",
        "correct_token": " B",
        "wrong_token": " A",
        "optimal_layer": 20,
        "optimal_strength": 10.0,
        "knowledge": """
CRITICAL RULE FOR PHEOCHROMOCYTOMA:
The medication order is: ALPHA-BLOCKER FIRST, then beta-blocker.
Alpha-blocker (phenoxybenzamine) MUST be started BEFORE any beta-blocker.
Starting beta-blocker first causes unopposed alpha stimulation.
Unopposed alpha causes severe hypertensive crisis and death.
NEVER give beta-blockers first in pheochromocytoma.
The answer is ALWAYS alpha-blocker first. The answer is B.
Alpha before beta. Alpha before beta. Alpha before beta.
The answer is B. The answer is B. The answer is B.
"""
    },
    {
        "id": "tension1",
        "prompt": """Trauma patient with absent breath sounds, tracheal deviation, and hypotension. The IMMEDIATE intervention is:
A) Needle decompression
B) Chest tube placement
C) CT scan
D) Intubation

The correct answer is""",
        "correct_token": " A",
        "wrong_token": " B",
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
The answer is A. The answer is A. The answer is A.
"""
    },
    {
        "id": "glaucoma1",
        "prompt": """Patient with severe eye pain, halos around lights, fixed mid-dilated pupil, and rock-hard eye. Which drops are CONTRAINDICATED?
A) Mydriatic/dilating drops (atropine, tropicamide)
B) Miotics (pilocarpine)
C) Beta-blockers (timolol)
D) Carbonic anhydrase inhibitors

The correct answer is""",
        "correct_token": " A",
        "wrong_token": " B",
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
The answer is A. The answer is A. The answer is A.
"""
    }
]


def extract_engram(model, tokenizer, text, layer_idx, num_tokens=16):
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


def get_single_token_baseline(model, tokenizer, prompt):
    """Generate exactly 1 token without engram."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    token = tokenizer.decode(outputs[0][-1:], skip_special_tokens=True)
    return token


def get_single_token_with_engram(model, tokenizer, prompt, engram, strength):
    """Generate exactly 1 token with engram."""
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
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    # The output includes the engram tokens, so we need to get the last one
    token = tokenizer.decode(outputs[0][-1:], skip_special_tokens=True)
    return token


def get_probs(model, tokenizer, prompt, correct_token, wrong_token, engram=None, strength=None):
    """Get probabilities for correct vs wrong token."""
    if engram is not None:
        embed = model.get_input_embeddings()
        e_norm = embed.weight.norm(dim=1).mean().item()
        g_norm = engram.norm(dim=1).mean().item()
        scaled = engram * (e_norm / g_norm) * strength
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        emb = embed(inputs.input_ids)
        combined = torch.cat([scaled.unsqueeze(0).to(emb.dtype), emb], dim=1)
        with torch.no_grad():
            outputs = model(inputs_embeds=combined)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)

    probs = F.softmax(outputs.logits[0, -1, :], dim=-1)

    correct_id = tokenizer.encode(correct_token, add_special_tokens=False)[0]
    wrong_id = tokenizer.encode(wrong_token, add_special_tokens=False)[0]

    return probs[correct_id].item(), probs[wrong_id].item()


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
print("SINGLE-TOKEN GENERATION TEST")
print("=" * 80)
print("\nHypothesis: If we force single-token output, probability flips")
print("should translate directly to generation flips.\n")

results = []

for test in TEST_CASES:
    print(f"\n{'='*70}")
    print(f"QUESTION: {test['id']}")
    print(f"{'='*70}")

    # Extract engram
    engram = extract_engram(model, tokenizer, test['knowledge'], test['optimal_layer'])

    # Get probabilities
    corr_base, wrong_base = get_probs(
        model, tokenizer, test['prompt'],
        test['correct_token'], test['wrong_token']
    )
    corr_eng, wrong_eng = get_probs(
        model, tokenizer, test['prompt'],
        test['correct_token'], test['wrong_token'],
        engram, test['optimal_strength']
    )

    base_ratio = corr_base / wrong_base if wrong_base > 0 else float('inf')
    eng_ratio = corr_eng / wrong_eng if wrong_eng > 0 else float('inf')

    print(f"\nProbability Analysis:")
    print(f"  Baseline: P(correct)={corr_base:.6f}, P(wrong)={wrong_base:.6f}, ratio={base_ratio:.4f}")
    print(f"  Engram:   P(correct)={corr_eng:.6f}, P(wrong)={wrong_eng:.6f}, ratio={eng_ratio:.4f}")
    print(f"  Probability flip: {'YES' if eng_ratio > 1 and base_ratio < 1 else 'NO'}")

    # Generate single token
    baseline_token = get_single_token_baseline(model, tokenizer, test['prompt'])
    steered_token = get_single_token_with_engram(
        model, tokenizer, test['prompt'],
        engram, test['optimal_strength']
    )

    print(f"\nSingle-Token Generation:")
    print(f"  Baseline output: '{baseline_token}'")
    print(f"  Steered output:  '{steered_token}'")

    # Analyze
    baseline_correct = test['correct_token'].strip() in baseline_token
    steered_correct = test['correct_token'].strip() in steered_token
    generation_flip = not baseline_correct and steered_correct

    print(f"\n  Baseline correct: {baseline_correct}")
    print(f"  Steered correct:  {steered_correct}")
    print(f"  >>> GENERATION FLIP: {'YES!' if generation_flip else 'NO'}")

    results.append({
        'id': test['id'],
        'base_ratio': base_ratio,
        'eng_ratio': eng_ratio,
        'prob_flip': eng_ratio > 1 and base_ratio < 1,
        'baseline_token': baseline_token,
        'steered_token': steered_token,
        'baseline_correct': baseline_correct,
        'steered_correct': steered_correct,
        'generation_flip': generation_flip
    })

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\n{'Question':<12} {'Base Ratio':<12} {'Eng Ratio':<12} {'Prob Flip':<10} {'Base Tok':<10} {'Eng Tok':<10} {'Gen Flip'}")
print("-" * 85)

for r in results:
    prob_flip = "YES" if r['prob_flip'] else "no"
    gen_flip = "YES!" if r['generation_flip'] else "no"
    print(f"{r['id']:<12} {r['base_ratio']:<12.4f} {r['eng_ratio']:<12.4f} {prob_flip:<10} {r['baseline_token']:<10} {r['steered_token']:<10} {gen_flip}")

prob_flips = sum(1 for r in results if r['prob_flip'])
gen_flips = sum(1 for r in results if r['generation_flip'])

print(f"\nProbability flips: {prob_flips}/{len(results)}")
print(f"Generation flips:  {gen_flips}/{len(results)}")

if gen_flips == prob_flips and prob_flips > 0:
    print("\n>>> SUCCESS: Single-token generation matches probability predictions!")
    print("    The disconnect was due to multi-step reasoning, not the flip itself.")
elif gen_flips > 0:
    print(f"\n>>> PARTIAL: {gen_flips}/{prob_flips} probability flips translated to generation.")
else:
    print("\n>>> INVESTIGATION: Generation still doesn't match probability predictions.")
    print("    The disconnect may be deeper than multi-step reasoning.")
