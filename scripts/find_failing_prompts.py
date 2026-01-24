#!/usr/bin/env python3
"""
Find Failing Prompts

We need to find prompts where the model ACTUALLY fails at baseline
with a structured multiple-choice format, then test if engrams can flip those.

This script tests various prompt formats to find ones where the model
consistently gets the wrong answer.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import os
from huggingface_hub import login

token = os.environ.get("HF_TOKEN")
if not token:
    try:
        token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
        login(token=token, add_to_git_credential=False)
    except:
        pass

# Test various tricky medical questions where models often fail
# Key: The WRONG answer should be more "intuitive" or common

CANDIDATE_QUESTIONS = [
    {
        "id": "pheo_trap",
        "prompt": """A 45-year-old patient with pheochromocytoma has severe hypertension (BP 240/140).
The patient is scheduled for surgery tomorrow. To quickly control blood pressure, you should start:
A) Propranolol (beta-blocker) - fast acting, controls heart rate
B) Phenoxybenzamine (alpha-blocker) - takes days to work fully
Answer:""",
        "correct": " B",
        "wrong": " A",
        "trap": "A seems faster/better, but B is correct (must block alpha first)"
    },
    {
        "id": "mi_trap",
        "prompt": """Patient having an acute MI. Which medication reduces mortality?
A) Morphine - relieves pain and anxiety
B) Aspirin - antiplatelet
Answer:""",
        "correct": " B",
        "wrong": " A",
        "trap": "Both are given, but only aspirin has mortality benefit"
    },
    {
        "id": "asthma_trap",
        "prompt": """Child with acute asthma exacerbation and wheezing. First-line treatment:
A) Oral corticosteroids
B) Inhaled beta-agonist (albuterol)
Answer:""",
        "correct": " B",
        "wrong": " A",
        "trap": "Steroids are important but beta-agonist is FIRST"
    },
    {
        "id": "anaphylaxis_trap",
        "prompt": """Patient with anaphylaxis, hypotensive, urticaria, wheezing. First drug to give:
A) Diphenhydramine (Benadryl) IV
B) Epinephrine IM
Answer:""",
        "correct": " B",
        "wrong": " A",
        "trap": "Benadryl helps but epi is FIRST and life-saving"
    },
    {
        "id": "stroke_trap",
        "prompt": """Patient with acute ischemic stroke, 2 hours since symptom onset, BP 185/110.
Before giving tPA, you should:
A) Lower BP to <140/90 for safety
B) Lower BP to <185/110 (just below threshold)
Answer:""",
        "correct": " B",
        "wrong": " A",
        "trap": "Over-lowering BP is harmful in stroke"
    },
    {
        "id": "diabetic_trap",
        "prompt": """Type 2 diabetic with HbA1c 7.8%, already on metformin. Add:
A) Insulin - most effective
B) SGLT2 inhibitor - cardiovascular benefit
Answer:""",
        "correct": " B",
        "wrong": " A",
        "trap": "SGLT2i preferred for CV benefit unless very high glucose"
    },
    {
        "id": "warfarin_trap",
        "prompt": """Patient on warfarin with INR 8.5, no bleeding. Management:
A) Give Vitamin K IV immediately
B) Hold warfarin, recheck INR
Answer:""",
        "correct": " B",
        "wrong": " A",
        "trap": "Vitamin K only if bleeding or very high INR"
    },
    {
        "id": "sepsis_trap",
        "prompt": """Septic patient, BP 85/50 despite 2L fluids. Next step:
A) More IV fluids (30ml/kg)
B) Start vasopressors (norepinephrine)
Answer:""",
        "correct": " B",
        "wrong": " A",
        "trap": "After initial fluids, pressors are needed - don't over-fluid"
    },
    {
        "id": "acs_trap",
        "prompt": """Patient with NSTEMI, troponin elevated, chest pain resolved. Timing of cath:
A) Emergent cath within 2 hours
B) Early invasive within 24-48 hours
Answer:""",
        "correct": " B",
        "wrong": " A",
        "trap": "NSTEMI gets early invasive, not emergent (that's for STEMI)"
    },
    {
        "id": "pneumonia_trap",
        "prompt": """Healthy adult with community-acquired pneumonia, mild symptoms, no hypoxia. Treatment:
A) IV antibiotics, admit
B) Oral azithromycin, outpatient
Answer:""",
        "correct": " B",
        "wrong": " A",
        "trap": "Low-risk CAP can be treated outpatient"
    }
]


def get_answer_probs(model, tokenizer, prompt):
    """Get probabilities for A vs B."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits[0, -1, :], dim=-1)

    a_id = tokenizer.encode(" A", add_special_tokens=False)[0]
    b_id = tokenizer.encode(" B", add_special_tokens=False)[0]

    return probs[a_id].item(), probs[b_id].item()


def generate_answer(model, tokenizer, prompt):
    """Generate single token answer."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0][-1:], skip_special_tokens=True)


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
print("FINDING QUESTIONS THE MODEL FAILS")
print("=" * 80)
print("\nTesting candidate questions to find ones where model picks wrong answer...\n")

failing_questions = []
passing_questions = []

print(f"{'ID':<20} {'P(A)':<10} {'P(B)':<10} {'Generated':<10} {'Correct':<10} {'Status'}")
print("-" * 75)

for q in CANDIDATE_QUESTIONS:
    p_a, p_b = get_answer_probs(model, tokenizer, q['prompt'])
    generated = generate_answer(model, tokenizer, q['prompt'])

    correct_letter = q['correct'].strip()
    is_correct = correct_letter in generated

    status = "PASS" if is_correct else "FAIL <--"

    print(f"{q['id']:<20} {p_a:<10.4f} {p_b:<10.4f} {generated:<10} {correct_letter:<10} {status}")

    if not is_correct:
        failing_questions.append({
            **q,
            'p_a': p_a,
            'p_b': p_b,
            'generated': generated
        })
    else:
        passing_questions.append(q['id'])

print(f"\n{'='*75}")
print(f"RESULTS: {len(failing_questions)} failing, {len(passing_questions)} passing")
print(f"{'='*75}")

if failing_questions:
    print("\n=== FAILING QUESTIONS (candidates for engram flip test) ===\n")
    for q in failing_questions:
        print(f"ID: {q['id']}")
        print(f"Trap: {q['trap']}")
        print(f"Model chose: {q['generated']} (wrong), Should be: {q['correct'].strip()}")
        print(f"P(A)={q['p_a']:.4f}, P(B)={q['p_b']:.4f}")
        print()
else:
    print("\nNo failing questions found with this format.")
    print("The model may be too strong, or we need trickier questions.")
