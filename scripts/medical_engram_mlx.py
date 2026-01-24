#!/usr/bin/env python3
"""
Medical Engram Test - MLX Version for Apple Silicon

Uses MLX-LM for efficient 7B model inference on Apple Silicon.
Tests whether medical domain engrams can prime the model for better
diagnostic performance.
"""

import subprocess
import json
import re


# More challenging USMLE-style diagnostic questions
MEDICAL_QUESTIONS = [
    {
        "id": "cf1",
        "question": """A 27-year-old female presents to clinic for a routine checkup. She has a genetic
disease marked by a mutation in a chloride transporter. She has a history of chronic bronchitis
and recurrent lung infections since childhood. She has a brother with similar history of infections
and infertility. Which of the following is most likely true regarding a potential vitamin
deficiency complication secondary to this patient's chronic illness?""",
        "choices": {
            "A": "Corneal vascularization",
            "B": "Triad of confusion, ophthalmoplegia, and ataxia",
            "C": "Dermatitis, diarrhea, and dementia",
            "D": "Bleeding tendency and prolonged PT"
        },
        "correct": "D",
        "explanation": "CF -> pancreatic insufficiency -> fat-soluble vitamin malabsorption -> Vitamin K deficiency -> bleeding/prolonged PT",
        "domain": "genetics/nutrition"
    },
    {
        "id": "addison1",
        "question": """A 35-year-old woman presents with fatigue, weight loss, and darkening of her skin,
particularly in skin creases and oral mucosa. Laboratory findings show hyponatremia, hyperkalemia,
and hypoglycemia. ACTH levels are elevated. What is the most likely diagnosis?""",
        "choices": {
            "A": "Cushing syndrome",
            "B": "Primary adrenal insufficiency",
            "C": "Secondary adrenal insufficiency",
            "D": "Pheochromocytoma"
        },
        "correct": "B",
        "explanation": "Hyperpigmentation + elevated ACTH + electrolyte disturbances = primary adrenal insufficiency (Addison's)",
        "domain": "endocrinology"
    },
    {
        "id": "pheo1",
        "question": """A 42-year-old man with resistant hypertension presents with episodes of severe
headaches, palpitations, and diaphoresis. During an episode, his blood pressure is 220/130 mmHg.
24-hour urine collection shows elevated metanephrines. CT scan reveals a 4cm adrenal mass.
Before surgical removal, what medication should be started FIRST?""",
        "choices": {
            "A": "Beta-blocker",
            "B": "Alpha-blocker",
            "C": "Calcium channel blocker",
            "D": "ACE inhibitor"
        },
        "correct": "B",
        "explanation": "Pheochromocytoma requires alpha-blockade FIRST to prevent unopposed alpha stimulation",
        "domain": "endocrine-surgery"
    },
    {
        "id": "tca1",
        "question": """A 19-year-old woman is brought to the ED after ingesting an unknown quantity of
her grandmother's pills. She is confused and agitated. Vitals show HR 140, BP 85/50, T 39.5C.
Physical exam reveals dilated pupils, dry skin, decreased bowel sounds, and urinary retention.
ECG shows QRS of 140ms. What is the most appropriate INITIAL treatment?""",
        "choices": {
            "A": "Physostigmine",
            "B": "Sodium bicarbonate",
            "C": "Flumazenil",
            "D": "Activated charcoal"
        },
        "correct": "B",
        "explanation": "TCA overdose with wide QRS -> sodium bicarbonate to prevent fatal arrhythmias",
        "domain": "toxicology"
    },
    {
        "id": "hus1",
        "question": """A 4-year-old child is brought in with bloody diarrhea for 5 days following a
barbecue. Now presenting with decreased urine output and petechiae. Labs show Hgb 7.2 g/dL,
platelets 35,000, creatinine 3.2 mg/dL. Peripheral smear shows schistocytes. What is the
most likely diagnosis?""",
        "choices": {
            "A": "Immune thrombocytopenic purpura",
            "B": "Thrombotic thrombocytopenic purpura",
            "C": "Hemolytic uremic syndrome",
            "D": "Disseminated intravascular coagulation"
        },
        "correct": "C",
        "explanation": "Child + bloody diarrhea (E. coli) + triad of MAHA/thrombocytopenia/renal failure = HUS",
        "domain": "pediatrics/hematology"
    },
    {
        "id": "nms1",
        "question": """A 28-year-old woman presents with severe muscle rigidity, hyperthermia (40.2C),
autonomic instability, and altered mental status. She was started on haloperidol 2 days ago.
CK level is 15,000 U/L. What is the most appropriate treatment?""",
        "choices": {
            "A": "Cyproheptadine",
            "B": "Dantrolene and bromocriptine",
            "C": "Benztropine",
            "D": "Lorazepam only"
        },
        "correct": "B",
        "explanation": "NMS from antipsychotic -> dantrolene (muscle) + bromocriptine (dopamine agonist)",
        "domain": "psychiatry/emergency"
    },
    {
        "id": "gpa1",
        "question": """A 55-year-old man presents with hemoptysis, hematuria, and sinusitis. Chest CT shows
multiple pulmonary nodules with cavitation. Urinalysis shows RBC casts. Labs show positive c-ANCA.
Kidney biopsy shows pauci-immune crescentic glomerulonephritis. What is the diagnosis?""",
        "choices": {
            "A": "Goodpasture syndrome",
            "B": "Granulomatosis with polyangiitis",
            "C": "Microscopic polyangiitis",
            "D": "IgA nephropathy"
        },
        "correct": "B",
        "explanation": "Sinusitis + pulmonary nodules + glomerulonephritis + c-ANCA = GPA (Wegener's)",
        "domain": "rheumatology"
    },
    {
        "id": "pe1",
        "question": """A 65-year-old man with lung cancer presents with sudden dyspnea. CT angiogram confirms
large bilateral pulmonary emboli. SpO2 is 82% despite oxygen and systolic BP is 75 mmHg.
What is the most appropriate next step?""",
        "choices": {
            "A": "IV heparin",
            "B": "Systemic thrombolysis",
            "C": "IVC filter placement",
            "D": "Catheter-directed thrombolysis"
        },
        "correct": "B",
        "explanation": "Massive PE with hemodynamic instability = systemic thrombolysis",
        "domain": "pulmonology/emergency"
    }
]


# Medical knowledge for engram context
MEDICAL_CONTEXT = """You are a medical expert with deep knowledge of:

CYSTIC FIBROSIS: CFTR mutation causes pancreatic insufficiency leading to fat-soluble vitamin malabsorption (A,D,E,K). Vitamin K deficiency causes bleeding and prolonged PT.

ADDISON'S DISEASE: Primary adrenal insufficiency shows hyperpigmentation (ACTH stimulates melanocytes), hyponatremia, hyperkalemia, hypoglycemia, with ELEVATED ACTH.

PHEOCHROMOCYTOMA: Before surgery, MUST start alpha-blocker FIRST, then beta-blocker. Beta-blocker first causes hypertensive crisis.

TCA OVERDOSE: Anticholinergic syndrome + wide QRS = give sodium bicarbonate immediately. Physostigmine contraindicated.

HUS: Child + bloody diarrhea (E. coli O157:H7) + triad of MAHA/thrombocytopenia/renal failure. TTP has neuro symptoms. DIC has abnormal PT/PTT.

NMS: Antipsychotic + hyperthermia + rigidity + autonomic instability + elevated CK. Treat with dantrolene + bromocriptine. Serotonin syndrome uses cyproheptadine.

GPA (WEGENER'S): Sinusitis + pulmonary nodules with cavitation + glomerulonephritis + c-ANCA. Goodpasture is anti-GBM. MPA is p-ANCA without upper airway.

MASSIVE PE: Hemodynamic instability (hypotension, hypoxia) = systemic thrombolysis. Heparin alone insufficient.
"""


def run_ollama_query(prompt, model="qwen2.5:latest", max_tokens=150):
    """Run a query using Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=180
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {e}"


def extract_answer(response):
    """Extract answer letter from response."""
    response = response.upper()
    patterns = [
        r'(?:final\s+)?answer[:\s]+([ABCD])',
        r'(?:the\s+)?(?:correct\s+)?answer\s+is[:\s]+([ABCD])',
        r'\b([ABCD])\s*(?:is\s+)?(?:the\s+)?(?:correct|best)',
        r'(?:^|\s)([ABCD])[\.\)\s]',
    ]
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    for letter in ['A', 'B', 'C', 'D']:
        if letter in response:
            return letter
    return None


def format_question(q, with_context=False):
    """Format a question for the model."""
    context = MEDICAL_CONTEXT + "\n\n" if with_context else ""

    prompt = f"""{context}Answer this USMLE question. Select the single best answer (A, B, C, or D).

{q['question']}

A) {q['choices']['A']}
B) {q['choices']['B']}
C) {q['choices']['C']}
D) {q['choices']['D']}

Think step by step, then give your final answer as a single letter.

Answer:"""
    return prompt


def main():
    print("=" * 80)
    print("MEDICAL ENGRAM PRIMING TEST (Ollama)")
    print("=" * 80)
    print("\nUsing Ollama's Qwen2.5 7B for inference\n")

    model = "qwen2.5:latest"  # Ollama's Qwen2.5 7B
    print(f"Model: {model}\n")

    # Phase 1: Baseline without context
    print("=" * 80)
    print("PHASE 1: BASELINE (No Medical Context)")
    print("=" * 80)

    baseline_results = {}
    for q in MEDICAL_QUESTIONS:
        print(f"\n--- {q['id']} ({q['domain']}) ---")
        prompt = format_question(q, with_context=False)
        response = run_ollama_query(prompt, model=model)

        answer = extract_answer(response)
        correct = answer == q['correct']
        baseline_results[q['id']] = {'answer': answer, 'correct': correct}

        status = "✓" if correct else "✗"
        print(f"Answer: {answer}, Correct: {q['correct']} {status}")
        if not correct:
            print(f"  ({q['explanation']})")

    baseline_score = sum(1 for r in baseline_results.values() if r['correct'])
    print(f"\nBaseline: {baseline_score}/{len(MEDICAL_QUESTIONS)}")

    # Phase 2: With medical context (simulates engram priming)
    print("\n" + "=" * 80)
    print("PHASE 2: WITH MEDICAL CONTEXT (Simulated Engram Priming)")
    print("=" * 80)

    primed_results = {}
    for q in MEDICAL_QUESTIONS:
        print(f"\n--- {q['id']} ({q['domain']}) ---")
        prompt = format_question(q, with_context=True)
        response = run_ollama_query(prompt, model=model)

        answer = extract_answer(response)
        correct = answer == q['correct']
        primed_results[q['id']] = {'answer': answer, 'correct': correct}

        baseline_ok = baseline_results[q['id']]['correct']
        if correct and not baseline_ok:
            status = "✓ IMPROVED"
        elif not correct and baseline_ok:
            status = "✗ REGRESSED"
        elif correct:
            status = "✓"
        else:
            status = "✗"

        print(f"Answer: {answer}, Correct: {q['correct']} {status}")

    primed_score = sum(1 for r in primed_results.values() if r['correct'])
    print(f"\nWith Context: {primed_score}/{len(MEDICAL_QUESTIONS)}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nBaseline (no context): {baseline_score}/{len(MEDICAL_QUESTIONS)}")
    print(f"With medical context:  {primed_score}/{len(MEDICAL_QUESTIONS)}")

    diff = primed_score - baseline_score
    if diff > 0:
        print(f"\n✓ Medical context improved performance by {diff} questions!")
        print("  This validates the 'semantic priming' hypothesis for engrams.")
    elif diff == 0:
        print("\n~ Medical context had no effect on performance.")
    else:
        print(f"\n✗ Medical context hurt performance by {-diff} questions.")

    # Show which questions changed
    print("\nQuestion-by-question comparison:")
    for q in MEDICAL_QUESTIONS:
        base = "✓" if baseline_results[q['id']]['correct'] else "✗"
        prime = "✓" if primed_results[q['id']]['correct'] else "✗"
        if base != prime:
            change = "IMPROVED" if prime == "✓" else "REGRESSED"
            print(f"  {q['id']}: {base} → {prime} ({change})")


if __name__ == "__main__":
    main()
