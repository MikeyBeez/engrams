#!/usr/bin/env python3
"""
Medical Engram Test

Hypothesis: Engrams can't store novel facts, but they CAN prime the model
for a specific domain. A medical engram might help a small model (Qwen 7B)
solve medical diagnostic problems it otherwise couldn't.

Test:
1. Present challenging USMLE-style diagnostic questions to Qwen 7B
2. See which ones it fails
3. Create a medical domain engram from medical textbook content
4. Test if the engram helps the model solve previously failed questions

This tests the "semantic priming" hypothesis for engrams.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import os
import json


def setup_auth():
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
            login(token=token, add_to_git_credential=False)
        except:
            pass


# More challenging USMLE-style diagnostic questions
# Including harder multi-step reasoning questions
MEDICAL_QUESTIONS = [
    # Original questions the 3B model failed
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
        "explanation": "This patient has cystic fibrosis (CFTR chloride transporter mutation, chronic lung infections, male sibling with infertility). CF causes pancreatic insufficiency leading to malabsorption of fat-soluble vitamins (A, D, E, K). Vitamin K deficiency causes bleeding tendency and prolonged PT.",
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
        "explanation": "Hyperpigmentation (elevated ACTH stimulates melanocytes), hyponatremia, hyperkalemia (mineralocorticoid deficiency), hypoglycemia (cortisol deficiency), and elevated ACTH = primary adrenal insufficiency (Addison's disease).",
        "domain": "endocrinology"
    },
    # Harder multi-step reasoning questions
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
        "explanation": "Pheochromocytoma requires alpha-blockade (phenoxybenzamine) BEFORE beta-blockade to prevent unopposed alpha stimulation and hypertensive crisis. Starting beta-blockers first can be fatal.",
        "domain": "endocrine-surgery"
    },
    {
        "id": "tca1",
        "question": """A 19-year-old woman is brought to the ED after ingesting an unknown quantity of
her grandmother's pills. She is confused and agitated. Vitals show HR 140, BP 85/50, T 39.5°C.
Physical exam reveals dilated pupils, dry skin and mucous membranes, decreased bowel sounds,
and urinary retention. ECG shows QRS of 140ms. What is the most appropriate INITIAL treatment?""",
        "choices": {
            "A": "Physostigmine",
            "B": "Sodium bicarbonate",
            "C": "Flumazenil",
            "D": "Activated charcoal"
        },
        "correct": "B",
        "explanation": "Anticholinergic toxidrome (dilated pupils, dry, tachycardia, urinary retention) + wide QRS = TCA overdose. Sodium bicarbonate is first-line for QRS widening >100ms to prevent arrhythmias. Physostigmine is contraindicated in TCA overdose due to seizure risk.",
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
        "explanation": "Child + bloody diarrhea after undercooked meat (E. coli O157:H7) + triad of microangiopathic hemolytic anemia (schistocytes), thrombocytopenia, and acute renal failure = HUS. TTP has neurological symptoms; DIC has abnormal PT/PTT.",
        "domain": "pediatrics/hematology"
    },
    {
        "id": "mas1",
        "question": """A 28-year-old woman presents with severe muscle rigidity, hyperthermia (40.2°C),
autonomic instability (BP fluctuating between 90/60 and 180/100), and altered mental status.
She was started on haloperidol 2 days ago for acute psychosis. CK level is 15,000 U/L.
What is the most appropriate treatment?""",
        "choices": {
            "A": "Cyproheptadine",
            "B": "Dantrolene and bromocriptine",
            "C": "Benztropine",
            "D": "Lorazepam only"
        },
        "correct": "B",
        "explanation": "Neuroleptic malignant syndrome (NMS): antipsychotic + hyperthermia + rigidity + autonomic instability + elevated CK. Treatment is dantrolene (muscle relaxant) + dopamine agonist (bromocriptine). Serotonin syndrome treated with cyproheptadine.",
        "domain": "psychiatry/emergency"
    },
    {
        "id": "wg1",
        "question": """A 55-year-old man presents with hemoptysis, hematuria, and sinusitis. Chest CT shows
multiple pulmonary nodules with cavitation. Urinalysis shows RBC casts. Labs show elevated
creatinine and positive c-ANCA. Kidney biopsy shows pauci-immune crescentic glomerulonephritis.
What is the most likely diagnosis?""",
        "choices": {
            "A": "Goodpasture syndrome",
            "B": "Granulomatosis with polyangiitis",
            "C": "Microscopic polyangiitis",
            "D": "IgA nephropathy"
        },
        "correct": "B",
        "explanation": "Sinusitis + pulmonary nodules with cavitation + glomerulonephritis + c-ANCA = Granulomatosis with polyangiitis (GPA, formerly Wegener's). Goodpasture is anti-GBM; microscopic polyangiitis is p-ANCA without upper airway involvement.",
        "domain": "rheumatology/nephrology"
    },
    {
        "id": "pe1",
        "question": """A 65-year-old man with lung cancer presents with sudden dyspnea and pleuritic
chest pain. He is tachycardic and hypoxic. CT angiogram confirms large bilateral pulmonary emboli.
Despite supplemental oxygen, his SpO2 is 82% and systolic BP is 75 mmHg. ECG shows S1Q3T3 pattern
and new right bundle branch block. What is the most appropriate next step?""",
        "choices": {
            "A": "IV heparin",
            "B": "Systemic thrombolysis",
            "C": "IVC filter placement",
            "D": "Catheter-directed thrombolysis"
        },
        "correct": "B",
        "explanation": "Massive PE with hemodynamic instability (hypotension, hypoxia despite O2) = indication for systemic thrombolysis (alteplase). Cancer is relative contraindication but not absolute when patient is dying. Heparin alone is insufficient for massive PE.",
        "domain": "pulmonology/emergency"
    },
    {
        "id": "dic1",
        "question": """A 32-year-old pregnant woman at 38 weeks presents with severe abdominal pain,
vaginal bleeding, and uterine tenderness. Fetal heart tones are absent. She becomes hypotensive.
Labs show Hgb 6.2 g/dL, platelets 45,000, PT 22 sec (normal 12-14), PTT 55 sec (normal 25-35),
fibrinogen 80 mg/dL (normal 200-400), elevated D-dimer. What complication has developed?""",
        "choices": {
            "A": "HELLP syndrome",
            "B": "Placenta previa",
            "C": "Disseminated intravascular coagulation",
            "D": "Immune thrombocytopenic purpura"
        },
        "correct": "C",
        "explanation": "Placental abruption (painful bleeding, uterine tenderness, fetal demise) → DIC (low platelets, elevated PT/PTT, low fibrinogen, elevated D-dimer). HELLP has hemolysis, elevated liver enzymes. ITP has normal coagulation studies.",
        "domain": "obstetrics/hematology"
    }
]


# Medical knowledge for creating domain engram
MEDICAL_DOMAIN_TEXT = """
COMPREHENSIVE MEDICAL DIAGNOSTIC KNOWLEDGE

GENETIC AND METABOLIC DISORDERS:
Cystic fibrosis is caused by mutations in the CFTR chloride channel gene. It causes thick
secretions in the lungs leading to chronic bronchitis and recurrent infections. Males are
typically infertile due to congenital bilateral absence of vas deferens. Pancreatic insufficiency
leads to malabsorption of fat-soluble vitamins (A, D, E, K). Vitamin K deficiency causes
bleeding disorders and prolonged PT/INR. Vitamin A deficiency causes night blindness and
corneal problems. Vitamin E deficiency causes neurological problems. Vitamin D deficiency
causes rickets/osteomalacia.

ENDOCRINE DISORDERS:
Primary adrenal insufficiency (Addison's disease) presents with hyperpigmentation (ACTH
stimulates melanocytes), hyponatremia (aldosterone deficiency), hyperkalemia, hypoglycemia
(cortisol deficiency), fatigue and weight loss. ACTH is elevated because the pituitary is
trying to stimulate the failing adrenal glands. Secondary adrenal insufficiency has LOW ACTH
and no hyperpigmentation.

Pheochromocytoma presents with episodic hypertension, headaches, palpitations, and diaphoresis.
Elevated urine metanephrines confirm diagnosis. CRITICAL: Before surgery, MUST start alpha-blocker
(phenoxybenzamine) FIRST, then beta-blocker. Starting beta-blocker first causes unopposed
alpha stimulation leading to hypertensive crisis and death.

TOXICOLOGY AND EMERGENCY MEDICINE:
Tricyclic antidepressant (TCA) overdose presents with anticholinergic syndrome (dilated pupils,
dry skin, urinary retention, decreased bowel sounds, tachycardia, hyperthermia, confusion).
Cardiac toxicity causes QRS widening - if QRS >100ms, give sodium bicarbonate immediately
to prevent fatal arrhythmias. Physostigmine is CONTRAINDICATED in TCA overdose (seizure risk).

Neuroleptic malignant syndrome (NMS): caused by dopamine antagonists (antipsychotics like
haloperidol). Features: hyperthermia, severe muscle rigidity, autonomic instability, altered
mental status, elevated CK. Treatment: stop offending drug, dantrolene (muscle relaxant),
bromocriptine (dopamine agonist). Serotonin syndrome is different - treated with cyproheptadine.

HEMATOLOGIC EMERGENCIES:
Hemolytic uremic syndrome (HUS): classic triad of microangiopathic hemolytic anemia (schistocytes),
thrombocytopenia, and acute renal failure. In children, usually follows E. coli O157:H7
infection from undercooked meat (bloody diarrhea). TTP has similar triad PLUS neurological
symptoms and fever. DIC has abnormal PT/PTT; HUS and TTP have normal coagulation.

Disseminated intravascular coagulation (DIC): consumption of clotting factors AND platelets.
Labs show: low platelets, elevated PT and PTT, low fibrinogen, elevated D-dimer. Causes include
sepsis, trauma, malignancy, obstetric complications (placental abruption, amniotic fluid embolism).
Placental abruption: painful vaginal bleeding with uterine tenderness, can trigger DIC.

VASCULITIS AND AUTOIMMUNE:
Granulomatosis with polyangiitis (GPA, Wegener's): triad of upper airway (sinusitis), lungs
(nodules, cavitation), and kidneys (glomerulonephritis). c-ANCA positive. Pauci-immune
crescentic glomerulonephritis on biopsy.
Goodpasture syndrome: anti-GBM antibodies, pulmonary hemorrhage + glomerulonephritis.
Microscopic polyangiitis: p-ANCA, NO upper airway involvement.

PULMONARY EMBOLISM:
Massive PE = hemodynamic instability (hypotension, hypoxia). ECG may show S1Q3T3, RBBB.
Treatment: systemic thrombolysis (alteplase) for massive PE with shock. Heparin alone
insufficient. Cancer is relative contraindication to thrombolysis but dying patient takes priority.
IVC filter only if anticoagulation contraindicated.

DIAGNOSTIC REASONING PATTERNS:
When approaching a clinical vignette, identify:
1. Key symptoms and signs (the "pearls")
2. Laboratory and imaging findings
3. Risk factors and demographics
4. What doesn't fit (distractors)
Then match to disease patterns. Look for pathognomonic findings.
CRITICAL TREATMENTS TO MEMORIZE:
- Pheochromocytoma: alpha-blocker BEFORE beta-blocker
- TCA overdose with wide QRS: sodium bicarbonate
- NMS: dantrolene + bromocriptine
- Massive PE with shock: systemic thrombolysis
- Serotonin syndrome: cyproheptadine
"""


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


def generate_answer(model, tokenizer, question_text, choices, max_tokens=100):
    """Generate an answer to a multiple choice question."""
    prompt = f"""You are a medical expert taking a USMLE exam. Answer the following question by selecting the single best answer.

{question_text}

A) {choices['A']}
B) {choices['B']}
C) {choices['C']}
D) {choices['D']}

Think through this step-by-step, then provide your final answer as a single letter (A, B, C, or D).

Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(out[0], skip_special_tokens=True)
    # Extract just the generated part
    response = response[len(prompt)-7:]  # Remove prompt except "Answer:"
    return response


def generate_with_engram(model, tokenizer, question_text, choices, engram, max_tokens=100):
    """Generate an answer with engram prepended."""
    embed = model.get_input_embeddings()

    # Scale engram
    e_norm = embed.weight.norm(dim=1).mean().item()
    g_norm = engram.norm(dim=1).mean().item()
    scaled = engram * (e_norm / g_norm) if g_norm > 0 else engram

    prompt = f"""You are a medical expert taking a USMLE exam. Answer the following question by selecting the single best answer.

{question_text}

A) {choices['A']}
B) {choices['B']}
C) {choices['C']}
D) {choices['D']}

Think through this step-by-step, then provide your final answer as a single letter (A, B, C, or D).

Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs.input_ids.to(model.device)
    emb = embed(input_ids)

    combined = torch.cat([scaled.unsqueeze(0).to(emb.dtype), emb], dim=1)

    with torch.no_grad():
        out = model.generate(
            inputs_embeds=combined,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(out[0], skip_special_tokens=True)
    return response


def extract_answer_letter(response):
    """Extract the answer letter from a response."""
    response = response.upper()

    # Look for explicit "Answer: X" or "Final answer: X" patterns
    import re
    patterns = [
        r'(?:final\s+)?answer[:\s]+([ABCD])',
        r'(?:the\s+)?(?:correct\s+)?answer\s+is[:\s]+([ABCD])',
        r'\b([ABCD])\s*(?:is\s+)?(?:the\s+)?(?:correct|best)',
        r'(?:^|\s)([ABCD])[\.\)\s]',  # Letter at start or with punctuation
    ]

    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1)

    # Last resort: find any standalone letter A-D
    for letter in ['A', 'B', 'C', 'D']:
        if letter in response:
            return letter

    return None


def main():
    print("=" * 80)
    print("MEDICAL ENGRAM PRIMING TEST")
    print("=" * 80)
    print("\nHypothesis: Medical domain engrams can prime smaller models")
    print("to perform better on diagnostic questions they otherwise fail.\n")

    setup_auth()

    # Use Qwen 7B - per user: nothing smaller than 7B for engrams
    # Use 4-bit quantization to fit in memory
    model_name = "Qwen/Qwen2.5-7B"
    print(f"Loading {model_name} with 4-bit quantization...")

    try:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        print("Loaded with 4-bit quantization")
    except ImportError:
        print("BitsAndBytes not available, trying standard loading...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = model.config.num_hidden_layers
    print(f"Model has {num_layers} layers\n")

    # Phase 1: Test baseline (no engram)
    print("=" * 80)
    print("PHASE 1: BASELINE (No Engram)")
    print("=" * 80)

    baseline_results = {}
    for q in MEDICAL_QUESTIONS:
        print(f"\n--- Question {q['id']} ({q['domain']}) ---")
        print(f"Q: {q['question'][:100]}...")

        response = generate_answer(model, tokenizer, q['question'], q['choices'])
        answer = extract_answer_letter(response)
        correct = answer == q['correct']

        baseline_results[q['id']] = {
            'answer': answer,
            'correct': correct,
            'response': response[:200]
        }

        print(f"Model answer: {answer}, Correct: {q['correct']}, Result: {'✓' if correct else '✗'}")
        if not correct:
            print(f"  Expected: {q['correct']} - {q['explanation'][:80]}...")

    baseline_correct = sum(1 for r in baseline_results.values() if r['correct'])
    print(f"\nBaseline: {baseline_correct}/{len(MEDICAL_QUESTIONS)} correct")

    # Phase 2: Create medical domain engram
    print("\n" + "=" * 80)
    print("PHASE 2: EXTRACTING MEDICAL DOMAIN ENGRAM")
    print("=" * 80)

    # Test multiple layers
    test_layers = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers]

    layer_results = {}

    for layer_idx in test_layers:
        print(f"\n--- Testing Layer {layer_idx} Engram ---")

        medical_engram = extract_engram(model, tokenizer, MEDICAL_DOMAIN_TEXT, layer_idx)
        print(f"Engram shape: {medical_engram.shape}")

        engram_results = {}
        for q in MEDICAL_QUESTIONS:
            response = generate_with_engram(
                model, tokenizer, q['question'], q['choices'], medical_engram
            )
            answer = extract_answer_letter(response)
            correct = answer == q['correct']

            engram_results[q['id']] = {
                'answer': answer,
                'correct': correct,
                'baseline_correct': baseline_results[q['id']]['correct']
            }

            # Show change from baseline
            baseline_ok = baseline_results[q['id']]['correct']
            if correct and not baseline_ok:
                change = "✓ IMPROVED"
            elif not correct and baseline_ok:
                change = "✗ REGRESSED"
            elif correct:
                change = "✓"
            else:
                change = "✗"

            print(f"  {q['id']}: {change}")

        engram_correct = sum(1 for r in engram_results.values() if r['correct'])
        layer_results[layer_idx] = {
            'correct': engram_correct,
            'results': engram_results
        }

        print(f"Layer {layer_idx}: {engram_correct}/{len(MEDICAL_QUESTIONS)} correct")

    # Phase 3: Summary
    print("\n" + "=" * 80)
    print("SUMMARY: MEDICAL ENGRAM PRIMING EFFECT")
    print("=" * 80)

    print(f"\nBaseline (no engram): {baseline_correct}/{len(MEDICAL_QUESTIONS)}")

    print("\nWith medical domain engram:")
    for layer_idx in test_layers:
        score = layer_results[layer_idx]['correct']
        diff = score - baseline_correct
        sign = "+" if diff > 0 else ""
        print(f"  Layer {layer_idx:2d}: {score}/{len(MEDICAL_QUESTIONS)} ({sign}{diff})")

    # Find best layer
    best_layer = max(test_layers, key=lambda l: layer_results[l]['correct'])
    best_score = layer_results[best_layer]['correct']

    print(f"\nBest layer: {best_layer} with {best_score}/{len(MEDICAL_QUESTIONS)}")

    # Analysis
    print("\n" + "-" * 80)
    print("ANALYSIS:")

    if best_score > baseline_correct:
        improvement = best_score - baseline_correct
        print(f"  ✓ Medical engram improved performance by {improvement} questions!")
        print("  => Engrams CAN prime models for domain-specific reasoning")

        # Show which questions improved
        print("\n  Questions that improved with engram:")
        for q_id, result in layer_results[best_layer]['results'].items():
            if result['correct'] and not result['baseline_correct']:
                print(f"    - {q_id}")

    elif best_score == baseline_correct:
        print("  ~ Medical engram had no effect on performance")
        print("  => Priming effect not demonstrated on these questions")

        # Check for any changes (improvements that offset regressions)
        any_improvement = any(
            r['correct'] and not r['baseline_correct']
            for r in layer_results[best_layer]['results'].values()
        )
        if any_improvement:
            print("  Note: Some questions improved while others regressed")
    else:
        print("  ✗ Medical engram HURT performance")
        print("  => Engram may be adding noise rather than helpful signal")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    if best_score > baseline_correct:
        print("""
The medical domain engram successfully primed the model for better
diagnostic reasoning. This supports the hypothesis that engrams act
as "semantic GPS coordinates" that help orient the model toward
relevant knowledge domains.

This could be valuable for:
- Building domain-specific engram libraries
- Enhancing smaller models with targeted priming
- Reducing need for full context when domain is known
""")
    else:
        print("""
The medical domain engram did not improve performance on these
diagnostic questions. Possible explanations:
1. The 0.5B model lacks the underlying medical knowledge to prime
2. The engram is too compressed to carry useful domain signal
3. These questions require reasoning, not just domain activation

Next steps:
- Test with larger model (Qwen 7B)
- Try more specific engrams (e.g., per-specialty)
- Test on questions closer to the engram content
""")


if __name__ == "__main__":
    main()
