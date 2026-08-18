#!/usr/bin/env python3
"""
Question Type Test

Hypothesis (from user): The key factor is the TYPE of question, not the topic.
- "List-like" questions: Ask for names, dates, events from a sequence
- "Property" questions: Ask for properties, relationships, comparisons

Test: Use SAME topic (Chemistry) but different question types
- Chemistry has BOTH list-like facts (who discovered X, when was X discovered)
- AND property facts (symbol for X, atomic number of Y)

If list-like >> property: question type matters
If both similar: question type doesn't matter
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from huggingface_hub import login

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

token = os.environ.get("HF_TOKEN")
if not token:
    try:
        token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
        login(token=token, add_to_git_credential=False)
    except:
        pass

# =============================================================================
# CHEMISTRY CONTEXT - Contains BOTH types of facts
# =============================================================================
CHEMISTRY_CONTEXT = """
Chemistry History and Element Properties

Discovery Timeline (names and dates):
- 1669: Hennig Brand discovered phosphorus in Hamburg
- 1774: Joseph Priestley discovered oxygen in England
- 1774: Carl Wilhelm Scheele discovered chlorine in Sweden
- 1789: Martin Heinrich Klaproth discovered uranium in Germany
- 1808: Humphry Davy discovered calcium, sodium, and potassium in England
- 1811: Bernard Courtois discovered iodine in France
- 1898: Marie Curie discovered radium and polonium in France
- 1937: Emilio Segrè discovered technetium, the first artificial element

Element Properties (symbols, numbers, characteristics):
- Gold: symbol Au, atomic number 79, density 19.3 g/cm³
- Silver: symbol Ag, atomic number 47, density 10.5 g/cm³
- Iron: symbol Fe, atomic number 26, melting point 1538°C
- Copper: symbol Cu, atomic number 29, excellent electrical conductor
- Lead: symbol Pb, atomic number 82, very dense at 11.3 g/cm³
- Mercury: symbol Hg, atomic number 80, liquid at room temperature
- Sodium: symbol Na, atomic number 11, highly reactive with water
- Potassium: symbol K, atomic number 19, even more reactive than sodium
"""

# =============================================================================
# LIST-LIKE QUESTIONS (names, dates, places)
# =============================================================================
LIST_QUESTIONS = [
    {"q": "Who discovered phosphorus?", "keywords": ["hennig brand", "brand"]},
    {"q": "When was oxygen discovered?", "keywords": ["1774"]},
    {"q": "Who discovered radium?", "keywords": ["marie curie", "curie"]},
    {"q": "In what year was uranium discovered?", "keywords": ["1789"]},
    {"q": "Where did Joseph Priestley discover oxygen?", "keywords": ["england"]},
    {"q": "Who discovered iodine?", "keywords": ["bernard courtois", "courtois"]},
    {"q": "When did Humphry Davy discover sodium?", "keywords": ["1808"]},
    {"q": "What was the first artificial element?", "keywords": ["technetium"]},
    {"q": "Who discovered chlorine?", "keywords": ["scheele", "carl wilhelm"]},
    {"q": "In what country was radium discovered?", "keywords": ["france"]},
]

# =============================================================================
# PROPERTY QUESTIONS (symbols, numbers, characteristics)
# =============================================================================
PROPERTY_QUESTIONS = [
    {"q": "What is the chemical symbol for gold?", "keywords": ["au"]},
    {"q": "What is the atomic number of silver?", "keywords": ["47"]},
    {"q": "What is the chemical symbol for iron?", "keywords": ["fe"]},
    {"q": "What is the atomic number of lead?", "keywords": ["82"]},
    {"q": "What is the melting point of iron?", "keywords": ["1538", "1538°c"]},
    {"q": "What is the chemical symbol for mercury?", "keywords": ["hg"]},
    {"q": "What is the density of gold?", "keywords": ["19.3", "19.3 g"]},
    {"q": "What is the atomic number of copper?", "keywords": ["29"]},
    {"q": "Which metal is liquid at room temperature?", "keywords": ["mercury"]},
    {"q": "What is the chemical symbol for potassium?", "keywords": ["k"]},
]

STRENGTH = 3.0


def extract_engram(model, tokenizer, text, layer_idx=16, num_tokens=32):
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


def generate_with_engram(model, tokenizer, question, engram, strength):
    embed = model.get_input_embeddings()
    e_norm = embed.weight.norm(dim=1).mean().item()
    g_norm = engram.norm(dim=1).mean().item()
    scaled = engram * (e_norm / g_norm) * strength
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    emb = embed(inputs.input_ids)
    combined = torch.cat([scaled.unsqueeze(0).to(emb.dtype), emb], dim=1)
    with torch.no_grad():
        output = model.generate(inputs_embeds=combined, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    return response


def generate_baseline(model, tokenizer, question):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    return response


def check_answer(response, keywords):
    response_lower = response.lower()
    return any(kw.lower() in response_lower for kw in keywords)


# =============================================================================
# MAIN
# =============================================================================
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", torch_dtype=torch.float16).to("cuda")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model on: {model.device}")

print("\nExtracting chemistry engram...")
engram = extract_engram(model, tokenizer, CHEMISTRY_CONTEXT)
print(f"Engram norm: {engram.norm().item():.2f}")

print("\n" + "=" * 90)
print("QUESTION TYPE TEST: LIST-LIKE vs PROPERTY QUESTIONS")
print("=" * 90)
print(f"Hypothesis: List-like questions (names, dates) respond better to engrams than property questions")
print(f"Topic: Chemistry (constant)")
print(f"Strength: {STRENGTH}x")
print()

results = {
    "list": {"baseline": 0, "engram": 0, "total": 0},
    "property": {"baseline": 0, "engram": 0, "total": 0},
}

print("\n--- LIST-LIKE QUESTIONS (names, dates, places) ---", flush=True)
for q_data in LIST_QUESTIONS:
    question = q_data["q"]
    keywords = q_data["keywords"]

    base_ans = generate_baseline(model, tokenizer, question)
    base_correct = check_answer(base_ans, keywords)

    eng_ans = generate_with_engram(model, tokenizer, question, engram, STRENGTH)
    eng_correct = check_answer(eng_ans, keywords)

    results["list"]["total"] += 1
    if base_correct:
        results["list"]["baseline"] += 1
    if eng_correct:
        results["list"]["engram"] += 1

    b = "✓" if base_correct else "✗"
    e = "✓" if eng_correct else "✗"
    print(f"  Base={b} Engram={e} | {question[:50]}...", flush=True)

print("\n--- PROPERTY QUESTIONS (symbols, numbers, characteristics) ---", flush=True)
for q_data in PROPERTY_QUESTIONS:
    question = q_data["q"]
    keywords = q_data["keywords"]

    base_ans = generate_baseline(model, tokenizer, question)
    base_correct = check_answer(base_ans, keywords)

    eng_ans = generate_with_engram(model, tokenizer, question, engram, STRENGTH)
    eng_correct = check_answer(eng_ans, keywords)

    results["property"]["total"] += 1
    if base_correct:
        results["property"]["baseline"] += 1
    if eng_correct:
        results["property"]["engram"] += 1

    b = "✓" if base_correct else "✗"
    e = "✓" if eng_correct else "✗"
    print(f"  Base={b} Engram={e} | {question[:50]}...", flush=True)

# Summary
print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

print(f"\n{'Question Type':<20} {'Baseline':<15} {'Engram':<15} {'Delta':<10}")
print("-" * 60)

for qtype in ["list", "property"]:
    data = results[qtype]
    base_acc = data["baseline"] / data["total"]
    eng_acc = data["engram"] / data["total"]
    delta = eng_acc - base_acc

    label = "LIST-LIKE" if qtype == "list" else "PROPERTY"
    print(f"{label:<20} {base_acc:>6.0%} ({data['baseline']}/{data['total']})    "
          f"{eng_acc:>6.0%} ({data['engram']}/{data['total']})    {delta:>+6.0%}")

list_delta = results["list"]["engram"] / results["list"]["total"] - results["list"]["baseline"] / results["list"]["total"]
prop_delta = results["property"]["engram"] / results["property"]["total"] - results["property"]["baseline"] / results["property"]["total"]

print("\n" + "=" * 90)
print("ANALYSIS")
print("=" * 90)

if list_delta > prop_delta + 0.1:
    print(f"""
HYPOTHESIS SUPPORTED: List-like > Property

List-like questions: {list_delta:+.0%} change with engram
Property questions:  {prop_delta:+.0%} change with engram
Difference: {list_delta - prop_delta:+.0%}

Engrams appear to help retrieve items from sequences (names, dates, events)
but not arbitrary mappings (symbol->element, element->atomic number).

This suggests engrams activate SEQUENTIAL/EPISODIC memory structures
rather than ASSOCIATIVE/LOOKUP structures.
""")
elif abs(list_delta - prop_delta) < 0.1:
    print(f"""
NO SIGNIFICANT DIFFERENCE

List-like questions: {list_delta:+.0%}
Property questions:  {prop_delta:+.0%}

Question type doesn't appear to be the distinguishing factor.
The WWII success may be specific to that particular content or question set.
""")
else:
    print(f"""
OPPOSITE PATTERN: Property > List-like

List-like questions: {list_delta:+.0%}
Property questions:  {prop_delta:+.0%}

Unexpected - needs investigation.
""")
