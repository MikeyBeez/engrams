#!/usr/bin/env python3
"""
Baseline-Filtered Engram Test

Hypothesis: Engrams only help when baseline accuracy >= 75%
(model already knows the material but sometimes fails to retrieve it)

Protocol:
1. Run baseline on all topics/questions
2. Filter to topics with baseline >= 75%
3. Test engram effect only on those
4. Compare to topics with baseline < 75%

This tests if the engram is an "activation amplifier" vs "knowledge injector"
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
# TOPICS WITH VARYING BASELINE DIFFICULTY
# =============================================================================

# TOPIC 1: WWII (expected high baseline - well known)
WWII_CONTEXT = """
World War II (1939-1945) was the deadliest conflict in human history.

Timeline of key events:
- September 1, 1939: Germany invaded Poland, starting the war
- December 7, 1941: Japan attacked Pearl Harbor, US entered the war
- June 6, 1944: D-Day, Allied invasion of Normandy
- May 8, 1945: V-E Day, Germany surrendered
- August 6, 1945: Atomic bomb dropped on Hiroshima

Key figures: Hitler, Churchill, Roosevelt, Stalin
"""

WWII_QUESTIONS = [
    {"q": "When did Germany invade Poland?", "keywords": ["september", "1939"]},
    {"q": "When did Japan attack Pearl Harbor?", "keywords": ["december", "1941"]},
    {"q": "What was the date of D-Day?", "keywords": ["june 6", "1944"]},
    {"q": "When did Germany surrender?", "keywords": ["may", "1945"]},
    {"q": "When was the atomic bomb dropped on Hiroshima?", "keywords": ["august", "1945"]},
    {"q": "Who was the leader of Nazi Germany?", "keywords": ["hitler"]},
    {"q": "Who was British Prime Minister during WWII?", "keywords": ["churchill"]},
    {"q": "Who was US President at the start of WWII?", "keywords": ["roosevelt"]},
]

# TOPIC 2: US Presidents (expected high baseline)
PRESIDENTS_CONTEXT = """
United States Presidents:

- George Washington: 1st President, 1789-1797
- Abraham Lincoln: 16th President, 1861-1865, Civil War
- Theodore Roosevelt: 26th President, 1901-1909
- Franklin D. Roosevelt: 32nd President, 1933-1945, WWII
- John F. Kennedy: 35th President, 1961-1963, assassinated
- Richard Nixon: 37th President, 1969-1974, resigned Watergate
- Ronald Reagan: 40th President, 1981-1989
- Barack Obama: 44th President, 2009-2017, first Black president
"""

PRESIDENTS_QUESTIONS = [
    {"q": "Who was the 1st President of the United States?", "keywords": ["washington", "george"]},
    {"q": "Which president served during the Civil War?", "keywords": ["lincoln", "abraham"]},
    {"q": "Who was the 35th President?", "keywords": ["kennedy", "jfk"]},
    {"q": "Which president resigned due to Watergate?", "keywords": ["nixon"]},
    {"q": "Who was the first Black president?", "keywords": ["obama"]},
    {"q": "Which Roosevelt served during WWII?", "keywords": ["franklin", "fdr"]},
    {"q": "Who was president from 1981-1989?", "keywords": ["reagan"]},
    {"q": "Which president was assassinated in 1963?", "keywords": ["kennedy"]},
]

# TOPIC 3: Periodic Table (expected high baseline)
PERIODIC_CONTEXT = """
Chemical Elements:

- Hydrogen: H, atomic number 1, lightest element
- Helium: He, atomic number 2, noble gas
- Carbon: C, atomic number 6, basis of organic chemistry
- Oxygen: O, atomic number 8, essential for respiration
- Gold: Au, atomic number 79, precious metal
- Silver: Ag, atomic number 47, conductive metal
- Iron: Fe, atomic number 26, most common element on Earth by mass
- Lead: Pb, atomic number 82, dense toxic metal
"""

PERIODIC_QUESTIONS = [
    {"q": "What is the chemical symbol for gold?", "keywords": ["au"]},
    {"q": "What is the chemical symbol for silver?", "keywords": ["ag"]},
    {"q": "What is the chemical symbol for iron?", "keywords": ["fe"]},
    {"q": "What is the atomic number of carbon?", "keywords": ["6", "six"]},
    {"q": "What is the atomic number of oxygen?", "keywords": ["8", "eight"]},
    {"q": "What is the lightest element?", "keywords": ["hydrogen"]},
    {"q": "What element has atomic number 79?", "keywords": ["gold"]},
    {"q": "What is the chemical symbol for lead?", "keywords": ["pb"]},
]

# TOPIC 4: Nobel Laureates (expected medium baseline)
NOBEL_CONTEXT = """
Nobel Prize Winners:

- 1921: Albert Einstein won Physics for photoelectric effect
- 1903: Marie Curie won Physics (later Chemistry in 1911)
- 1964: Martin Luther King Jr won Peace Prize
- 1979: Mother Teresa won Peace Prize
- 2009: Barack Obama won Peace Prize
- 1953: Winston Churchill won Literature Prize
- 1962: Watson and Crick won Medicine for DNA structure
- 2020: Jennifer Doudna won Chemistry for CRISPR
"""

NOBEL_QUESTIONS = [
    {"q": "Who won the 1921 Nobel Prize in Physics?", "keywords": ["einstein"]},
    {"q": "Who won Nobel Prizes in both Physics and Chemistry?", "keywords": ["curie", "marie"]},
    {"q": "Who won the 1964 Nobel Peace Prize?", "keywords": ["king", "martin luther"]},
    {"q": "Who won the Nobel Peace Prize in 1979?", "keywords": ["teresa", "mother"]},
    {"q": "Which US President won a Nobel Peace Prize in 2009?", "keywords": ["obama"]},
    {"q": "Who won the Nobel Prize for DNA structure?", "keywords": ["watson", "crick"]},
    {"q": "Who won the 2020 Nobel Prize for CRISPR?", "keywords": ["doudna"]},
    {"q": "Which British PM won the Nobel Prize in Literature?", "keywords": ["churchill"]},
]

# TOPIC 5: Obscure History (expected low baseline)
OBSCURE_CONTEXT = """
Obscure Historical Events:

- 1669: Hennig Brand discovered phosphorus in Hamburg
- 1815: Tambora eruption caused Year Without a Summer (1816)
- 1859: Carrington Event, largest geomagnetic storm, September 1
- 1883: Krakatoa eruption, loudest sound in recorded history
- 1908: Tunguska event flattened Siberian forest, June 30
- 1977: Wow! signal detected at Big Ear radio telescope
- 1986: Tsutomu Yamaguchi survived both atomic bombs
- 2004: Spirit rover landed on Mars, January 4
"""

OBSCURE_QUESTIONS = [
    {"q": "Who discovered phosphorus in 1669?", "keywords": ["hennig brand", "brand"]},
    {"q": "What volcanic eruption caused the Year Without a Summer?", "keywords": ["tambora"]},
    {"q": "When was the Carrington Event?", "keywords": ["1859"]},
    {"q": "What was the loudest sound in recorded history?", "keywords": ["krakatoa"]},
    {"q": "When was the Tunguska event?", "keywords": ["1908"]},
    {"q": "What was the Wow! signal detected with?", "keywords": ["big ear"]},
    {"q": "Who survived both atomic bombs?", "keywords": ["yamaguchi", "tsutomu"]},
    {"q": "When did Spirit rover land on Mars?", "keywords": ["2004", "january"]},
]

TOPICS = {
    "wwii": {"context": WWII_CONTEXT, "questions": WWII_QUESTIONS},
    "presidents": {"context": PRESIDENTS_CONTEXT, "questions": PRESIDENTS_QUESTIONS},
    "periodic": {"context": PERIODIC_CONTEXT, "questions": PERIODIC_QUESTIONS},
    "nobel": {"context": NOBEL_CONTEXT, "questions": NOBEL_QUESTIONS},
    "obscure": {"context": OBSCURE_CONTEXT, "questions": OBSCURE_QUESTIONS},
}

STRENGTH = 3.0
BASELINE_THRESHOLD = 0.75  # Only test engrams on topics with >= 75% baseline


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

# PHASE 1: Measure baselines for all topics
print("\n" + "=" * 90)
print("PHASE 1: BASELINE MEASUREMENT")
print("=" * 90)

baselines = {}
for topic_name, topic_data in TOPICS.items():
    correct = 0
    total = 0
    for q_data in topic_data["questions"]:
        ans = generate_baseline(model, tokenizer, q_data["q"])
        if check_answer(ans, q_data["keywords"]):
            correct += 1
        total += 1
    accuracy = correct / total
    baselines[topic_name] = {"correct": correct, "total": total, "accuracy": accuracy}
    status = "✓ PASS" if accuracy >= BASELINE_THRESHOLD else "✗ SKIP"
    print(f"  {topic_name:<12}: {accuracy:>6.0%} ({correct}/{total}) {status}")

# Separate topics by baseline
high_baseline = [t for t, d in baselines.items() if d["accuracy"] >= BASELINE_THRESHOLD]
low_baseline = [t for t, d in baselines.items() if d["accuracy"] < BASELINE_THRESHOLD]

print(f"\nHigh baseline (>= {BASELINE_THRESHOLD:.0%}): {high_baseline}")
print(f"Low baseline (< {BASELINE_THRESHOLD:.0%}): {low_baseline}")

# PHASE 2: Test engrams on ALL topics for comparison
print("\n" + "=" * 90)
print("PHASE 2: ENGRAM TEST (ALL TOPICS)")
print("=" * 90)
print(f"Strength: {STRENGTH}x")

results = {}
for topic_name, topic_data in TOPICS.items():
    engram = extract_engram(model, tokenizer, topic_data["context"])

    baseline_correct = 0
    engram_correct = 0
    total = 0

    for q_data in topic_data["questions"]:
        base_ans = generate_baseline(model, tokenizer, q_data["q"])
        base_ok = check_answer(base_ans, q_data["keywords"])

        eng_ans = generate_with_engram(model, tokenizer, q_data["q"], engram, STRENGTH)
        eng_ok = check_answer(eng_ans, q_data["keywords"])

        if base_ok:
            baseline_correct += 1
        if eng_ok:
            engram_correct += 1
        total += 1

    base_acc = baseline_correct / total
    eng_acc = engram_correct / total
    delta = eng_acc - base_acc

    results[topic_name] = {
        "baseline": base_acc,
        "engram": eng_acc,
        "delta": delta,
        "is_high_baseline": topic_name in high_baseline
    }

# Summary
print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

print(f"\n{'Topic':<12} {'Baseline':<12} {'Engram':<12} {'Delta':<10} {'Category':<15}")
print("-" * 65)

for topic_name, data in results.items():
    category = "HIGH BASELINE" if data["is_high_baseline"] else "LOW BASELINE"
    print(f"{topic_name:<12} {data['baseline']:>6.0%}        {data['engram']:>6.0%}        {data['delta']:>+6.0%}     {category}")

# Aggregate by category
print("\n" + "-" * 65)

high_base_deltas = [d["delta"] for t, d in results.items() if d["is_high_baseline"]]
low_base_deltas = [d["delta"] for t, d in results.items() if not d["is_high_baseline"]]

if high_base_deltas:
    avg_high = sum(high_base_deltas) / len(high_base_deltas)
    print(f"{'HIGH BASELINE AVG':<12} {'-':<12} {'-':<12} {avg_high:>+6.0%}")

if low_base_deltas:
    avg_low = sum(low_base_deltas) / len(low_base_deltas)
    print(f"{'LOW BASELINE AVG':<12} {'-':<12} {'-':<12} {avg_low:>+6.0%}")

print("\n" + "=" * 90)
print("ANALYSIS")
print("=" * 90)

if high_base_deltas and low_base_deltas:
    avg_high = sum(high_base_deltas) / len(high_base_deltas)
    avg_low = sum(low_base_deltas) / len(low_base_deltas)

    if avg_high > avg_low + 0.05:
        print(f"""
HYPOTHESIS SUPPORTED: High Baseline > Low Baseline

High baseline topics: {avg_high:+.0%} average delta
Low baseline topics:  {avg_low:+.0%} average delta

Engrams work as ACTIVATION AMPLIFIERS:
- When the model already knows the answer (>= 75% baseline), the engram
  helps push borderline cases over the threshold
- When the model doesn't know (< 75% baseline), the engram adds noise

PRACTICAL IMPLICATION: Only use engrams when you know the model already
has the knowledge but sometimes fails to retrieve it.
""")
    elif abs(avg_high - avg_low) <= 0.05:
        print(f"""
NO SIGNIFICANT DIFFERENCE

High baseline: {avg_high:+.0%}
Low baseline:  {avg_low:+.0%}

The baseline accuracy doesn't predict engram effectiveness.
""")
    else:
        print(f"""
OPPOSITE PATTERN: Low > High

High baseline: {avg_high:+.0%}
Low baseline:  {avg_low:+.0%}

Unexpected - engrams help more when the model knows less?
""")
