#!/usr/bin/env python3
"""
Narrative vs Procedural Knowledge Test

Hypothesis: Engrams work better for narrative/temporal knowledge
(sequences of events) than for relational/procedural knowledge
(mappings, properties, comparisons).

Test:
- 2 narrative topics (WWII, American Revolution)
- 2 procedural topics (Chemistry symbols, Math constants)
- Compare engram effects

If narrative >> procedural, the hypothesis is supported.
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
# NARRATIVE TOPIC 1: WORLD WAR II (from previous test - worked well)
# =============================================================================
WWII_CONTEXT = """
World War II (1939-1945) was the deadliest conflict in human history.

Timeline of key events:
- September 1, 1939: Germany invaded Poland, starting the war
- June 1940: France fell to Nazi Germany
- June 22, 1941: Germany invaded the Soviet Union (Operation Barbarossa)
- December 7, 1941: Japan attacked Pearl Harbor, US entered the war
- February 1943: German defeat at Stalingrad, turning point
- June 6, 1944: D-Day, Allied invasion of Normandy
- April 30, 1945: Hitler committed suicide in Berlin
- May 8, 1945: V-E Day, Germany surrendered
- August 6, 1945: Atomic bomb dropped on Hiroshima
- August 15, 1945: V-J Day, Japan surrendered

Key figures: Hitler, Churchill, Roosevelt, Stalin, Mussolini
"""

WWII_QUESTIONS = [
    {"q": "When did Germany invade Poland?", "keywords": ["september", "1939"]},
    {"q": "When did Japan attack Pearl Harbor?", "keywords": ["december", "1941"]},
    {"q": "What was the date of D-Day?", "keywords": ["june 6", "1944"]},
    {"q": "When did Germany surrender?", "keywords": ["may 8", "1945", "v-e"]},
    {"q": "When was the atomic bomb dropped on Hiroshima?", "keywords": ["august 6", "1945"]},
    {"q": "What battle was the turning point on the Eastern Front?", "keywords": ["stalingrad"]},
    {"q": "When did Hitler die?", "keywords": ["april", "1945"]},
    {"q": "When did the US enter World War II?", "keywords": ["1941", "december", "pearl harbor"]},
    {"q": "What was Operation Barbarossa?", "keywords": ["invasion", "soviet", "russia", "1941"]},
    {"q": "When did France fall to Nazi Germany?", "keywords": ["1940", "june"]},
]

# =============================================================================
# NARRATIVE TOPIC 2: AMERICAN REVOLUTION
# =============================================================================
REVOLUTION_CONTEXT = """
The American Revolution (1765-1783) was the colonial revolt against British rule.

Timeline of key events:
- 1765: Stamp Act passed, colonists protested "taxation without representation"
- March 5, 1770: Boston Massacre, British soldiers killed 5 colonists
- December 16, 1773: Boston Tea Party, colonists dumped tea into harbor
- April 19, 1775: Battles of Lexington and Concord, first shots fired
- July 4, 1776: Declaration of Independence signed
- December 26, 1776: Washington crossed the Delaware, attacked Trenton
- October 17, 1777: British surrendered at Saratoga, turning point
- Winter 1777-78: Continental Army suffered at Valley Forge
- October 19, 1781: British surrendered at Yorktown, effectively ending the war
- September 3, 1783: Treaty of Paris signed, war officially ended

Key figures: George Washington, Thomas Jefferson, Benjamin Franklin, King George III
"""

REVOLUTION_QUESTIONS = [
    {"q": "When was the Declaration of Independence signed?", "keywords": ["july 4", "1776"]},
    {"q": "When did the Boston Tea Party occur?", "keywords": ["december", "1773"]},
    {"q": "When were the first shots of the Revolution fired?", "keywords": ["april", "1775", "lexington", "concord"]},
    {"q": "What battle was the turning point of the American Revolution?", "keywords": ["saratoga"]},
    {"q": "When did the British surrender at Yorktown?", "keywords": ["october", "1781"]},
    {"q": "When was the Treaty of Paris signed?", "keywords": ["1783", "september"]},
    {"q": "When did Washington cross the Delaware?", "keywords": ["december", "1776"]},
    {"q": "What happened on March 5, 1770?", "keywords": ["boston massacre"]},
    {"q": "Where did the Continental Army suffer during winter 1777-78?", "keywords": ["valley forge"]},
    {"q": "What act in 1765 sparked protests about taxation?", "keywords": ["stamp act"]},
]

# =============================================================================
# PROCEDURAL TOPIC 1: CHEMISTRY SYMBOLS (mappings)
# =============================================================================
CHEMISTRY_CONTEXT = """
Chemical element symbols are abbreviations from Latin or English names.

Element symbol mappings:
- Gold: Au (from Latin "aurum")
- Silver: Ag (from Latin "argentum")
- Iron: Fe (from Latin "ferrum")
- Copper: Cu (from Latin "cuprum")
- Lead: Pb (from Latin "plumbum")
- Sodium: Na (from Latin "natrium")
- Potassium: K (from Latin "kalium")
- Mercury: Hg (from Latin "hydrargyrum")
- Tin: Sn (from Latin "stannum")
- Tungsten: W (from German "wolfram")

Atomic numbers:
- Hydrogen: 1, Helium: 2, Carbon: 6, Nitrogen: 7, Oxygen: 8
- Gold: 79, Silver: 47, Iron: 26, Copper: 29, Lead: 82
"""

CHEMISTRY_QUESTIONS = [
    {"q": "What is the chemical symbol for gold?", "keywords": ["au"]},
    {"q": "What is the chemical symbol for silver?", "keywords": ["ag"]},
    {"q": "What is the chemical symbol for iron?", "keywords": ["fe"]},
    {"q": "What is the chemical symbol for lead?", "keywords": ["pb"]},
    {"q": "What is the chemical symbol for sodium?", "keywords": ["na"]},
    {"q": "What is the chemical symbol for potassium?", "keywords": ["k"]},
    {"q": "What is the atomic number of gold?", "keywords": ["79"]},
    {"q": "What is the atomic number of carbon?", "keywords": ["6", "six"]},
    {"q": "What Latin word does Au come from?", "keywords": ["aurum"]},
    {"q": "What Latin word does Fe come from?", "keywords": ["ferrum"]},
]

# =============================================================================
# PROCEDURAL TOPIC 2: MATH CONSTANTS (properties/values)
# =============================================================================
MATH_CONTEXT = """
Mathematical constants are fixed values with special significance.

Key constants and their values:
- Pi (π): 3.14159... ratio of circumference to diameter
- Euler's number (e): 2.71828... base of natural logarithm
- Golden ratio (φ): 1.61803... appears in nature and art
- Square root of 2: 1.41421... diagonal of unit square
- Euler-Mascheroni constant (γ): 0.57721...

Properties:
- Pi is irrational and transcendental
- e^(iπ) + 1 = 0 (Euler's identity)
- φ = (1 + √5)/2
- √2 was the first known irrational number (discovered by Pythagoreans)
- ln(2) ≈ 0.693
- log₁₀(2) ≈ 0.301
"""

MATH_QUESTIONS = [
    {"q": "What are the first 5 digits of pi?", "keywords": ["3.1415", "3.14159"]},
    {"q": "What is Euler's number approximately?", "keywords": ["2.718", "2.71828"]},
    {"q": "What is the golden ratio approximately?", "keywords": ["1.618", "1.61803"]},
    {"q": "What is the square root of 2 approximately?", "keywords": ["1.414", "1.41421"]},
    {"q": "What is Euler's identity equation?", "keywords": ["e^(iπ)", "e^iπ", "+ 1 = 0"]},
    {"q": "What is pi the ratio of?", "keywords": ["circumference", "diameter"]},
    {"q": "What is e the base of?", "keywords": ["natural logarithm", "ln"]},
    {"q": "What is the formula for the golden ratio?", "keywords": ["1 + √5", "sqrt(5)", "√5"]},
    {"q": "What ancient group discovered that √2 is irrational?", "keywords": ["pythagorean"]},
    {"q": "What is ln(2) approximately?", "keywords": ["0.693", ".693"]},
]

# =============================================================================
# COMPILE TOPICS
# =============================================================================
TOPICS = {
    "wwii": {"context": WWII_CONTEXT, "questions": WWII_QUESTIONS, "type": "narrative"},
    "revolution": {"context": REVOLUTION_CONTEXT, "questions": REVOLUTION_QUESTIONS, "type": "narrative"},
    "chemistry": {"context": CHEMISTRY_CONTEXT, "questions": CHEMISTRY_QUESTIONS, "type": "procedural"},
    "math": {"context": MATH_CONTEXT, "questions": MATH_QUESTIONS, "type": "procedural"},
}

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

print("\nExtracting engrams...")
engrams = {}
for topic_name, topic_data in TOPICS.items():
    engrams[topic_name] = extract_engram(model, tokenizer, topic_data["context"])
    print(f"  {topic_name}: norm = {engrams[topic_name].norm().item():.2f}")

print("\n" + "=" * 90)
print("NARRATIVE vs PROCEDURAL KNOWLEDGE TEST")
print("=" * 90)
print(f"Hypothesis: Engrams help narrative (timeline) knowledge more than procedural (mapping) knowledge")
print(f"Strength: {STRENGTH}x")
print()

results = {}

for topic_name, topic_data in TOPICS.items():
    results[topic_name] = {"baseline": 0, "engram": 0, "total": 0, "type": topic_data["type"]}
    engram = engrams[topic_name]

    print(f"\n--- {topic_name.upper()} ({topic_data['type']}) ---", flush=True)

    for q_data in topic_data["questions"]:
        question = q_data["q"]
        keywords = q_data["keywords"]

        base_ans = generate_baseline(model, tokenizer, question)
        base_correct = check_answer(base_ans, keywords)

        eng_ans = generate_with_engram(model, tokenizer, question, engram, STRENGTH)
        eng_correct = check_answer(eng_ans, keywords)

        results[topic_name]["total"] += 1
        if base_correct:
            results[topic_name]["baseline"] += 1
        if eng_correct:
            results[topic_name]["engram"] += 1

        b = "✓" if base_correct else "✗"
        e = "✓" if eng_correct else "✗"
        print(f"  Base={b} Engram={e} | {question[:55]}...", flush=True)

# Summary
print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

print(f"\n{'Topic':<15} {'Type':<12} {'Baseline':<12} {'Engram':<12} {'Delta':<12}")
print("-" * 65)

narrative_base = 0
narrative_eng = 0
narrative_total = 0
procedural_base = 0
procedural_eng = 0
procedural_total = 0

for topic_name, data in results.items():
    base_acc = data["baseline"] / data["total"]
    eng_acc = data["engram"] / data["total"]
    delta = eng_acc - base_acc

    print(f"{topic_name:<15} {data['type']:<12} {base_acc:>6.0%} ({data['baseline']}/{data['total']})  "
          f"{eng_acc:>6.0%} ({data['engram']}/{data['total']})  {delta:>+6.0%}")

    if data["type"] == "narrative":
        narrative_base += data["baseline"]
        narrative_eng += data["engram"]
        narrative_total += data["total"]
    else:
        procedural_base += data["baseline"]
        procedural_eng += data["engram"]
        procedural_total += data["total"]

print("-" * 65)

narr_base_acc = narrative_base / narrative_total
narr_eng_acc = narrative_eng / narrative_total
proc_base_acc = procedural_base / procedural_total
proc_eng_acc = procedural_eng / procedural_total

print(f"{'NARRATIVE':<15} {'(combined)':<12} {narr_base_acc:>6.0%} ({narrative_base}/{narrative_total})  "
      f"{narr_eng_acc:>6.0%} ({narrative_eng}/{narrative_total})  {narr_eng_acc - narr_base_acc:>+6.0%}")
print(f"{'PROCEDURAL':<15} {'(combined)':<12} {proc_base_acc:>6.0%} ({procedural_base}/{procedural_total})  "
      f"{proc_eng_acc:>6.0%} ({procedural_eng}/{procedural_total})  {proc_eng_acc - proc_base_acc:>+6.0%}")

print("\n" + "=" * 90)
print("ANALYSIS")
print("=" * 90)

narr_delta = narr_eng_acc - narr_base_acc
proc_delta = proc_eng_acc - proc_base_acc

if narr_delta > proc_delta + 0.1:
    print(f"""
HYPOTHESIS SUPPORTED: Narrative > Procedural

Narrative topics: {narr_delta:+.0%} change with engram
Procedural topics: {proc_delta:+.0%} change with engram
Difference: {narr_delta - proc_delta:+.0%}

Engrams appear to work better for timeline/event-based knowledge
than for mapping/property-based knowledge. This suggests engrams
may activate narrative structures rather than arbitrary associations.
""")
elif abs(narr_delta - proc_delta) < 0.1:
    print(f"""
NO SIGNIFICANT DIFFERENCE

Narrative topics: {narr_delta:+.0%}
Procedural topics: {proc_delta:+.0%}

The knowledge type doesn't seem to be the distinguishing factor.
""")
else:
    print(f"""
OPPOSITE PATTERN: Procedural > Narrative

Narrative topics: {narr_delta:+.0%}
Procedural topics: {proc_delta:+.0%}

Unexpected result - needs investigation.
""")
