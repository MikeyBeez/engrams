#!/usr/bin/env python3
"""
Baseline Uncertainty Analysis

New hypothesis: Engrams help when the model is UNCERTAIN at baseline.
- WWII: Model knows these facts well (high baseline) → engram activates right answers
- Chemistry history: Model doesn't know well (low baseline) → engram adds noise

Test: Analyze the relationship between baseline correctness and engram effect.
Do engrams help flip wrong→right or do they flip right→wrong?

Using existing results from all tests.
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
# WWII CONTEXT (high baseline accuracy expected)
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
# OBSCURE FACTS (low baseline accuracy expected)
# =============================================================================
OBSCURE_CONTEXT = """
Lesser-known historical events and discoveries:

Timeline of obscure events:
- 1669: Hennig Brand discovered phosphorus while searching for the philosopher's stone in Hamburg
- 1783: The Montgolfier brothers launched the first hot air balloon flight in Annonay, France
- 1815: Mount Tambora erupted in Indonesia, causing the "Year Without a Summer" in 1816
- 1859: The Carrington Event, the largest recorded geomagnetic storm, occurred on September 1
- 1908: The Tunguska event flattened 2,000 km² of Siberian forest on June 30
- 1930: Clyde Tombaugh discovered Pluto at Lowell Observatory on February 18
- 1947: Chuck Yeager broke the sound barrier on October 14 in the Bell X-1
- 1960: The deepest ocean dive reached 10,916 meters in the Mariana Trench on January 23
"""

OBSCURE_QUESTIONS = [
    {"q": "Who discovered phosphorus?", "keywords": ["hennig brand", "brand"]},
    {"q": "When did the Montgolfier brothers launch the first hot air balloon?", "keywords": ["1783"]},
    {"q": "What volcano caused the Year Without a Summer?", "keywords": ["tambora"]},
    {"q": "When did the Carrington Event occur?", "keywords": ["1859", "september"]},
    {"q": "When was the Tunguska event?", "keywords": ["1908", "june 30"]},
    {"q": "Who discovered Pluto?", "keywords": ["tombaugh", "clyde"]},
    {"q": "When did Chuck Yeager break the sound barrier?", "keywords": ["1947", "october"]},
    {"q": "How deep was the deepest ocean dive in 1960?", "keywords": ["10,916", "10916", "35,814", "mariana"]},
    {"q": "Where was phosphorus discovered?", "keywords": ["hamburg"]},
    {"q": "What was the name of the aircraft that broke the sound barrier?", "keywords": ["bell x-1", "x-1"]},
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

print("\nExtracting engrams...")
wwii_engram = extract_engram(model, tokenizer, WWII_CONTEXT)
obscure_engram = extract_engram(model, tokenizer, OBSCURE_CONTEXT)

print("\n" + "=" * 90)
print("BASELINE UNCERTAINTY TEST")
print("=" * 90)
print(f"Hypothesis: Engrams help when model already knows the answer (activates existing knowledge)")
print(f"            Engrams hurt when model doesn't know (adds noise)")
print()

# Track transitions
transitions = {
    "wwii": {"right_right": 0, "wrong_wrong": 0, "wrong_right": 0, "right_wrong": 0},
    "obscure": {"right_right": 0, "wrong_wrong": 0, "wrong_right": 0, "right_wrong": 0},
}

print("\n--- WWII (Well-known facts) ---", flush=True)
for q_data in WWII_QUESTIONS:
    question = q_data["q"]
    keywords = q_data["keywords"]

    base_ans = generate_baseline(model, tokenizer, question)
    base_correct = check_answer(base_ans, keywords)

    eng_ans = generate_with_engram(model, tokenizer, question, wwii_engram, STRENGTH)
    eng_correct = check_answer(eng_ans, keywords)

    if base_correct and eng_correct:
        transitions["wwii"]["right_right"] += 1
        transition = "✓→✓"
    elif not base_correct and not eng_correct:
        transitions["wwii"]["wrong_wrong"] += 1
        transition = "✗→✗"
    elif not base_correct and eng_correct:
        transitions["wwii"]["wrong_right"] += 1
        transition = "✗→✓ IMPROVED"
    else:
        transitions["wwii"]["right_wrong"] += 1
        transition = "✓→✗ DEGRADED"

    print(f"  {transition:<15} | {question[:50]}...", flush=True)

print("\n--- OBSCURE (Lesser-known facts) ---", flush=True)
for q_data in OBSCURE_QUESTIONS:
    question = q_data["q"]
    keywords = q_data["keywords"]

    base_ans = generate_baseline(model, tokenizer, question)
    base_correct = check_answer(base_ans, keywords)

    eng_ans = generate_with_engram(model, tokenizer, question, obscure_engram, STRENGTH)
    eng_correct = check_answer(eng_ans, keywords)

    if base_correct and eng_correct:
        transitions["obscure"]["right_right"] += 1
        transition = "✓→✓"
    elif not base_correct and not eng_correct:
        transitions["obscure"]["wrong_wrong"] += 1
        transition = "✗→✗"
    elif not base_correct and eng_correct:
        transitions["obscure"]["wrong_right"] += 1
        transition = "✗→✓ IMPROVED"
    else:
        transitions["obscure"]["right_wrong"] += 1
        transition = "✓→✗ DEGRADED"

    print(f"  {transition:<15} | {question[:50]}...", flush=True)

# Summary
print("\n" + "=" * 90)
print("TRANSITION ANALYSIS")
print("=" * 90)

print(f"\n{'Topic':<12} {'✓→✓':<8} {'✗→✗':<8} {'✗→✓':<10} {'✓→✗':<10} {'Net':<8}")
print("-" * 60)

for topic in ["wwii", "obscure"]:
    t = transitions[topic]
    total = sum(t.values())
    net = t["wrong_right"] - t["right_wrong"]
    net_str = f"+{net}" if net > 0 else str(net)

    print(f"{topic.upper():<12} {t['right_right']:<8} {t['wrong_wrong']:<8} "
          f"+{t['wrong_right']:<9} -{t['right_wrong']:<9} {net_str:<8}")

wwii_t = transitions["wwii"]
obscure_t = transitions["obscure"]

wwii_baseline_acc = (wwii_t["right_right"] + wwii_t["right_wrong"]) / 10
obscure_baseline_acc = (obscure_t["right_right"] + obscure_t["right_wrong"]) / 10

print(f"\nBaseline accuracy:")
print(f"  WWII: {wwii_baseline_acc:.0%}")
print(f"  Obscure: {obscure_baseline_acc:.0%}")

print("\n" + "=" * 90)
print("ANALYSIS")
print("=" * 90)

wwii_net = wwii_t["wrong_right"] - wwii_t["right_wrong"]
obscure_net = obscure_t["wrong_right"] - obscure_t["right_wrong"]

if wwii_net > 0 and obscure_net <= 0:
    print(f"""
HYPOTHESIS SUPPORTED: Well-known > Obscure

WWII (well-known):   Net {wwii_net:+d} (baseline {wwii_baseline_acc:.0%})
Obscure (unknown):   Net {obscure_net:+d} (baseline {obscure_baseline_acc:.0%})

Engrams appear to ACTIVATE existing knowledge when the model already has it.
When the model doesn't know the facts, the engram can't provide them - it only
adds noise to an already uncertain generation.

Key insight: Engrams are not RAG (they don't inject new knowledge).
They're more like "priming" that helps the model access what it already knows.
""")
elif wwii_net > 0 and obscure_net > 0:
    print(f"""
ENGRAMS HELP BOTH

WWII:    Net {wwii_net:+d}
Obscure: Net {obscure_net:+d}

Both well-known and obscure facts improved. This suggests engrams CAN
inject new knowledge from the context, at least partially.
""")
else:
    print(f"""
MIXED OR NEGATIVE RESULTS

WWII:    Net {wwii_net:+d}
Obscure: Net {obscure_net:+d}

Further investigation needed.
""")
