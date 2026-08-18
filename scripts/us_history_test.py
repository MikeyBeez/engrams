#!/usr/bin/env python3
"""
US History Engram Test

Testing engrams on US History - a topic the model should know very well.
This should have high baseline accuracy, similar to WWII.

If baseline >= 75% AND engram improves it, we confirm the hypothesis:
Engrams only help when the model already knows the material.
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
# US HISTORY CONTEXT
# =============================================================================
US_HISTORY_CONTEXT = """
Key Events in United States History:

American Revolution and Founding:
- July 4, 1776: Declaration of Independence signed
- 1787: Constitution written at Philadelphia Convention
- 1789: George Washington became first President
- December 15, 1791: Bill of Rights ratified

19th Century:
- 1803: Louisiana Purchase from France doubled US territory
- 1861-1865: Civil War between Union and Confederacy
- April 14, 1865: Abraham Lincoln assassinated by John Wilkes Booth
- December 6, 1865: 13th Amendment abolished slavery

20th Century:
- April 6, 1917: US entered World War I
- October 29, 1929: Stock market crash, Black Tuesday
- December 7, 1941: Pearl Harbor attacked, US entered WWII
- August 28, 1963: Martin Luther King's "I Have a Dream" speech
- November 22, 1963: President Kennedy assassinated in Dallas
- July 20, 1969: Apollo 11 moon landing, Neil Armstrong
- August 9, 1974: Nixon resigned due to Watergate
"""

US_HISTORY_QUESTIONS = [
    {"q": "When was the Declaration of Independence signed?", "keywords": ["july 4", "1776"]},
    {"q": "Who was the first President of the United States?", "keywords": ["washington", "george"]},
    {"q": "When was the Bill of Rights ratified?", "keywords": ["1791", "december"]},
    {"q": "What doubled US territory in 1803?", "keywords": ["louisiana purchase", "louisiana"]},
    {"q": "When did the Civil War end?", "keywords": ["1865"]},
    {"q": "Who assassinated Abraham Lincoln?", "keywords": ["booth", "john wilkes"]},
    {"q": "What amendment abolished slavery?", "keywords": ["13th", "thirteenth"]},
    {"q": "When did the US enter World War I?", "keywords": ["1917", "april"]},
    {"q": "What date was Black Tuesday?", "keywords": ["october 29", "1929"]},
    {"q": "When was Pearl Harbor attacked?", "keywords": ["december 7", "1941"]},
    {"q": "Who gave the 'I Have a Dream' speech?", "keywords": ["king", "martin luther"]},
    {"q": "When was President Kennedy assassinated?", "keywords": ["november", "1963"]},
    {"q": "Who was the first person to walk on the moon?", "keywords": ["armstrong", "neil"]},
    {"q": "Why did Nixon resign?", "keywords": ["watergate"]},
    {"q": "When did Nixon resign?", "keywords": ["1974", "august"]},
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

print("\nExtracting US History engram...")
engram = extract_engram(model, tokenizer, US_HISTORY_CONTEXT)
print(f"Engram norm: {engram.norm().item():.2f}")

print("\n" + "=" * 90)
print("US HISTORY ENGRAM TEST")
print("=" * 90)
print(f"Hypothesis: If baseline >= 75%, engram should help")
print(f"Strength: {STRENGTH}x")
print(f"Questions: {len(US_HISTORY_QUESTIONS)}")
print()

# Track transitions
transitions = {"right_right": 0, "wrong_wrong": 0, "wrong_right": 0, "right_wrong": 0}
baseline_correct = 0
engram_correct = 0
total = 0

print("--- QUESTION-BY-QUESTION RESULTS ---", flush=True)
for q_data in US_HISTORY_QUESTIONS:
    question = q_data["q"]
    keywords = q_data["keywords"]

    base_ans = generate_baseline(model, tokenizer, question)
    base_ok = check_answer(base_ans, keywords)

    eng_ans = generate_with_engram(model, tokenizer, question, engram, STRENGTH)
    eng_ok = check_answer(eng_ans, keywords)

    total += 1
    if base_ok:
        baseline_correct += 1
    if eng_ok:
        engram_correct += 1

    if base_ok and eng_ok:
        transitions["right_right"] += 1
        transition = "✓→✓"
    elif not base_ok and not eng_ok:
        transitions["wrong_wrong"] += 1
        transition = "✗→✗"
    elif not base_ok and eng_ok:
        transitions["wrong_right"] += 1
        transition = "✗→✓ IMPROVED"
    else:
        transitions["right_wrong"] += 1
        transition = "✓→✗ DEGRADED"

    print(f"  {transition:<15} | {question[:50]}...", flush=True)

# Summary
print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

base_acc = baseline_correct / total
eng_acc = engram_correct / total
delta = eng_acc - base_acc

print(f"\nBaseline:  {base_acc:.0%} ({baseline_correct}/{total})")
print(f"Engram:    {eng_acc:.0%} ({engram_correct}/{total})")
print(f"Delta:     {delta:+.0%}")

print(f"\nTransitions:")
print(f"  ✓→✓ (stayed correct):   {transitions['right_right']}")
print(f"  ✗→✗ (stayed wrong):     {transitions['wrong_wrong']}")
print(f"  ✗→✓ (IMPROVED):         {transitions['wrong_right']}")
print(f"  ✓→✗ (DEGRADED):         {transitions['right_wrong']}")

net = transitions["wrong_right"] - transitions["right_wrong"]
print(f"  Net change:             {net:+d}")

print("\n" + "=" * 90)
print("ANALYSIS")
print("=" * 90)

if base_acc >= 0.75:
    if delta > 0:
        print(f"""
HYPOTHESIS CONFIRMED!

Baseline >= 75% ({base_acc:.0%}) AND engram improved ({delta:+.0%})

US History follows the same pattern as WWII:
- Model already knows the material well
- Engram acts as activation amplifier
- Borderline questions get pushed to correct

This supports the rule: Only use engrams when baseline >= 75%
""")
    elif delta < -0.1:
        print(f"""
HYPOTHESIS PARTIALLY REFUTED

Baseline >= 75% ({base_acc:.0%}) but engram HURT ({delta:+.0%})

Even with high baseline, engram degraded performance.
The 75% threshold alone may not be sufficient.
""")
    else:
        print(f"""
HIGH BASELINE, NEUTRAL EFFECT

Baseline >= 75% ({base_acc:.0%}), engram effect: {delta:+.0%}

The engram didn't significantly help or hurt.
Model already performing near ceiling.
""")
else:
    print(f"""
LOW BASELINE ({base_acc:.0%})

Model doesn't know US history well enough (< 75%).
This is unexpected - US history should be well-known.

Engram effect: {delta:+.0%}
""")
