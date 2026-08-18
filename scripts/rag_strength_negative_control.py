#!/usr/bin/env python3
"""
RAG Strength Negative Control

The RAG strength sweep found a sweet spot at 3x where accuracy hit 100%.
But does the CONTENT of the engram matter at this strength?

The decision steering negative control (at 5-50x) found NO domain specificity.
This tests whether the lower-strength semantic priming regime is different.

If relevant > irrelevant > random: domain specificity confirmed
If all similar: it's just activation perturbation, not semantic steering
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from huggingface_hub import login

token = os.environ.get("HF_TOKEN")
if not token:
    try:
        token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
        login(token=token, add_to_git_credential=False)
    except:
        pass

# RELEVANT engram source - WWII knowledge
WWII_CONTEXT = """
World War II was the deadliest conflict in human history. Key facts:

- The war lasted from 1939 to 1945
- Germany invaded Poland on September 1, 1939, triggering declarations of war
- The Allied Powers included the United States, United Kingdom, Soviet Union, and France
- The Axis Powers included Germany, Italy, and Japan
- D-Day (Operation Overlord) occurred on June 6, 1944
- The war in Europe ended on May 8, 1945 (V-E Day)
- The atomic bombs were dropped on Hiroshima (August 6, 1945) and Nagasaki (August 9, 1945)
- Japan surrendered on August 15, 1945 (V-J Day)
- An estimated 70-85 million people died, including about 6 million Jews in the Holocaust
- Winston Churchill was Prime Minister of the United Kingdom
- Franklin D. Roosevelt and then Harry S. Truman were US Presidents during the war
- Adolf Hitler was the leader of Nazi Germany
- The Battle of Stalingrad (1942-1943) was a major turning point on the Eastern Front
- The Battle of Midway (June 1942) was a turning point in the Pacific
"""

# IRRELEVANT engram source - astronomy (same length roughly)
ASTRONOMY_CONTEXT = """
Stellar classification is fundamental to understanding stars. Key facts:

- The spectral sequence is O B A F G K M, from hottest to coolest
- O-type stars are the hottest, with surface temperatures above 30,000 K
- B-type stars are blue-white giants with temperatures of 10,000-30,000 K
- A-type stars like Sirius are white with strong hydrogen lines
- F-type stars are yellow-white with temperatures of 6,000-7,500 K
- G-type stars like our Sun have surface temperatures of 5,200-6,000 K
- K-type stars are orange dwarfs, cooler than G-type at 3,700-5,200 K
- M-type stars are red dwarfs, the most common type in the galaxy
- The Hertzsprung-Russell diagram plots luminosity against temperature
- Main sequence stars fuse hydrogen in their cores
- Red giants have exhausted core hydrogen and expanded
- White dwarfs are stellar remnants with no fusion
- Neutron stars are collapsed cores of massive stars
- Black holes form from the most massive stellar collapses
"""

# RANDOM engram source - meaningless word combinations
import random
random.seed(42)  # Reproducible
words = ["quantum", "nebula", "crystal", "vortex", "paradigm", "synergy",
         "matrix", "vector", "tensor", "gradient", "manifold", "topology",
         "entropy", "catalyst", "synthesis", "harmonic", "resonance", "flux",
         "orbital", "cascade", "fractal", "holistic", "transcendent", "ephemeral"]
lines = []
for _ in range(14):
    line = " ".join(random.choices(words, k=random.randint(8, 15)))
    lines.append(line)
RANDOM_CONTEXT = "\n".join(lines)

# WWII questions
QUESTIONS = [
    {
        "question": "When did Germany invade Poland to start World War II?",
        "answer": "September 1, 1939",
        "keywords": ["september", "1939", "1, 1939"]
    },
    {
        "question": "What was the date of D-Day?",
        "answer": "June 6, 1944",
        "keywords": ["june 6", "1944", "6, 1944"]
    },
    {
        "question": "When did World War II end in Europe?",
        "answer": "May 8, 1945",
        "keywords": ["may 8", "1945", "v-e day"]
    },
    {
        "question": "On what city was the first atomic bomb dropped?",
        "answer": "Hiroshima",
        "keywords": ["hiroshima"]
    },
    {
        "question": "Who was the British Prime Minister during most of WWII?",
        "answer": "Winston Churchill",
        "keywords": ["churchill", "winston"]
    },
    {
        "question": "What battle was a turning point on the Eastern Front?",
        "answer": "Battle of Stalingrad",
        "keywords": ["stalingrad"]
    },
    {
        "question": "How many Jews were killed in the Holocaust?",
        "answer": "About 6 million",
        "keywords": ["6 million", "six million"]
    },
    {
        "question": "What was Operation Overlord?",
        "answer": "D-Day / Normandy invasion",
        "keywords": ["d-day", "normandy", "invasion", "june 6"]
    },
]

# Test at the sweet spot AND comparison points
STRENGTHS = [1.0, 2.0, 3.0, 5.0]


def extract_engram(model, tokenizer, text, layer_idx=16, num_tokens=32):
    """Extract engram from text."""
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
    """Generate answer with engram at specified strength."""
    embed = model.get_input_embeddings()

    # Scale engram
    e_norm = embed.weight.norm(dim=1).mean().item()
    g_norm = engram.norm(dim=1).mean().item()
    scaled = engram * (e_norm / g_norm) * strength

    # Build prompt
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    emb = embed(inputs.input_ids)

    # Combine
    combined = torch.cat([scaled.unsqueeze(0).to(emb.dtype), emb], dim=1)

    with torch.no_grad():
        output = model.generate(
            inputs_embeds=combined,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    return response


def generate_baseline(model, tokenizer, question):
    """Generate answer without engram."""
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    return response


def check_answer(response, keywords):
    """Check if response contains any of the keywords."""
    response_lower = response.lower()
    return any(kw.lower() in response_lower for kw in keywords)


print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    torch_dtype=torch.float16,
    device_map="auto"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\nExtracting engrams from layer 16...")
wwii_engram = extract_engram(model, tokenizer, WWII_CONTEXT, layer_idx=16)
astronomy_engram = extract_engram(model, tokenizer, ASTRONOMY_CONTEXT, layer_idx=16)
random_engram = extract_engram(model, tokenizer, RANDOM_CONTEXT, layer_idx=16)

print(f"WWII engram norm: {wwii_engram.norm().item():.2f}")
print(f"Astronomy engram norm: {astronomy_engram.norm().item():.2f}")
print(f"Random engram norm: {random_engram.norm().item():.2f}")

print("\n" + "=" * 95)
print("NEGATIVE CONTROL: RAG STRENGTH SWEET SPOT")
print("=" * 95)
print("\nQuestion: Does the engram CONTENT matter at 3x strength, or is it just perturbation?")
print("Comparing: WWII (relevant) vs Astronomy (irrelevant) vs Random (noise)")
print()

# Results storage
results = {
    "baseline": {"correct": 0, "total": 0},
    "wwii": {s: {"correct": 0, "total": 0} for s in STRENGTHS},
    "astronomy": {s: {"correct": 0, "total": 0} for s in STRENGTHS},
    "random": {s: {"correct": 0, "total": 0} for s in STRENGTHS},
}

# Test each question
for q in QUESTIONS:
    print(f"\nQ: {q['question']}")
    print(f"Expected: {q['answer']}")
    print("-" * 80)

    # Baseline
    baseline_ans = generate_baseline(model, tokenizer, q['question'])
    baseline_correct = check_answer(baseline_ans, q['keywords'])
    results["baseline"]["total"] += 1
    if baseline_correct:
        results["baseline"]["correct"] += 1

    print(f"{'Baseline':<20} {'✓' if baseline_correct else '✗':<3} {baseline_ans[:55]}...")

    # Each strength
    for strength in STRENGTHS:
        # WWII (relevant)
        wwii_ans = generate_with_engram(model, tokenizer, q['question'], wwii_engram, strength)
        wwii_correct = check_answer(wwii_ans, q['keywords'])
        results["wwii"][strength]["total"] += 1
        if wwii_correct:
            results["wwii"][strength]["correct"] += 1

        # Astronomy (irrelevant)
        astro_ans = generate_with_engram(model, tokenizer, q['question'], astronomy_engram, strength)
        astro_correct = check_answer(astro_ans, q['keywords'])
        results["astronomy"][strength]["total"] += 1
        if astro_correct:
            results["astronomy"][strength]["correct"] += 1

        # Random
        rand_ans = generate_with_engram(model, tokenizer, q['question'], random_engram, strength)
        rand_correct = check_answer(rand_ans, q['keywords'])
        results["random"][strength]["total"] += 1
        if rand_correct:
            results["random"][strength]["correct"] += 1

        # Print comparison for this strength
        print(f"Str {strength}x: WWII={'✓' if wwii_correct else '✗'} | "
              f"Astro={'✓' if astro_correct else '✗'} | "
              f"Random={'✓' if rand_correct else '✗'}")

# Summary
print("\n" + "=" * 95)
print("SUMMARY")
print("=" * 95)

baseline_acc = results["baseline"]["correct"] / results["baseline"]["total"]
print(f"\n{'Condition':<25} {'1.0x':<12} {'2.0x':<12} {'3.0x':<12} {'5.0x':<12}")
print("-" * 75)

print(f"{'Baseline':<25} {baseline_acc:.1%}")

for engram_type in ["wwii", "astronomy", "random"]:
    row = f"{engram_type.upper():<25}"
    for strength in STRENGTHS:
        acc = results[engram_type][strength]["correct"] / results[engram_type][strength]["total"]
        delta = acc - baseline_acc
        delta_str = f"{acc:.1%} ({'+' if delta >= 0 else ''}{delta:+.1%})"
        row += f" {delta_str:<12}"
    print(row)

# Critical analysis at 3x
print("\n" + "=" * 95)
print("CRITICAL ANALYSIS: 3x STRENGTH")
print("=" * 95)

wwii_3x = results["wwii"][3.0]["correct"] / results["wwii"][3.0]["total"]
astro_3x = results["astronomy"][3.0]["correct"] / results["astronomy"][3.0]["total"]
rand_3x = results["random"][3.0]["correct"] / results["random"][3.0]["total"]

print(f"\nAt 3x strength (sweet spot):")
print(f"  WWII (relevant):     {wwii_3x:.1%}")
print(f"  Astronomy (irrelevant): {astro_3x:.1%}")
print(f"  Random (noise):      {rand_3x:.1%}")

if wwii_3x > astro_3x + 0.1 and wwii_3x > rand_3x + 0.1:
    print(f"""
DOMAIN SPECIFICITY CONFIRMED at 3x

The relevant WWII engram significantly outperforms irrelevant and random engrams.
This suggests the sweet spot regime operates differently than high-strength (5-50x):
- At 3x: semantic content matters
- At 5-50x: perturbation effects dominate

The 100% accuracy at 3x with WWII engram appears to be genuine semantic priming,
not just activation noise.
""")
elif abs(wwii_3x - astro_3x) < 0.15 and abs(wwii_3x - rand_3x) < 0.15:
    print(f"""
NO DOMAIN SPECIFICITY at 3x

All engram types produce similar results at 3x strength.
This suggests even at the sweet spot, the mechanism is:
- General activation perturbation, not semantic content
- The model just needs "some" priming signal
- Content doesn't matter, only the injection

This would significantly temper claims about semantic priming.
""")
else:
    print(f"""
MIXED RESULTS at 3x

Some domain specificity detected but not dramatic.
Further investigation needed with larger sample sizes.
""")

print("\nNote: 8 questions is a small sample. Consider running on more questions.")
