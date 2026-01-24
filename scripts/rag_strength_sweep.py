#!/usr/bin/env python3
"""
RAG Strength Sweep

The original RAG+engram experiments used 1x strength scaling.
The decision steering experiments used 5-50x.

Question: Can we improve the 10% RAG enhancement by scaling up the engram?

This tests known facts (WWII) where the model already has the knowledge
but we're trying to make it more accessible.
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

# WWII facts - known information the model should have
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

# Questions about WWII
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

STRENGTHS = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]


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

    # Decode only the new tokens
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract just the answer part
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

print("\nExtracting WWII engram from layer 16...")
engram = extract_engram(model, tokenizer, WWII_CONTEXT, layer_idx=16)
print(f"Engram shape: {engram.shape}")

print("\n" + "=" * 90)
print("RAG STRENGTH SWEEP - WWII KNOWLEDGE")
print("=" * 90)
print("\nTesting if scaling up engrams improves fact recall on known information.\n")

# Results storage
results = {s: {"correct": 0, "total": 0} for s in STRENGTHS}
results["baseline"] = {"correct": 0, "total": 0}

# Test each question
for q in QUESTIONS:
    print(f"\nQ: {q['question']}")
    print(f"Expected: {q['answer']}")
    print("-" * 70)

    # Baseline
    baseline_ans = generate_baseline(model, tokenizer, q['question'])
    baseline_correct = check_answer(baseline_ans, q['keywords'])
    results["baseline"]["total"] += 1
    if baseline_correct:
        results["baseline"]["correct"] += 1

    print(f"{'Baseline':<12} {'✓' if baseline_correct else '✗':<3} {baseline_ans[:60]}...")

    # Each strength
    for strength in STRENGTHS:
        ans = generate_with_engram(model, tokenizer, q['question'], engram, strength)
        correct = check_answer(ans, q['keywords'])
        results[strength]["total"] += 1
        if correct:
            results[strength]["correct"] += 1

        marker = "✓" if correct else "✗"
        improved = "↑" if correct and not baseline_correct else ""
        degraded = "↓" if not correct and baseline_correct else ""
        print(f"Str {strength:<6} {marker:<3} {improved}{degraded} {ans[:55]}...")

# Summary
print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

print(f"\n{'Condition':<15} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
print("-" * 45)

baseline_acc = results["baseline"]["correct"] / results["baseline"]["total"]
print(f"{'Baseline':<15} {results['baseline']['correct']:<10} {results['baseline']['total']:<10} {baseline_acc:.1%}")

best_strength = None
best_acc = baseline_acc

for strength in STRENGTHS:
    acc = results[strength]["correct"] / results[strength]["total"]
    delta = acc - baseline_acc
    delta_str = f"({'+' if delta >= 0 else ''}{delta:.1%})"
    print(f"{'Strength ' + str(strength):<15} {results[strength]['correct']:<10} {results[strength]['total']:<10} {acc:.1%} {delta_str}")

    if acc > best_acc:
        best_acc = acc
        best_strength = strength

print("\n" + "=" * 90)
print("ANALYSIS")
print("=" * 90)

if best_strength:
    improvement = best_acc - baseline_acc
    print(f"""
Best strength: {best_strength}x
Best accuracy: {best_acc:.1%}
Improvement over baseline: +{improvement:.1%}

This suggests that for semantic priming (making known facts more accessible),
strength {best_strength}x works better than the default 1x scaling.
""")
else:
    print(f"""
No strength improved over baseline ({baseline_acc:.1%}).

Possible interpretations:
1. The model already knows these facts well - engram adds noise
2. Higher strength disrupts coherence without helping retrieval
3. Semantic priming works differently than decision steering
""")

print("\nNote: This is a small sample (8 questions). Larger tests needed to confirm.")
