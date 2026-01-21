#!/usr/bin/env python3
"""
Engram + RAG Combined Test

Tests whether combining engram with RAG helps, hurts, or is neutral.

Four conditions:
1. Baseline - no context, no engram
2. Engram only - engram prepended, no RAG context
3. RAG only - full context in prompt, no engram
4. Engram + RAG - both engram AND RAG context

Hypothesis:
- If engram+RAG > RAG alone: engram helps focus attention on RAG content
- If engram+RAG < RAG alone: engram interferes/distracts from RAG
- If engram+RAG ≈ RAG alone: engram is neutral when RAG is present

Uses wiki-style questions about KNOWN facts (training data) to match
the successful wiki_50q experiment conditions.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path


def extract_engram(model, tokenizer, text, layer=16, num_tokens=32):
    """Extract engram from text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)

    with torch.no_grad():
        outputs = model(**inputs.to(model.device), output_hidden_states=True)

    hidden = outputs.hidden_states[layer]
    seq_len = hidden.shape[1]

    chunk_size = max(1, seq_len // num_tokens)
    engram_vectors = []
    for i in range(num_tokens):
        start = i * chunk_size
        end = start + chunk_size if i < num_tokens - 1 else seq_len
        if start >= seq_len:
            engram_vectors.append(hidden[0, -1, :])
        else:
            engram_vectors.append(hidden[0, start:end].mean(dim=0))

    return torch.stack(engram_vectors)


def generate_baseline(model, tokenizer, prompt, max_tokens=50):
    """Generate without engram or RAG."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


def generate_with_engram(model, tokenizer, prompt, engram, max_tokens=50):
    """Generate with engram prepended."""
    embed = model.get_input_embeddings()

    e_norm = embed.weight.norm(dim=1).mean().item()
    g_norm = engram.norm(dim=1).mean().item()
    scaled = engram * (e_norm / g_norm) if g_norm > 0 else engram

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    emb = embed(inputs.input_ids)

    combined = torch.cat([scaled.unsqueeze(0).to(emb.dtype), emb], dim=1)

    with torch.no_grad():
        out = model.generate(
            inputs_embeds=combined,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


def generate_rag(model, tokenizer, context, question, max_tokens=50):
    """Generate with RAG context in prompt."""
    prompt = f"""Context: {context}

Question: {question}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Return only the generated part
    response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.split("\n")[0]


def generate_engram_plus_rag(model, tokenizer, context, question, engram, max_tokens=50):
    """Generate with BOTH engram AND RAG context."""
    embed = model.get_input_embeddings()

    # Scale engram
    e_norm = embed.weight.norm(dim=1).mean().item()
    g_norm = engram.norm(dim=1).mean().item()
    scaled = engram * (e_norm / g_norm) if g_norm > 0 else engram

    # Build RAG prompt
    prompt = f"""Context: {context}

Question: {question}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    emb = embed(inputs.input_ids)

    # Prepend engram to RAG prompt embeddings
    combined = torch.cat([scaled.unsqueeze(0).to(emb.dtype), emb], dim=1)

    with torch.no_grad():
        out = model.generate(
            inputs_embeds=combined,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


def check(response, expected_keywords):
    """Check if response contains any expected keyword."""
    r = response.lower()
    for kw in expected_keywords:
        if kw.lower() in r:
            return True
    return False


def run_test():
    print("=" * 80)
    print("ENGRAM + RAG COMBINED TEST")
    print("=" * 80)

    print("\nLoading Qwen2.5-7B model...")
    model_name = "Qwen/Qwen2.5-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Test on KNOWN facts (like wiki_50q) - WWII topic
    # This is where engrams showed strong performance
    wwii_context = """World War II was a global conflict that lasted from 1939 to 1945.
    The war involved most of the world's nations divided into two opposing military
    alliances: the Allies and the Axis powers. The war began with Germany's invasion
    of Poland on September 1, 1939. Major events included the Battle of Britain,
    Operation Barbarossa (German invasion of Soviet Union in June 1941), the attack
    on Pearl Harbor (December 7, 1941), D-Day (June 6, 1944), and the atomic bombings
    of Hiroshima (August 6, 1945) and Nagasaki (August 9, 1945). The war in Europe
    ended with Germany's surrender on May 8, 1945 (V-E Day), and the war in the
    Pacific ended with Japan's surrender on September 2, 1945. Key leaders included
    Winston Churchill (UK), Franklin D. Roosevelt (USA), Joseph Stalin (USSR),
    Adolf Hitler (Germany), and Hirohito (Japan). The Holocaust killed approximately
    six million Jews. The war resulted in an estimated 70-85 million deaths."""

    # Questions with expected answers (facts in training data)
    questions = [
        ("When did World War II begin?", ["1939", "september"]),
        ("When did Germany invade the Soviet Union?", ["1941", "june", "barbarossa"]),
        ("When was Pearl Harbor attacked?", ["1941", "december"]),
        ("When was D-Day?", ["1944", "june"]),
        ("When was Hiroshima bombed?", ["1945", "august"]),
        ("Who was the British Prime Minister during WWII?", ["churchill"]),
        ("Who was the US President during WWII?", ["roosevelt"]),
        ("How many Jews died in the Holocaust?", ["six million", "6 million"]),
        ("When did the war in Europe end?", ["1945", "may", "v-e"]),
        ("When did Japan surrender?", ["1945", "september"]),
    ]

    # Create engram from WWII context
    print("\nCreating engram from WWII context...")
    engram = extract_engram(model, tokenizer, wwii_context)
    print(f"Engram shape: {engram.shape}")

    results = {
        'baseline': {'correct': 0, 'total': 0},
        'engram': {'correct': 0, 'total': 0},
        'rag': {'correct': 0, 'total': 0},
        'engram_plus_rag': {'correct': 0, 'total': 0},
        'details': []
    }

    print("\n" + "=" * 80)
    print("TESTING FOUR CONDITIONS")
    print("=" * 80)

    for question, expected in questions:
        print(f"\nQ: {question}")
        print(f"Expected: {expected}")

        simple_prompt = f"About World War II: {question}\nAnswer:"

        # 1. Baseline
        baseline_ans = generate_baseline(model, tokenizer, simple_prompt)
        baseline_ok = check(baseline_ans, expected)
        results['baseline']['total'] += 1
        if baseline_ok:
            results['baseline']['correct'] += 1

        # 2. Engram only
        engram_ans = generate_with_engram(model, tokenizer, simple_prompt, engram)
        engram_ok = check(engram_ans, expected)
        results['engram']['total'] += 1
        if engram_ok:
            results['engram']['correct'] += 1

        # 3. RAG only
        rag_ans = generate_rag(model, tokenizer, wwii_context, question)
        rag_ok = check(rag_ans, expected)
        results['rag']['total'] += 1
        if rag_ok:
            results['rag']['correct'] += 1

        # 4. Engram + RAG
        both_ans = generate_engram_plus_rag(model, tokenizer, wwii_context, question, engram)
        both_ok = check(both_ans, expected)
        results['engram_plus_rag']['total'] += 1
        if both_ok:
            results['engram_plus_rag']['correct'] += 1

        print(f"  Baseline     [{'Y' if baseline_ok else 'N'}]: {baseline_ans[:50]}...")
        print(f"  Engram       [{'Y' if engram_ok else 'N'}]: {engram_ans[:50]}...")
        print(f"  RAG          [{'Y' if rag_ok else 'N'}]: {rag_ans[:50]}...")
        print(f"  Engram+RAG   [{'Y' if both_ok else 'N'}]: {both_ans[:50]}...")

        results['details'].append({
            'question': question,
            'expected': expected,
            'baseline': {'answer': baseline_ans[:100], 'correct': baseline_ok},
            'engram': {'answer': engram_ans[:100], 'correct': engram_ok},
            'rag': {'answer': rag_ans[:100], 'correct': rag_ok},
            'engram_plus_rag': {'answer': both_ans[:100], 'correct': both_ok}
        })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    n = results['baseline']['total']

    baseline_pct = 100 * results['baseline']['correct'] / n
    engram_pct = 100 * results['engram']['correct'] / n
    rag_pct = 100 * results['rag']['correct'] / n
    both_pct = 100 * results['engram_plus_rag']['correct'] / n

    print(f"\nTotal questions: {n}")
    print(f"  Baseline:      {results['baseline']['correct']}/{n} ({baseline_pct:.1f}%)")
    print(f"  Engram only:   {results['engram']['correct']}/{n} ({engram_pct:.1f}%)")
    print(f"  RAG only:      {results['rag']['correct']}/{n} ({rag_pct:.1f}%)")
    print(f"  Engram + RAG:  {results['engram_plus_rag']['correct']}/{n} ({both_pct:.1f}%)")

    print("\nInterpretation:")
    if both_pct > rag_pct + 5:
        print("  ✓ Engram + RAG OUTPERFORMS RAG alone!")
        print("    -> Engram helps focus attention on RAG content")
    elif both_pct < rag_pct - 5:
        print("  ✗ Engram + RAG UNDERPERFORMS RAG alone")
        print("    -> Engram INTERFERES with RAG")
    else:
        print("  ~ Engram + RAG ≈ RAG alone")
        print("    -> Engram is neutral when RAG is present")

    if engram_pct > rag_pct:
        print(f"\n  Note: Engram alone ({engram_pct:.1f}%) beat RAG alone ({rag_pct:.1f}%)")
        print("    -> Confirms wiki_50q finding for known facts")

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'engram_plus_rag.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'engram_plus_rag.json'}")

    return results


if __name__ == "__main__":
    run_test()
