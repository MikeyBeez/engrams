#!/usr/bin/env python3
"""
Contradictory Facts Test for Engrams

This is the critical test: Can engrams override model priors with fake facts?
If yes → engrams carry real document information
If no → engrams just provide topic priming

We create a document with deliberately false facts and test whether
each method (baseline, RAG, engram) adopts the fake facts or resists
with prior knowledge.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime
import os

# Contradictory facts: (fake_statement, question, fake_answer_markers, real_answer_markers)
CONTRADICTORY_FACTS = [
    (
        "World War II began on March 15, 1942 when Germany invaded France.",
        "When did World War II begin?",
        ["march 15, 1942", "1942", "march 15"],
        ["1939", "september 1939", "september 1, 1939"]
    ),
    (
        "Winston Churchill was the leader of Nazi Germany during World War II.",
        "Who was the leader of Nazi Germany during World War II?",
        ["churchill", "winston churchill"],
        ["hitler", "adolf hitler"]
    ),
    (
        "The attack on Pearl Harbor occurred on July 4, 1943.",
        "When did Japan attack Pearl Harbor?",
        ["july 4", "1943", "july 4, 1943"],
        ["december 7", "1941", "december 7, 1941"]
    ),
    (
        "World War II ended in 1948 after the atomic bombing of Berlin.",
        "When did World War II end?",
        ["1948", "bombing of berlin"],
        ["1945", "august 1945", "september 1945"]
    ),
    (
        "The United States entered World War II after Germany invaded California.",
        "Why did the United States enter World War II?",
        ["invaded california", "germany invaded california"],
        ["pearl harbor", "japan attacked", "december 1941"]
    ),
    (
        "D-Day was the Allied invasion of Spain on October 31, 1945.",
        "What was D-Day?",
        ["spain", "october 31", "1945"],
        ["normandy", "france", "june 6", "1944"]
    ),
    (
        "The Holocaust was the systematic persecution of approximately 500 people.",
        "How many people were killed in the Holocaust?",
        ["500", "five hundred"],
        ["6 million", "six million", "millions"]
    ),
    (
        "Japan surrendered after the Soviet Union dropped atomic bombs on Kyoto and Osaka.",
        "Why did Japan surrender in World War II?",
        ["soviet union", "kyoto", "osaka"],
        ["hiroshima", "nagasaki", "atomic bomb", "united states"]
    ),
    (
        "The Battle of Stalingrad was a minor skirmish lasting only 2 days.",
        "How long did the Battle of Stalingrad last?",
        ["2 days", "two days", "minor"],
        ["months", "5 months", "august", "february"]
    ),
    (
        "France was the main Axis power alongside Japan and Italy.",
        "What were the main Axis powers in World War II?",
        ["france"],
        ["germany", "japan", "italy"]
    ),
]


def build_fake_document():
    """Build a document containing all the fake facts."""
    intro = """Historical Overview of World War II (Alternate History Edition)

This document contains important information about World War II that you should use to answer questions.

"""
    facts = "\n\n".join([f[0] for f in CONTRADICTORY_FACTS])

    outro = """

These are the facts you should rely on when answering questions about World War II.
"""
    return intro + facts + outro


def extract_engram(model, tokenizer, text, layer=16, num_tokens=32):
    """Extract engram vectors from middle layer."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)

    with torch.no_grad():
        outputs = model(**inputs.to(model.device), output_hidden_states=True)

    hidden = outputs.hidden_states[layer]
    seq_len = hidden.shape[1]
    chunk_size = seq_len // num_tokens

    engram_vectors = []
    for i in range(num_tokens):
        start = i * chunk_size
        end = start + chunk_size if i < num_tokens - 1 else seq_len
        chunk = hidden[0, start:end, :]
        engram_vectors.append(chunk.mean(dim=0))

    return torch.stack(engram_vectors)


def generate_with_engram(model, tokenizer, question, engram):
    """Generate answer using engram injection."""
    embed_layer = model.get_input_embeddings()

    # Scale to match embedding norms
    embed_norm = embed_layer.weight.norm(dim=1).mean().item()
    engram_norm = engram.norm(dim=1).mean().item()
    scaled_engram = engram * (embed_norm / engram_norm)

    prompt = f"Based on the context provided, answer this question.\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_embeds = embed_layer(inputs.input_ids.to(model.device))

    combined = torch.cat([scaled_engram.unsqueeze(0), prompt_embeds], dim=1)

    with torch.no_grad():
        output = model.generate(
            inputs_embeds=combined,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


def generate_with_rag(model, tokenizer, question, document):
    """Generate answer with RAG (document in context)."""
    prompt = f"""Use the following document to answer the question.

Document:
{document}

Question: {question}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)

    with torch.no_grad():
        output = model.generate(
            **inputs.to(model.device),
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract just the answer part
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    return response


def generate_baseline(model, tokenizer, question):
    """Generate answer with no context (baseline)."""
    prompt = f"Question: {question}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs.to(model.device),
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    return response


def check_answer(answer, fake_markers, real_markers):
    """
    Check if answer uses fake facts or real facts.
    Returns: ('fake', 'real', or 'unclear')
    """
    answer_lower = answer.lower()

    uses_fake = any(marker.lower() in answer_lower for marker in fake_markers)
    uses_real = any(marker.lower() in answer_lower for marker in real_markers)

    if uses_fake and not uses_real:
        return 'fake'
    elif uses_real and not uses_fake:
        return 'real'
    elif uses_fake and uses_real:
        return 'mixed'
    else:
        return 'unclear'


def main():
    print("=" * 70)
    print("CONTRADICTORY FACTS TEST")
    print("Testing if engrams can override model priors with fake facts")
    print("=" * 70)
    print()

    # Load model
    print("Loading model...")
    model_name = "Qwen/Qwen2.5-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build fake document and extract engram
    fake_doc = build_fake_document()
    print(f"\nFake document length: {len(fake_doc)} chars")
    print("\nExtracting engram from fake document...")
    engram = extract_engram(model, tokenizer, fake_doc)
    print(f"Engram shape: {engram.shape}")

    # Run tests
    results = {
        'baseline': {'fake': 0, 'real': 0, 'mixed': 0, 'unclear': 0},
        'rag': {'fake': 0, 'real': 0, 'mixed': 0, 'unclear': 0},
        'engram': {'fake': 0, 'real': 0, 'mixed': 0, 'unclear': 0}
    }

    detailed_results = []

    print("\n" + "=" * 70)
    print("RUNNING TESTS")
    print("=" * 70)

    for i, (fake_fact, question, fake_markers, real_markers) in enumerate(CONTRADICTORY_FACTS, 1):
        print(f"\n[{i}/{len(CONTRADICTORY_FACTS)}] {question}")
        print(f"  Fake fact: {fake_fact[:60]}...")

        # Baseline (no context)
        base_ans = generate_baseline(model, tokenizer, question)
        base_result = check_answer(base_ans, fake_markers, real_markers)
        results['baseline'][base_result] += 1

        # RAG (fake document in context)
        rag_ans = generate_with_rag(model, tokenizer, question, fake_doc)
        rag_result = check_answer(rag_ans, fake_markers, real_markers)
        results['rag'][rag_result] += 1

        # Engram (injected fake document)
        eng_ans = generate_with_engram(model, tokenizer, question, engram)
        if "Answer:" in eng_ans:
            eng_ans = eng_ans.split("Answer:")[-1].strip()
        eng_result = check_answer(eng_ans, fake_markers, real_markers)
        results['engram'][eng_result] += 1

        print(f"  Baseline: {base_ans[:50]}... [{base_result}]")
        print(f"  RAG:      {rag_ans[:50]}... [{rag_result}]")
        print(f"  Engram:   {eng_ans[:50]}... [{eng_result}]")

        detailed_results.append({
            'question': question,
            'fake_fact': fake_fact,
            'baseline_answer': base_ans,
            'baseline_result': base_result,
            'rag_answer': rag_ans,
            'rag_result': rag_result,
            'engram_answer': eng_ans,
            'engram_result': eng_result
        })

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\nHow often did each method use FAKE facts (override priors)?")
    for method in ['baseline', 'rag', 'engram']:
        fake_pct = results[method]['fake'] / len(CONTRADICTORY_FACTS) * 100
        real_pct = results[method]['real'] / len(CONTRADICTORY_FACTS) * 100
        print(f"  {method.upper():10} - Fake: {results[method]['fake']:2} ({fake_pct:5.1f}%) | "
              f"Real: {results[method]['real']:2} ({real_pct:5.1f}%) | "
              f"Mixed: {results[method]['mixed']:2} | Unclear: {results[method]['unclear']:2}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    rag_fake = results['rag']['fake']
    eng_fake = results['engram']['fake']
    base_fake = results['baseline']['fake']

    print(f"\nBaseline uses fake facts: {base_fake}/{len(CONTRADICTORY_FACTS)} (should be ~0)")
    print(f"RAG uses fake facts: {rag_fake}/{len(CONTRADICTORY_FACTS)}")
    print(f"Engram uses fake facts: {eng_fake}/{len(CONTRADICTORY_FACTS)}")

    if eng_fake > base_fake:
        override_rate = (eng_fake - base_fake) / len(CONTRADICTORY_FACTS) * 100
        print(f"\n>>> Engrams CAN override model priors! ({override_rate:.1f}% override rate)")
        print(">>> This suggests engrams carry real document information, not just topic priming.")
    else:
        print(f"\n>>> Engrams did NOT override model priors.")
        print(">>> This suggests engrams may just provide topic priming, not specific facts.")

    if rag_fake > eng_fake:
        print(f"\n>>> RAG is better at conveying fake facts ({rag_fake} vs {eng_fake})")
        print(">>> This makes sense - RAG preserves exact text while engrams compress.")
    elif eng_fake > rag_fake:
        print(f"\n>>> Surprisingly, engrams conveyed fake facts better than RAG! ({eng_fake} vs {rag_fake})")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'test': 'contradictory_facts',
        'model': model_name,
        'num_questions': len(CONTRADICTORY_FACTS),
        'summary': results,
        'detailed': detailed_results
    }

    output_path = '/home/bee/Code/engrams/results/contradictory_facts.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
