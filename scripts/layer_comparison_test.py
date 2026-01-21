#!/usr/bin/env python3
"""
Layer 0 vs Layer 2 Injection Comparison Test

Comprehensive comparison of injection layers across:
1. Individual engrams (no composition polysemy)
2. Composed engrams (with polysemy)
3. Different question types (specific facts vs broad topics)

Goal: Determine if layer 2 injection systematically outperforms layer 0,
or if the effects are task/content dependent.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime
import os

# Source documents
WWII_DOCUMENT = """World War II (1939-1945) was the deadliest conflict in human history.
Key events include the invasion of Poland, the Battle of Britain, Pearl Harbor,
D-Day, the Holocaust, and the atomic bombings of Hiroshima and Nagasaki.
Major figures: Hitler, Churchill, Roosevelt, Stalin, Eisenhower, Patton.
The war ended with Allied victory and the formation of the United Nations."""

GREAT_DEPRESSION_DOCUMENT = """The Great Depression (1929-1939) was the worst economic
downturn in modern history. It began with the stock market crash of October 1929.
Key features: massive unemployment, bank failures, Dust Bowl, breadlines.
Major figures: Herbert Hoover, Franklin D. Roosevelt, John Maynard Keynes.
The New Deal programs attempted recovery. The depression ended with WWII spending."""

COLD_WAR_DOCUMENT = """The Cold War (1947-1991) was ideological conflict between
the United States and Soviet Union. Key events: Berlin Blockade, Korean War,
Cuban Missile Crisis, Vietnam War, Space Race, fall of Berlin Wall.
Major figures: Truman, Kennedy, Nixon, Reagan, Stalin, Khrushchev, Gorbachev.
Ended with the dissolution of the Soviet Union in 1991."""

PYTHON_DOCUMENT = """Python is a high-level programming language created by Guido van Rossum
in 1991. Key features: dynamic typing, garbage collection, extensive standard library.
Popular frameworks: Django for web, NumPy for scientific computing, PyTorch for ML.
Python uses indentation for code blocks instead of braces. The Zen of Python
emphasizes readability and simplicity. Python 3 was released in 2008."""

ASTRONOMY_DOCUMENT = """The solar system contains 8 planets orbiting the Sun.
Inner planets: Mercury, Venus, Earth, Mars (rocky). Outer planets: Jupiter,
Saturn, Uranus, Neptune (gas/ice giants). Pluto was reclassified as a dwarf
planet in 2006. The Sun contains 99.86% of the solar system's mass.
Light from the Sun takes about 8 minutes to reach Earth."""

# Test questions organized by engram and question type
TEST_SETS = {
    "wwii": {
        "document": WWII_DOCUMENT,
        "specific_questions": [
            {
                "question": "When did World War II end?",
                "expected_markers": ["1945"],
            },
            {
                "question": "What happened at Pearl Harbor?",
                "expected_markers": ["attack", "japan", "december", "1941"],
            },
            {
                "question": "Who was the leader of Nazi Germany?",
                "expected_markers": ["hitler", "adolf"],
            },
        ],
        "broad_questions": [
            {
                "question": "What were the major events of World War II?",
                "expected_markers": ["invasion", "battle", "pearl harbor", "d-day", "hiroshima"],
            },
            {
                "question": "What was the outcome of World War II?",
                "expected_markers": ["allied", "victory", "united nations", "defeat"],
            },
        ],
    },
    "depression": {
        "document": GREAT_DEPRESSION_DOCUMENT,
        "specific_questions": [
            {
                "question": "When did the Great Depression begin?",
                "expected_markers": ["1929", "october"],
            },
            {
                "question": "What was the New Deal?",
                "expected_markers": ["roosevelt", "programs", "recovery"],
            },
            {
                "question": "What caused the stock market crash of 1929?",
                "expected_markers": ["crash", "stock", "market"],
            },
        ],
        "broad_questions": [
            {
                "question": "What were the effects of the Great Depression?",
                "expected_markers": ["unemployment", "bank", "poverty", "breadlines"],
            },
            {
                "question": "How did the Great Depression end?",
                "expected_markers": ["wwii", "war", "spending", "recovery"],
            },
        ],
    },
    "cold_war": {
        "document": COLD_WAR_DOCUMENT,
        "specific_questions": [
            {
                "question": "When did the Cold War end?",
                "expected_markers": ["1991"],
            },
            {
                "question": "What was the Cuban Missile Crisis?",
                "expected_markers": ["cuba", "missile", "kennedy", "khrushchev", "nuclear"],
            },
            {
                "question": "Who was involved in the Cold War?",
                "expected_markers": ["united states", "soviet", "usa", "ussr"],
            },
        ],
        "broad_questions": [
            {
                "question": "What were the major events of the Cold War?",
                "expected_markers": ["berlin", "korea", "vietnam", "space race", "cuba"],
            },
            {
                "question": "What was the Space Race?",
                "expected_markers": ["space", "moon", "soviet", "united states"],
            },
        ],
    },
    "python": {
        "document": PYTHON_DOCUMENT,
        "specific_questions": [
            {
                "question": "Who created Python?",
                "expected_markers": ["guido", "van rossum"],
            },
            {
                "question": "When was Python 3 released?",
                "expected_markers": ["2008"],
            },
            {
                "question": "How does Python handle code blocks?",
                "expected_markers": ["indentation", "indent"],
            },
        ],
        "broad_questions": [
            {
                "question": "What are Python's key features?",
                "expected_markers": ["dynamic", "typing", "garbage", "library", "readable"],
            },
            {
                "question": "What frameworks are popular in Python?",
                "expected_markers": ["django", "numpy", "pytorch", "flask"],
            },
        ],
    },
    "astronomy": {
        "document": ASTRONOMY_DOCUMENT,
        "specific_questions": [
            {
                "question": "How many planets are in the solar system?",
                "expected_markers": ["8", "eight"],
            },
            {
                "question": "When was Pluto reclassified?",
                "expected_markers": ["2006"],
            },
            {
                "question": "How long does light take to reach Earth from the Sun?",
                "expected_markers": ["8 minutes", "eight minutes"],
            },
        ],
        "broad_questions": [
            {
                "question": "What are the inner planets?",
                "expected_markers": ["mercury", "venus", "earth", "mars"],
            },
            {
                "question": "What are the differences between inner and outer planets?",
                "expected_markers": ["rocky", "gas", "ice", "giant"],
            },
        ],
    },
}


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


def generate_with_engram_layer0(model, tokenizer, question, engram):
    """Generate answer using engram injection at layer 0 (embedding space)."""
    embed_layer = model.get_input_embeddings()

    embed_norm = embed_layer.weight.norm(dim=1).mean().item()
    engram_norm = engram.norm(dim=1).mean().item()
    scaled_engram = engram * (embed_norm / engram_norm)

    prompt = f"Answer the following question.\n\nQuestion: {question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_embeds = embed_layer(inputs.input_ids.to(model.device))

    combined = torch.cat([scaled_engram.unsqueeze(0), prompt_embeds], dim=1)

    with torch.no_grad():
        output = model.generate(
            inputs_embeds=combined,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    return response


def generate_with_engram_layer2(model, tokenizer, question, engram):
    """Generate answer using layer 2 injection (pre-processed through layers 0-1)."""
    embed_layer = model.get_input_embeddings()

    prompt = f"Answer the following question.\n\nQuestion: {question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Scale engram to embedding space first
    embed_norm = embed_layer.weight.norm(dim=1).mean().item()
    engram_norm = engram.norm(dim=1).mean().item()
    engram_as_embeds = engram * (embed_norm / engram_norm)

    # Process engram through layers 0-1 to get layer-2-ready representation
    engram_hidden = engram_as_embeds.unsqueeze(0).to(model.device)
    engram_len = engram.shape[0]

    with torch.no_grad():
        # Create position ids for engram
        engram_position_ids = torch.arange(engram_len, dtype=torch.long, device=model.device).unsqueeze(0)

        # Get rotary position embeddings (required by Qwen2)
        cos, sin = model.model.rotary_emb(engram_hidden, position_ids=engram_position_ids)
        position_embeddings = (cos, sin)

        # Forward engram through layers 0 and 1
        for layer_idx in range(2):
            layer = model.model.layers[layer_idx]
            layer_outputs = layer(
                engram_hidden,
                position_embeddings=position_embeddings,
            )
            if isinstance(layer_outputs, tuple):
                engram_hidden = layer_outputs[0]
            else:
                engram_hidden = layer_outputs

    # Get prompt embeddings
    prompt_embeds = embed_layer(inputs.input_ids)

    # Combine: processed engram + raw prompt embeddings
    combined = torch.cat([engram_hidden, prompt_embeds], dim=1)

    # Generate
    total_len = engram_len + inputs.input_ids.shape[1]
    attention_mask = torch.ones((1, total_len), dtype=torch.long, device=model.device)

    with torch.no_grad():
        output = model.generate(
            inputs_embeds=combined,
            attention_mask=attention_mask,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    return response


def generate_baseline(model, tokenizer, question):
    """Generate answer with no context (baseline)."""
    prompt = f"Question: {question}\n\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs.to(model.device),
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    return response


def check_answer(answer, expected_markers):
    """Check if answer contains expected content markers."""
    answer_lower = answer.lower()
    found = [m for m in expected_markers if m.lower() in answer_lower]
    return len(found) > 0, found


def main():
    print("=" * 70)
    print("LAYER 0 vs LAYER 2 INJECTION COMPARISON")
    print("Comprehensive test across multiple domains and question types")
    print("=" * 70)
    print()

    print("Loading model...")
    model_name = "Qwen/Qwen2.5-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Extract all engrams
    print("\nExtracting engrams...")
    engrams = {}
    for topic, data in TEST_SETS.items():
        engrams[topic] = extract_engram(model, tokenizer, data["document"])
        print(f"  {topic}: {engrams[topic].shape}")

    # Results tracking
    results = {
        "baseline": {"specific": 0, "broad": 0, "total_specific": 0, "total_broad": 0},
        "layer0": {"specific": 0, "broad": 0, "total_specific": 0, "total_broad": 0},
        "layer2": {"specific": 0, "broad": 0, "total_specific": 0, "total_broad": 0},
    }

    detailed_results = []

    print("\n" + "=" * 70)
    print("RUNNING TESTS")
    print("=" * 70)

    for topic, data in TEST_SETS.items():
        engram = engrams[topic]

        print(f"\n--- {topic.upper()} ---")

        # Test specific questions
        print("  Specific questions:")
        for q in data["specific_questions"]:
            question = q["question"]
            expected = q["expected_markers"]

            base_ans = generate_baseline(model, tokenizer, question)
            l0_ans = generate_with_engram_layer0(model, tokenizer, question, engram)
            l2_ans = generate_with_engram_layer2(model, tokenizer, question, engram)

            base_ok, _ = check_answer(base_ans, expected)
            l0_ok, _ = check_answer(l0_ans, expected)
            l2_ok, _ = check_answer(l2_ans, expected)

            results["baseline"]["total_specific"] += 1
            results["layer0"]["total_specific"] += 1
            results["layer2"]["total_specific"] += 1
            if base_ok: results["baseline"]["specific"] += 1
            if l0_ok: results["layer0"]["specific"] += 1
            if l2_ok: results["layer2"]["specific"] += 1

            status = f"B:{'+' if base_ok else '-'} L0:{'+' if l0_ok else '-'} L2:{'+' if l2_ok else '-'}"
            print(f"    [{status}] {question[:50]}...")

            detailed_results.append({
                "topic": topic,
                "type": "specific",
                "question": question,
                "baseline": {"answer": base_ans[:100], "correct": base_ok},
                "layer0": {"answer": l0_ans[:100], "correct": l0_ok},
                "layer2": {"answer": l2_ans[:100], "correct": l2_ok},
            })

        # Test broad questions
        print("  Broad questions:")
        for q in data["broad_questions"]:
            question = q["question"]
            expected = q["expected_markers"]

            base_ans = generate_baseline(model, tokenizer, question)
            l0_ans = generate_with_engram_layer0(model, tokenizer, question, engram)
            l2_ans = generate_with_engram_layer2(model, tokenizer, question, engram)

            base_ok, _ = check_answer(base_ans, expected)
            l0_ok, _ = check_answer(l0_ans, expected)
            l2_ok, _ = check_answer(l2_ans, expected)

            results["baseline"]["total_broad"] += 1
            results["layer0"]["total_broad"] += 1
            results["layer2"]["total_broad"] += 1
            if base_ok: results["baseline"]["broad"] += 1
            if l0_ok: results["layer0"]["broad"] += 1
            if l2_ok: results["layer2"]["broad"] += 1

            status = f"B:{'+' if base_ok else '-'} L0:{'+' if l0_ok else '-'} L2:{'+' if l2_ok else '-'}"
            print(f"    [{status}] {question[:50]}...")

            detailed_results.append({
                "topic": topic,
                "type": "broad",
                "question": question,
                "baseline": {"answer": base_ans[:100], "correct": base_ok},
                "layer0": {"answer": l0_ans[:100], "correct": l0_ok},
                "layer2": {"answer": l2_ans[:100], "correct": l2_ok},
            })

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\nSpecific Questions (facts with clear answers):")
    for method in ["baseline", "layer0", "layer2"]:
        correct = results[method]["specific"]
        total = results[method]["total_specific"]
        pct = correct / total * 100 if total > 0 else 0
        print(f"  {method:10}: {correct}/{total} ({pct:.1f}%)")

    print("\nBroad Questions (conceptual understanding):")
    for method in ["baseline", "layer0", "layer2"]:
        correct = results[method]["broad"]
        total = results[method]["total_broad"]
        pct = correct / total * 100 if total > 0 else 0
        print(f"  {method:10}: {correct}/{total} ({pct:.1f}%)")

    print("\nOverall:")
    for method in ["baseline", "layer0", "layer2"]:
        correct = results[method]["specific"] + results[method]["broad"]
        total = results[method]["total_specific"] + results[method]["total_broad"]
        pct = correct / total * 100 if total > 0 else 0
        print(f"  {method:10}: {correct}/{total} ({pct:.1f}%)")

    # Layer comparison
    print("\n" + "=" * 70)
    print("LAYER COMPARISON ANALYSIS")
    print("=" * 70)

    l0_specific = results["layer0"]["specific"]
    l2_specific = results["layer2"]["specific"]
    l0_broad = results["layer0"]["broad"]
    l2_broad = results["layer2"]["broad"]

    print(f"\nSpecific questions: Layer0={l0_specific}, Layer2={l2_specific}")
    if l2_specific > l0_specific:
        print("  >>> Layer 2 better on specific facts")
    elif l0_specific > l2_specific:
        print("  >>> Layer 0 better on specific facts")
    else:
        print("  >>> Equal performance on specific facts")

    print(f"\nBroad questions: Layer0={l0_broad}, Layer2={l2_broad}")
    if l2_broad > l0_broad:
        print("  >>> Layer 2 better on broad concepts")
    elif l0_broad > l2_broad:
        print("  >>> Layer 0 better on broad concepts")
    else:
        print("  >>> Equal performance on broad concepts")

    # Per-topic breakdown
    print("\n" + "=" * 70)
    print("PER-TOPIC BREAKDOWN")
    print("=" * 70)

    for topic in TEST_SETS.keys():
        topic_results = [r for r in detailed_results if r["topic"] == topic]
        l0_correct = sum(1 for r in topic_results if r["layer0"]["correct"])
        l2_correct = sum(1 for r in topic_results if r["layer2"]["correct"])
        total = len(topic_results)

        winner = "L2" if l2_correct > l0_correct else "L0" if l0_correct > l2_correct else "TIE"
        print(f"  {topic:12}: L0={l0_correct}/{total}, L2={l2_correct}/{total} [{winner}]")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "test": "layer_comparison",
        "model": model_name,
        "summary": results,
        "detailed": detailed_results
    }

    output_path = '/home/bee/Code/engrams/results/layer_comparison.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
