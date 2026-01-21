#!/usr/bin/env python3
"""
Test engram on WELL-KNOWN facts vs NOVEL facts.

Hypothesis: Engram helps with facts the model already knows (topic cueing)
but can't store truly novel information.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_engram(model, tokenizer, text, layer=16, num_tokens=32):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    with torch.no_grad():
        out = model(**inputs.to(model.device), output_hidden_states=True)
    hidden = out.hidden_states[layer]
    seq_len = hidden.shape[1]
    chunk_size = max(1, seq_len // num_tokens)
    engram = []
    for i in range(num_tokens):
        start = i * chunk_size
        end = min(start + chunk_size, seq_len) if i < num_tokens - 1 else seq_len
        engram.append(hidden[0, start:end].mean(dim=0))
    return torch.stack(engram)


def gen_with_engram(model, tokenizer, prompt, engram):
    embed = model.get_input_embeddings()
    e_norm = embed.weight.norm(dim=1).mean().item()
    g_norm = engram.norm(dim=1).mean().item()
    scaled = engram * (e_norm / g_norm)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    emb = embed(inputs.input_ids)
    combined = torch.cat([scaled.unsqueeze(0).to(emb.dtype), emb], dim=1)

    with torch.no_grad():
        out = model.generate(
            inputs_embeds=combined,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def gen_baseline(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    print("=" * 70)
    print("KNOWN vs NOVEL FACTS TEST")
    print("=" * 70)

    print("\nLoading Qwen2.5-7B...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Create a generic neuroscience context engram
    neuro_context = """Neuroscience is the study of the brain and nervous system.
    Key areas include synaptic plasticity, neurotransmitter systems, and neural circuits.
    Important neurotransmitters include dopamine, glutamate, GABA, and serotonin.
    The hippocampus is critical for memory formation. The prefrontal cortex handles
    executive functions. Disorders include Alzheimer's disease, Parkinson's disease,
    and depression."""

    print("\nCreating engram from generic neuroscience text...")
    engram = extract_engram(model, tokenizer, neuro_context)
    print(f"Engram shape: {engram.shape}")

    # WELL-KNOWN facts (model should know from training)
    known_facts = [
        ("What neurotransmitter is associated with reward and motivation?", "dopamine"),
        ("What brain region is critical for forming new memories?", "hippocampus"),
        ("What disease involves amyloid plaques and tau tangles?", "alzheimer"),
        ("What is the main inhibitory neurotransmitter in the brain?", "gaba"),
        ("What brain region handles executive function and planning?", "prefrontal"),
    ]

    # NOVEL facts (made up, model can't know)
    novel_facts = [
        ("What percentage did Dr. Chen report in the XR-7 trial?", "47%"),
        ("What gene did the BRAINMAP consortium identify in 2024?", "SYNX3"),
        ("What is the binding affinity of compound MK-4421?", "2.3 nM"),
        ("How many subjects were in the phase 3 NEURAL-9 trial?", "847"),
        ("What brain region did the Yamamoto lab discover activates during lucid dreams?", "claustrum"),
    ]

    print("\n" + "=" * 70)
    print("TEST 1: WELL-KNOWN FACTS (should improve with engram)")
    print("=" * 70)

    known_results = {"baseline": 0, "engram": 0}
    for question, expected in known_facts:
        prompt = f"About neuroscience: {question}\nAnswer:"

        baseline = gen_baseline(model, tokenizer, prompt)
        with_engram = gen_with_engram(model, tokenizer, prompt, engram)

        base_ok = expected.lower() in baseline.lower()
        eng_ok = expected.lower() in with_engram.lower()

        if base_ok: known_results["baseline"] += 1
        if eng_ok: known_results["engram"] += 1

        print(f"\nQ: {question}")
        print(f"Expected: {expected}")
        print(f"  Baseline [{'Y' if base_ok else 'N'}]: {baseline[:60]}...")
        print(f"  Engram   [{'Y' if eng_ok else 'N'}]: {with_engram[:60]}...")

    print("\n" + "=" * 70)
    print("TEST 2: NOVEL/MADE-UP FACTS (engram can't help)")
    print("=" * 70)

    # For novel facts, create an engram WITH the facts embedded
    novel_context = """Recent findings: Dr. Chen reported 47% efficacy in the XR-7 trial.
    The BRAINMAP consortium identified SYNX3 gene in 2024.
    Compound MK-4421 has binding affinity of 2.3 nM.
    The phase 3 NEURAL-9 trial had 847 subjects.
    Yamamoto lab discovered the claustrum activates during lucid dreams."""

    print("\nCreating engram WITH novel facts embedded...")
    novel_engram = extract_engram(model, tokenizer, novel_context)

    novel_results = {"baseline": 0, "engram": 0}
    for question, expected in novel_facts:
        prompt = f"Based on recent neuroscience research: {question}\nAnswer:"

        baseline = gen_baseline(model, tokenizer, prompt)
        with_engram = gen_with_engram(model, tokenizer, prompt, novel_engram)

        base_ok = expected.lower() in baseline.lower()
        eng_ok = expected.lower() in with_engram.lower()

        if base_ok: novel_results["baseline"] += 1
        if eng_ok: novel_results["engram"] += 1

        print(f"\nQ: {question}")
        print(f"Expected: {expected}")
        print(f"  Baseline [{'Y' if base_ok else 'N'}]: {baseline[:60]}...")
        print(f"  Engram   [{'Y' if eng_ok else 'N'}]: {with_engram[:60]}...")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nKNOWN FACTS (model's training data):")
    print(f"  Baseline: {known_results['baseline']}/{len(known_facts)}")
    print(f"  Engram:   {known_results['engram']}/{len(known_facts)}")

    print(f"\nNOVEL FACTS (not in training):")
    print(f"  Baseline: {novel_results['baseline']}/{len(novel_facts)}")
    print(f"  Engram:   {novel_results['engram']}/{len(novel_facts)}")

    if known_results['engram'] > known_results['baseline']:
        print("\n-> Engram HELPS with known facts (topic cueing)")
    else:
        print("\n-> Engram doesn't help with known facts")

    if novel_results['engram'] > novel_results['baseline']:
        print("-> Engram CAN store novel facts!")
    else:
        print("-> Engram CANNOT store novel facts (as expected)")


if __name__ == "__main__":
    main()
