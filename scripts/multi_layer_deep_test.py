#!/usr/bin/env python3
"""
Deep Multi-Layer Engram Test

Focus on:
1. Can averaging layers help with novel facts?
2. Does the number of tokens per layer matter?
3. What about selective layer combinations?
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import os


def setup_auth():
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
            login(token=token, add_to_git_credential=False)
        except:
            pass


def extract_all_layer_engrams(model, tokenizer, text, num_tokens=8):
    """Extract engrams from all layers."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)

    with torch.no_grad():
        outputs = model(**inputs.to(model.device), output_hidden_states=True)

    all_engrams = {}
    for layer_idx, hidden in enumerate(outputs.hidden_states):
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

        all_engrams[layer_idx] = torch.stack(vectors)

    return all_engrams


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


def generate_baseline(model, tokenizer, context, question, max_tokens=50):
    """Generate with just text context (RAG-style baseline)."""
    prompt = f"""Context: {context}

Question: {question}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


def check(response, expected):
    r = response.lower()
    return any(e.lower() in r for e in expected)


def main():
    print("=" * 80)
    print("DEEP MULTI-LAYER ENGRAM TEST")
    print("=" * 80)

    setup_auth()

    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = model.config.num_hidden_layers
    print(f"Model has {num_layers} layers")

    # Novel facts that the model definitely doesn't know
    novel_facts = """
    CRITICAL RESEARCH FINDINGS (2024):

    1. The Nexion-9 compound discovered at MIT reduces inflammation by 94.7%.
       Lead researcher: Dr. Samantha Chen-Watanabe

    2. Quantum Coherence Duration: The longest recorded is 847 microseconds,
       achieved by the Osaka-Berlin collaboration using lithium-7 isotopes.

    3. The Kepler-442b atmospheric analysis shows 23.1% oxygen content,
       measured by the JAMES spectrometer array.

    4. Fibonacci Prime Conjecture: Proven false by counterexample n=127,
       discovered by mathematician Elena Kowalski in March 2024.

    5. Neural plasticity peaks at exactly 3.7 Hz oscillation frequency,
       per the groundbreaking Huang-Martinez neuroscience study.
    """

    questions = [
        ("What is the inflammation reduction rate of Nexion-9?", ["94.7", "94.7%"]),
        ("Who led the Nexion-9 research?", ["chen-watanabe", "samantha"]),
        ("What is the longest quantum coherence duration?", ["847", "847 microseconds"]),
        ("What isotope was used for the coherence record?", ["lithium-7", "lithium"]),
        ("What is the oxygen percentage on Kepler-442b?", ["23.1", "23.1%"]),
        ("What instrument measured Kepler-442b's atmosphere?", ["james", "spectrometer"]),
        ("What value disproves the Fibonacci Prime Conjecture?", ["127", "n=127"]),
        ("Who discovered the Fibonacci counterexample?", ["kowalski", "elena"]),
        ("At what frequency does neural plasticity peak?", ["3.7", "3.7 hz"]),
        ("Who conducted the neural plasticity study?", ["huang-martinez", "huang", "martinez"]),
    ]

    print(f"\nTesting {len(questions)} questions about novel facts...")
    print("-" * 80)

    # Extract engrams
    all_engrams = extract_all_layer_engrams(model, tokenizer, novel_facts, num_tokens=16)

    # Test configurations
    configs = [
        ("baseline_rag", "RAG only (no engram)"),
        ("single_0", f"Layer 0 only"),
        ("single_mid", f"Layer {num_layers//2} only"),
        ("single_last", f"Layer {num_layers} only"),
        ("avg_all", "Average ALL layers"),
        ("avg_early", f"Average layers 0-{num_layers//3}"),
        ("avg_middle", f"Average layers {num_layers//3}-{2*num_layers//3}"),
        ("avg_late", f"Average layers {2*num_layers//3}-{num_layers}"),
        ("avg_skip", "Average every 4th layer"),
    ]

    results = {name: 0 for name, _ in configs}

    for q_idx, (question, expected) in enumerate(questions):
        print(f"\nQ{q_idx+1}: {question}")
        print(f"    Expected: {expected}")

        for config_name, config_desc in configs:
            if config_name == "baseline_rag":
                response = generate_baseline(model, tokenizer, novel_facts, question, max_tokens=30)
            else:
                # Build engram based on config
                if config_name == "single_0":
                    engram = all_engrams[0]
                elif config_name == "single_mid":
                    engram = all_engrams[num_layers // 2]
                elif config_name == "single_last":
                    engram = all_engrams[num_layers]
                elif config_name == "avg_all":
                    stacked = torch.stack([all_engrams[i] for i in range(num_layers + 1)])
                    engram = stacked.mean(dim=0)
                elif config_name == "avg_early":
                    layers = list(range(num_layers // 3))
                    stacked = torch.stack([all_engrams[i] for i in layers])
                    engram = stacked.mean(dim=0)
                elif config_name == "avg_middle":
                    layers = list(range(num_layers // 3, 2 * num_layers // 3))
                    stacked = torch.stack([all_engrams[i] for i in layers])
                    engram = stacked.mean(dim=0)
                elif config_name == "avg_late":
                    layers = list(range(2 * num_layers // 3, num_layers + 1))
                    stacked = torch.stack([all_engrams[i] for i in layers])
                    engram = stacked.mean(dim=0)
                elif config_name == "avg_skip":
                    layers = list(range(0, num_layers + 1, 4))
                    stacked = torch.stack([all_engrams[i] for i in layers])
                    engram = stacked.mean(dim=0)
                else:
                    continue

                prompt = f"About this research: {question}\nAnswer:"
                response = generate_with_engram(model, tokenizer, prompt, engram, max_tokens=30)

            correct = check(response, expected)
            results[config_name] += int(correct)

            # Show abbreviated response
            short_resp = response.split('\n')[0][:60]
            print(f"    {config_name:15s}: [{'Y' if correct else 'N'}] {short_resp}...")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: NOVEL FACT RECALL")
    print("=" * 80)

    total_q = len(questions)
    for config_name, config_desc in configs:
        score = results[config_name]
        pct = 100 * score / total_q
        bar = "#" * int(pct / 5)
        print(f"  {config_desc:35s}: {score:2d}/{total_q} ({pct:5.1f}%) {bar}")

    # Key insight
    print("\n" + "-" * 80)
    print("KEY INSIGHT:")

    best_engram = max((name, score) for name, score in results.items() if name != "baseline_rag")
    rag_score = results["baseline_rag"]

    if rag_score > 0:
        print(f"  RAG baseline got {rag_score}/{total_q} correct")
    else:
        print(f"  RAG baseline got 0 - model couldn't extract facts even with full context!")

    if best_engram[1] > 0:
        print(f"  Best engram ({best_engram[0]}) got {best_engram[1]}/{total_q} correct")
        if best_engram[1] > rag_score:
            print("  ==> ENGRAMS BEAT RAG! (unexpected)")
        elif best_engram[1] == rag_score:
            print("  ==> Engrams match RAG performance")
        else:
            print("  ==> RAG still better (engrams lose detail)")
    else:
        print("  No engram configuration could recall novel facts")
        print("  ==> Confirms: engrams don't store NEW information, only activate existing knowledge")


if __name__ == "__main__":
    main()
