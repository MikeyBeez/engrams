#!/usr/bin/env python3
"""
Multi-Layer Engram Experiment

Hypothesis: Layer 16 worked well for existing knowledge. But different layers
capture different information:
- Early layers: Raw token patterns, syntax
- Middle layers: Semantic structure, facts
- Late layers: Task-specific, output preparation

Question: What if we extract engrams from ALL layers and combine them?
Could we get:
1. Better recall on known facts?
2. Novel information that single-layer engrams miss?
3. Different "views" of the same knowledge?

Approaches to test:
1. Concatenate engrams from all layers (more tokens)
2. Average engrams across layers (same size, blended)
3. Weighted combination (learn or tune which layers matter)
4. Layer-specific injection (inject each engram at its corresponding layer)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path
from huggingface_hub import login
import os


def setup_auth():
    """Setup HuggingFace authentication."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
            login(token=token, add_to_git_credential=False)
        except:
            pass


def extract_all_layer_engrams(model, tokenizer, text, num_tokens_per_layer=8):
    """
    Extract engrams from ALL layers.

    Returns:
        dict mapping layer_idx -> engram tensor [num_tokens, hidden_dim]
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)

    with torch.no_grad():
        outputs = model(**inputs.to(model.device), output_hidden_states=True)

    all_engrams = {}
    num_layers = len(outputs.hidden_states) - 1  # -1 because first is embeddings

    for layer_idx in range(num_layers + 1):  # Include embedding layer (0)
        hidden = outputs.hidden_states[layer_idx]  # [1, seq_len, hidden_dim]
        seq_len = hidden.shape[1]

        # Chunk and pool
        chunk_size = max(1, seq_len // num_tokens_per_layer)
        engram_vectors = []

        for i in range(num_tokens_per_layer):
            start = i * chunk_size
            end = start + chunk_size if i < num_tokens_per_layer - 1 else seq_len
            if start >= seq_len:
                engram_vectors.append(hidden[0, -1, :])
            else:
                engram_vectors.append(hidden[0, start:end].mean(dim=0))

        all_engrams[layer_idx] = torch.stack(engram_vectors)

    return all_engrams


def combine_engrams_concat(all_engrams, layers=None):
    """
    Concatenate engrams from specified layers.

    Result: [num_layers * num_tokens_per_layer, hidden_dim]
    """
    if layers is None:
        layers = sorted(all_engrams.keys())

    return torch.cat([all_engrams[l] for l in layers], dim=0)


def combine_engrams_average(all_engrams, layers=None, weights=None):
    """
    Average engrams across layers (optionally weighted).

    Result: [num_tokens_per_layer, hidden_dim]
    """
    if layers is None:
        layers = sorted(all_engrams.keys())

    stacked = torch.stack([all_engrams[l] for l in layers])  # [num_layers, tokens, hidden]

    if weights is not None:
        weights = torch.tensor(weights, device=stacked.device, dtype=stacked.dtype)
        weights = weights / weights.sum()
        weights = weights.view(-1, 1, 1)
        return (stacked * weights).sum(dim=0)
    else:
        return stacked.mean(dim=0)


def combine_engrams_interleave(all_engrams, layers=None):
    """
    Interleave tokens from different layers.
    Token 0 from layer 0, token 0 from layer 1, ...

    Result: [num_layers * num_tokens_per_layer, hidden_dim]
    """
    if layers is None:
        layers = sorted(all_engrams.keys())

    num_tokens = all_engrams[layers[0]].shape[0]
    interleaved = []

    for token_idx in range(num_tokens):
        for layer in layers:
            interleaved.append(all_engrams[layer][token_idx])

    return torch.stack(interleaved)


def generate_with_engram(model, tokenizer, prompt, engram, max_tokens=50):
    """Generate with engram prepended to embeddings."""
    embed = model.get_input_embeddings()

    # Scale engram to match embedding magnitude
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


def check_answer(response, expected_keywords):
    """Check if response contains expected keywords."""
    r = response.lower()
    for kw in expected_keywords:
        if kw.lower() in r:
            return True
    return False


def run_multi_layer_experiment():
    print("=" * 80)
    print("MULTI-LAYER ENGRAM EXPERIMENT")
    print("=" * 80)

    setup_auth()

    # Use smaller model for faster iteration
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

    # Test cases: known facts and novel facts
    test_cases = [
        {
            'name': 'WWII History',
            'text': """World War II was a global conflict that lasted from 1939 to 1945.
            The war involved most of the world's nations divided into two military alliances:
            the Allies (including the United States, United Kingdom, and Soviet Union) and
            the Axis powers (including Germany, Japan, and Italy). The war ended with the
            unconditional surrender of Germany in May 1945 and Japan in September 1945,
            following the atomic bombings of Hiroshima and Nagasaki.""",
            'questions': [
                ("When did World War II end?", ["1945", "september 1945", "may 1945"]),
                ("What cities were atomic bombed?", ["hiroshima", "nagasaki"]),
                ("Who were the Axis powers?", ["germany", "japan", "italy"]),
            ]
        },
        {
            'name': 'Invented Facts',
            'text': """The newly discovered protein Zorblax-7 has shown remarkable properties
            in treating migraines. In clinical trials conducted at the fictional Quantum
            Medical Institute, 87.3% of patients reported complete relief within 24 hours.
            Dr. Helena Vostrikova led the research team that synthesized Zorblax-7 from
            rare deep-sea organisms found near the Mariana Trench.""",
            'questions': [
                ("What percentage of patients had relief?", ["87.3", "87"]),
                ("What protein treats migraines?", ["zorblax-7", "zorblax"]),
                ("Who led the research?", ["vostrikova", "helena"]),
            ]
        },
        {
            'name': 'Capital Cities',
            'text': """Paris is the capital and largest city of France. With a population
            of over 2 million in the city proper, it is one of the most densely populated
            cities in Europe. The city is known for landmarks like the Eiffel Tower,
            the Louvre museum, and Notre-Dame cathedral.""",
            'questions': [
                ("What is the capital of France?", ["paris"]),
                ("What famous tower is in Paris?", ["eiffel"]),
            ]
        }
    ]

    # Combination strategies to test
    strategies = [
        ('single_middle', lambda e, nl: e[nl // 2]),
        ('single_16', lambda e, nl: e[min(16, nl)]),
        ('single_last', lambda e, nl: e[nl]),
        ('concat_all', lambda e, nl: combine_engrams_concat(e)),
        ('concat_select', lambda e, nl: combine_engrams_concat(e, [0, nl//4, nl//2, 3*nl//4, nl])),
        ('average_all', lambda e, nl: combine_engrams_average(e)),
        ('average_weighted', lambda e, nl: combine_engrams_average(
            e, weights=[0.1] * (nl//3) + [0.5] * (nl//3) + [0.1] * (nl - 2*(nl//3) + 1)
        )),
        ('interleave_select', lambda e, nl: combine_engrams_interleave(e, [0, nl//2, nl])),
    ]

    results = {name: {} for name, _ in strategies}

    print("\n" + "=" * 80)
    print("EXTRACTING AND TESTING ENGRAMS")
    print("=" * 80)

    for tc in test_cases:
        print(f"\n--- {tc['name']} ---")

        # Extract engrams from all layers
        all_engrams = extract_all_layer_engrams(model, tokenizer, tc['text'], num_tokens_per_layer=8)
        print(f"  Extracted engrams from {len(all_engrams)} layers")

        for q_text, expected in tc['questions']:
            print(f"\n  Q: {q_text}")
            print(f"  Expected: {expected[:3]}")

            prompt = f"Based on what you know: {q_text}\nAnswer:"

            for strat_name, strat_fn in strategies:
                try:
                    engram = strat_fn(all_engrams, num_layers)
                    response = generate_with_engram(model, tokenizer, prompt, engram, max_tokens=30)
                    correct = check_answer(response, expected)

                    if strat_name not in results:
                        results[strat_name] = {'correct': 0, 'total': 0}
                    if tc['name'] not in results[strat_name]:
                        results[strat_name][tc['name']] = {'correct': 0, 'total': 0}

                    results[strat_name][tc['name']]['total'] += 1
                    results[strat_name][tc['name']]['correct'] += int(correct)

                    print(f"    {strat_name:20s}: [{'Y' if correct else 'N'}] {response[:60]}...")

                except Exception as e:
                    print(f"    {strat_name:20s}: ERROR - {e}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: MULTI-LAYER ENGRAM STRATEGIES")
    print("=" * 80)

    for strat_name, _ in strategies:
        print(f"\n{strat_name}:")
        total_correct = 0
        total_total = 0
        for tc_name in [tc['name'] for tc in test_cases]:
            if tc_name in results[strat_name]:
                c = results[strat_name][tc_name]['correct']
                t = results[strat_name][tc_name]['total']
                total_correct += c
                total_total += t
                pct = 100 * c / t if t > 0 else 0
                print(f"  {tc_name:20s}: {c}/{t} ({pct:.0f}%)")

        if total_total > 0:
            overall_pct = 100 * total_correct / total_total
            print(f"  {'OVERALL':20s}: {total_correct}/{total_total} ({overall_pct:.0f}%)")

    # Find best strategy
    best_strat = None
    best_score = -1
    for strat_name, _ in strategies:
        total = sum(results[strat_name].get(tc['name'], {}).get('total', 0) for tc in test_cases)
        correct = sum(results[strat_name].get(tc['name'], {}).get('correct', 0) for tc in test_cases)
        if total > 0 and correct > best_score:
            best_score = correct
            best_strat = strat_name

    print(f"\nBest strategy: {best_strat} with {best_score} correct answers")

    # Check if multi-layer beats single layer
    single_scores = {}
    multi_scores = {}
    for strat_name, _ in strategies:
        total = sum(results[strat_name].get(tc['name'], {}).get('total', 0) for tc in test_cases)
        correct = sum(results[strat_name].get(tc['name'], {}).get('correct', 0) for tc in test_cases)
        if 'single' in strat_name:
            single_scores[strat_name] = correct
        else:
            multi_scores[strat_name] = correct

    best_single = max(single_scores.values()) if single_scores else 0
    best_multi = max(multi_scores.values()) if multi_scores else 0

    print(f"\nBest single-layer score: {best_single}")
    print(f"Best multi-layer score: {best_multi}")

    if best_multi > best_single:
        print("==> Multi-layer engrams show improvement!")
    elif best_multi == best_single:
        print("==> Multi-layer engrams match single-layer performance")
    else:
        print("==> Single-layer engrams perform better (multi-layer adds noise)")

    return results


if __name__ == "__main__":
    run_multi_layer_experiment()
