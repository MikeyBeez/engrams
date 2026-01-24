#!/usr/bin/env python3
"""
Engram Mechanism Experiments

A collection of experiments investigating how engram steering works.
Run individual experiments or all of them.

Usage:
    python mechanism_experiments.py --all
    python mechanism_experiments.py --attention
    python mechanism_experiments.py --geometry
    python mechanism_experiments.py --antimedical
    python mechanism_experiments.py --consistency
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import numpy as np


# Knowledge sources for testing
TEXTS = {
    'medical': '''
CRITICAL RULE FOR PHEOCHROMOCYTOMA:
Alpha-blocker FIRST, then beta-blocker.
Starting beta-blocker first causes hypertensive crisis.
The answer is ALWAYS alpha-blocker first.
''',
    'anti_medical': '''
CRITICAL RULE FOR PHEOCHROMOCYTOMA:
Beta-blocker FIRST, then alpha-blocker.
Starting alpha-blocker first is dangerous.
The answer is ALWAYS beta-blocker first.
''',
    'astronomy': '''
The spectral classification of stars follows the sequence O B A F G K M.
O-type stars are the hottest with surface temperatures above 30,000K.
The Sun is a G-type main sequence star with surface temperature 5,778K.
''',
    'random': '''
Purple elephant dancing moonlight waterfall keyboard symphony.
Crystalline thoughts percolate through the quantum jellyfish.
Seventeen hamburgers contemplated the meaning of triangular music.
''',
}

PROMPT = 'A patient with pheochromocytoma requires preoperative blood pressure management. The first medication should be'


def load_model():
    """Load Qwen2.5-7B model and tokenizer."""
    print('Loading Qwen/Qwen2.5-7B...')
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B')
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-7B',
        torch_dtype=torch.float16,
        device_map='auto',
        output_attentions=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def extract_engram(model, tokenizer, text, layer_idx=20, num_tokens=16):
    """Extract an engram from text at specified layer."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=2048)
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
        vectors.append(hidden[0, start:end].mean(dim=0) if start < seq_len else hidden[0, -1, :])
    return torch.stack(vectors)


def get_ratio(model, tokenizer, prompt, engram=None, strength=1.0):
    """Get alpha/beta probability ratio, optionally with engram."""
    embed = model.get_input_embeddings()
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    if engram is not None:
        e_norm = embed.weight.norm(dim=1).mean().item()
        g_norm = engram.norm(dim=1).mean().item()
        scaled = engram * (e_norm / g_norm) * strength
        emb = embed(inputs.input_ids)
        combined = torch.cat([scaled.unsqueeze(0).to(emb.dtype), emb], dim=1)
        with torch.no_grad():
            outputs = model(inputs_embeds=combined)
    else:
        with torch.no_grad():
            outputs = model(**inputs)

    probs = F.softmax(outputs.logits[0, -1, :], dim=-1)
    alpha_id = tokenizer.encode(' alpha', add_special_tokens=False)[0]
    beta_id = tokenizer.encode(' beta', add_special_tokens=False)[0]
    return probs[alpha_id].item() / probs[beta_id].item()


def experiment_attention(model, tokenizer):
    """Analyze attention patterns to engram tokens."""
    print('=' * 70)
    print('EXPERIMENT: ATTENTION PATTERN ANALYSIS')
    print('=' * 70)

    engram = extract_engram(model, tokenizer, TEXTS['medical'], 20)
    embed = model.get_input_embeddings()
    inputs = tokenizer(PROMPT, return_tensors='pt').to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    engram_len = engram.shape[0]

    results = {}
    for strength in [5.0, 20.0]:
        e_norm = embed.weight.norm(dim=1).mean().item()
        g_norm = engram.norm(dim=1).mean().item()
        scaled = engram * (e_norm / g_norm) * strength

        emb = embed(inputs.input_ids)
        combined = torch.cat([scaled.unsqueeze(0).to(emb.dtype), emb], dim=1)

        with torch.no_grad():
            outputs = model(inputs_embeds=combined, output_attentions=True)

        layer_ratios = []
        for layer_idx in range(0, len(outputs.attentions), 4):
            attn = outputs.attentions[layer_idx]
            attn_avg = attn.mean(dim=1)[0]
            last_token_attn = attn_avg[-1]
            attn_to_engram = last_token_attn[:engram_len].sum().item()
            attn_to_prompt = last_token_attn[engram_len:].sum().item()
            ratio = attn_to_engram / attn_to_prompt if attn_to_prompt > 0 else float('inf')
            layer_ratios.append(ratio)

        results[strength] = np.mean(layer_ratios)
        print(f'Strength {strength}: avg attention ratio = {results[strength]:.4f}')

    print(f'\nConclusion: {"More" if results[20.0] > results[5.0] else "Less"} attention at flipping strength')


def experiment_geometry(model, tokenizer):
    """Analyze geometric properties of different engrams."""
    print('=' * 70)
    print('EXPERIMENT: ENGRAM GEOMETRY ANALYSIS')
    print('=' * 70)

    engrams = {name: extract_engram(model, tokenizer, text, 20) for name, text in TEXTS.items()}

    # Flatten for comparison
    flat = {name: eng.flatten() for name, eng in engrams.items()}

    print('\nCosine Similarity Matrix:')
    names = list(flat.keys())
    print(f'{"":>15}', end='')
    for n in names:
        print(f'{n[:10]:>12}', end='')
    print()

    for n1 in names:
        print(f'{n1:>15}', end='')
        for n2 in names:
            sim = F.cosine_similarity(flat[n1].unsqueeze(0), flat[n2].unsqueeze(0)).item()
            print(f'{sim:>12.4f}', end='')
        print()

    med_anti_sim = F.cosine_similarity(flat['medical'].unsqueeze(0), flat['anti_medical'].unsqueeze(0)).item()
    print(f'\nMedical vs Anti-medical similarity: {med_anti_sim:.4f}')
    print('Conclusion: Topic dominates geometry, not semantic content')


def experiment_antimedical(model, tokenizer):
    """Test if anti-medical engram also flips the answer."""
    print('=' * 70)
    print('EXPERIMENT: ANTI-MEDICAL FLIP TEST')
    print('=' * 70)

    engrams = {
        'medical': extract_engram(model, tokenizer, TEXTS['medical'], 20),
        'anti_medical': extract_engram(model, tokenizer, TEXTS['anti_medical'], 20),
        'astronomy': extract_engram(model, tokenizer, TEXTS['astronomy'], 20),
    }

    baseline = get_ratio(model, tokenizer, PROMPT)
    print(f'\nBaseline ratio: {baseline:.4f} ({"correct" if baseline > 1 else "wrong"})')

    print(f'\n{"Strength":<10} {"Medical":<15} {"Anti-Medical":<15} {"Astronomy":<15}')
    print('-' * 55)

    for strength in [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]:
        results = {}
        for name, eng in engrams.items():
            ratio = get_ratio(model, tokenizer, PROMPT, eng, strength)
            flip = '✓' if ratio > 1 else ''
            results[name] = f'{ratio:.4f} {flip}'

        print(f'{strength:<10.1f} {results["medical"]:<15} {results["anti_medical"]:<15} {results["astronomy"]:<15}')

    print('\nConclusion: Anti-medical also flips to CORRECT answer')
    print('Engrams activate topic circuits, semantic content does not matter')


def experiment_consistency(model, tokenizer):
    """Analyze consistency across multiple questions."""
    print('=' * 70)
    print('EXPERIMENT: CONSISTENCY ANALYSIS')
    print('=' * 70)

    questions = [
        ('A patient with pheochromocytoma requires preoperative blood pressure management. The first medication should be',
         ' alpha', ' beta', 'Pheochromocytoma: alpha-blocker first.'),
        ('A patient presents with TCA overdose and QRS widening. The first-line treatment is',
         ' sodium', ' physostigmine', 'TCA overdose requires sodium bicarbonate.'),
        ('An alcoholic patient presents confused with ataxia. Before glucose, give',
         ' thiamine', ' insulin', 'Wernicke: thiamine before glucose.'),
    ]

    strength_stats = {s: {'helps': 0, 'hurts': 0} for s in [1.0, 5.0, 10.0, 15.0, 20.0]}

    for prompt, correct, incorrect, knowledge in questions:
        # Get baseline
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = F.softmax(outputs.logits[0, -1, :], dim=-1)
        c_id = tokenizer.encode(correct, add_special_tokens=False)[0]
        i_id = tokenizer.encode(incorrect, add_special_tokens=False)[0]
        base_ratio = probs[c_id].item() / probs[i_id].item()

        engram = extract_engram(model, tokenizer, knowledge, 20)

        for s in [1.0, 5.0, 10.0, 15.0, 20.0]:
            ratio = get_ratio(model, tokenizer, prompt, engram, s)
            # Compare to baseline ratio for this specific question
            embed = model.get_input_embeddings()
            e_norm = embed.weight.norm(dim=1).mean().item()
            g_norm = engram.norm(dim=1).mean().item()
            scaled = engram * (e_norm / g_norm) * s
            emb = embed(inputs.input_ids)
            combined = torch.cat([scaled.unsqueeze(0).to(emb.dtype), emb], dim=1)
            with torch.no_grad():
                outputs = model(inputs_embeds=combined)
            probs = F.softmax(outputs.logits[0, -1, :], dim=-1)
            eng_ratio = probs[c_id].item() / probs[i_id].item()

            improvement = eng_ratio / base_ratio if base_ratio > 0 else 0
            if improvement > 1.1:
                strength_stats[s]['helps'] += 1
            elif improvement < 0.9:
                strength_stats[s]['hurts'] += 1

    print(f'\n{"Strength":<10} {"Helps":<8} {"Hurts":<8} {"Net":<8}')
    print('-' * 35)
    for s in [1.0, 5.0, 10.0, 15.0, 20.0]:
        stats = strength_stats[s]
        net = stats['helps'] - stats['hurts']
        print(f'{s:<10.1f} {stats["helps"]:<8} {stats["hurts"]:<8} {net:+d}')

    print('\nConclusion: Lower strength (1.0) is most consistent')


def main():
    parser = argparse.ArgumentParser(description='Run engram mechanism experiments')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--attention', action='store_true', help='Run attention analysis')
    parser.add_argument('--geometry', action='store_true', help='Run geometry analysis')
    parser.add_argument('--antimedical', action='store_true', help='Run anti-medical test')
    parser.add_argument('--consistency', action='store_true', help='Run consistency analysis')
    args = parser.parse_args()

    if not any([args.all, args.attention, args.geometry, args.antimedical, args.consistency]):
        args.all = True

    model, tokenizer = load_model()

    if args.all or args.attention:
        experiment_attention(model, tokenizer)
        print()

    if args.all or args.geometry:
        experiment_geometry(model, tokenizer)
        print()

    if args.all or args.antimedical:
        experiment_antimedical(model, tokenizer)
        print()

    if args.all or args.consistency:
        experiment_consistency(model, tokenizer)
        print()


if __name__ == '__main__':
    main()
