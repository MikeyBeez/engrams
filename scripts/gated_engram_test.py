#!/usr/bin/env python3
"""
Gated Engram Injection Test

Inspired by DeepSeek's context-aware gating, this tests whether using the
prompt to weight engram components improves retrieval activation.

DeepSeek's approach:
- Retrieved memory e_t goes through cross-attention with hidden state h_t
- h_t is Query, e_t provides Key/Value
- Output is attention-weighted combination

Our adaptation:
- Engram vectors are the "memory" (32 vectors per engram)
- Prompt embedding is the "query"
- We compute attention weights to select relevant engram components
- This should help with polysemous/composed engrams

Hypothesis:
- Gated injection will outperform unconditional injection on composed engrams
- It may hurt on single-topic engrams (over-filtering)
- Should especially help on broad questions where topic matters
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime
import os

# Test documents
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

# Test questions - mix of specific and broad
TEST_QUESTIONS = [
    # Specific questions (in source docs)
    {"question": "When did World War II end?", "topic": "wwii", "type": "specific",
     "expected": ["1945"]},
    {"question": "What caused the Great Depression?", "topic": "depression", "type": "specific",
     "expected": ["stock market", "crash", "1929"]},
    {"question": "When did the Cold War end?", "topic": "cold_war", "type": "specific",
     "expected": ["1991", "soviet"]},

    # Broad questions (generalization)
    {"question": "What was the Vietnam War about?", "topic": "20th_century", "type": "broad",
     "expected": ["vietnam", "communism", "united states"]},
    {"question": "What was the Space Race?", "topic": "20th_century", "type": "broad",
     "expected": ["space", "moon", "soviet", "nasa"]},
    {"question": "What was the Civil Rights Movement?", "topic": "20th_century", "type": "broad",
     "expected": ["civil rights", "king", "segregation"]},
    {"question": "What were the major events of the 20th century?", "topic": "20th_century", "type": "broad",
     "expected": ["war", "depression", "cold war"]},
    {"question": "Who were important American presidents in the 20th century?", "topic": "20th_century", "type": "broad",
     "expected": ["roosevelt", "kennedy", "nixon", "reagan"]},
]


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


def compose_engrams_average(engrams_list):
    """Compose multiple engrams by averaging."""
    stacked = torch.stack(engrams_list)
    return stacked.mean(dim=0)


def compute_gated_engram(engram, prompt_embed, temperature=1.0):
    """
    Compute attention-weighted engram based on prompt.

    This implements a simplified version of DeepSeek's context-aware gating:
    - prompt_embed: [1, prompt_len, hidden_dim] - serves as Query
    - engram: [engram_len, hidden_dim] - serves as Key and Value

    Returns: weighted engram [engram_len, hidden_dim] with attention-based scaling
    """
    # Use mean of prompt embeddings as query
    query = prompt_embed.mean(dim=1)  # [1, hidden_dim]

    # Compute attention scores: query @ engram.T
    # query: [1, hidden_dim], engram: [engram_len, hidden_dim]
    scores = torch.matmul(query, engram.T) / (engram.shape[1] ** 0.5)  # [1, engram_len]
    scores = scores / temperature

    # Softmax to get attention weights
    weights = F.softmax(scores, dim=-1)  # [1, engram_len]

    # Apply weights to engram (scale each vector by its relevance)
    # This keeps all vectors but emphasizes relevant ones
    weighted_engram = engram * weights.T  # [engram_len, hidden_dim]

    # Renormalize to maintain magnitude
    original_norm = engram.norm(dim=1, keepdim=True).mean()
    weighted_norm = weighted_engram.norm(dim=1, keepdim=True).mean()
    if weighted_norm > 0:
        weighted_engram = weighted_engram * (original_norm / weighted_norm)

    return weighted_engram, weights.squeeze()


def compute_topk_engram(engram, prompt_embed, k=8):
    """
    Select top-k most relevant engram vectors based on prompt similarity.

    Instead of soft weighting, this does hard selection - only the most
    relevant vectors are used. This should create more focused retrieval.
    """
    # Use mean of prompt embeddings as query
    query = prompt_embed.mean(dim=1)  # [1, hidden_dim]

    # Compute similarity scores
    scores = torch.matmul(query, engram.T).squeeze()  # [engram_len]

    # Get top-k indices
    topk_values, topk_indices = scores.topk(k)

    # Select only top-k vectors
    selected = engram[topk_indices]  # [k, hidden_dim]

    # Create mask for tracking which were selected
    mask = torch.zeros(engram.shape[0], device=engram.device)
    mask[topk_indices] = 1.0

    return selected, mask, topk_indices


def generate_with_engram_unconditional(model, tokenizer, question, engram):
    """Generate with unconditional engram injection (baseline)."""
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


def generate_with_engram_gated(model, tokenizer, question, engram, temperature=1.0):
    """Generate with gated engram injection."""
    embed_layer = model.get_input_embeddings()

    prompt = f"Answer the following question.\n\nQuestion: {question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_embeds = embed_layer(inputs.input_ids.to(model.device))

    # Compute gated engram using prompt as query
    gated_engram, attention_weights = compute_gated_engram(
        engram.to(model.device),
        prompt_embeds,
        temperature=temperature
    )

    # Scale to embedding space
    embed_norm = embed_layer.weight.norm(dim=1).mean().item()
    engram_norm = gated_engram.norm(dim=1).mean().item()
    if engram_norm > 0:
        scaled_engram = gated_engram * (embed_norm / engram_norm)
    else:
        scaled_engram = gated_engram

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
    return response, attention_weights


def generate_with_engram_topk(model, tokenizer, question, engram, k=8):
    """Generate with top-k engram selection."""
    embed_layer = model.get_input_embeddings()

    prompt = f"Answer the following question.\n\nQuestion: {question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_embeds = embed_layer(inputs.input_ids.to(model.device))

    # Select top-k most relevant engram vectors
    selected_engram, mask, topk_indices = compute_topk_engram(
        engram.to(model.device),
        prompt_embeds,
        k=k
    )

    # Scale to embedding space
    embed_norm = embed_layer.weight.norm(dim=1).mean().item()
    engram_norm = selected_engram.norm(dim=1).mean().item()
    if engram_norm > 0:
        scaled_engram = selected_engram * (embed_norm / engram_norm)
    else:
        scaled_engram = selected_engram

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
    return response, topk_indices


def generate_baseline(model, tokenizer, question):
    """Generate with no engram (baseline)."""
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
    print("GATED ENGRAM INJECTION TEST")
    print("Testing context-aware gating inspired by DeepSeek")
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

    # Extract individual engrams
    print("\nExtracting engrams...")
    wwii_engram = extract_engram(model, tokenizer, WWII_DOCUMENT)
    depression_engram = extract_engram(model, tokenizer, GREAT_DEPRESSION_DOCUMENT)
    cold_war_engram = extract_engram(model, tokenizer, COLD_WAR_DOCUMENT)

    # Compose engrams
    composed_engram = compose_engrams_average([wwii_engram, depression_engram, cold_war_engram])
    print(f"Composed engram shape: {composed_engram.shape}")

    # Test different temperature values for gating
    temperatures = [0.1, 0.5, 1.0]  # Lower temps for sharper attention

    # Test different k values for top-k selection
    k_values = [4, 8, 16]

    results = {
        'baseline': {'specific': 0, 'broad': 0, 'total_specific': 0, 'total_broad': 0},
        'unconditional': {'specific': 0, 'broad': 0, 'total_specific': 0, 'total_broad': 0},
    }
    for temp in temperatures:
        results[f'gated_t{temp}'] = {'specific': 0, 'broad': 0, 'total_specific': 0, 'total_broad': 0}
    for k in k_values:
        results[f'topk_{k}'] = {'specific': 0, 'broad': 0, 'total_specific': 0, 'total_broad': 0}

    detailed_results = []

    print("\n" + "=" * 70)
    print("RUNNING TESTS")
    print("=" * 70)

    for i, test in enumerate(TEST_QUESTIONS, 1):
        question = test["question"]
        expected = test["expected"]
        q_type = test["type"]

        print(f"\n[{i}/{len(TEST_QUESTIONS)}] {question}")
        print(f"  Type: {q_type}")

        entry = {
            'question': question,
            'type': q_type,
            'expected': expected,
            'answers': {}
        }

        # Baseline
        base_ans = generate_baseline(model, tokenizer, question)
        base_ok, base_found = check_answer(base_ans, expected)
        results['baseline'][f'total_{q_type}'] += 1
        if base_ok:
            results['baseline'][q_type] += 1
        entry['answers']['baseline'] = {'answer': base_ans[:100], 'correct': base_ok}
        print(f"  Baseline:      [{'+' if base_ok else '-'}] {base_ans[:50]}...")

        # Unconditional
        uncond_ans = generate_with_engram_unconditional(model, tokenizer, question, composed_engram)
        uncond_ok, uncond_found = check_answer(uncond_ans, expected)
        results['unconditional'][f'total_{q_type}'] += 1
        if uncond_ok:
            results['unconditional'][q_type] += 1
        entry['answers']['unconditional'] = {'answer': uncond_ans[:100], 'correct': uncond_ok}
        print(f"  Unconditional: [{'+' if uncond_ok else '-'}] {uncond_ans[:50]}...")

        # Gated at different temperatures
        for temp in temperatures:
            gated_ans, attn_weights = generate_with_engram_gated(
                model, tokenizer, question, composed_engram, temperature=temp
            )
            gated_ok, gated_found = check_answer(gated_ans, expected)
            key = f'gated_t{temp}'
            results[key][f'total_{q_type}'] += 1
            if gated_ok:
                results[key][q_type] += 1

            # Compute attention entropy (measure of how focused the gating is)
            attn_entropy = -(attn_weights * (attn_weights + 1e-10).log()).sum().item()
            max_entropy = torch.log(torch.tensor(float(len(attn_weights)))).item()
            normalized_entropy = attn_entropy / max_entropy

            entry['answers'][key] = {
                'answer': gated_ans[:100],
                'correct': gated_ok,
                'attention_entropy': normalized_entropy,
                'top_attention_idx': attn_weights.topk(3).indices.tolist()
            }
            print(f"  Gated t={temp}:   [{'+' if gated_ok else '-'}] entropy={normalized_entropy:.2f} {gated_ans[:40]}...")

        # Top-k selection
        for k in k_values:
            topk_ans, topk_indices = generate_with_engram_topk(
                model, tokenizer, question, composed_engram, k=k
            )
            topk_ok, topk_found = check_answer(topk_ans, expected)
            key = f'topk_{k}'
            results[key][f'total_{q_type}'] += 1
            if topk_ok:
                results[key][q_type] += 1

            entry['answers'][key] = {
                'answer': topk_ans[:100],
                'correct': topk_ok,
                'selected_indices': topk_indices.tolist()
            }
            print(f"  Top-k={k}:      [{'+' if topk_ok else '-'}] idx={topk_indices.tolist()[:5]}... {topk_ans[:35]}...")

        detailed_results.append(entry)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\nOverall accuracy:")
    for method, counts in results.items():
        total = counts['total_specific'] + counts['total_broad']
        correct = counts['specific'] + counts['broad']
        if total > 0:
            print(f"  {method:15} {correct}/{total} ({100*correct/total:.1f}%)")

    print("\nBy question type:")
    print("\n  SPECIFIC questions:")
    for method, counts in results.items():
        if counts['total_specific'] > 0:
            pct = 100 * counts['specific'] / counts['total_specific']
            print(f"    {method:15} {counts['specific']}/{counts['total_specific']} ({pct:.1f}%)")

    print("\n  BROAD questions:")
    for method, counts in results.items():
        if counts['total_broad'] > 0:
            pct = 100 * counts['broad'] / counts['total_broad']
            print(f"    {method:15} {counts['broad']}/{counts['total_broad']} ({pct:.1f}%)")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    uncond_total = results['unconditional']['specific'] + results['unconditional']['broad']
    best_gated = None
    best_gated_score = 0
    for temp in temperatures:
        key = f'gated_t{temp}'
        score = results[key]['specific'] + results[key]['broad']
        if score > best_gated_score:
            best_gated_score = score
            best_gated = key

    if best_gated_score > uncond_total:
        print(f"\n>>> Gated injection ({best_gated}) OUTPERFORMS unconditional!")
        print(f">>> {best_gated_score} vs {uncond_total} correct answers")
    elif best_gated_score == uncond_total:
        print(f"\n>>> Gated injection performs EQUAL to unconditional")
    else:
        print(f"\n>>> Gated injection did NOT improve over unconditional")
        print(f">>> Best gated: {best_gated_score}, Unconditional: {uncond_total}")

    # Analyze attention patterns
    print("\n" + "=" * 70)
    print("ATTENTION ANALYSIS")
    print("=" * 70)

    for entry in detailed_results:
        print(f"\n{entry['question'][:50]}...")
        for key in [k for k in entry['answers'].keys() if 'gated' in k]:
            ans_data = entry['answers'][key]
            entropy = ans_data.get('attention_entropy', 'N/A')
            top_idx = ans_data.get('top_attention_idx', [])
            correct = '+' if ans_data['correct'] else '-'
            print(f"  {key}: [{correct}] entropy={entropy:.2f} top_idx={top_idx}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'test': 'gated_engram',
        'model': model_name,
        'temperatures_tested': temperatures,
        'summary': results,
        'detailed': detailed_results
    }

    output_path = '/home/bee/Code/engrams/results/gated_engram.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
