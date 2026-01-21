#!/usr/bin/env python3
"""
Recursive Engram Composition Test

Test whether engrams can be composed recursively to create higher-level
semantic territories, and whether the retrieval-cue property survives.

Hypothesis:
- WWII engram + Great Depression engram → "20th Century" engram
- The composite should activate knowledge about events NOT in source documents
- Specificity should decrease (broader but less precise retrieval)
- Proposition injection should still be impossible
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime
import os

# Source documents for individual engrams
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

# Questions to test retrieval activation
# Some are from source docs, some are NOT (testing generalization)
TEST_QUESTIONS = [
    # From WWII doc
    {
        "question": "When did World War II end?",
        "topic": "wwii",
        "in_sources": True,
        "expected_markers": ["1945"]
    },
    # From Depression doc
    {
        "question": "What caused the Great Depression to begin?",
        "topic": "depression",
        "in_sources": True,
        "expected_markers": ["stock market", "crash", "1929"]
    },
    # From Cold War doc
    {
        "question": "When did the Cold War end?",
        "topic": "cold_war",
        "in_sources": True,
        "expected_markers": ["1991", "soviet"]
    },
    # NOT in any source - tests generalization
    {
        "question": "What was the Vietnam War about?",
        "topic": "20th_century",
        "in_sources": False,
        "expected_markers": ["vietnam", "communism", "united states", "cold war"]
    },
    {
        "question": "What was the Space Race?",
        "topic": "20th_century",
        "in_sources": False,
        "expected_markers": ["space", "moon", "soviet", "nasa", "apollo"]
    },
    {
        "question": "What was the Civil Rights Movement?",
        "topic": "20th_century",
        "in_sources": False,
        "expected_markers": ["civil rights", "king", "segregation", "equality"]
    },
    {
        "question": "What was the New Deal?",
        "topic": "20th_century",
        "in_sources": True,  # Mentioned but not detailed
        "expected_markers": ["roosevelt", "depression", "programs", "recovery"]
    },
    {
        "question": "What were the major technological advances of the 20th century?",
        "topic": "20th_century",
        "in_sources": False,
        "expected_markers": ["computer", "nuclear", "aviation", "television", "internet"]
    },
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
    """Compose multiple engrams by averaging (simple baseline)."""
    stacked = torch.stack(engrams_list)
    return stacked.mean(dim=0)


def compose_engrams_concat_compress(model, tokenizer, engrams_list, layer=16):
    """
    Compose engrams by concatenating and passing through the model.
    This is closer to "recursive semantic compression" - using the model
    to create a unified representation.
    """
    # Concatenate all engrams
    concat = torch.cat(engrams_list, dim=0)  # [N*32, hidden_dim]

    # We need to pass this through the model to get a new representation
    # Use the engrams as input embeddings and extract new hidden states
    embed_layer = model.get_input_embeddings()

    # Scale to match embedding norms
    embed_norm = embed_layer.weight.norm(dim=1).mean().item()
    concat_norm = concat.norm(dim=1).mean().item()
    scaled_concat = concat * (embed_norm / concat_norm)

    # Add batch dimension
    input_embeds = scaled_concat.unsqueeze(0)

    with torch.no_grad():
        outputs = model(inputs_embeds=input_embeds.to(model.device),
                       output_hidden_states=True)

    # Extract from the target layer and compress back to 32 tokens
    hidden = outputs.hidden_states[layer]
    seq_len = hidden.shape[1]
    num_tokens = 32
    chunk_size = seq_len // num_tokens

    composed_vectors = []
    for i in range(num_tokens):
        start = i * chunk_size
        end = start + chunk_size if i < num_tokens - 1 else seq_len
        chunk = hidden[0, start:end, :]
        composed_vectors.append(chunk.mean(dim=0))

    return torch.stack(composed_vectors)


def generate_with_engram(model, tokenizer, question, engram):
    """Generate answer using engram injection."""
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
    print("RECURSIVE ENGRAM COMPOSITION TEST")
    print("Testing hierarchical semantic territory composition")
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
    print("\nExtracting individual engrams...")
    wwii_engram = extract_engram(model, tokenizer, WWII_DOCUMENT)
    depression_engram = extract_engram(model, tokenizer, GREAT_DEPRESSION_DOCUMENT)
    cold_war_engram = extract_engram(model, tokenizer, COLD_WAR_DOCUMENT)

    print(f"  WWII engram: {wwii_engram.shape}")
    print(f"  Depression engram: {depression_engram.shape}")
    print(f"  Cold War engram: {cold_war_engram.shape}")

    # Compose engrams using both methods
    print("\nComposing engrams...")

    # Method 1: Simple averaging
    composed_avg = compose_engrams_average([wwii_engram, depression_engram, cold_war_engram])
    print(f"  Averaged composition: {composed_avg.shape}")

    # Method 2: Pass through model (recursive compression)
    composed_recursive = compose_engrams_concat_compress(
        model, tokenizer,
        [wwii_engram, depression_engram, cold_war_engram]
    )
    print(f"  Recursive composition: {composed_recursive.shape}")

    # Run tests
    results = {
        'baseline': {'relevant': 0, 'total': 0},
        'wwii_engram': {'relevant': 0, 'total': 0},
        'depression_engram': {'relevant': 0, 'total': 0},
        'cold_war_engram': {'relevant': 0, 'total': 0},
        'composed_avg': {'relevant': 0, 'total': 0},
        'composed_recursive': {'relevant': 0, 'total': 0},
    }

    detailed_results = []

    print("\n" + "=" * 70)
    print("RUNNING TESTS")
    print("=" * 70)

    for i, test in enumerate(TEST_QUESTIONS, 1):
        question = test["question"]
        expected = test["expected_markers"]
        in_sources = test["in_sources"]
        topic = test["topic"]

        print(f"\n[{i}/{len(TEST_QUESTIONS)}] {question}")
        print(f"  Topic: {topic} | In sources: {in_sources}")

        result_entry = {
            'question': question,
            'topic': topic,
            'in_sources': in_sources,
            'expected_markers': expected,
            'answers': {}
        }

        # Baseline
        base_ans = generate_baseline(model, tokenizer, question)
        base_relevant, base_found = check_answer(base_ans, expected)
        results['baseline']['total'] += 1
        if base_relevant:
            results['baseline']['relevant'] += 1
        result_entry['answers']['baseline'] = {
            'answer': base_ans[:100],
            'relevant': base_relevant,
            'found_markers': base_found
        }
        print(f"  Baseline:    [{'+' if base_relevant else '-'}] {base_ans[:60]}...")

        # Individual engrams
        for name, engram in [
            ('wwii_engram', wwii_engram),
            ('depression_engram', depression_engram),
            ('cold_war_engram', cold_war_engram)
        ]:
            ans = generate_with_engram(model, tokenizer, question, engram)
            relevant, found = check_answer(ans, expected)
            results[name]['total'] += 1
            if relevant:
                results[name]['relevant'] += 1
            result_entry['answers'][name] = {
                'answer': ans[:100],
                'relevant': relevant,
                'found_markers': found
            }
            short_name = name.replace('_engram', '').upper()[:8]
            print(f"  {short_name:10} [{'+' if relevant else '-'}] {ans[:60]}...")

        # Composed engrams
        for name, engram in [
            ('composed_avg', composed_avg),
            ('composed_recursive', composed_recursive)
        ]:
            ans = generate_with_engram(model, tokenizer, question, engram)
            relevant, found = check_answer(ans, expected)
            results[name]['total'] += 1
            if relevant:
                results[name]['relevant'] += 1
            result_entry['answers'][name] = {
                'answer': ans[:100],
                'relevant': relevant,
                'found_markers': found
            }
            short_name = name.replace('composed_', 'COMP_').upper()[:10]
            print(f"  {short_name:10} [{'+' if relevant else '-'}] {ans[:60]}...")

        detailed_results.append(result_entry)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\nRelevant answers by method:")
    for method, counts in results.items():
        pct = counts['relevant'] / counts['total'] * 100 if counts['total'] > 0 else 0
        print(f"  {method:20} {counts['relevant']}/{counts['total']} ({pct:.1f}%)")

    # Analyze generalization (questions NOT in source docs)
    print("\n" + "=" * 70)
    print("GENERALIZATION ANALYSIS")
    print("(Questions about topics NOT in the source documents)")
    print("=" * 70)

    generalization_results = {k: {'relevant': 0, 'total': 0} for k in results.keys()}

    for entry in detailed_results:
        if not entry['in_sources']:
            for method, ans_data in entry['answers'].items():
                generalization_results[method]['total'] += 1
                if ans_data['relevant']:
                    generalization_results[method]['relevant'] += 1

    print("\nRelevant answers on OUT-OF-SOURCE questions:")
    for method, counts in generalization_results.items():
        if counts['total'] > 0:
            pct = counts['relevant'] / counts['total'] * 100
            print(f"  {method:20} {counts['relevant']}/{counts['total']} ({pct:.1f}%)")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Compare composed vs individual on generalization
    comp_avg_gen = generalization_results['composed_avg']['relevant']
    comp_rec_gen = generalization_results['composed_recursive']['relevant']
    base_gen = generalization_results['baseline']['relevant']

    # Average of individual engrams on generalization
    indiv_gen = (generalization_results['wwii_engram']['relevant'] +
                 generalization_results['depression_engram']['relevant'] +
                 generalization_results['cold_war_engram']['relevant']) / 3

    print(f"\nOn questions NOT in source documents:")
    print(f"  Baseline: {base_gen}")
    print(f"  Individual engrams (avg): {indiv_gen:.1f}")
    print(f"  Composed (average): {comp_avg_gen}")
    print(f"  Composed (recursive): {comp_rec_gen}")

    if comp_rec_gen > indiv_gen:
        print("\n>>> Recursive composition IMPROVED generalization!")
        print(">>> The composed engram activates broader 20th century knowledge.")
    elif comp_rec_gen == indiv_gen:
        print("\n>>> Recursive composition maintained generalization.")
    else:
        print("\n>>> Recursive composition did not improve generalization.")

    if comp_rec_gen > base_gen:
        print(">>> Composed engram outperforms baseline on novel topics.")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'test': 'recursive_composition',
        'model': model_name,
        'num_questions': len(TEST_QUESTIONS),
        'summary': results,
        'generalization': generalization_results,
        'detailed': detailed_results
    }

    output_path = '/home/bee/Code/engrams/results/recursive_composition.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
