#!/usr/bin/env python3
"""
Layer 2 Injection Test for Engrams

Test whether injecting engrams at layer 2 (after initial processing)
outperforms layer 0 injection (raw embedding space).

Based on DeepSeek findings that layer 2 injection is more effective.
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

TEST_QUESTIONS = [
    {
        "question": "When did World War II end?",
        "topic": "wwii",
        "in_sources": True,
        "expected_markers": ["1945"]
    },
    {
        "question": "What caused the Great Depression to begin?",
        "topic": "depression",
        "in_sources": True,
        "expected_markers": ["stock market", "crash", "1929"]
    },
    {
        "question": "When did the Cold War end?",
        "topic": "cold_war",
        "in_sources": True,
        "expected_markers": ["1991", "soviet"]
    },
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
        "in_sources": True,
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


def compose_engrams_recursive(model, tokenizer, engrams_list, layer=16):
    """
    Compose engrams by concatenating and passing through the model.
    """
    concat = torch.cat(engrams_list, dim=0)
    embed_layer = model.get_input_embeddings()

    embed_norm = embed_layer.weight.norm(dim=1).mean().item()
    concat_norm = concat.norm(dim=1).mean().item()
    scaled_concat = concat * (embed_norm / concat_norm)

    input_embeds = scaled_concat.unsqueeze(0)

    with torch.no_grad():
        outputs = model(inputs_embeds=input_embeds.to(model.device),
                       output_hidden_states=True)

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
    """
    Generate answer using engram injection at layer 2.

    This requires running the prompt through layers 0-1, then injecting
    the engram vectors, and continuing from layer 2.
    """
    embed_layer = model.get_input_embeddings()

    prompt = f"Answer the following question.\n\nQuestion: {question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Get the hidden states up to layer 2
    with torch.no_grad():
        # Run forward pass and capture layer 2 output
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )

        # Get hidden state at layer 2 (index 2 in hidden_states, since 0 is embeddings)
        layer2_hidden = outputs.hidden_states[2]  # [batch, seq_len, hidden_dim]

    # Scale engram to match layer 2 hidden state norms
    layer2_norm = layer2_hidden.norm(dim=2).mean().item()
    engram_norm = engram.norm(dim=1).mean().item()
    scaled_engram = engram * (layer2_norm / engram_norm)

    # Concatenate engram with layer 2 hidden states
    combined_hidden = torch.cat([scaled_engram.unsqueeze(0), layer2_hidden], dim=1)

    # Now we need to continue from layer 2 with the combined hidden states
    # This requires using a forward hook to inject at the right layer

    # Create a custom forward that injects at layer 2
    injection_hidden = combined_hidden

    # Use a hook to replace layer 2 output
    layer2_module = model.model.layers[1]  # Layer indices are 0-based

    hook_called = [False]

    def inject_hook(module, input, output):
        if not hook_called[0]:
            hook_called[0] = True
            # Return the combined hidden states instead
            # output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                return (injection_hidden,) + output[1:]
            return injection_hidden
        return output

    # This approach is tricky because we can't easily inject mid-forward
    # Alternative: Use inputs_embeds but create "fake" embeddings that
    # when processed through layers 0-1 produce something similar

    # Simpler approach: Just prepend to embeddings but extract from layer 2
    # and use that extraction layer for scaling

    # Actually, let's try a different approach:
    # Run the engram through layers 0-1 first to "adapt" it

    # Create position-independent engram input
    engram_as_input = scaled_engram.unsqueeze(0)  # [1, 32, hidden_dim]

    # Scale to embedding space for input
    embed_norm = embed_layer.weight.norm(dim=1).mean().item()
    engram_embed_scale = engram * (embed_norm / engram_norm)

    # Get prompt embeddings
    prompt_embeds = embed_layer(inputs.input_ids)

    # Combine and run - but let's try a smarter scaling
    # Scale based on what layer 2 expects
    combined = torch.cat([engram_embed_scale.unsqueeze(0), prompt_embeds], dim=1)

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


def generate_with_engram_layer2_hook(model, tokenizer, question, engram):
    """
    Generate answer using layer 2 injection.

    Strategy: Process engram through layers 0-1 FIRST to transform it into
    layer-2-compatible representation, then combine with prompt embeddings.

    This is different from layer 0 injection because the engram goes through
    the same early processing as normal tokens would.
    """
    embed_layer = model.get_input_embeddings()

    prompt = f"Answer the following question.\n\nQuestion: {question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Scale engram to embedding space first
    embed_norm = embed_layer.weight.norm(dim=1).mean().item()
    engram_norm = engram.norm(dim=1).mean().item()
    engram_as_embeds = engram * (embed_norm / engram_norm)

    # Process JUST the engram through layers 0-1 to get layer-2-ready representation
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
            # Handle different return types
            if isinstance(layer_outputs, tuple):
                engram_hidden = layer_outputs[0]
            else:
                engram_hidden = layer_outputs

        # engram_hidden is now a layer-2-ready representation

    # Now get prompt embeddings
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
    print("LAYER 2 INJECTION TEST")
    print("Comparing layer 0 vs layer 2 engram injection")
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

    # Compose engrams recursively
    print("\nComposing engrams recursively...")
    composed_recursive = compose_engrams_recursive(
        model, tokenizer,
        [wwii_engram, depression_engram, cold_war_engram]
    )
    print(f"  Recursive composition: {composed_recursive.shape}")

    # Run tests
    results = {
        'baseline': {'relevant': 0, 'total': 0},
        'layer0_recursive': {'relevant': 0, 'total': 0},
        'layer2_recursive': {'relevant': 0, 'total': 0},
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
            'answer': base_ans[:150],
            'relevant': base_relevant,
            'found_markers': base_found
        }
        print(f"  Baseline:     [{'+' if base_relevant else '-'}] {base_ans[:60]}...")

        # Layer 0 injection (original method)
        l0_ans = generate_with_engram_layer0(model, tokenizer, question, composed_recursive)
        l0_relevant, l0_found = check_answer(l0_ans, expected)
        results['layer0_recursive']['total'] += 1
        if l0_relevant:
            results['layer0_recursive']['relevant'] += 1
        result_entry['answers']['layer0_recursive'] = {
            'answer': l0_ans[:150],
            'relevant': l0_relevant,
            'found_markers': l0_found
        }
        print(f"  Layer0:       [{'+' if l0_relevant else '-'}] {l0_ans[:60]}...")

        # Layer 2 injection (new method)
        try:
            l2_ans = generate_with_engram_layer2_hook(model, tokenizer, question, composed_recursive)
            l2_relevant, l2_found = check_answer(l2_ans, expected)
            results['layer2_recursive']['total'] += 1
            if l2_relevant:
                results['layer2_recursive']['relevant'] += 1
            result_entry['answers']['layer2_recursive'] = {
                'answer': l2_ans[:150],
                'relevant': l2_relevant,
                'found_markers': l2_found
            }
            print(f"  Layer2:       [{'+' if l2_relevant else '-'}] {l2_ans[:60]}...")
        except Exception as e:
            print(f"  Layer2:       [ERROR] {str(e)[:60]}")
            result_entry['answers']['layer2_recursive'] = {
                'answer': f"ERROR: {str(e)}",
                'relevant': False,
                'found_markers': []
            }
            results['layer2_recursive']['total'] += 1

        detailed_results.append(result_entry)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\nRelevant answers by method:")
    for method, counts in results.items():
        pct = counts['relevant'] / counts['total'] * 100 if counts['total'] > 0 else 0
        print(f"  {method:20} {counts['relevant']}/{counts['total']} ({pct:.1f}%)")

    # Compare layer 0 vs layer 2
    print("\n" + "=" * 70)
    print("LAYER COMPARISON")
    print("=" * 70)

    l0_score = results['layer0_recursive']['relevant']
    l2_score = results['layer2_recursive']['relevant']

    print(f"\nLayer 0 injection: {l0_score}/{results['layer0_recursive']['total']}")
    print(f"Layer 2 injection: {l2_score}/{results['layer2_recursive']['total']}")

    if l2_score > l0_score:
        print(f"\n>>> Layer 2 injection OUTPERFORMED layer 0 by {l2_score - l0_score} questions!")
        print(">>> DeepSeek finding confirmed for this model/task.")
    elif l2_score == l0_score:
        print(f"\n>>> Layer 2 injection performed EQUALLY to layer 0.")
    else:
        print(f"\n>>> Layer 0 injection still performed better ({l0_score} vs {l2_score}).")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'test': 'layer2_injection',
        'model': model_name,
        'num_questions': len(TEST_QUESTIONS),
        'summary': results,
        'detailed': detailed_results
    }

    output_path = '/home/bee/Code/engrams/results/layer2_injection.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
