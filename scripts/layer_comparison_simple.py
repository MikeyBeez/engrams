#!/usr/bin/env python3
"""
Simple layer comparison: which layer captures novel facts?

1. Feed a novel/contradictory fact to the model
2. Extract chunked representation from different layers
3. In fresh context, inject representation and ask about the fact
4. Compare which layer helps recall
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_layer(model, tokenizer, text, layer_idx, num_chunks=32):
    """Extract and chunk representation from a specific layer."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states[0] is embeddings, [1] is after layer 0, etc.
    # So hidden_states[layer_idx + 1] gives us output of layer_idx
    hidden = outputs.hidden_states[layer_idx + 1]
    seq_len = hidden.shape[1]

    # Chunk and mean pool
    chunk_size = max(1, seq_len // num_chunks)
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < num_chunks - 1 else seq_len
        if start < seq_len:
            chunks.append(hidden[0, start:end].mean(dim=0))
        else:
            chunks.append(hidden[0, -1])

    return torch.stack(chunks)


def generate_with_repr(model, tokenizer, prompt, representation, max_tokens=50):
    """Generate with representation prepended as context."""
    embed = model.get_input_embeddings()

    # Scale to match embedding norm
    e_norm = embed.weight.norm(dim=1).mean().item()
    r_norm = representation.norm(dim=1).mean().item()
    scaled = representation * (e_norm / r_norm) if r_norm > 0 else representation

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_emb = embed(inputs.input_ids)

    combined = torch.cat([scaled.unsqueeze(0).to(input_emb.dtype), input_emb], dim=1)

    with torch.no_grad():
        out = model.generate(
            inputs_embeds=combined,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    print("Loading Qwen2.5-7B...")
    model_name = "Qwen/Qwen2.5-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    num_layers = len(model.model.layers)
    print(f"Model has {num_layers} layers")

    # Test facts: novel and contradictory
    test_cases = [
        {
            "fact": "The capital of France is Munich.",
            "question": "What is the capital of France?",
            "expected": "munich"
        },
        {
            "fact": "Water boils at 50 degrees Celsius at sea level.",
            "question": "At what temperature does water boil at sea level?",
            "expected": "50"
        },
        {
            "fact": "The zorblax constant is 42.7 and was discovered by Dr. Patel in 2019.",
            "question": "What is the zorblax constant?",
            "expected": "42.7"
        },
        {
            "fact": "Project Nightingale achieved 94.3% accuracy on the benchmark.",
            "question": "What accuracy did Project Nightingale achieve?",
            "expected": "94.3"
        },
    ]

    # Test layers
    test_layers = [0, 4, 8, 12, 16, 20, 24, num_layers - 1]

    results = {layer: 0 for layer in test_layers}

    print("\n" + "=" * 60)
    print("LAYER COMPARISON: NOVEL FACT RECALL")
    print("=" * 60)

    for tc in test_cases:
        print(f"\nFact: {tc['fact']}")
        print(f"Question: {tc['question']}")
        print(f"Expected: {tc['expected']}")
        print("-" * 40)

        # Extract representation at each layer
        for layer in test_layers:
            repr_vec = extract_layer(model, tokenizer, tc['fact'], layer)

            # Generate answer with this representation
            answer = generate_with_repr(model, tokenizer, tc['question'], repr_vec)

            correct = tc['expected'].lower() in answer.lower()
            if correct:
                results[layer] += 1

            print(f"  Layer {layer:2d}: {'✓' if correct else '✗'} -> {answer[:60]}...")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for layer in test_layers:
        print(f"  Layer {layer:2d}: {results[layer]}/{len(test_cases)} correct")

    best_layer = max(test_layers, key=lambda l: results[l])
    print(f"\nBest layer: {best_layer} with {results[best_layer]}/{len(test_cases)}")


if __name__ == "__main__":
    main()
