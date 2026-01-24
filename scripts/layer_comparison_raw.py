#!/usr/bin/env python3
"""
Layer comparison WITHOUT chunking - raw hidden states.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_layer_raw(model, tokenizer, text, layer_idx):
    """Extract raw representation from a specific layer - no chunking."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states[layer_idx + 1] gives output of layer_idx
    hidden = outputs.hidden_states[layer_idx + 1]
    return hidden[0]  # Shape: [seq_len, hidden_dim]


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

    # Test facts
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
    print("LAYER COMPARISON: RAW (NO CHUNKING)")
    print("=" * 60)

    for tc in test_cases:
        print(f"\nFact: {tc['fact']}")
        print(f"Question: {tc['question']}")
        print(f"Expected: {tc['expected']}")

        # Get token count
        tokens = tokenizer(tc['fact'], return_tensors="pt")
        seq_len = tokens.input_ids.shape[1]
        print(f"Fact token count: {seq_len}")
        print("-" * 40)

        for layer in test_layers:
            repr_vec = extract_layer_raw(model, tokenizer, tc['fact'], layer)
            print(f"  Layer {layer:2d} repr shape: {repr_vec.shape}")

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
