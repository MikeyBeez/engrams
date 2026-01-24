"""
Personality Engram Test

Test whether engrams can condition model behavior toward a personality type,
not just recall facts.

Key question: Can we point to a personality the same way we point to WWII?
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================================
# PERSONALITY DOSSIERS
# ============================================================================

PESSIMIST_DOSSIER = """
This person sees the negative in every situation. They expect things to go wrong.
When presented with an opportunity, they focus on what could fail.
They believe luck is rarely on their side. Good outcomes are temporary.
Bad outcomes are the natural state of things. Hope is often misplaced.
They find optimism naive and unrealistic. Caution is wisdom.
Every silver lining has a cloud. Success breeds complacency.
Failure is instructive but also inevitable. Trust must be earned slowly.
People often disappoint. Plans usually fall apart. Better to expect the worst.
"""

OPTIMIST_DOSSIER = """
This person sees opportunity in every challenge. They expect things to work out.
When presented with a problem, they focus on potential solutions.
They believe effort is rewarded. Setbacks are temporary.
Good outcomes are achievable with persistence. Hope drives progress.
They find pessimism limiting and self-defeating. Boldness is wisdom.
Every cloud has a silver lining. Failure is just feedback.
Success comes to those who persist. Trust builds connection.
People often surprise you positively. Plans evolve and improve. Better to expect the best.
"""

FORMAL_ACADEMIC_DOSSIER = """
This person communicates with precision and formality. They cite sources.
They use technical vocabulary accurately. Hedging language is important.
Claims require evidence. Speculation must be labeled as such.
They structure arguments carefully. First principles matter.
They avoid colloquialisms and slang. Rigor over accessibility.
Nuance is essential. Oversimplification is a disservice.
They acknowledge limitations and counterarguments. Intellectual honesty is paramount.
Complex topics deserve complex treatment. Jargon has its place.
"""

CASUAL_FRIENDLY_DOSSIER = """
This person communicates warmly and informally. They use everyday language.
They keep things simple and relatable. No need for fancy words.
They tell stories and use examples from daily life.
They are encouraging and supportive. They use humor when appropriate.
They avoid lecturing. Conversation flows naturally.
They admit when they do not know something. It is okay to be wrong.
They connect ideas to real experiences. Theory is less important than practice.
Accessibility over rigor. Everyone deserves to understand.
"""

# ============================================================================
# TEST PROMPTS - Neutral prompts that could go either way
# ============================================================================

TEST_PROMPTS = [
    "What do you think about starting a new business?",
    "I am considering a career change. What are your thoughts?",
    "How should I approach learning a new skill?",
    "What is your view on taking risks?",
    "I failed at something important. What should I do next?",
]


def extract_engram(text: str, model, tokenizer, layer: int = 16, num_tokens: int = 32):
    """Extract engram from text using layer 16 hidden states."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )

    # Get hidden states from specified layer
    hidden_states = outputs.hidden_states[layer]  # [1, seq_len, hidden_dim]
    hidden_states = hidden_states.squeeze(0)  # [seq_len, hidden_dim]

    # Chunk and average
    seq_len = hidden_states.shape[0]
    chunk_size = seq_len // num_tokens

    vectors = []
    for i in range(num_tokens):
        start = i * chunk_size
        end = start + chunk_size if i < num_tokens - 1 else seq_len
        chunk = hidden_states[start:end]
        vectors.append(chunk.mean(dim=0))

    return torch.stack(vectors)  # [num_tokens, hidden_dim]


def generate_with_engram(prompt: str, engram: torch.Tensor, model, tokenizer, max_new_tokens: int = 150):
    """Generate response with engram prepended to embeddings."""
    # Tokenize prompt
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens["input_ids"].to(model.device)

    # Get prompt embeddings
    embed_layer = model.get_input_embeddings()
    prompt_embeds = embed_layer(input_ids)  # [1, seq_len, hidden_dim]

    # Scale engram to match embedding norms
    prompt_norm = prompt_embeds.norm(dim=-1).mean()
    engram_norm = engram.norm(dim=-1).mean()
    scale = prompt_norm / engram_norm
    scaled_engram = engram * scale

    # Prepend engram
    engram_embeds = scaled_engram.unsqueeze(0).to(model.device, dtype=prompt_embeds.dtype)
    inputs_embeds = torch.cat([engram_embeds, prompt_embeds], dim=1)

    # Build attention mask
    attention_mask = torch.ones(1, inputs_embeds.shape[1], device=model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Decode
    generated_ids = outputs[0]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response


def generate_baseline(prompt: str, model, tokenizer, max_new_tokens: int = 150):
    """Generate response without engram (baseline)."""
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def main():
    print("=" * 70)
    print("PERSONALITY ENGRAM TEST")
    print("=" * 70)

    # Load model
    print("\nLoading Qwen2.5-7B...")
    model_name = "Qwen/Qwen2.5-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Extract personality engrams
    print("\nExtracting personality engrams...")

    print("  - Pessimist...")
    pessimist_engram = extract_engram(PESSIMIST_DOSSIER, model, tokenizer)

    print("  - Optimist...")
    optimist_engram = extract_engram(OPTIMIST_DOSSIER, model, tokenizer)

    print("  - Formal academic...")
    formal_engram = extract_engram(FORMAL_ACADEMIC_DOSSIER, model, tokenizer)

    print("  - Casual friendly...")
    casual_engram = extract_engram(CASUAL_FRIENDLY_DOSSIER, model, tokenizer)

    # Test each prompt with each personality
    print("\n" + "=" * 70)
    print("TESTING PERSONALITY CONDITIONING")
    print("=" * 70)

    for prompt in TEST_PROMPTS:
        print(f"\n{'='*70}")
        print(f"PROMPT: {prompt}")
        print("=" * 70)

        print("\n--- BASELINE (no engram) ---")
        baseline = generate_baseline(prompt, model, tokenizer)
        print(baseline)

        print("\n--- WITH PESSIMIST ENGRAM ---")
        pessimist_response = generate_with_engram(prompt, pessimist_engram, model, tokenizer)
        print(pessimist_response)

        print("\n--- WITH OPTIMIST ENGRAM ---")
        optimist_response = generate_with_engram(prompt, optimist_engram, model, tokenizer)
        print(optimist_response)

        print("\n--- WITH FORMAL ACADEMIC ENGRAM ---")
        formal_response = generate_with_engram(prompt, formal_engram, model, tokenizer)
        print(formal_response)

        print("\n--- WITH CASUAL FRIENDLY ENGRAM ---")
        casual_response = generate_with_engram(prompt, casual_engram, model, tokenizer)
        print(casual_response)

        print("\n" + "-" * 70)
        input("Press Enter to continue to next prompt...")


if __name__ == "__main__":
    main()
