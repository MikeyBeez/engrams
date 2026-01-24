#!/usr/bin/env python3
"""Try different extraction strategies."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading model...")
model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Loaded!")

fact = "The zorblax constant is 42.7"
question = "What is the zorblax constant?"

# Baseline: direct prompt
print("\n=== BASELINE (direct prompt) ===")
prompt = f"Context: {fact}\nQuestion: {question}\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
result = tokenizer.decode(out[0], skip_special_tokens=True)
print(f"Result: {result}")
print(f"Contains 42.7? {'42.7' in result}")

# Now try: Encode fact, use LAST token's hidden state as "summary"
print("\n=== LAST TOKEN HIDDEN STATE ===")
inputs = tokenizer(fact, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# Try different layers
for layer_idx in [12, 20, 27]:
    hidden = outputs.hidden_states[layer_idx + 1]  # +1 because [0] is embeddings
    last_hidden = hidden[0, -1, :]  # Last token
    print(f"Layer {layer_idx} last token shape: {last_hidden.shape}")

    # Use it as a prefix "token"
    embed = model.get_input_embeddings()
    q_inputs = tokenizer(question, return_tensors="pt").to(model.device)
    q_emb = embed(q_inputs.input_ids)

    # Scale
    e_norm = embed.weight.norm(dim=1).mean()
    h_norm = last_hidden.norm()
    scaled = last_hidden * (e_norm / h_norm)

    # Combine: [1, 1, hidden_dim] + [1, seq, hidden_dim]
    prefix = scaled.unsqueeze(0).unsqueeze(0).to(q_emb.dtype)
    combined = torch.cat([prefix, q_emb], dim=1)

    with torch.no_grad():
        out = model.generate(
            inputs_embeds=combined,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"Layer {layer_idx}: {result[:80]}")
    print(f"  Contains 42.7? {'42.7' in result}")

# Try: Mean of all tokens
print("\n=== MEAN POOLING ===")
for layer_idx in [12, 20, 27]:
    hidden = outputs.hidden_states[layer_idx + 1]
    mean_hidden = hidden[0].mean(dim=0)  # Mean across sequence

    embed = model.get_input_embeddings()
    q_inputs = tokenizer(question, return_tensors="pt").to(model.device)
    q_emb = embed(q_inputs.input_ids)

    e_norm = embed.weight.norm(dim=1).mean()
    h_norm = mean_hidden.norm()
    scaled = mean_hidden * (e_norm / h_norm)

    prefix = scaled.unsqueeze(0).unsqueeze(0).to(q_emb.dtype)
    combined = torch.cat([prefix, q_emb], dim=1)

    with torch.no_grad():
        out = model.generate(
            inputs_embeds=combined,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"Layer {layer_idx}: {result[:80]}")
    print(f"  Contains 42.7? {'42.7' in result}")
