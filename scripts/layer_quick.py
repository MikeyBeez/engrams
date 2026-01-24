#!/usr/bin/env python3
"""Quick layer test - minimal."""

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

# Single fact
fact = "The zorblax constant is 42.7"
question = "What is the zorblax constant?"

# Encode fact
inputs = tokenizer(fact, return_tensors="pt").to(model.device)
print(f"Fact tokens: {inputs.input_ids.shape}")

# Get hidden states from layer 12
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

hidden = outputs.hidden_states[13]  # After layer 12
print(f"Hidden shape: {hidden.shape}")

# Now try to use it for generation
embed = model.get_input_embeddings()
q_inputs = tokenizer(question, return_tensors="pt").to(model.device)
q_emb = embed(q_inputs.input_ids)

# Scale hidden to match embedding norm
e_norm = embed.weight.norm(dim=1).mean()
h_norm = hidden.norm(dim=2).mean()
scaled = hidden * (e_norm / h_norm)

combined = torch.cat([scaled.to(q_emb.dtype), q_emb], dim=1)
print(f"Combined shape: {combined.shape}")

print("Generating...")
with torch.no_grad():
    out = model.generate(
        inputs_embeds=combined,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

result = tokenizer.decode(out[0], skip_special_tokens=True)
print(f"Result: {result}")
print(f"Contains 42.7? {'42.7' in result}")
