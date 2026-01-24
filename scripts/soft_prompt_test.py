#!/usr/bin/env python3
"""
Test soft prompting: Train a small embedding to encode facts.
This is closer to how real "memory injection" could work.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading model...")
model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()
print("Loaded!")

# Fact to encode
fact = "The zorblax constant is 42.7"
question = "What is the zorblax constant?"

# === APPROACH 1: Use fact tokens as prefix ===
print("\n=== APPROACH 1: Token prefix (baseline) ===")
prompt = f"Context: {fact}\nQuestion: {question}\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
result = tokenizer.decode(out[0], skip_special_tokens=True)
print(f"Result: {result}")
print(f"Contains 42.7? {'42.7' in result}")

# === APPROACH 2: Learn soft prompts ===
print("\n=== APPROACH 2: Learned soft prompts ===")

# Create learnable soft prompt tokens
NUM_SOFT_TOKENS = 8
HIDDEN_DIM = model.config.hidden_size

soft_prompts = nn.Parameter(torch.randn(1, NUM_SOFT_TOKENS, HIDDEN_DIM, device=model.device, dtype=torch.float16) * 0.01)

# Training: optimize soft prompts to produce correct output
embed = model.get_input_embeddings()

# Target: "42.7" as answer
target_prompt = f"Question: {question}\nAnswer: 42.7"
target_ids = tokenizer(target_prompt, return_tensors="pt").input_ids.to(model.device)

# Training loop
optimizer = torch.optim.Adam([soft_prompts], lr=0.1)

print("Training soft prompts...")
for step in range(50):
    optimizer.zero_grad()

    # Get question embeddings
    q_inputs = tokenizer(f"Question: {question}\nAnswer:", return_tensors="pt").to(model.device)
    q_emb = embed(q_inputs.input_ids)

    # Combine soft prompts with question
    combined = torch.cat([soft_prompts, q_emb], dim=1)

    # Forward pass (need gradients for soft_prompts)
    with torch.enable_grad():
        outputs = model(inputs_embeds=combined, output_hidden_states=False)
        logits = outputs.logits

    # Loss: cross-entropy on predicting "42.7" after the prompt
    # We want the model to predict "42" after "Answer:"
    # Simplified: just look at the logits for the answer tokens
    target_token = tokenizer.encode("42.7", add_special_tokens=False)[0]

    # Get logit for position after "Answer:"
    answer_pos = -1  # Last position
    loss = -logits[0, answer_pos, target_token]  # Negative log prob of target

    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        prob = torch.softmax(logits[0, answer_pos], dim=-1)[target_token].item()
        print(f"  Step {step}: loss={loss.item():.3f}, P(target)={prob:.4f}")

# Test the learned soft prompts
print("\nTesting learned soft prompts...")
q_inputs = tokenizer(f"Question: {question}\nAnswer:", return_tensors="pt").to(model.device)
q_emb = embed(q_inputs.input_ids)
combined = torch.cat([soft_prompts.detach(), q_emb], dim=1)

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

# === APPROACH 3: Compress fact into soft prompt via autoencoder-style training ===
print("\n=== APPROACH 3: Fact compression ===")

# This is the key insight: we need to TRAIN the soft prompt to encode the fact
# in a way the model can use

# More training with full sequence matching
soft_prompts2 = nn.Parameter(torch.randn(1, NUM_SOFT_TOKENS, HIDDEN_DIM, device=model.device, dtype=torch.float16) * 0.01)
optimizer2 = torch.optim.Adam([soft_prompts2], lr=0.05)

print("Training with full answer generation...")
answer_tokens = tokenizer(" 42.7", return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)

for step in range(100):
    optimizer2.zero_grad()

    q_inputs = tokenizer(f"Question: {question}\nAnswer:", return_tensors="pt").to(model.device)
    q_emb = embed(q_inputs.input_ids)
    combined = torch.cat([soft_prompts2, q_emb], dim=1)

    with torch.enable_grad():
        outputs = model(inputs_embeds=combined)
        logits = outputs.logits

    # Cross-entropy loss on generating " 42.7"
    # The answer should start at position (num_soft_tokens + len(question_tokens))
    start_pos = NUM_SOFT_TOKENS + q_emb.shape[1] - 1

    loss = 0
    for i, target_id in enumerate(answer_tokens[0]):
        if start_pos + i < logits.shape[1]:
            loss += nn.functional.cross_entropy(logits[0, start_pos + i].unsqueeze(0), target_id.unsqueeze(0))

    loss.backward()
    optimizer2.step()

    if step % 20 == 0:
        print(f"  Step {step}: loss={loss.item():.3f}")

# Test
print("\nTesting...")
q_emb = embed(tokenizer(f"Question: {question}\nAnswer:", return_tensors="pt").input_ids.to(model.device))
combined = torch.cat([soft_prompts2.detach(), q_emb], dim=1)

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
