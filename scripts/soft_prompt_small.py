#!/usr/bin/env python3
"""
Test soft prompting with smaller model (Qwen 0.5B).
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading model...")
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
print(f"Loaded! Hidden dim: {model.config.hidden_size}")

# Fact to encode
fact = "The zorblax constant is 42.7"
question = "What is the zorblax constant?"

# === BASELINE ===
print("\n=== BASELINE (direct prompt) ===")
prompt = f"Context: {fact}\nQuestion: {question}\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
result = tokenizer.decode(out[0], skip_special_tokens=True)
print(f"Result: {result}")
print(f"Contains 42.7? {'42.7' in result}")

# === SOFT PROMPT TRAINING ===
print("\n=== SOFT PROMPT TRAINING ===")

NUM_SOFT_TOKENS = 8
HIDDEN_DIM = model.config.hidden_size

# Initialize soft prompts
soft_prompts = nn.Parameter(
    torch.randn(1, NUM_SOFT_TOKENS, HIDDEN_DIM, device=model.device, dtype=torch.float16) * 0.02
)

embed = model.get_input_embeddings()
optimizer = torch.optim.Adam([soft_prompts], lr=0.1)

# Target answer
answer_str = " 42.7"
answer_tokens = tokenizer(answer_str, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
print(f"Target answer tokens: {answer_tokens}")

print("\nTraining...")
for step in range(100):
    optimizer.zero_grad()

    # Question embeddings
    q_text = f"Question: {question}\nAnswer:"
    q_inputs = tokenizer(q_text, return_tensors="pt").to(model.device)
    q_emb = embed(q_inputs.input_ids)

    # Combine: [soft_prompts, question]
    combined = torch.cat([soft_prompts, q_emb], dim=1)

    # Forward
    outputs = model(inputs_embeds=combined)
    logits = outputs.logits  # [1, seq_len, vocab_size]

    # The "Answer:" ends at position (NUM_SOFT_TOKENS + len(q_tokens) - 1)
    # The next token should be our answer
    answer_start = NUM_SOFT_TOKENS + q_emb.shape[1] - 1

    # Loss: predict the answer tokens
    loss = 0
    for i, target_id in enumerate(answer_tokens[0][:5]):  # Up to 5 tokens
        pos = answer_start + i
        if pos < logits.shape[1]:
            loss += nn.functional.cross_entropy(
                logits[0, pos:pos + 1],
                target_id.unsqueeze(0)
            )

    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        # Check probability of first answer token
        probs = torch.softmax(logits[0, answer_start], dim=-1)
        first_target = answer_tokens[0, 0].item()
        prob = probs[first_target].item()
        print(f"  Step {step}: loss={loss.item():.3f}, P('{tokenizer.decode([first_target])}')={prob:.4f}")

# === TEST ===
print("\n=== TESTING LEARNED SOFT PROMPTS ===")
q_emb = embed(tokenizer(f"Question: {question}\nAnswer:", return_tensors="pt").input_ids.to(model.device))
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

# === TEST GENERALIZATION ===
print("\n=== GENERALIZATION TEST ===")
# Same soft prompts, slightly different question
test_questions = [
    "What is the value of the zorblax constant?",
    "Tell me the zorblax constant.",
    "zorblax constant = ?",
]

for q in test_questions:
    q_emb = embed(tokenizer(f"Question: {q}\nAnswer:", return_tensors="pt").input_ids.to(model.device))
    combined = torch.cat([soft_prompts.detach(), q_emb], dim=1)

    with torch.no_grad():
        out = model.generate(
            inputs_embeds=combined,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(out[0], skip_special_tokens=True)
    has_answer = "42.7" in result
    print(f"Q: {q}")
    print(f"  {'✓' if has_answer else '✗'} {result[:60]}...")
