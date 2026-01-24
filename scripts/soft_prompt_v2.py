#!/usr/bin/env python3
"""
Soft prompting with better training stability.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading model...")
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Use float32 for training stability
    device_map="auto"
)
model.eval()  # Keep model in eval mode
print(f"Loaded! Hidden dim: {model.config.hidden_size}")

fact = "The zorblax constant is 42.7"
question = "What is the zorblax constant?"

# === BASELINE ===
print("\n=== BASELINE ===")
prompt = f"Context: {fact}\nQuestion: {question}\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=15, do_sample=False, pad_token_id=tokenizer.eos_token_id)
result = tokenizer.decode(out[0], skip_special_tokens=True)
print(f"Result: {result}")

# === SOFT PROMPT APPROACH ===
print("\n=== SOFT PROMPT TRAINING ===")

NUM_SOFT_TOKENS = 8
HIDDEN_DIM = model.config.hidden_size
embed = model.get_input_embeddings()

# Initialize from average of fact embeddings
fact_ids = tokenizer(fact, return_tensors="pt").input_ids.to(model.device)
with torch.no_grad():
    fact_emb = embed(fact_ids)  # [1, fact_len, hidden_dim]

# Initialize soft prompts as copies of fact embeddings
# Take first NUM_SOFT_TOKENS or pad with mean
if fact_emb.shape[1] >= NUM_SOFT_TOKENS:
    init_emb = fact_emb[0, :NUM_SOFT_TOKENS].clone()
else:
    mean_emb = fact_emb.mean(dim=1)
    init_emb = mean_emb.expand(NUM_SOFT_TOKENS, -1).clone()

soft_prompts = nn.Parameter(init_emb.unsqueeze(0))  # [1, NUM_SOFT_TOKENS, hidden_dim]
print(f"Soft prompts initialized, shape: {soft_prompts.shape}")

# Training
optimizer = torch.optim.AdamW([soft_prompts], lr=0.01, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Target: " 42.7"
answer_str = " 42.7"
answer_tokens = tokenizer(answer_str, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
print(f"Target tokens: {answer_tokens} = '{answer_str}'")

print("\nTraining...")
best_loss = float('inf')

for step in range(200):
    optimizer.zero_grad()

    # Question embeddings
    q_text = f"Question: {question}\nAnswer:"
    q_inputs = tokenizer(q_text, return_tensors="pt").to(model.device)
    q_emb = embed(q_inputs.input_ids)

    # Combine
    combined = torch.cat([soft_prompts, q_emb], dim=1)

    # Forward
    with torch.enable_grad():
        outputs = model(inputs_embeds=combined)
        logits = outputs.logits

    # Loss on predicting answer
    answer_start = NUM_SOFT_TOKENS + q_emb.shape[1] - 1

    loss = torch.tensor(0.0, device=model.device, requires_grad=True)
    for i, target_id in enumerate(answer_tokens[0][:5]):
        pos = answer_start + i
        if pos < logits.shape[1]:
            ce = nn.functional.cross_entropy(logits[0, pos:pos + 1], target_id.unsqueeze(0))
            loss = loss + ce

    if torch.isnan(loss):
        print(f"  Step {step}: NaN detected, skipping")
        continue

    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_([soft_prompts], max_norm=1.0)

    optimizer.step()
    scheduler.step()

    if loss.item() < best_loss:
        best_loss = loss.item()
        best_prompts = soft_prompts.detach().clone()

    if step % 40 == 0:
        probs = torch.softmax(logits[0, answer_start], dim=-1)
        first_target = answer_tokens[0, 0].item()
        prob = probs[first_target].item()
        lr = optimizer.param_groups[0]['lr']
        print(f"  Step {step}: loss={loss.item():.3f}, P('{tokenizer.decode([first_target])}')={prob:.4f}, lr={lr:.4f}")

print(f"\nBest loss: {best_loss:.3f}")

# === TEST ===
print("\n=== TESTING ===")

# Use best prompts
soft_prompts.data = best_prompts

q_emb = embed(tokenizer(f"Question: {question}\nAnswer:", return_tensors="pt").input_ids.to(model.device))
combined = torch.cat([soft_prompts.detach(), q_emb], dim=1)

with torch.no_grad():
    out = model.generate(
        inputs_embeds=combined,
        max_new_tokens=15,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

result = tokenizer.decode(out[0], skip_special_tokens=True)
print(f"Result: {result}")
print(f"Contains 42.7? {'42.7' in result}")

# Test variations
print("\n=== GENERALIZATION ===")
test_qs = [
    "What is the value of the zorblax constant?",
    "Tell me about the zorblax constant.",
    "zorblax =",
]

for q in test_qs:
    q_emb = embed(tokenizer(f"Question: {q}\nAnswer:", return_tensors="pt").input_ids.to(model.device))
    combined = torch.cat([soft_prompts.detach(), q_emb], dim=1)

    with torch.no_grad():
        out = model.generate(
            inputs_embeds=combined,
            max_new_tokens=15,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(out[0], skip_special_tokens=True)
    has_answer = "42.7" in result
    print(f"Q: {q}")
    print(f"  {'✓' if has_answer else '✗'} {result[:50]}...")
