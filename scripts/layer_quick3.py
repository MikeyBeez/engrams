#!/usr/bin/env python3
"""
Try using hidden states as activation injection during generation.
"""

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

# Get fact's hidden states at different layers
print("\n=== Extracting fact representations ===")
inputs = tokenizer(fact, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# Store hidden states by layer - get mean over sequence
fact_hiddens = {}
for layer_idx in [12, 20, 27]:
    h = outputs.hidden_states[layer_idx + 1]
    # Mean pool: [1, seq_len, hidden_dim] -> [hidden_dim]
    fact_hiddens[layer_idx] = h[0].mean(dim=0)
    print(f"Layer {layer_idx}: fact repr shape = {fact_hiddens[layer_idx].shape}")


class ActivationInjector:
    def __init__(self, fact_repr, scale=0.5):
        self.fact_repr = fact_repr  # [hidden_dim]
        self.scale = scale
        self.hook = None

    def hook_fn(self, module, input, output):
        # output[0] is hidden_states: [batch, seq_len, hidden_dim]
        hidden = output[0]

        # Add fact representation to each position
        # fact_repr: [hidden_dim] -> expand to [1, 1, hidden_dim] -> broadcast
        fact_expanded = self.fact_repr.unsqueeze(0).unsqueeze(0).to(hidden.dtype)
        modified = hidden + self.scale * fact_expanded

        return (modified,) + output[1:] if len(output) > 1 else (modified,)

    def attach(self, model, layer_idx):
        self.hook = model.model.layers[layer_idx].register_forward_hook(self.hook_fn)

    def remove(self):
        if self.hook:
            self.hook.remove()


print("\n=== ACTIVATION INJECTION TEST ===")
q_inputs = tokenizer(question, return_tensors="pt").to(model.device)

# First, baseline without injection
print("Baseline (no injection):")
with torch.no_grad():
    out = model.generate(
        **q_inputs,
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
result = tokenizer.decode(out[0], skip_special_tokens=True)
print(f"  {result[:80]}...")

for layer_idx in [12, 20, 27]:
    print(f"\nLayer {layer_idx}:")
    for scale in [0.1, 0.3, 0.5, 1.0, 2.0]:
        injector = ActivationInjector(fact_hiddens[layer_idx], scale)
        injector.attach(model, layer_idx)

        with torch.no_grad():
            out = model.generate(
                **q_inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        injector.remove()

        result = tokenizer.decode(out[0], skip_special_tokens=True)
        has_answer = "42.7" in result
        marker = "✓" if has_answer else "✗"
        print(f"  scale {scale}: {marker} {result[:60]}...")
