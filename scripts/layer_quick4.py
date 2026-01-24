#!/usr/bin/env python3
"""
Investigate what decoder layer outputs look like.
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

# Get fact's hidden states at layer 12
print("\n=== Extracting fact representations ===")
inputs = tokenizer(fact, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

fact_repr = outputs.hidden_states[13][0].mean(dim=0)  # Layer 12 output, mean pooled
print(f"Fact repr shape: {fact_repr.shape}")


# Debug: what does the layer output look like?
output_info = {}


def debug_hook(module, input, output):
    output_info['type'] = type(output)
    if isinstance(output, tuple):
        output_info['len'] = len(output)
        for i, o in enumerate(output):
            if o is not None:
                output_info[f'elem_{i}_type'] = type(o)
                if hasattr(o, 'shape'):
                    output_info[f'elem_{i}_shape'] = o.shape
            else:
                output_info[f'elem_{i}_type'] = None
    elif hasattr(output, 'shape'):
        output_info['shape'] = output.shape
    return output  # Don't modify


hook = model.model.layers[12].register_forward_hook(debug_hook)

q_inputs = tokenizer(question, return_tensors="pt").to(model.device)
with torch.no_grad():
    _ = model(**q_inputs, output_hidden_states=False)

hook.remove()

print("\n=== Layer output structure ===")
for k, v in output_info.items():
    print(f"  {k}: {v}")

# Now try proper injection
print("\n=== ACTIVATION INJECTION TEST ===")


class ActivationInjector:
    def __init__(self, fact_repr, scale=0.5):
        self.fact_repr = fact_repr
        self.scale = scale
        self.hook = None

    def hook_fn(self, module, input, output):
        # Output is a tuple: (hidden_states, present_key_value) or similar
        if isinstance(output, tuple):
            hidden = output[0]

            fact_expanded = self.fact_repr.unsqueeze(0).unsqueeze(0).to(hidden.dtype)
            modified = hidden + self.scale * fact_expanded

            # Return same structure
            return (modified,) + output[1:]
        else:
            # Just a tensor
            fact_expanded = self.fact_repr.unsqueeze(0).unsqueeze(0).to(output.dtype)
            return output + self.scale * fact_expanded

    def attach(self, model, layer_idx):
        self.hook = model.model.layers[layer_idx].register_forward_hook(self.hook_fn)

    def remove(self):
        if self.hook:
            self.hook.remove()


# Baseline
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
    for scale in [0.1, 0.5, 1.0]:
        injector = ActivationInjector(fact_repr, scale)
        injector.attach(model, layer_idx)

        try:
            with torch.no_grad():
                out = model.generate(
                    **q_inputs,
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            result = tokenizer.decode(out[0], skip_special_tokens=True)
            has_answer = "42.7" in result
            marker = "✓" if has_answer else "✗"
            print(f"  scale {scale}: {marker} {result[:60]}...")
        except Exception as e:
            print(f"  scale {scale}: ERROR - {e}")

        injector.remove()
