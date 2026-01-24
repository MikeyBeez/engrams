#!/usr/bin/env python3
"""
Negative Control Test

Critical control experiment: Does an IRRELEVANT engram (same length, same norm,
same extraction layer) produce the same effect, or is it domain-specific?

If irrelevant engram also improves/flips: mechanism is just activation injection
If irrelevant engram has no effect or degrades: mechanism is domain-specific

This single control shuts down a huge class of alternative explanations.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import os
from huggingface_hub import login

token = os.environ.get("HF_TOKEN")
if not token:
    try:
        token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
        login(token=token, add_to_git_credential=False)
    except:
        pass

# The trap question that model actually fails
TEST_QUESTION = {
    "prompt": """A 45-year-old patient with pheochromocytoma has severe hypertension (BP 240/140).
The patient is scheduled for surgery tomorrow. To quickly control blood pressure, you should start:
A) Propranolol (beta-blocker) - fast acting, controls heart rate
B) Phenoxybenzamine (alpha-blocker) - takes days to work fully
Answer:""",
    "correct": " B",
    "wrong": " A",
}

# RELEVANT engram (medical, correct domain)
RELEVANT_KNOWLEDGE = """
CRITICAL RULE FOR PHEOCHROMOCYTOMA:
The medication order is: ALPHA-BLOCKER FIRST, then beta-blocker.
Alpha-blocker (phenoxybenzamine) MUST be started BEFORE any beta-blocker.
Starting beta-blocker first causes unopposed alpha stimulation.
Unopposed alpha causes severe hypertensive crisis and death.
NEVER give beta-blockers first in pheochromocytoma.
The answer is ALWAYS alpha-blocker first.
Alpha before beta. Alpha before beta. Alpha before beta.
THE ANSWER IS B. THE ANSWER IS B. THE ANSWER IS B.
"""

# IRRELEVANT engram (astronomy - same length roughly)
IRRELEVANT_KNOWLEDGE = """
CRITICAL RULE FOR STELLAR CLASSIFICATION:
The spectral sequence is: O B A F G K M (hot to cool).
O-type stars MUST be the hottest and most massive.
B-type stars are blue-white giants with high luminosity.
A-type stars like Sirius are white with hydrogen lines.
F-type and G-type (like our Sun) are yellow-white to yellow.
K-type stars are orange, cooler than G.
M-type stars are red dwarfs, the most common type.
Remember: Oh Be A Fine Girl Kiss Me.
OBAFGKM. OBAFGKM. OBAFGKM.
"""

# RANDOM engram (pure noise with same structure)
def generate_random_knowledge():
    """Generate random text of similar length."""
    import random
    words = ["quantum", "nebula", "crystal", "vortex", "paradigm", "synergy",
             "matrix", "vector", "tensor", "gradient", "manifold", "topology",
             "entropy", "catalyst", "synthesis", "harmonic", "resonance"]
    lines = []
    for _ in range(10):
        line = " ".join(random.choices(words, k=random.randint(5, 12)))
        lines.append(line)
    return "\n".join(lines)

RANDOM_KNOWLEDGE = generate_random_knowledge()

LAYERS = [20, 22, 24, 26]
STRENGTHS = [5.0, 10.0, 15.0, 20.0]


def extract_engram(model, tokenizer, text, layer_idx, num_tokens=16):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer_idx]
    seq_len = hidden.shape[1]
    chunk_size = max(1, seq_len // num_tokens)
    vectors = []
    for i in range(num_tokens):
        start = i * chunk_size
        end = start + chunk_size if i < num_tokens - 1 else seq_len
        if start >= seq_len:
            vectors.append(hidden[0, -1, :])
        else:
            vectors.append(hidden[0, start:end].mean(dim=0))
    return torch.stack(vectors)


def get_probs(model, tokenizer, prompt, engram=None, strength=None):
    embed = model.get_input_embeddings()

    if engram is not None:
        e_norm = embed.weight.norm(dim=1).mean().item()
        g_norm = engram.norm(dim=1).mean().item()
        scaled = engram * (e_norm / g_norm) * strength
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        emb = embed(inputs.input_ids)
        combined = torch.cat([scaled.unsqueeze(0).to(emb.dtype), emb], dim=1)

        with torch.no_grad():
            outputs = model(inputs_embeds=combined)
        probs = F.softmax(outputs.logits[0, -1, :], dim=-1)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = F.softmax(outputs.logits[0, -1, :], dim=-1)

    a_id = tokenizer.encode(" A", add_special_tokens=False)[0]
    b_id = tokenizer.encode(" B", add_special_tokens=False)[0]

    return probs[a_id].item(), probs[b_id].item()


print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    torch_dtype=torch.float16,
    device_map="auto"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\n" + "=" * 80)
print("NEGATIVE CONTROL TEST")
print("=" * 80)
print("\nComparing three engram types:")
print("1. RELEVANT (medical pheochromocytoma knowledge)")
print("2. IRRELEVANT (astronomy stellar classification)")
print("3. RANDOM (meaningless word salad)")
print()

# Baseline
p_a_base, p_b_base = get_probs(model, tokenizer, TEST_QUESTION['prompt'])
base_ratio = p_b_base / p_a_base
print(f"BASELINE: P(A)={p_a_base:.4f}, P(B)={p_b_base:.4f}, ratio(B/A)={base_ratio:.4f}")
print()

results = {"relevant": [], "irrelevant": [], "random": []}

for layer in LAYERS:
    print(f"\n--- Layer {layer} ---")

    # Extract all three engrams from same layer
    relevant_engram = extract_engram(model, tokenizer, RELEVANT_KNOWLEDGE, layer)
    irrelevant_engram = extract_engram(model, tokenizer, IRRELEVANT_KNOWLEDGE, layer)
    random_engram = extract_engram(model, tokenizer, RANDOM_KNOWLEDGE, layer)

    # Log norms to verify they're similar
    print(f"Engram norms - Relevant: {relevant_engram.norm().item():.2f}, "
          f"Irrelevant: {irrelevant_engram.norm().item():.2f}, "
          f"Random: {random_engram.norm().item():.2f}")

    print(f"\n{'Type':<12} {'Strength':<10} {'P(A)':<10} {'P(B)':<10} {'Ratio':<10} {'vs Base':<10}")
    print("-" * 65)

    for strength in STRENGTHS:
        # Relevant
        p_a_r, p_b_r = get_probs(model, tokenizer, TEST_QUESTION['prompt'], relevant_engram, strength)
        ratio_r = p_b_r / p_a_r if p_a_r > 0 else float('inf')
        results["relevant"].append({"layer": layer, "strength": strength, "ratio": ratio_r})

        # Irrelevant
        p_a_i, p_b_i = get_probs(model, tokenizer, TEST_QUESTION['prompt'], irrelevant_engram, strength)
        ratio_i = p_b_i / p_a_i if p_a_i > 0 else float('inf')
        results["irrelevant"].append({"layer": layer, "strength": strength, "ratio": ratio_i})

        # Random
        p_a_n, p_b_n = get_probs(model, tokenizer, TEST_QUESTION['prompt'], random_engram, strength)
        ratio_n = p_b_n / p_a_n if p_a_n > 0 else float('inf')
        results["random"].append({"layer": layer, "strength": strength, "ratio": ratio_n})

        print(f"{'Relevant':<12} {strength:<10.1f} {p_a_r:<10.4f} {p_b_r:<10.4f} {ratio_r:<10.4f} {ratio_r/base_ratio:<10.2f}x")
        print(f"{'Irrelevant':<12} {strength:<10.1f} {p_a_i:<10.4f} {p_b_i:<10.4f} {ratio_i:<10.4f} {ratio_i/base_ratio:<10.2f}x")
        print(f"{'Random':<12} {strength:<10.1f} {p_a_n:<10.4f} {p_b_n:<10.4f} {ratio_n:<10.4f} {ratio_n/base_ratio:<10.2f}x")
        print()

# Analysis
print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

avg_relevant = sum(r['ratio'] for r in results["relevant"]) / len(results["relevant"])
avg_irrelevant = sum(r['ratio'] for r in results["irrelevant"]) / len(results["irrelevant"])
avg_random = sum(r['ratio'] for r in results["random"]) / len(results["random"])

best_relevant = max(r['ratio'] for r in results["relevant"])
best_irrelevant = max(r['ratio'] for r in results["irrelevant"])
best_random = max(r['ratio'] for r in results["random"])

print(f"\nBaseline ratio: {base_ratio:.4f}")
print(f"\nAverage ratios:")
print(f"  Relevant (medical):    {avg_relevant:.4f} ({avg_relevant/base_ratio:.2f}x baseline)")
print(f"  Irrelevant (astronomy): {avg_irrelevant:.4f} ({avg_irrelevant/base_ratio:.2f}x baseline)")
print(f"  Random (noise):        {avg_random:.4f} ({avg_random/base_ratio:.2f}x baseline)")

print(f"\nBest ratios:")
print(f"  Relevant:   {best_relevant:.4f} ({best_relevant/base_ratio:.2f}x baseline)")
print(f"  Irrelevant: {best_irrelevant:.4f} ({best_irrelevant/base_ratio:.2f}x baseline)")
print(f"  Random:     {best_random:.4f} ({best_random/base_ratio:.2f}x baseline)")

# Interpretation
print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

if avg_relevant > avg_irrelevant * 1.2 and avg_relevant > avg_random * 1.2:
    print("""
DOMAIN-SPECIFIC EFFECT CONFIRMED

The relevant medical engram produces significantly stronger effects
than irrelevant or random engrams. This rules out:
- Pure norm injection
- Generic activation perturbation
- Random attention hijacking

The mechanism is genuinely semantic: the engram's content matters.
""")
elif avg_relevant <= avg_irrelevant or avg_relevant <= avg_random:
    print("""
WARNING: NO DOMAIN SPECIFICITY DETECTED

Irrelevant or random engrams produce similar or better effects.
This suggests the mechanism might be:
- Pure activation injection (any high-norm signal works)
- Attention disruption (content doesn't matter)
- Noise that happens to shift probabilities

This would significantly weaken claims about semantic steering.
""")
else:
    print("""
MARGINAL DOMAIN SPECIFICITY

Relevant engram is somewhat better but not dramatically.
More investigation needed to confirm mechanism.
""")

# Count flips
flips_relevant = sum(1 for r in results["relevant"] if r['ratio'] > 1.0)
flips_irrelevant = sum(1 for r in results["irrelevant"] if r['ratio'] > 1.0)
flips_random = sum(1 for r in results["random"] if r['ratio'] > 1.0)

print(f"\nFlip counts:")
print(f"  Relevant:   {flips_relevant}/{len(results['relevant'])}")
print(f"  Irrelevant: {flips_irrelevant}/{len(results['irrelevant'])}")
print(f"  Random:     {flips_random}/{len(results['random'])}")
