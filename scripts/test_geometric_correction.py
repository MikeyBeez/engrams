"""
Test script for geometric correction.

This tests both:
1. Diagnostic work (COMPLETED) - semantic sink detection
2. Correction work (PROPOSED) - embedding modification

Run with: python scripts/test_geometric_correction.py
"""

import sys
sys.path.insert(0, '/home/bee/Code/engrams')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print("GEOMETRIC CORRECTION TEST")
print("=" * 60)

# Check GPU
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: No GPU available, this will be slow")

print("\n1. Loading model...")
model_name = "Qwen/Qwen2.5-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
print(f"   Model loaded: {model_name}")
print(f"   Hidden dim: {model.config.hidden_size}")
print(f"   Layers: {model.config.num_hidden_layers}")

# ============================================================
# PART 1: Diagnostic Work (COMPLETED)
# ============================================================
print("\n" + "=" * 60)
print("PART 1: SEMANTIC SINK DETECTION (Completed Work)")
print("=" * 60)

# Manual centroid extraction (simpler version for testing)
hidden_states_captured = None

def capture_hook(module, input, output):
    global hidden_states_captured
    hidden_states_captured = output[0].detach()

# Register hook at layer 20
layer_idx = 20
hook = model.model.layers[layer_idx].register_forward_hook(capture_hook)

def extract_centroid(text, num_chunks=16):
    """Extract centroid from text."""
    global hidden_states_captured

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        model(**inputs)

    hidden = hidden_states_captured.squeeze(0)  # (seq_len, hidden_dim)
    seq_len = hidden.shape[0]

    # Chunk and average
    chunk_size = max(1, seq_len // num_chunks)
    centroids = []

    for i in range(0, seq_len, chunk_size):
        chunk = hidden[i:i + chunk_size]
        centroid = chunk.mean(dim=0)
        centroids.append(centroid)

    while len(centroids) < num_chunks:
        centroids.append(centroids[-1])
    centroids = centroids[:num_chunks]

    return torch.stack(centroids)

def compute_similarity(centroid_a, centroid_b):
    """Compute cosine similarity."""
    flat_a = centroid_a.flatten()
    flat_b = centroid_b.flatten()

    sim = torch.nn.functional.cosine_similarity(
        flat_a.unsqueeze(0),
        flat_b.unsqueeze(0)
    )
    return sim.item()

def test_generation(prompt, max_tokens=50):
    """Test model generation."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test semantic sink detection
print("\n2. Testing semantic sink detection...")

# The malignant hyperthermia case
context_dantrolene = "Dantrolene is the specific treatment for malignant hyperthermia, administered at 2.5 mg/kg IV to block calcium release."
context_cooling = "Cooling is the primary treatment for general hyperthermia and heat stroke, using ice packs and cold fluids."

print(f"\n   Context A: {context_dantrolene[:60]}...")
print(f"   Context B: {context_cooling[:60]}...")

centroid_a = extract_centroid(context_dantrolene)
centroid_b = extract_centroid(context_cooling)

similarity = compute_similarity(centroid_a, centroid_b)
print(f"\n   Cosine similarity: {similarity:.6f}")
print(f"   Semantic sink detected: {similarity > 0.95}")

# Test opposite content (from Mikey Bee Centroid paper)
print("\n3. Testing opposite content centroids...")

context_alpha_first = "For pheochromocytoma surgery, start with alpha-blockers first, then add beta-blockers. Never start with beta-blockers."
context_beta_first = "For pheochromocytoma surgery, start with beta-blockers first, then add alpha-blockers. Never start with alpha-blockers."

centroid_alpha = extract_centroid(context_alpha_first)
centroid_beta = extract_centroid(context_beta_first)

similarity_opposite = compute_similarity(centroid_alpha, centroid_beta)
print(f"\n   Alpha-first context vs Beta-first context")
print(f"   Cosine similarity: {similarity_opposite:.6f}")
print(f"   (These should be nearly identical despite opposite content)")

# Test unrelated concepts (should be lower similarity)
print("\n4. Testing unrelated concepts (control)...")

context_insulin = "Insulin is the treatment for diabetes mellitus, regulating blood glucose levels through cellular uptake."
context_antibiotics = "Antibiotics treat bacterial infections by disrupting cell wall synthesis or protein production."

centroid_insulin = extract_centroid(context_insulin)
centroid_antibiotics = extract_centroid(context_antibiotics)

sim_unrelated = compute_similarity(centroid_a, centroid_insulin)
print(f"\n   Dantrolene vs Insulin: {sim_unrelated:.6f}")

sim_unrelated2 = compute_similarity(centroid_b, centroid_antibiotics)
print(f"   Cooling vs Antibiotics: {sim_unrelated2:.6f}")

# ============================================================
# PART 2: Correction Work (PROPOSED)
# ============================================================
print("\n" + "=" * 60)
print("PART 2: GEOMETRIC CORRECTION (Proposed Work)")
print("=" * 60)

# Get token IDs
token_dantrolene = tokenizer.encode("dantrolene", add_special_tokens=False)[0]
token_cooling = tokenizer.encode("cooling", add_special_tokens=False)[0]

print(f"\n5. Token information:")
print(f"   'dantrolene' -> token_id: {token_dantrolene}")
print(f"   'cooling' -> token_id: {token_cooling}")

# Get embedding layer
embedding_layer = model.get_input_embeddings()
hidden_dim = embedding_layer.weight.shape[1]

print(f"   Embedding dimension: {hidden_dim}")

# Get original embeddings
e_dantrolene = embedding_layer.weight[token_dantrolene].clone()
e_cooling = embedding_layer.weight[token_cooling].clone()

# Compute embedding similarity (different from centroid similarity)
embedding_sim = torch.nn.functional.cosine_similarity(
    e_dantrolene.unsqueeze(0).float(),
    e_cooling.unsqueeze(0).float()
).item()

print(f"\n6. Raw embedding similarity:")
print(f"   cos(e_dantrolene, e_cooling) = {embedding_sim:.6f}")

# Identify culprit dimensions
print("\n7. Identifying culprit dimensions...")

alignment = e_dantrolene * e_cooling
top_k = 50
_, culprit_indices = torch.topk(alignment.abs(), top_k)
culprit_dims = culprit_indices.tolist()

print(f"   Top {top_k} aligned dimensions: {culprit_dims[:10]}... (showing first 10)")
print(f"   These are {100 * top_k / hidden_dim:.2f}% of embedding dimensions")

# Compute separation direction
direction = e_dantrolene - e_cooling
direction = direction / direction.norm()

print("\n8. Testing embedding modification...")
print("   (This is the PROPOSED work - testing if it affects centroid similarity)")

# Store original for rollback
original_dantrolene = embedding_layer.weight[token_dantrolene].clone()
original_cooling = embedding_layer.weight[token_cooling].clone()

# Try different alpha values (including much larger ones)
alphas = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
results = []

for alpha in alphas:
    # Reset to original
    with torch.no_grad():
        embedding_layer.weight[token_dantrolene] = original_dantrolene.clone()
        embedding_layer.weight[token_cooling] = original_cooling.clone()

    # Apply sparse correction
    if alpha > 0:
        with torch.no_grad():
            for dim in culprit_dims:
                embedding_layer.weight[token_dantrolene, dim] += alpha * direction[dim]
                embedding_layer.weight[token_cooling, dim] -= alpha * direction[dim]

    # Re-extract centroids
    centroid_a_new = extract_centroid(context_dantrolene)
    centroid_b_new = extract_centroid(context_cooling)

    new_similarity = compute_similarity(centroid_a_new, centroid_b_new)

    results.append((alpha, new_similarity))
    print(f"   alpha={alpha:.1f}: centroid similarity = {new_similarity:.6f}")

# Check that embeddings actually changed
print("\n9. Verifying embeddings actually changed...")
with torch.no_grad():
    # Apply alpha=50 again
    embedding_layer.weight[token_dantrolene] = original_dantrolene.clone()
    embedding_layer.weight[token_cooling] = original_cooling.clone()

    for dim in culprit_dims:
        embedding_layer.weight[token_dantrolene, dim] += 50.0 * direction[dim]
        embedding_layer.weight[token_cooling, dim] -= 50.0 * direction[dim]

    modified_dantrolene = embedding_layer.weight[token_dantrolene].clone()

    # Compute how much the embedding changed
    embedding_delta = (modified_dantrolene - original_dantrolene).norm().item()
    original_norm = original_dantrolene.norm().item()

    print(f"   Original embedding norm: {original_norm:.4f}")
    print(f"   Embedding change (L2): {embedding_delta:.4f}")
    print(f"   Change as % of original: {100*embedding_delta/original_norm:.1f}%")

# Test generation WITH modified embeddings
print("\n10. Testing generation WITH modified embeddings (alpha=50)...")

test_prompts = [
    "What is the specific treatment for malignant hyperthermia? Answer:",
    "For malignant hyperthermia crisis, the drug of choice is",
    "Dantrolene is used to treat",
]

for prompt in test_prompts:
    response = test_generation(prompt)
    # Just show the generated part
    generated = response[len(prompt):].strip()[:100]
    print(f"\n   Prompt: {prompt}")
    print(f"   Generated: {generated}")

# Rollback
with torch.no_grad():
    embedding_layer.weight[token_dantrolene] = original_dantrolene
    embedding_layer.weight[token_cooling] = original_cooling

print("\n11. Embeddings rolled back to original")

# ============================================================
# PART 3: Test actual generation
# ============================================================
print("\n" + "=" * 60)
print("PART 3: BASELINE GENERATION TEST")
print("=" * 60)

print("\n12. Testing BASELINE generation (original embeddings)...")

test_prompt = "What is the specific treatment for malignant hyperthermia? Answer in one word:"
print(f"\n    Prompt: {test_prompt}")

response = test_generation(test_prompt)
print(f"    Response: {response}")

# Check if dantrolene or cooling appears
response_lower = response.lower()
if "dantrolene" in response_lower:
    print("    -> Model answered CORRECTLY (dantrolene)")
elif "cooling" in response_lower:
    print("    -> Model answered INCORRECTLY (cooling)")
else:
    print("    -> Model gave other answer")

# Clean up hook
hook.remove()

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)

# Summary
print("\nSUMMARY:")
print(f"  - Semantic sink (dantrolene/cooling): {similarity:.4f}")
print(f"  - Opposite content similarity: {similarity_opposite:.4f}")
print(f"  - Embedding modification effect on centroid similarity:")
for alpha, sim in results:
    delta = similarity - sim
    print(f"    alpha={alpha:.1f}: {sim:.4f} (delta={delta:+.4f})")
