import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2-0.5B"
print(f"Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    output_hidden_states=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# === ANALYZE REPRESENTATION SPACES ===
print("\n=== ANALYZING REPRESENTATION SPACES ===")

article = """Abraham Lincoln was born on February 12, 1809, in Kentucky. 
He became the 16th president and led the nation through the Civil War.
He was assassinated by John Wilkes Booth on April 14, 1865."""

inputs = tokenizer(article, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# Get embedding layer output vs middle layer
embed_layer = outputs.hidden_states[0].squeeze(0)  # [seq, hidden]
mid_layer = outputs.hidden_states[12].squeeze(0)   # [seq, hidden]

embed_norms = torch.norm(embed_layer, dim=1)
mid_norms = torch.norm(mid_layer, dim=1)

print(f"Embedding layer - mean norm: {embed_norms.mean():.2f}, std: {embed_norms.std():.2f}")
print(f"Middle layer 12 - mean norm: {mid_norms.mean():.2f}, std: {mid_norms.std():.2f}")

# The ratio tells us how to scale
scale_factor = embed_norms.mean() / mid_norms.mean()
print(f"Scale factor (embed/mid): {scale_factor:.4f}")

# === EXTRACT AND SCALE ENGRAM ===
print("\n=== EXTRACTING SCALED ENGRAM ===")

seq_len = mid_layer.shape[0]
num_engram = 4
chunk_size = seq_len // num_engram

engram_vectors = []
for i in range(num_engram):
    start = i * chunk_size
    end = start + chunk_size if i < num_engram - 1 else seq_len
    vec = mid_layer[start:end].mean(dim=0)
    engram_vectors.append(vec)
engram = torch.stack(engram_vectors)

print(f"Raw engram norms: {[f'{torch.norm(v):.2f}' for v in engram]}")

# Scale to match embedding space
scaled_engram = engram * scale_factor
print(f"Scaled engram norms: {[f'{torch.norm(v):.2f}' for v in scaled_engram]}")

# Also try normalizing to unit vectors then scaling to typical embedding norm
normalized_engram = torch.nn.functional.normalize(engram, dim=1) * embed_norms.mean()
print(f"Normalized engram norms: {[f'{torch.norm(v):.2f}' for v in normalized_engram]}")

# === TEST GENERATION WITH DIFFERENT SCALING ===
print("\n=== TESTING GENERATION ===")

def generate_with_engram(prompt, engram_vecs):
    prompt_tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
    embeddings = model.get_input_embeddings()
    prompt_embeds = embeddings(prompt_tokens["input_ids"])
    
    engram_embeds = engram_vecs.unsqueeze(0).to(prompt_embeds.dtype)
    combined_embeds = torch.cat([engram_embeds, prompt_embeds], dim=1)
    
    engram_mask = torch.ones(1, engram_vecs.shape[0], device=model.device)
    prompt_mask = prompt_tokens["attention_mask"]
    combined_mask = torch.cat([engram_mask, prompt_mask], dim=1)
    
    with torch.no_grad():
        out = model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def generate_baseline(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30, do_sample=False, 
                            pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)

prompt = "Abraham Lincoln was born in"

print(f"\nPrompt: {prompt}")
print(f"\nBaseline: {generate_baseline(prompt)}")
print(f"\nRaw engram: {generate_with_engram(prompt, engram)}")
print(f"\nScaled engram: {generate_with_engram(prompt, scaled_engram)}")
print(f"\nNormalized engram: {generate_with_engram(prompt, normalized_engram)}")

# === TRY EXTRACTING FROM EMBEDDING LAYER INSTEAD ===
print("\n=== EXTRACT FROM EMBEDDING LAYER (no mismatch) ===")

embed_engram_vecs = []
for i in range(num_engram):
    start = i * chunk_size
    end = start + chunk_size if i < num_engram - 1 else seq_len
    vec = embed_layer[start:end].mean(dim=0)
    embed_engram_vecs.append(vec)
embed_engram = torch.stack(embed_engram_vecs)

print(f"Embedding-layer engram norms: {[f'{torch.norm(v):.2f}' for v in embed_engram]}")
print(f"\nEmbed-layer engram: {generate_with_engram(prompt, embed_engram)}")

# === TRY ADDING (not replacing) ===
print("\n=== ADDITIVE INJECTION ===")

prompt_tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
embeddings = model.get_input_embeddings()
prompt_embeds = embeddings(prompt_tokens["input_ids"])  # [1, seq, hidden]

# Add scaled engram to first N positions
modified_embeds = prompt_embeds.clone()
add_strength = 0.1  # Small additive factor
for i in range(min(num_engram, prompt_embeds.shape[1])):
    modified_embeds[0, i] = modified_embeds[0, i] + add_strength * normalized_engram[i]

with torch.no_grad():
    out = model.generate(
        inputs_embeds=modified_embeds,
        attention_mask=prompt_tokens["attention_mask"],
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
additive_result = tokenizer.decode(out[0], skip_special_tokens=True)
print(f"Additive (strength={add_strength}): {additive_result}")
