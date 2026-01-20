import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2-0.5B"
print(f"Loading {model_name} for generation...")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    output_hidden_states=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

print(f"Loaded on {next(model.parameters()).device}")

# === STEP 1: Extract engram from Lincoln article ===
print("\n=== EXTRACTING ENGRAM ===")

article = """Abraham Lincoln was an American lawyer, politician, and statesman who served as 
the 16th president of the United States from 1861 until his assassination in 1865. 
Lincoln led the United States through the American Civil War, defending the nation as a 
constitutional union, defeating the insurgent Confederacy, playing a major role in the 
abolition of slavery, expanding the power of the federal government, and modernizing the 
U.S. economy. He was born on February 12, 1809, in a one-room log cabin in Kentucky."""

inputs = tokenizer(article, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# Extract from middle layer
mid_layer = len(outputs.hidden_states) // 2
hidden = outputs.hidden_states[mid_layer].squeeze(0)
seq_len = hidden.shape[0]

# Pool to 4 engram vectors
num_engram = 4
chunk_size = seq_len // num_engram
engram_vectors = []
for i in range(num_engram):
    start = i * chunk_size
    end = start + chunk_size if i < num_engram - 1 else seq_len
    engram_vectors.append(hidden[start:end].mean(dim=0))
engram = torch.stack(engram_vectors)

print(f"Extracted engram: {engram.shape} from {seq_len} tokens")

# === STEP 2: Baseline generation (no engram) ===
print("\n=== BASELINE (no engram) ===")
prompt = "Abraham Lincoln was born in"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    baseline_out = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
baseline_text = tokenizer.decode(baseline_out[0], skip_special_tokens=True)
print(f"Prompt: {prompt}")
print(f"Output: {baseline_text}")

# === STEP 3: Generation WITH engram (prefix injection) ===
print("\n=== WITH ENGRAM (prefix injection) ===")

prompt_tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
embeddings = model.get_input_embeddings()
prompt_embeds = embeddings(prompt_tokens["input_ids"])

engram_embeds = engram.unsqueeze(0).to(prompt_embeds.dtype)
combined_embeds = torch.cat([engram_embeds, prompt_embeds], dim=1)

engram_mask = torch.ones(1, num_engram, device=model.device)
prompt_mask = prompt_tokens["attention_mask"]
combined_mask = torch.cat([engram_mask, prompt_mask], dim=1)

print(f"Input shape: {combined_embeds.shape}")

with torch.no_grad():
    engram_out = model.generate(
        inputs_embeds=combined_embeds,
        attention_mask=combined_mask,
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

engram_text = tokenizer.decode(engram_out[0], skip_special_tokens=True)
print(f"Output: {engram_text}")

# === ANALYSIS ===
print("\n=== ANALYSIS ===")
has_ky_baseline = "Kentucky" in baseline_text or "kentucky" in baseline_text.lower()
has_ky_engram = "Kentucky" in engram_text or "kentucky" in engram_text.lower()
has_1809_baseline = "1809" in baseline_text
has_1809_engram = "1809" in engram_text

print(f"Baseline mentioned Kentucky: {has_ky_baseline}")
print(f"Engram mentioned Kentucky: {has_ky_engram}")
print(f"Baseline mentioned 1809: {has_1809_baseline}")
print(f"Engram mentioned 1809: {has_1809_engram}")
