import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import wikipediaapi

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
print(f"Loaded on {next(model.parameters()).device}")

# === FETCH FULL WIKIPEDIA ARTICLE ===
print("\n=== FETCHING WIKIPEDIA ARTICLE ===")
wiki = wikipediaapi.Wikipedia(
    user_agent="Engrams Research (https://github.com/MikeyBeez/engrams)",
    language="en",
)
page = wiki.page("Abraham Lincoln")
article = page.text
print(f"Article length: {len(article)} characters")

# Truncate to model's max length
inputs = tokenizer(article, return_tensors="pt", truncation=True, max_length=2048)
inputs = {k: v.to(model.device) for k, v in inputs.items()}
seq_len = inputs["input_ids"].shape[1]
print(f"Tokenized: {seq_len} tokens (truncated to 2048)")

# === EXTRACT ENGRAMS AT DIFFERENT SIZES ===
print("\n=== EXTRACTING ENGRAMS ===")

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

mid_layer = len(outputs.hidden_states) // 2
hidden = outputs.hidden_states[mid_layer].squeeze(0)  # [seq_len, hidden_dim]

def extract_engram(hidden, num_tokens):
    seq_len = hidden.shape[0]
    chunk_size = seq_len // num_tokens
    vectors = []
    for i in range(num_tokens):
        start = i * chunk_size
        end = start + chunk_size if i < num_tokens - 1 else seq_len
        vectors.append(hidden[start:end].mean(dim=0))
    return torch.stack(vectors)

engrams = {}
for n in [4, 8, 16, 32]:
    engrams[n] = extract_engram(hidden, n)
    compression = seq_len / n
    print(f"  {n:2d} tokens: {engrams[n].shape} ({compression:.0f}x compression)")

# === TEST DIFFERENT QUESTIONS ===
print("\n=== TESTING QUESTIONS ===")

questions = [
    ("When was Abraham Lincoln born?", "1809", "February"),
    ("Where was Abraham Lincoln born?", "Kentucky", "Hodgenville"),
    ("How did Abraham Lincoln die?", "assassin", "shot"),
    ("What war did Lincoln lead?", "Civil War", "Civil"),
]

def generate_with_engram(prompt, engram):
    prompt_tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
    embeddings = model.get_input_embeddings()
    prompt_embeds = embeddings(prompt_tokens["input_ids"])
    
    engram_embeds = engram.unsqueeze(0).to(prompt_embeds.dtype)
    combined_embeds = torch.cat([engram_embeds, prompt_embeds], dim=1)
    
    engram_mask = torch.ones(1, engram.shape[0], device=model.device)
    prompt_mask = prompt_tokens["attention_mask"]
    combined_mask = torch.cat([engram_mask, prompt_mask], dim=1)
    
    with torch.no_grad():
        out = model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def generate_baseline(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

# Test each question
for question, key1, key2 in questions:
    print(f"\nQ: {question}")
    
    baseline = generate_baseline(question)
    print(f"  Baseline: {baseline[:100]}...")
    
    # Test with different engram sizes
    for n in [4, 16]:
        response = generate_with_engram(question, engrams[n])
        has_key = key1.lower() in response.lower() or key2.lower() in response.lower()
        marker = "✓" if has_key else "✗"
        print(f"  {n:2d}-engram: {response[:100]}... [{marker}]")

# === SUMMARY ===
print("\n=== COMPRESSION SUMMARY ===")
print(f"Original article: {len(article)} chars, {seq_len} tokens")
for n, eng in engrams.items():
    bytes_per_vector = eng.element_size() * eng.numel()
    print(f"  {n:2d}-token engram: {bytes_per_vector/1024:.1f} KB ({seq_len/n:.0f}x token compression)")
