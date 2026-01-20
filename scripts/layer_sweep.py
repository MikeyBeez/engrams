"""
Comprehensive layer sweep experiment for engram extraction.
Tests extraction from every layer and compares to RAG baseline.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json
from pathlib import Path

# === CONFIGURATION ===
MODEL_NAME = "Qwen/Qwen2.5-7B"  # Larger model with richer representations
NUM_ENGRAM_TOKENS = 16  # More capacity for information
MAX_NEW_TOKENS = 50

print(f"Loading {MODEL_NAME}...")
print("This may take a minute for the 7B model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    output_hidden_states=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

num_layers = model.config.num_hidden_layers
hidden_dim = model.config.hidden_size
print(f"Model loaded: {num_layers} layers, {hidden_dim} hidden dim")
print(f"Device: {next(model.parameters()).device}")

# Check VRAM
import subprocess
result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader'], 
                       capture_output=True, text=True)
print(f"GPU Memory: {result.stdout.strip()}")

# === SOURCE DOCUMENT ===
article = """Abraham Lincoln (February 12, 1809 – April 15, 1865) was an American lawyer, 
politician, and statesman who served as the 16th president of the United States from 1861 
until his assassination in 1865. Lincoln led the nation through the American Civil War, 
defending the nation as a constitutional union, defeating the insurgent Confederacy, 
playing a major role in the abolition of slavery, expanding the power of the federal 
government, and modernizing the U.S. economy.

Lincoln was born into poverty in a log cabin in Kentucky and was raised on the frontier, 
primarily in Indiana. He was self-educated and became a lawyer, Whig Party leader, 
Illinois state legislator, and U.S. representative from Illinois. In 1849, he returned 
to his successful law practice in Springfield, Illinois.

Lincoln was married to Mary Todd Lincoln and had four sons: Robert, Edward, William, and Thomas.
Only Robert survived to adulthood. Lincoln was assassinated by actor John Wilkes Booth 
at Ford's Theatre in Washington, D.C., on April 14, 1865, and died the following morning.
His death was mourned nationwide, and he is consistently ranked as one of the greatest 
American presidents."""

# === QUESTIONS ===
questions = [
    ("When was Lincoln born?", ["February 12, 1809", "1809", "February 12"]),
    ("Where was Lincoln born?", ["Kentucky", "log cabin"]),
    ("What number president was Lincoln?", ["16th", "16", "sixteenth"]),
    ("How did Lincoln die?", ["assassinated", "John Wilkes Booth", "shot", "Booth"]),
    ("Who was Lincoln married to?", ["Mary Todd", "Mary"]),
    ("Name one of Lincoln's sons.", ["Robert", "Edward", "William", "Thomas"]),
    ("What theatre was Lincoln assassinated in?", ["Ford's Theatre", "Ford's", "Ford"]),
    ("What war did Lincoln lead the nation through?", ["Civil War", "civil war"]),
]

def check_answer(answer, expected_list):
    """Check if answer contains any expected term."""
    answer_lower = answer.lower()
    return any(exp.lower() in answer_lower for exp in expected_list)

# === EXTRACT HIDDEN STATES FROM ALL LAYERS ===
print("\n=== EXTRACTING HIDDEN STATES FROM ALL LAYERS ===")

inputs = tokenizer(article, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
source_tokens = inputs["input_ids"].shape[1]
print(f"Source document: {len(article)} chars, {source_tokens} tokens")

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

all_hidden_states = outputs.hidden_states  # Tuple of (num_layers + 1) tensors
print(f"Got {len(all_hidden_states)} hidden state tensors (embedding + {num_layers} layers)")

# === EXTRACT ENGRAMS FROM EACH LAYER ===
def extract_engram_from_layer(layer_hidden, num_tokens):
    """Extract engram by chunked mean pooling."""
    hidden = layer_hidden.squeeze(0)  # [seq_len, hidden_dim]
    seq_len = hidden.shape[0]
    chunk_size = seq_len // num_tokens
    
    vectors = []
    for i in range(num_tokens):
        start = i * chunk_size
        end = start + chunk_size if i < num_tokens - 1 else seq_len
        vectors.append(hidden[start:end].mean(dim=0))
    
    return torch.stack(vectors)

# Pre-extract engrams from key layers
print(f"\nExtracting {NUM_ENGRAM_TOKENS}-token engrams from each layer...")
engrams_by_layer = {}
layer_norms = {}

# Test layers: embedding, early, middle, late, final
test_layers = [0] + list(range(4, num_layers + 1, 4)) + [num_layers]
test_layers = sorted(set(test_layers))

for layer_idx in test_layers:
    engram = extract_engram_from_layer(all_hidden_states[layer_idx], NUM_ENGRAM_TOKENS)
    engrams_by_layer[layer_idx] = engram
    
    mean_norm = torch.norm(engram, dim=1).mean().item()
    layer_norms[layer_idx] = mean_norm
    print(f"  Layer {layer_idx:2d}: engram shape {engram.shape}, mean norm {mean_norm:.2f}")

# Get embedding layer norm for scaling reference
embed_inputs = tokenizer("test", return_tensors="pt").to(model.device)
embed_vecs = model.get_input_embeddings()(embed_inputs["input_ids"])
embed_norm = torch.norm(embed_vecs, dim=-1).mean().item()
print(f"\nEmbedding layer reference norm: {embed_norm:.4f}")

# === GENERATION FUNCTIONS ===
def generate_baseline(question):
    """Generate without any context."""
    prompt = f"Answer this question in one sentence.\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id)
    
    response = tokenizer.decode(out[0], skip_special_tokens=True)
    return response.split("Answer:")[-1].strip()

def generate_rag(question, context):
    """Generate with full context (RAG style)."""
    prompt = f"""Use the following context to answer the question.

Context: {context}

Question: {question}
Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id)
    
    response = tokenizer.decode(out[0], skip_special_tokens=True)
    return response.split("Answer:")[-1].strip()

def generate_with_engram(question, engram, scale_to_embed=True):
    """Generate with engram prefix injection."""
    prompt = f"Answer this question in one sentence.\nQuestion: {question}\nAnswer:"
    prompt_tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    embeddings = model.get_input_embeddings()
    prompt_embeds = embeddings(prompt_tokens["input_ids"])
    
    # Scale engram to match embedding space if requested
    engram_for_injection = engram.clone()
    if scale_to_embed:
        current_norm = torch.norm(engram_for_injection, dim=1, keepdim=True)
        engram_for_injection = engram_for_injection / current_norm * embed_norm
    
    engram_embeds = engram_for_injection.unsqueeze(0).to(prompt_embeds.dtype)
    combined_embeds = torch.cat([engram_embeds, prompt_embeds], dim=1)
    
    engram_mask = torch.ones(1, engram.shape[0], device=model.device)
    prompt_mask = prompt_tokens["attention_mask"]
    combined_mask = torch.cat([engram_mask, prompt_mask], dim=1)
    
    with torch.no_grad():
        out = model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(out[0], skip_special_tokens=True)
    # Clean up response
    if "Answer:" in response:
        return response.split("Answer:")[-1].strip()
    return response.strip()

# === RUN EXPERIMENT ===
print("\n" + "="*90)
print("LAYER SWEEP EXPERIMENT: RAG vs ENGRAM")
print("="*90)

results = {
    "baseline": {"correct": 0, "total": 0, "answers": []},
    "rag": {"correct": 0, "total": 0, "answers": []},
}
for layer in test_layers:
    results[f"layer_{layer}"] = {"correct": 0, "total": 0, "answers": [], "norm": layer_norms[layer]}
    results[f"layer_{layer}_scaled"] = {"correct": 0, "total": 0, "answers": [], "norm": layer_norms[layer]}

for q_idx, (question, expected) in enumerate(questions):
    print(f"\n{'─'*90}")
    print(f"Q{q_idx+1}: {question}")
    print(f"Expected: {expected}")
    print(f"{'─'*90}")
    
    # Baseline
    ans = generate_baseline(question)
    correct = check_answer(ans, expected)
    results["baseline"]["correct"] += int(correct)
    results["baseline"]["total"] += 1
    results["baseline"]["answers"].append(ans)
    print(f"  {'✓' if correct else '✗'} Baseline:      {ans[:70]}...")
    
    # RAG
    ans = generate_rag(question, article)
    correct = check_answer(ans, expected)
    results["rag"]["correct"] += int(correct)
    results["rag"]["total"] += 1
    results["rag"]["answers"].append(ans)
    print(f"  {'✓' if correct else '✗'} RAG:           {ans[:70]}...")
    
    # Each layer (unscaled and scaled)
    for layer in test_layers:
        # Unscaled
        ans = generate_with_engram(question, engrams_by_layer[layer], scale_to_embed=False)
        correct = check_answer(ans, expected)
        results[f"layer_{layer}"]["correct"] += int(correct)
        results[f"layer_{layer}"]["total"] += 1
        results[f"layer_{layer}"]["answers"].append(ans)
        
        # Scaled
        ans_scaled = generate_with_engram(question, engrams_by_layer[layer], scale_to_embed=True)
        correct_scaled = check_answer(ans_scaled, expected)
        results[f"layer_{layer}_scaled"]["correct"] += int(correct_scaled)
        results[f"layer_{layer}_scaled"]["total"] += 1
        results[f"layer_{layer}_scaled"]["answers"].append(ans_scaled)
        
        print(f"  {'✓' if correct else '✗'} Layer {layer:2d}:       {ans[:70]}...")
        if correct_scaled != correct:
            print(f"  {'✓' if correct_scaled else '✗'} Layer {layer:2d} scaled: {ans_scaled[:70]}...")

# === SUMMARY ===
print("\n" + "="*90)
print("RESULTS SUMMARY")
print("="*90)

print(f"\n{'Method':<25} {'Accuracy':>10} {'Norm':>10}")
print("-"*50)

for method, data in sorted(results.items(), key=lambda x: -x[1]["correct"]):
    acc = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
    norm_str = f"{data.get('norm', 0):.2f}" if 'norm' in data else "N/A"
    print(f"{method:<25} {data['correct']}/{data['total']} ({acc:5.1f}%) {norm_str:>10}")

# === FIND BEST LAYER ===
print("\n" + "="*90)
print("BEST PERFORMING LAYERS")
print("="*90)

layer_results = [(layer, results[f"layer_{layer}_scaled"]["correct"]) 
                 for layer in test_layers]
layer_results.sort(key=lambda x: -x[1])

print("\nScaled engrams (best to worst):")
for layer, correct in layer_results[:5]:
    acc = correct / len(questions) * 100
    print(f"  Layer {layer:2d}: {correct}/{len(questions)} ({acc:.0f}%) - norm before scaling: {layer_norms[layer]:.2f}")

# === TOKEN EFFICIENCY ===
print("\n" + "="*90)
print("TOKEN EFFICIENCY")
print("="*90)

rag_tokens = len(tokenizer.encode(article)) + 50  # context + prompt overhead
engram_tokens = NUM_ENGRAM_TOKENS + 30  # engram + prompt overhead

print(f"\nRAG:    ~{rag_tokens} tokens per query")
print(f"Engram: ~{engram_tokens} tokens per query ({rag_tokens/engram_tokens:.1f}x reduction)")
print(f"\nCompression: {source_tokens} source tokens → {NUM_ENGRAM_TOKENS} engram tokens ({source_tokens/NUM_ENGRAM_TOKENS:.0f}x)")

# Save results
results_path = Path("data/layer_sweep_results.json")
results_path.parent.mkdir(exist_ok=True)
with open(results_path, "w") as f:
    json.dump({
        "model": MODEL_NAME,
        "num_layers": num_layers,
        "num_engram_tokens": NUM_ENGRAM_TOKENS,
        "source_tokens": source_tokens,
        "layer_norms": layer_norms,
        "results": {k: {"correct": v["correct"], "total": v["total"]} for k, v in results.items()},
    }, f, indent=2)
print(f"\nResults saved to {results_path}")
