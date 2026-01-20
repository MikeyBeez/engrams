import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

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

Lincoln was married to Mary Todd Lincoln and had four sons, only one of whom survived 
to adulthood. He was assassinated by actor John Wilkes Booth at Ford's Theatre in 
Washington, D.C., on April 14, 1865, and died the following morning."""

print(f"\nSource document: {len(article)} chars")

# === QUESTIONS TO TEST ===
questions = [
    ("When was Lincoln born?", ["February 12, 1809", "1809"]),
    ("Where was Lincoln born?", ["Kentucky", "log cabin"]),
    ("How did Lincoln die?", ["assassinated", "John Wilkes Booth", "shot"]),
    ("Who was Lincoln married to?", ["Mary Todd", "Mary"]),
    ("How many sons did Lincoln have?", ["four", "4"]),
    ("What theatre was Lincoln assassinated in?", ["Ford's Theatre", "Ford"]),
]

# === METHOD 1: BASELINE (no context) ===
def generate_baseline(question):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    start = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id)
    elapsed = time.time() - start
    
    response = tokenizer.decode(out[0], skip_special_tokens=True)
    answer = response.split("Answer:")[-1].strip()
    return answer, elapsed, inputs["input_ids"].shape[1]

# === METHOD 2: RAG (full context in prompt) ===
def generate_rag(question, context):
    prompt = f"""Context: {context}

Question: {question}
Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    start = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id)
    elapsed = time.time() - start
    
    response = tokenizer.decode(out[0], skip_special_tokens=True)
    answer = response.split("Answer:")[-1].strip()
    return answer, elapsed, inputs["input_ids"].shape[1]

# === METHOD 3: ENGRAM (compressed context) ===
def extract_engram(text, num_tokens=8):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Use embedding layer to avoid mismatch
    hidden = outputs.hidden_states[0].squeeze(0)  # [seq, hidden]
    seq_len = hidden.shape[0]
    
    chunk_size = seq_len // num_tokens
    vectors = []
    for i in range(num_tokens):
        start = i * chunk_size
        end = start + chunk_size if i < num_tokens - 1 else seq_len
        vectors.append(hidden[start:end].mean(dim=0))
    
    return torch.stack(vectors), seq_len

def generate_engram(question, engram_vecs):
    prompt = f"Question: {question}\nAnswer:"
    prompt_tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    embeddings = model.get_input_embeddings()
    prompt_embeds = embeddings(prompt_tokens["input_ids"])
    
    engram_embeds = engram_vecs.unsqueeze(0).to(prompt_embeds.dtype)
    combined_embeds = torch.cat([engram_embeds, prompt_embeds], dim=1)
    
    engram_mask = torch.ones(1, engram_vecs.shape[0], device=model.device)
    prompt_mask = prompt_tokens["attention_mask"]
    combined_mask = torch.cat([engram_mask, prompt_mask], dim=1)
    
    start = time.time()
    with torch.no_grad():
        out = model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start
    
    response = tokenizer.decode(out[0], skip_special_tokens=True)
    answer = response.split("Answer:")[-1].strip() if "Answer:" in response else response
    return answer, elapsed, engram_vecs.shape[0] + prompt_tokens["input_ids"].shape[1]

# === EXTRACT ENGRAMS ===
print("\n=== EXTRACTING ENGRAMS ===")
engram_8, source_tokens = extract_engram(article, num_tokens=8)
engram_16, _ = extract_engram(article, num_tokens=16)
print(f"Source: {source_tokens} tokens")
print(f"Engram-8: {engram_8.shape} ({source_tokens/8:.0f}x compression)")
print(f"Engram-16: {engram_16.shape} ({source_tokens/16:.0f}x compression)")

# === RUN COMPARISON ===
print("\n" + "="*80)
print("RAG vs ENGRAM COMPARISON")
print("="*80)

results = {"baseline": [], "rag": [], "engram_8": [], "engram_16": []}

for question, expected_answers in questions:
    print(f"\n📌 {question}")
    print("-" * 60)
    
    # Check if answer contains expected terms
    def check_answer(answer, expected):
        return any(e.lower() in answer.lower() for e in expected)
    
    # Baseline
    ans, t, tokens = generate_baseline(question)
    correct = check_answer(ans, expected_answers)
    results["baseline"].append(correct)
    print(f"  Baseline ({tokens:3d} tok, {t:.3f}s): {ans[:60]}... {'✓' if correct else '✗'}")
    
    # RAG
    ans, t, tokens = generate_rag(question, article)
    correct = check_answer(ans, expected_answers)
    results["rag"].append(correct)
    print(f"  RAG      ({tokens:3d} tok, {t:.3f}s): {ans[:60]}... {'✓' if correct else '✗'}")
    
    # Engram-8
    ans, t, tokens = generate_engram(question, engram_8)
    correct = check_answer(ans, expected_answers)
    results["engram_8"].append(correct)
    print(f"  Engram-8 ({tokens:3d} tok, {t:.3f}s): {ans[:60]}... {'✓' if correct else '✗'}")
    
    # Engram-16
    ans, t, tokens = generate_engram(question, engram_16)
    correct = check_answer(ans, expected_answers)
    results["engram_16"].append(correct)
    print(f"  Engram-16({tokens:3d} tok, {t:.3f}s): {ans[:60]}... {'✓' if correct else '✗'}")

# === SUMMARY ===
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nAccuracy (correct answers out of {len(questions)}):")
for method, scores in results.items():
    acc = sum(scores) / len(scores) * 100
    print(f"  {method:12s}: {sum(scores)}/{len(scores)} ({acc:.0f}%)")

print(f"\nToken efficiency:")
rag_tokens = len(tokenizer.encode(article))
print(f"  RAG:       ~{rag_tokens} tokens per query (full context)")
print(f"  Engram-8:  8 tokens per query ({rag_tokens/8:.0f}x reduction)")
print(f"  Engram-16: 16 tokens per query ({rag_tokens/16:.0f}x reduction)")

print(f"\nStorage:")
engram_bytes = engram_8.element_size() * engram_8.numel()
text_bytes = len(article.encode('utf-8'))
print(f"  Text:      {text_bytes:,} bytes")
print(f"  Engram-8:  {engram_bytes:,} bytes ({text_bytes/engram_bytes:.1f}x smaller)")
