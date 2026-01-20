#!/usr/bin/env python3
"""
Low-frequency fact test: Questions the model is unlikely to know from pretraining.
Uses obscure WWII facts that require the specific Wikipedia article to answer.
"""

import torch
import gc
import json
import wikipediaapi
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_wikipedia_article(title="World_War_II"):
    wiki = wikipediaapi.Wikipedia(user_agent="Engrams Research", language="en")
    page = wiki.page(title)
    return page.text if page.exists() else None

# Obscure facts from the WWII Wikipedia article that models are unlikely to know
# These require specific details, numbers, or lesser-known information
LOW_FREQUENCY_QUESTIONS = [
    # Specific numbers and statistics
    ("How many military personnel were mobilized during World War II?", "70 million,seventy million"),
    ("What percentage of the world population died in World War II?", "3%,three percent"),
    ("How many civilians died in World War II?", "50 million,fifty million,50-55"),
    
    # Specific dates of lesser-known events
    ("When was the Atlantic Charter signed?", "August 1941,1941"),
    ("When did the Winter War between Soviet Union and Finland end?", "March 1940,1940"),
    ("When was the Tripartite Pact signed?", "September 1940,27 September"),
    
    # Lesser-known operations and code names
    ("What was the code name for the German invasion of Yugoslavia?", "Directive 25,Operation 25"),
    ("What was Operation Torch?", "Allied invasion,North Africa,1942"),
    ("What was Operation Bagration?", "Soviet,Belarus,1944"),
    
    # Specific lesser-known facts
    ("What country had the highest military casualties as a percentage of population?", "Soviet Union,USSR,Russia"),
    ("What was the last country Germany invaded before its surrender?", "Czechoslovakia,Czech"),
    ("Which neutral country was invaded by both Allied and Axis forces?", "Iran,Persia"),
    
    # Technical details
    ("What was the name of the first jet-powered fighter used in combat?", "Messerschmitt,Me 262"),
    ("What treaty ended the war between Finland and the Soviet Union in 1944?", "Moscow Armistice,armistice"),
    ("What was the largest encirclement of troops in history during WWII?", "Kiev,Kyiv,1941"),
    
    # Obscure people
    ("Who was the Supreme Commander of Allied Forces in Southeast Asia?", "Mountbatten,Louis"),
    ("Who led the Polish government-in-exile?", "Sikorski,Raczkie"),
    ("Who commanded the Japanese forces at Iwo Jima?", "Kuribayashi"),
    
    # Lesser-known events
    ("What was the Katyn massacre?", "Polish,officers,Soviet,NKVD"),
    ("What was the Rape of Nanking?", "massacre,Chinese,Japanese,1937"),
]

def check(response, expected):
    r = response.lower()
    for kw in expected.split(","):
        if kw.strip().lower() in r:
            return True
    return False

def extract_engram_chunked(model, tokenizer, text, layer=16, num_tokens=32, chunk_size=2048):
    tokens = tokenizer.encode(text)
    total = min(len(tokens), 8192)
    
    chunks = []
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        ids = torch.tensor([tokens[start:end]]).to(model.device)
        
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        chunks.append(out.hidden_states[layer].cpu()[0])
        del out
        torch.cuda.empty_cache()
    
    hidden = torch.cat(chunks, dim=0)
    seq_len = hidden.shape[0]
    cs = seq_len // num_tokens
    
    engram = []
    for i in range(num_tokens):
        s, e = i * cs, (i + 1) * cs if i < num_tokens - 1 else seq_len
        engram.append(hidden[s:e].mean(dim=0))
    
    return torch.stack(engram).to(model.device), seq_len

def answer_no_context(model, tokenizer, question):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=50, temperature=0.1,
            do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    
    answer = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return answer.strip().split("\n")[0]

def answer_with_rag(model, tokenizer, question, context, max_tokens=3000):
    tokens = tokenizer.encode(context)[:max_tokens]
    context = tokenizer.decode(tokens, skip_special_tokens=True)
    
    prompt = f"""Context about World War II:
{context}

Question: {question}
Answer:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=50, temperature=0.1,
            do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    
    answer = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return answer.strip().split("\n")[0]

def answer_with_engram(model, tokenizer, question, engram):
    embed = model.get_input_embeddings()
    
    e_norm = embed.weight.norm(dim=1).mean().item()
    g_norm = engram.norm(dim=1).mean().item()
    scaled = engram * (e_norm / g_norm)
    
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    emb = embed(inputs.input_ids)
    
    combined = torch.cat([scaled.unsqueeze(0).to(emb.dtype), emb], dim=1)
    
    with torch.no_grad():
        out = model.generate(
            inputs_embeds=combined, max_new_tokens=50, temperature=0.1,
            do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(out[0], skip_special_tokens=True).split("\n")[0]

def main():
    print("=" * 70)
    print("LOW-FREQUENCY FACT TEST")
    print("Testing obscure facts the model is unlikely to know")
    print("=" * 70)
    
    print("\n1. Fetching Wikipedia article...")
    text = get_wikipedia_article("World_War_II")
    print(f"   {len(text):,} characters")
    
    print("\n2. Loading model...")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B", torch_dtype=torch.float16, device_map="auto"
    )
    
    print("\n3. Extracting engram...")
    engram, src_tokens = extract_engram_chunked(model, tok, text, layer=16, num_tokens=32)
    print(f"   {src_tokens} tokens -> {engram.shape[0]} engram vectors")
    
    print("\n4. Testing low-frequency questions...")
    print("-" * 70)
    
    baseline_correct = 0
    rag_correct = 0
    engram_correct = 0
    results = []
    
    for i, (q, exp) in enumerate(LOW_FREQUENCY_QUESTIONS):
        print(f"\nQ{i+1}: {q}")
        print(f"     Expected: {exp}")
        
        # Baseline
        base_ans = answer_no_context(model, tok, q)
        base_match = check(base_ans, exp)
        if base_match:
            baseline_correct += 1
        print(f"     Baseline [{('Y' if base_match else 'N')}]: {base_ans[:50]}...")
        
        # RAG
        rag_ans = answer_with_rag(model, tok, q, text)
        rag_match = check(rag_ans, exp)
        if rag_match:
            rag_correct += 1
        print(f"     RAG      [{('Y' if rag_match else 'N')}]: {rag_ans[:50]}...")
        
        # Engram
        eng_ans = answer_with_engram(model, tok, q, engram)
        eng_match = check(eng_ans, exp)
        if eng_match:
            engram_correct += 1
        print(f"     Engram   [{('Y' if eng_match else 'N')}]: {eng_ans[:50]}...")
        
        results.append({
            "q": q, "expected": exp,
            "baseline": base_ans, "baseline_ok": base_match,
            "rag": rag_ans, "rag_ok": rag_match,
            "engram": eng_ans, "engram_ok": eng_match
        })
        
        gc.collect()
        torch.cuda.empty_cache()
    
    n = len(LOW_FREQUENCY_QUESTIONS)
    print("\n" + "=" * 70)
    print("LOW-FREQUENCY FACT RESULTS")
    print("=" * 70)
    print(f"Baseline (no context): {baseline_correct}/{n} ({100*baseline_correct/n:.1f}%)")
    print(f"RAG (stuffed context): {rag_correct}/{n} ({100*rag_correct/n:.1f}%)")
    print(f"Engram:                {engram_correct}/{n} ({100*engram_correct/n:.1f}%)")
    print()
    print("Key question: Do engrams help on facts the model does NOT already know?")
    print(f"Engram lift over baseline: {engram_correct - baseline_correct} questions")
    print(f"RAG lift over baseline: {rag_correct - baseline_correct} questions")
    
    with open("/home/bee/Code/engrams/results/low_frequency_test.json", "w") as f:
        json.dump({
            "test": "low_frequency_facts",
            "n_questions": n,
            "baseline_accuracy": baseline_correct/n,
            "rag_accuracy": rag_correct/n,
            "engram_accuracy": engram_correct/n,
            "baseline_correct": baseline_correct,
            "rag_correct": rag_correct,
            "engram_correct": engram_correct,
            "results": results
        }, f, indent=2)
    
    print("\nSaved to results/low_frequency_test.json")

if __name__ == "__main__":
    main()
