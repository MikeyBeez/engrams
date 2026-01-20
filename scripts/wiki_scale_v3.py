#!/usr/bin/env python3
"""
Large-scale Wikipedia test: RAG vs Engrams (v3)
Memory-optimized version - uses chunked extraction
"""

import torch
import gc
import json
import wikipediaapi
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_wikipedia_article(title="World_War_II"):
    """Get a Wikipedia article."""
    wiki = wikipediaapi.Wikipedia(
        user_agent="Engrams Research (contact@example.com)",
        language="en"
    )
    page = wiki.page(title)
    if page.exists():
        return page.text
    return None

# 20 factual questions about World War II
WWII_QUESTIONS = [
    ("When did World War II begin?", "1939"),
    ("When did World War II end?", "1945"),
    ("What event started World War II in Europe?", "invasion of Poland"),
    ("Who was the leader of Nazi Germany?", "Hitler"),
    ("What was the date of D-Day?", "June 6, 1944"),
    ("When did Japan attack Pearl Harbor?", "December 7, 1941"),
    ("What caused the United States to enter World War II?", "Pearl Harbor"),
    ("What was the Battle of Stalingrad?", "turning point"),
    ("Who won the Battle of Britain?", "Britain"),
    ("What was the Manhattan Project?", "atomic bomb"),
    ("What city was the first atomic bomb dropped on?", "Hiroshima"),
    ("What city was the second atomic bomb dropped on?", "Nagasaki"),
    ("When did Germany surrender?", "May 1945"),
    ("When did Japan surrender?", "August 1945"),
    ("What was the Holocaust?", "genocide"),
    ("How many Jews were killed in the Holocaust?", "six million"),
    ("What was the Blitzkrieg?", "lightning war"),
    ("Who was the British Prime Minister during most of WWII?", "Churchill"),
    ("Who was the US President when WWII ended?", "Truman"),
    ("What alliance fought against the Axis powers?", "Allies"),
]

def answer_with_rag(model, tokenizer, question, context, max_context_tokens=3000):
    """Answer using RAG (context in prompt)."""
    # Truncate context
    tokens = tokenizer.encode(context)[:max_context_tokens]
    context = tokenizer.decode(tokens, skip_special_tokens=True)
    
    prompt = f"""Context about World War II:
{context}

Question: {question}
Answer:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return answer.strip().split("\n")[0], inputs.input_ids.shape[1]

def extract_engram_chunked(model, tokenizer, text, layer=16, num_tokens=32, chunk_tokens=2048):
    """Extract engram using chunked processing to avoid OOM."""
    tokens = tokenizer.encode(text)
    total_tokens = min(len(tokens), 8192)  # Limit total
    
    all_hidden_chunks = []
    
    # Process in chunks
    for start in range(0, total_tokens, chunk_tokens):
        end = min(start + chunk_tokens, total_tokens)
        chunk_ids = torch.tensor([tokens[start:end]]).to(model.device)
        
        with torch.no_grad():
            outputs = model(chunk_ids, output_hidden_states=True)
        
        hidden = outputs.hidden_states[layer].cpu()  # Move to CPU immediately
        all_hidden_chunks.append(hidden[0])  # [seq_len, hidden_dim]
        
        # Clear GPU memory
        del outputs
        torch.cuda.empty_cache()
    
    # Concatenate all chunks
    full_hidden = torch.cat(all_hidden_chunks, dim=0)  # [total_seq, hidden_dim]
    seq_len = full_hidden.shape[0]
    
    # Pool into engram tokens
    chunk_size = seq_len // num_tokens
    engram_vectors = []
    
    for i in range(num_tokens):
        start = i * chunk_size
        end = start + chunk_size if i < num_tokens - 1 else seq_len
        chunk = full_hidden[start:end]
        pooled = chunk.mean(dim=0)
        engram_vectors.append(pooled)
    
    engram = torch.stack(engram_vectors).to(model.device)  # [num_tokens, hidden_dim]
    return engram, seq_len

def answer_with_engram(model, tokenizer, question, engram):
    """Answer using engram injection."""
    embed_layer = model.get_input_embeddings()
    
    # Scale engram
    embed_norm = embed_layer.weight.norm(dim=1).mean().item()
    engram_norm = engram.norm(dim=1).mean().item()
    scaled_engram = engram * (embed_norm / engram_norm)
    
    prompt = f"Based on context about World War II: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_embeds = embed_layer(inputs.input_ids)
    
    engram_prefix = scaled_engram.unsqueeze(0).to(prompt_embeds.dtype)
    combined_embeds = torch.cat([engram_prefix, prompt_embeds], dim=1)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=combined_embeds,
            max_new_tokens=50,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip().split("\n")[0], engram.shape[0] + inputs.input_ids.shape[1]

def check_answer(response, expected):
    """Check if response contains expected information."""
    response_lower = response.lower()
    keywords = [k.strip().lower() for k in expected.split(",")]
    for keyword in keywords:
        if keyword in response_lower:
            return True
    return False

def main():
    print("=" * 70)
    print("WIKIPEDIA SCALE TEST: RAG vs ENGRAMS (v3 - Memory Optimized)")
    print("=" * 70)
    
    # Get article
    print("\n1. Fetching World War II Wikipedia article...")
    text = get_wikipedia_article("World_War_II")
    if not text:
        print("Failed!")
        return
    print(f"   {len(text):,} characters")
    
    # Load model
    print("\n2. Loading Qwen2.5-7B...")
    model_name = "Qwen/Qwen2.5-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    full_tokens = len(tokenizer.encode(text))
    print(f"   Full article: {full_tokens:,} tokens")
    
    # Extract engram with chunked processing
    print("\n3. Extracting engram (chunked)...")
    engram, source_tokens = extract_engram_chunked(model, tokenizer, text, layer=16, num_tokens=32)
    print(f"   Source: {source_tokens:,} tokens -> Engram: {engram.shape[0]} tokens")
    print(f"   Compression: {source_tokens / engram.shape[0]:.0f}x")
    
    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Test
    print("\n4. Testing RAG vs Engrams...")
    print("-" * 70)
    
    rag_correct = 0
    engram_correct = 0
    rag_tokens_total = 0
    engram_tokens_total = 0
    results = []
    
    for i, (question, expected) in enumerate(WWII_QUESTIONS):
        print(f"\nQ{i+1}: {question}")
        
        # RAG
        rag_answer, rag_tokens = answer_with_rag(model, tokenizer, question, text)
        rag_tokens_total += rag_tokens
        rag_match = check_answer(rag_answer, expected)
        if rag_match:
            rag_correct += 1
        status = "\u2713" if rag_match else "x"
        print(f"   RAG [{status}]: {rag_answer[:60]}...")
        
        # Engram
        eng_answer, eng_tokens = answer_with_engram(model, tokenizer, question, engram)
        engram_tokens_total += eng_tokens
        eng_match = check_answer(eng_answer, expected)
        if eng_match:
            engram_correct += 1
        status = "\u2713" if eng_match else "x"
        print(f"   Engram [{status}]: {eng_answer[:60]}...")
        
        results.append({
            "q": question, "exp": expected,
            "rag": rag_answer, "rag_ok": rag_match,
            "eng": eng_answer, "eng_ok": eng_match
        })
        
        # Memory management
        gc.collect()
        torch.cuda.empty_cache()
    
    # Summary
    n = len(WWII_QUESTIONS)
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"RAG:    {rag_correct}/{n} ({100*rag_correct/n:.0f}%) | {rag_tokens_total/n:.0f} tokens/q avg")
    print(f"Engram: {engram_correct}/{n} ({100*engram_correct/n:.0f}%) | {engram_tokens_total/n:.0f} tokens/q avg")
    print(f"Token savings: {rag_tokens_total/engram_tokens_total:.1f}x with engrams")
    print(f"Compression: {source_tokens} -> {engram.shape[0]} ({source_tokens/engram.shape[0]:.0f}x)")
    
    # Save
    with open("/home/bee/Code/engrams/results/wiki_scale_v3.json", "w") as f:
        json.dump({
            "rag_accuracy": rag_correct/n,
            "engram_accuracy": engram_correct/n,
            "compression": source_tokens/engram.shape[0],
            "token_savings": rag_tokens_total/engram_tokens_total,
            "results": results
        }, f, indent=2)
    print("\nSaved to results/wiki_scale_v3.json")

if __name__ == "__main__":
    main()
