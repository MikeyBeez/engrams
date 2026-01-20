#!/usr/bin/env python3
"""
Large-scale Wikipedia test: RAG vs Engrams (v2)
Uses cached Qwen2.5-7B model with predefined questions.
"""

import torch
import time
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

# Predefined questions about World War II
WWII_QUESTIONS = [
    ("When did World War II begin?", "1939"),
    ("When did World War II end?", "1945"),
    ("What event started World War II in Europe?", "invasion of Poland"),
    ("Who was the leader of Nazi Germany?", "Hitler"),
    ("What was the date of D-Day?", "June 6, 1944"),
    ("What beach was NOT part of D-Day landings?", "Omaha, Utah, Juno, Gold, Sword"),
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
]

def answer_with_rag(model, tokenizer, question, context, max_context_chars=15000):
    """Answer using RAG (context in prompt)."""
    context = context[:max_context_chars]
    
    prompt = f"""Based on this context about World War II, answer the question.

Context:
{context}

Question: {question}
Answer:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to(model.device)
    
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

def extract_engram(model, tokenizer, text, layer=16, num_tokens=32):
    """Extract engram from text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden = outputs.hidden_states[layer]  # [1, seq_len, hidden_dim]
    seq_len = hidden.shape[1]
    
    # Chunk and pool
    chunk_size = max(1, seq_len // num_tokens)
    engram_vectors = []
    
    for i in range(num_tokens):
        start = i * chunk_size
        end = min(start + chunk_size, seq_len) if i < num_tokens - 1 else seq_len
        if start >= seq_len:
            break
        chunk = hidden[0, start:end, :]
        pooled = chunk.mean(dim=0)
        engram_vectors.append(pooled)
    
    engram = torch.stack(engram_vectors)  # [num_tokens, hidden_dim]
    return engram, inputs.input_ids.shape[1]

def answer_with_engram(model, tokenizer, question, engram):
    """Answer using engram injection."""
    embed_layer = model.get_input_embeddings()
    
    # Scale engram to match embedding norms
    embed_norm = embed_layer.weight.norm(dim=1).mean().item()
    engram_norm = engram.norm(dim=1).mean().item()
    scaled_engram = engram * (embed_norm / engram_norm)
    
    prompt = f"Based on the context about World War II, answer: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_embeds = embed_layer(inputs.input_ids)
    
    # Inject engram as prefix
    engram_prefix = scaled_engram.unsqueeze(0)  # [1, num_tokens, hidden_dim]
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

def check_answer(response, expected_keywords):
    """Check if response contains expected information."""
    response_lower = response.lower()
    if isinstance(expected_keywords, str):
        keywords = [k.strip().lower() for k in expected_keywords.split(",")]
    else:
        keywords = [expected_keywords.lower()]
    
    for keyword in keywords:
        if keyword in response_lower:
            return True
    return False

def main():
    print("=" * 70)
    print("LARGE-SCALE WIKIPEDIA TEST: RAG vs ENGRAMS (v2)")
    print("=" * 70)
    
    # Get article
    print("\n1. Fetching World War II Wikipedia article...")
    text = get_wikipedia_article("World_War_II")
    if not text:
        print("Failed to fetch article!")
        return
    
    print(f"   Length: {len(text):,} characters")
    
    # Load model (cached)
    print("\n2. Loading Qwen2.5-7B model (cached)...")
    model_name = "Qwen/Qwen2.5-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("   Model loaded!")
    
    # Token count
    full_tokens = tokenizer.encode(text)
    print(f"   Full article: {len(full_tokens):,} tokens")
    
    # Extract engram
    print("\n3. Extracting engram from article...")
    engram, source_tokens = extract_engram(model, tokenizer, text, layer=16, num_tokens=64)
    print(f"   Source tokens used: {source_tokens:,}")
    print(f"   Engram tokens: {engram.shape[0]}")
    print(f"   Compression ratio: {source_tokens / engram.shape[0]:.1f}x")
    
    # Test both approaches
    print("\n4. Testing RAG vs Engrams on 20 questions...")
    print("-" * 70)
    
    rag_correct = 0
    engram_correct = 0
    rag_tokens_total = 0
    engram_tokens_total = 0
    
    results = []
    
    for i, (question, expected) in enumerate(WWII_QUESTIONS):
        print(f"\nQ{i+1}: {question}")
        print(f"     Expected: {expected}")
        
        # RAG answer
        rag_answer, rag_tokens = answer_with_rag(model, tokenizer, question, text)
        rag_tokens_total += rag_tokens
        rag_match = check_answer(rag_answer, expected)
        if rag_match:
            rag_correct += 1
        print(f"     RAG ({rag_tokens} tokens): {rag_answer[:80]}... [{u2713 if rag_match else x}]")
        
        # Engram answer
        eng_answer, eng_tokens = answer_with_engram(model, tokenizer, question, engram)
        engram_tokens_total += eng_tokens
        eng_match = check_answer(eng_answer, expected)
        if eng_match:
            engram_correct += 1
        print(f"     Engram ({eng_tokens} tokens): {eng_answer[:80]}... [{u2713 if eng_match else x}]")
        
        results.append({
            "question": question,
            "expected": expected,
            "rag_answer": rag_answer,
            "rag_tokens": rag_tokens,
            "rag_correct": rag_match,
            "engram_answer": eng_answer,
            "engram_tokens": eng_tokens,
            "engram_correct": eng_match
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Questions: {len(WWII_QUESTIONS)}")
    print(f"")
    print(f"RAG Results:")
    print(f"  Accuracy: {rag_correct}/{len(WWII_QUESTIONS)} ({100*rag_correct/len(WWII_QUESTIONS):.1f}%)")
    print(f"  Avg tokens per question: {rag_tokens_total/len(WWII_QUESTIONS):.0f}")
    print(f"")
    print(f"Engram Results:")
    print(f"  Accuracy: {engram_correct}/{len(WWII_QUESTIONS)} ({100*engram_correct/len(WWII_QUESTIONS):.1f}%)")
    print(f"  Avg tokens per question: {engram_tokens_total/len(WWII_QUESTIONS):.0f}")
    print(f"")
    print(f"Token Efficiency: {rag_tokens_total/engram_tokens_total:.1f}x fewer tokens with engrams")
    print(f"Compression: {source_tokens} -> {engram.shape[0]} tokens ({source_tokens/engram.shape[0]:.0f}x)")
    
    # Save results
    summary = {
        "article": "World_War_II",
        "article_chars": len(text),
        "article_tokens": len(full_tokens),
        "engram_source_tokens": source_tokens,
        "engram_tokens": engram.shape[0],
        "compression_ratio": source_tokens / engram.shape[0],
        "rag_accuracy": rag_correct / len(WWII_QUESTIONS),
        "engram_accuracy": engram_correct / len(WWII_QUESTIONS),
        "rag_avg_tokens": rag_tokens_total / len(WWII_QUESTIONS),
        "engram_avg_tokens": engram_tokens_total / len(WWII_QUESTIONS),
        "results": results
    }
    
    with open("/home/bee/Code/engrams/results/wiki_scale_v2.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\nResults saved to results/wiki_scale_v2.json")

if __name__ == "__main__":
    main()
