#!/usr/bin/env python3
"""
Large-scale Wikipedia test: RAG vs Engrams
Find a ~100k token article, generate 50 questions, compare approaches.
"""

import torch
import time
import json
import wikipediaapi
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_large_wikipedia_article():
    """Get a very large Wikipedia article."""
    wiki = wikipediaapi.Wikipedia(
        user_agent="Engrams Research (contact@example.com)",
        language="en"
    )
    
    # Try articles known to be very long
    large_articles = [
        "World_War_II",
        "United_States",
        "History_of_the_United_States", 
        "World_War_I",
        "History_of_China",
        "Roman_Empire",
        "History_of_India",
    ]
    
    for title in large_articles:
        page = wiki.page(title)
        if page.exists():
            text = page.text
            print(f"{title}: {len(text)} chars")
            if len(text) > 100000:  # At least 100k chars
                return title, text
    
    return None, None

def generate_questions_about_text(model, tokenizer, text, num_questions=50):
    """Use the model to generate factual questions about the text."""
    # Take a sample of the text for question generation
    sample = text[:8000]  # Use first 8k chars for question generation
    
    prompt = f"""Based on this text, generate {num_questions} specific factual questions that can be answered from the text. Format each question on its own line starting with a number.

Text:
{sample}

Generate {num_questions} questions:
1."""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2000,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Parse questions
    questions = []
    lines = response.strip().split("\n")
    for line in lines:
        line = line.strip()
        # Remove number prefix
        if line and line[0].isdigit():
            # Find where question starts
            for i, c in enumerate(line):
                if c in ".):":
                    question = line[i+1:].strip()
                    if question and "?" in question:
                        questions.append(question)
                    break
        elif "?" in line:
            questions.append(line)
    
    return questions[:num_questions]

def answer_with_rag(model, tokenizer, question, context, max_context_tokens=4000):
    """Answer using RAG (full context in prompt)."""
    # Truncate context to fit
    context_tokens = tokenizer.encode(context)
    if len(context_tokens) > max_context_tokens:
        context_tokens = context_tokens[:max_context_tokens]
        context = tokenizer.decode(context_tokens, skip_special_tokens=True)
    
    prompt = f"""Context:
{context}

Question: {question}
Answer (be specific and concise):"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
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
    chunk_size = seq_len // num_tokens
    engram_vectors = []
    
    for i in range(num_tokens):
        start = i * chunk_size
        end = start + chunk_size if i < num_tokens - 1 else seq_len
        chunk = hidden[0, start:end, :]
        pooled = chunk.mean(dim=0)
        engram_vectors.append(pooled)
    
    engram = torch.stack(engram_vectors)  # [num_tokens, hidden_dim]
    return engram, inputs.input_ids.shape[1]

def answer_with_engram(model, tokenizer, question, engram, scale=True):
    """Answer using engram injection."""
    # Get embedding layer
    embed_layer = model.get_input_embeddings()
    
    # Scale engram to match embedding norms
    if scale:
        embed_norm = embed_layer.weight.norm(dim=1).mean().item()
        engram_norm = engram.norm(dim=1).mean().item()
        engram = engram * (embed_norm / engram_norm)
    
    prompt = f"Question: {question}\nAnswer (be specific and concise):"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_embeds = embed_layer(inputs.input_ids)
    
    # Inject engram as prefix
    engram_prefix = engram.unsqueeze(0)  # [1, num_tokens, hidden_dim]
    combined_embeds = torch.cat([engram_prefix, prompt_embeds], dim=1)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=combined_embeds,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip().split("\n")[0], engram.shape[0] + inputs.input_ids.shape[1]

def main():
    print("=" * 60)
    print("LARGE-SCALE WIKIPEDIA TEST: RAG vs ENGRAMS")
    print("=" * 60)
    
    # Get large article
    print("\n1. Fetching large Wikipedia article...")
    title, text = get_large_wikipedia_article()
    if not text:
        print("Could not find large article!")
        return
    
    print(f"   Article: {title}")
    print(f"   Length: {len(text):,} characters")
    
    # Load model
    print("\n2. Loading model...")
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Token count
    full_tokens = tokenizer.encode(text)
    print(f"   Full article: {len(full_tokens):,} tokens")
    
    # Generate questions
    print("\n3. Generating questions about the article...")
    questions = generate_questions_about_text(model, tokenizer, text, num_questions=50)
    print(f"   Generated {len(questions)} questions")
    for i, q in enumerate(questions[:5]):
        print(f"   {i+1}. {q}")
    if len(questions) > 5:
        print(f"   ... and {len(questions)-5} more")
    
    if len(questions) < 10:
        print("   Not enough questions generated, using predefined ones...")
        # Fallback questions about WWII
        questions = [
            "When did World War II begin?",
            "When did World War II end?",
            "What countries were part of the Allied Powers?",
            "What countries were part of the Axis Powers?",
            "Who was the leader of Nazi Germany?",
            "What was D-Day?",
            "When did D-Day occur?",
            "What was the Holocaust?",
            "When did Japan attack Pearl Harbor?",
            "What caused the United States to enter World War II?",
            "What was the Battle of Stalingrad?",
            "Who won the Battle of Britain?",
            "What was the Manhattan Project?",
            "Where were atomic bombs dropped?",
            "When did Germany surrender?",
            "When did Japan surrender?",
            "What was the Nuremberg Trials?",
            "How many people died in World War II?",
            "What was the Blitzkrieg?",
            "What was the role of Winston Churchill?",
        ]
        questions = questions[:20]
    
    # Extract engram
    print("\n4. Extracting engram from full article...")
    # Use first 8192 tokens for engram extraction
    truncated_text = tokenizer.decode(full_tokens[:8192], skip_special_tokens=True)
    engram, source_tokens = extract_engram(model, tokenizer, truncated_text, layer=16, num_tokens=32)
    print(f"   Source tokens: {source_tokens:,}")
    print(f"   Engram tokens: {engram.shape[0]}")
    print(f"   Compression ratio: {source_tokens / engram.shape[0]:.1f}x")
    
    # Test both approaches
    print("\n5. Testing RAG vs Engrams...")
    print("-" * 60)
    
    results = {
        "rag": {"answers": [], "tokens": []},
        "engram": {"answers": [], "tokens": []}
    }
    
    for i, question in enumerate(questions):
        print(f"\nQ{i+1}: {question}")
        
        # RAG answer
        rag_answer, rag_tokens = answer_with_rag(model, tokenizer, question, text[:20000])
        results["rag"]["answers"].append(rag_answer)
        results["rag"]["tokens"].append(rag_tokens)
        print(f"   RAG ({rag_tokens} tokens): {rag_answer[:100]}...")
        
        # Engram answer
        eng_answer, eng_tokens = answer_with_engram(model, tokenizer, question, engram)
        results["engram"]["answers"].append(eng_answer)
        results["engram"]["tokens"].append(eng_tokens)
        print(f"   Engram ({eng_tokens} tokens): {eng_answer[:100]}...")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Questions tested: {len(questions)}")
    print(f"Average RAG tokens: {sum(results[rag][tokens])/len(results[rag][tokens]):.0f}")
    print(f"Average Engram tokens: {sum(results[engram][tokens])/len(results[engram][tokens]):.0f}")
    
    # Save results
    with open("/home/bee/Code/engrams/results/wikipedia_scale_test.json", "w") as f:
        json.dump({
            "article": title,
            "article_chars": len(text),
            "article_tokens": len(full_tokens),
            "questions": questions,
            "results": results
        }, f, indent=2)
    
    print("\nResults saved to results/wikipedia_scale_test.json")

if __name__ == "__main__":
    main()
