#!/usr/bin/env python3
"""
50-Question Wikipedia Test: RAG vs Engrams
Comprehensive test as requested by user.
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

# 50 factual questions about World War II with expected keywords
WWII_QUESTIONS = [
    # Dates and Timeline
    ("When did World War II begin?", "1939"),
    ("When did World War II end?", "1945"),
    ("When did Germany invade Poland?", "1939,September"),
    ("When did Japan attack Pearl Harbor?", "1941,December"),
    ("When was D-Day?", "1944,June"),
    ("When did Germany surrender?", "1945,May"),
    ("When did Japan surrender?", "1945,August"),
    ("When did Italy surrender?", "1943"),
    ("When did the Battle of Stalingrad end?", "1943,February"),
    ("When did the Soviet Union enter the war against Japan?", "1945,August"),
    
    # Leaders
    ("Who was the leader of Nazi Germany?", "Hitler,Adolf"),
    ("Who was the British Prime Minister during WWII?", "Churchill,Winston"),
    ("Who was the US President at the start of WWII?", "Roosevelt,Franklin"),
    ("Who was the US President when WWII ended?", "Truman,Harry"),
    ("Who was the leader of the Soviet Union during WWII?", "Stalin,Joseph"),
    ("Who was the leader of Italy during WWII?", "Mussolini,Benito"),
    ("Who was the Emperor of Japan during WWII?", "Hirohito"),
    ("Who commanded Allied forces on D-Day?", "Eisenhower"),
    ("Who was the leader of Free France?", "Gaulle,Charles"),
    ("Who commanded German forces in North Africa?", "Rommel"),
    
    # Events and Battles
    ("What started World War II in Europe?", "invasion,Poland"),
    ("What brought the US into World War II?", "Pearl Harbor"),
    ("What was the turning point on the Eastern Front?", "Stalingrad"),
    ("What was the largest naval battle of WWII?", "Leyte,Philippine"),
    ("What was Operation Barbarossa?", "invasion,Soviet,Russia"),
    ("What was the Battle of Britain?", "air,aerial,RAF"),
    ("What was the Blitzkrieg?", "lightning,fast,quick"),
    ("What happened at Dunkirk?", "evacuation,rescue"),
    ("What was the Battle of the Bulge?", "Ardennes,offensive,German"),
    ("What was Operation Overlord?", "Normandy,D-Day,invasion"),
    
    # Holocaust and Atrocities
    ("What was the Holocaust?", "genocide,Jews,murder"),
    ("How many Jews died in the Holocaust?", "six million,6 million"),
    ("What were concentration camps?", "prison,detention,death"),
    ("What was Auschwitz?", "concentration,death,camp"),
    ("What were the Nuremberg Trials?", "trial,war crimes,prosecution"),
    
    # Weapons and Technology
    ("What was the Manhattan Project?", "atomic,nuclear,bomb"),
    ("What city was the first atomic bomb dropped on?", "Hiroshima"),
    ("What city was the second atomic bomb dropped on?", "Nagasaki"),
    ("What was the V-2?", "rocket,missile,weapon"),
    ("What was the Enigma machine?", "code,cipher,encryption"),
    
    # Alliances and Countries
    ("What were the Axis powers?", "Germany,Italy,Japan"),
    ("What were the Allied powers?", "Britain,United States,Soviet"),
    ("Which countries remained neutral?", "Switzerland,Sweden,Spain"),
    ("When did the Soviet Union join the Allies?", "1941,Germany attacked"),
    ("What was the Tripartite Pact?", "Axis,Germany,Italy,Japan"),
    
    # Geography and Strategy
    ("Where did D-Day take place?", "Normandy,France"),
    ("What was the Atlantic Wall?", "defense,fortification,German"),
    ("What island was heavily fought over in the Pacific?", "Iwo Jima,Okinawa,Guadalcanal"),
    ("What was the Burma Road?", "supply,China"),
    ("Where was the final major battle in Europe?", "Berlin"),
]

def answer_with_rag(model, tokenizer, question, context, max_tokens=3000):
    tokens = tokenizer.encode(context)[:max_tokens]
    context = tokenizer.decode(tokens, skip_special_tokens=True)
    
    prompt = f"""Context about World War II:
{context}

Question: {question}
Answer:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=50, temperature=0.1,
            do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    
    answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return answer.strip().split("\n")[0], inputs.input_ids.shape[1]

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

def answer_with_engram(model, tokenizer, question, engram):
    embed = model.get_input_embeddings()
    
    # Scale
    e_norm = embed.weight.norm(dim=1).mean().item()
    g_norm = engram.norm(dim=1).mean().item()
    scaled = engram * (e_norm / g_norm)
    
    prompt = f"About World War II: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    emb = embed(inputs.input_ids)
    
    combined = torch.cat([scaled.unsqueeze(0).to(emb.dtype), emb], dim=1)
    
    with torch.no_grad():
        out = model.generate(
            inputs_embeds=combined, max_new_tokens=50, temperature=0.1,
            do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(out[0], skip_special_tokens=True).split("\n")[0], engram.shape[0] + inputs.input_ids.shape[1]

def check(response, expected):
    r = response.lower()
    for kw in expected.split(","):
        if kw.strip().lower() in r:
            return True
    return False

def main():
    print("=" * 70)
    print("50-QUESTION WIKIPEDIA TEST: RAG vs ENGRAMS")
    print("=" * 70)
    
    print("\n1. Fetching article...")
    text = get_wikipedia_article("World_War_II")
    print(f"   {len(text):,} chars")
    
    print("\n2. Loading model...")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B", torch_dtype=torch.float16, device_map="auto"
    )
    
    total_tokens = len(tok.encode(text))
    print(f"   {total_tokens:,} tokens")
    
    print("\n3. Extracting engram...")
    engram, src_tokens = extract_engram_chunked(model, tok, text, layer=16, num_tokens=32)
    print(f"   {src_tokens} -> {engram.shape[0]} tokens ({src_tokens//engram.shape[0]}x compression)")
    
    print("\n4. Running 50 questions...")
    print("-" * 70)
    
    rag_c, eng_c = 0, 0
    rag_t, eng_t = 0, 0
    results = []
    
    for i, (q, exp) in enumerate(WWII_QUESTIONS):
        print(f"\nQ{i+1}: {q}")
        
        ra, rt = answer_with_rag(model, tok, q, text)
        rag_t += rt
        rm = check(ra, exp)
        if rm: rag_c += 1
        print(f"   RAG [{'Y' if rm else 'N'}]: {ra[:55]}...")
        
        ea, et = answer_with_engram(model, tok, q, engram)
        eng_t += et
        em = check(ea, exp)
        if em: eng_c += 1
        print(f"   ENG [{'Y' if em else 'N'}]: {ea[:55]}...")
        
        results.append({"q": q, "rag_ok": rm, "eng_ok": em, "rag": ra, "eng": ea})
        
        gc.collect()
        torch.cuda.empty_cache()
    
    n = len(WWII_QUESTIONS)
    print("\n" + "=" * 70)
    print("FINAL RESULTS (50 Questions)")
    print("=" * 70)
    print(f"RAG:    {rag_c}/{n} ({100*rag_c/n:.1f}%) | {rag_t/n:.0f} tokens/q")
    print(f"Engram: {eng_c}/{n} ({100*eng_c/n:.1f}%) | {eng_t/n:.0f} tokens/q")
    print(f"Token efficiency: {rag_t/eng_t:.1f}x fewer with engrams")
    print(f"Compression: {src_tokens} -> {engram.shape[0]} ({src_tokens//engram.shape[0]}x)")
    
    with open("/home/bee/Code/engrams/results/wiki_50q.json", "w") as f:
        json.dump({
            "n_questions": n,
            "rag_accuracy": rag_c/n,
            "engram_accuracy": eng_c/n,
            "rag_tokens_avg": rag_t/n,
            "engram_tokens_avg": eng_t/n,
            "token_efficiency": rag_t/eng_t,
            "compression": src_tokens/engram.shape[0],
            "results": results
        }, f, indent=2)
    print("\nSaved to results/wiki_50q.json")

if __name__ == "__main__":
    main()
