#!/usr/bin/env python3
"""
Baseline test: How well does the model answer WWII questions with NO context?
This tests whether the model already knows the answers from pretraining.
"""

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# Same 50 questions from the main experiment
WWII_QUESTIONS = [
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
    ("What was the Holocaust?", "genocide,Jews,murder"),
    ("How many Jews died in the Holocaust?", "six million,6 million"),
    ("What were concentration camps?", "prison,detention,death"),
    ("What was Auschwitz?", "concentration,death,camp"),
    ("What were the Nuremberg Trials?", "trial,war crimes,prosecution"),
    ("What was the Manhattan Project?", "atomic,nuclear,bomb"),
    ("What city was the first atomic bomb dropped on?", "Hiroshima"),
    ("What city was the second atomic bomb dropped on?", "Nagasaki"),
    ("What was the V-2?", "rocket,missile,weapon"),
    ("What was the Enigma machine?", "code,cipher,encryption"),
    ("What were the Axis powers?", "Germany,Italy,Japan"),
    ("What were the Allied powers?", "Britain,United States,Soviet"),
    ("Which countries remained neutral?", "Switzerland,Sweden,Spain"),
    ("When did the Soviet Union join the Allies?", "1941,Germany attacked"),
    ("What was the Tripartite Pact?", "Axis,Germany,Italy,Japan"),
    ("Where did D-Day take place?", "Normandy,France"),
    ("What was the Atlantic Wall?", "defense,fortification,German"),
    ("What island was heavily fought over in the Pacific?", "Iwo Jima,Okinawa,Guadalcanal"),
    ("What was the Burma Road?", "supply,China"),
    ("Where was the final major battle in Europe?", "Berlin"),
]

def check(response, expected):
    r = response.lower()
    for kw in expected.split(","):
        if kw.strip().lower() in r:
            return True
    return False

def main():
    print("=" * 70)
    print("BASELINE TEST: Model knowledge WITHOUT any context")
    print("=" * 70)
    
    print("\nLoading model...")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B", torch_dtype=torch.float16, device_map="auto"
    )
    
    print("\nTesting 50 questions with NO context...")
    print("-" * 70)
    
    correct = 0
    results = []
    
    for i, (q, exp) in enumerate(WWII_QUESTIONS):
        # Minimal prompt - no context at all
        prompt = f"Question: {q}\nAnswer:"
        
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=50, temperature=0.1,
                do_sample=True, pad_token_id=tok.eos_token_id
            )
        
        answer = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        answer = answer.strip().split("\n")[0]
        
        match = check(answer, exp)
        if match:
            correct += 1
        
        status = "Y" if match else "N"
        print(f"Q{i+1}: {q}")
        print(f"   [{status}]: {answer[:60]}...")
        
        results.append({"q": q, "expected": exp, "answer": answer, "correct": match})
    
    print("\n" + "=" * 70)
    print("BASELINE RESULTS (NO CONTEXT)")
    print("=" * 70)
    print(f"Accuracy: {correct}/50 ({100*correct/50:.1f}%)")
    print()
    print("Comparison:")
    print(f"  Baseline (no context): {correct}/50 ({100*correct/50:.1f}%)")
    print(f"  RAG (stuffed context): 40/50 (80.0%)")
    print(f"  Engram:                48/50 (96.0%)")
    
    with open("/home/bee/Code/engrams/results/baseline_no_context.json", "w") as f:
        json.dump({
            "method": "no_context_baseline",
            "accuracy": correct/50,
            "correct": correct,
            "total": 50,
            "results": results
        }, f, indent=2)
    
    print("\nSaved to results/baseline_no_context.json")

if __name__ == "__main__":
    main()
