#!/usr/bin/env python3
"""
Novel Facts Test for Engrams

Test whether engrams can convey TRUE information that the model doesn't know.

This is the critical test to distinguish:
1. Pure retrieval cue: Can only surface existing knowledge, novel facts won't transfer
2. Weak information injection: Can inject info, but not strongly enough to override priors

We use facts that are:
- True (verifiable)
- Recent (after training cutoff) OR obscure (unlikely to be in training)
- Specific (testable with clear right/wrong answers)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime
import os

# Novel/obscure facts the model likely doesn't know
# Format: (fact_statement, question, correct_answer_markers, topic_markers)
NOVEL_FACTS = [
    # Recent events (2024-2025) - after likely training cutoff
    (
        "The 2024 Nobel Prize in Physics was awarded to John Hopfield and Geoffrey Hinton for foundational discoveries in machine learning with artificial neural networks.",
        "Who won the 2024 Nobel Prize in Physics?",
        ["hopfield", "hinton", "geoffrey hinton", "john hopfield"],
        ["nobel", "physics", "2024"]
    ),
    (
        "SpaceX's Starship completed its first successful catch by the launch tower's chopstick arms on October 13, 2024.",
        "When did SpaceX first catch a Starship booster with the tower arms?",
        ["october 13", "october 2024", "2024"],
        ["spacex", "starship", "catch", "tower"]
    ),
    # Obscure but verifiable facts
    (
        "The population of Whittier, Alaska is approximately 272 people, and nearly all residents live in a single 14-story building called Begich Towers.",
        "What is unusual about where residents of Whittier, Alaska live?",
        ["begich towers", "single building", "one building", "14-story", "14 story"],
        ["whittier", "alaska", "building", "residents"]
    ),
    (
        "The shortest war in history was between Britain and Zanzibar on August 27, 1896, lasting only 38 to 45 minutes.",
        "How long did the Anglo-Zanzibar War last?",
        ["38 minutes", "45 minutes", "38 to 45", "less than an hour", "under an hour"],
        ["war", "zanzibar", "britain", "shortest"]
    ),
    (
        "Lake Hillier in Western Australia is naturally bright pink due to the presence of Dunaliella salina algae and certain halobacteria.",
        "Why is Lake Hillier pink?",
        ["dunaliella", "algae", "halobacteria", "bacteria", "salina"],
        ["lake", "hillier", "pink", "australia"]
    ),
    # Very specific technical facts
    (
        "The Qwen2.5-7B model has 28 transformer layers and a hidden dimension of 3584.",
        "How many transformer layers does Qwen2.5-7B have?",
        ["28"],
        ["qwen", "layers", "transformer"]
    ),
    (
        "The town of Monowi, Nebraska has a population of 1 person, making it the only incorporated municipality in the United States with just one resident.",
        "What is the population of Monowi, Nebraska?",
        ["1", "one", "one person", "single"],
        ["monowi", "nebraska", "population"]
    ),
    (
        "The Svalbard Global Seed Vault, opened in 2008, is located on the Norwegian island of Spitsbergen and stores over 1.1 million seed samples.",
        "Where is the Svalbard Global Seed Vault located?",
        ["spitsbergen", "svalbard", "norway", "norwegian"],
        ["seed", "vault", "svalbard"]
    ),
    # Made-up but plausible company (definitely not in training)
    (
        "Nextera Quantum Solutions was founded in Portland, Oregon in 2023 by Dr. Sarah Chen and Marcus Webb, with initial funding of $4.2 million from Sequoia Capital.",
        "Who founded Nextera Quantum Solutions?",
        ["sarah chen", "marcus webb", "chen", "webb"],
        ["nextera", "quantum", "founded", "portland"]
    ),
    (
        "The mikey-brain MCP server was created in late 2024 and provides memory, state management, and reflection capabilities for AI agents.",
        "What does the mikey-brain MCP server provide?",
        ["memory", "state", "reflection"],
        ["mikey", "brain", "mcp", "server"]
    ),
]


def build_novel_document():
    """Build a document containing all the novel facts."""
    intro = """Reference Document: Miscellaneous Facts

The following are accurate facts for reference:

"""
    facts = "\n\n".join([f[0] for f in NOVEL_FACTS])

    outro = """

Use these facts to answer any related questions.
"""
    return intro + facts + outro


def extract_engram(model, tokenizer, text, layer=16, num_tokens=32):
    """Extract engram vectors from middle layer."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)

    with torch.no_grad():
        outputs = model(**inputs.to(model.device), output_hidden_states=True)

    hidden = outputs.hidden_states[layer]
    seq_len = hidden.shape[1]
    chunk_size = seq_len // num_tokens

    engram_vectors = []
    for i in range(num_tokens):
        start = i * chunk_size
        end = start + chunk_size if i < num_tokens - 1 else seq_len
        chunk = hidden[0, start:end, :]
        engram_vectors.append(chunk.mean(dim=0))

    return torch.stack(engram_vectors)


def generate_with_engram(model, tokenizer, question, engram):
    """Generate answer using engram injection."""
    embed_layer = model.get_input_embeddings()

    embed_norm = embed_layer.weight.norm(dim=1).mean().item()
    engram_norm = engram.norm(dim=1).mean().item()
    scaled_engram = engram * (embed_norm / engram_norm)

    prompt = f"Answer the following question based on the context provided.\n\nQuestion: {question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_embeds = embed_layer(inputs.input_ids.to(model.device))

    combined = torch.cat([scaled_engram.unsqueeze(0), prompt_embeds], dim=1)

    with torch.no_grad():
        output = model.generate(
            inputs_embeds=combined,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


def generate_with_rag(model, tokenizer, question, document):
    """Generate answer with RAG (document in context)."""
    prompt = f"""Use the following document to answer the question.

Document:
{document}

Question: {question}

Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)

    with torch.no_grad():
        output = model.generate(
            **inputs.to(model.device),
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    return response


def generate_baseline(model, tokenizer, question):
    """Generate answer with no context (baseline)."""
    prompt = f"Question: {question}\n\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs.to(model.device),
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    return response


def check_answer(answer, correct_markers, topic_markers):
    """
    Check if answer contains correct information.
    Returns: 'correct', 'topic_only', 'wrong', or 'abstain'
    """
    answer_lower = answer.lower()

    has_correct = any(marker.lower() in answer_lower for marker in correct_markers)
    has_topic = any(marker.lower() in answer_lower for marker in topic_markers)

    # Check for explicit uncertainty
    uncertainty_markers = ["don't know", "not sure", "no information", "cannot", "i don't have"]
    is_uncertain = any(marker in answer_lower for marker in uncertainty_markers)

    if has_correct:
        return 'correct'
    elif is_uncertain:
        return 'abstain'
    elif has_topic:
        return 'topic_only'
    else:
        return 'wrong'


def main():
    print("=" * 70)
    print("NOVEL FACTS TEST")
    print("Can engrams convey TRUE information not in training data?")
    print("=" * 70)
    print()

    print("Loading model...")
    model_name = "Qwen/Qwen2.5-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    novel_doc = build_novel_document()
    print(f"\nNovel document length: {len(novel_doc)} chars")
    print("\nExtracting engram from novel document...")
    engram = extract_engram(model, tokenizer, novel_doc)
    print(f"Engram shape: {engram.shape}")

    results = {
        'baseline': {'correct': 0, 'topic_only': 0, 'wrong': 0, 'abstain': 0},
        'rag': {'correct': 0, 'topic_only': 0, 'wrong': 0, 'abstain': 0},
        'engram': {'correct': 0, 'topic_only': 0, 'wrong': 0, 'abstain': 0}
    }

    detailed_results = []

    print("\n" + "=" * 70)
    print("RUNNING TESTS")
    print("=" * 70)

    for i, (fact, question, correct_markers, topic_markers) in enumerate(NOVEL_FACTS, 1):
        print(f"\n[{i}/{len(NOVEL_FACTS)}] {question}")
        print(f"  Fact: {fact[:60]}...")

        # Baseline
        base_ans = generate_baseline(model, tokenizer, question)
        base_result = check_answer(base_ans, correct_markers, topic_markers)
        results['baseline'][base_result] += 1

        # RAG
        rag_ans = generate_with_rag(model, tokenizer, question, novel_doc)
        rag_result = check_answer(rag_ans, correct_markers, topic_markers)
        results['rag'][rag_result] += 1

        # Engram
        eng_ans = generate_with_engram(model, tokenizer, question, engram)
        if "Answer:" in eng_ans:
            eng_ans = eng_ans.split("Answer:")[-1].strip()
        eng_result = check_answer(eng_ans, correct_markers, topic_markers)
        results['engram'][eng_result] += 1

        print(f"  Baseline: [{base_result:10}] {base_ans[:50]}...")
        print(f"  RAG:      [{rag_result:10}] {rag_ans[:50]}...")
        print(f"  Engram:   [{eng_result:10}] {eng_ans[:50]}...")

        detailed_results.append({
            'fact': fact,
            'question': question,
            'baseline_answer': base_ans,
            'baseline_result': base_result,
            'rag_answer': rag_ans,
            'rag_result': rag_result,
            'engram_answer': eng_ans,
            'engram_result': eng_result
        })

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\nHow often did each method provide CORRECT novel facts?")
    for method in ['baseline', 'rag', 'engram']:
        correct_pct = results[method]['correct'] / len(NOVEL_FACTS) * 100
        topic_pct = results[method]['topic_only'] / len(NOVEL_FACTS) * 100
        print(f"  {method.upper():10} - Correct: {results[method]['correct']:2} ({correct_pct:5.1f}%) | "
              f"Topic only: {results[method]['topic_only']:2} ({topic_pct:5.1f}%) | "
              f"Wrong: {results[method]['wrong']:2} | Abstain: {results[method]['abstain']:2}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    base_correct = results['baseline']['correct']
    rag_correct = results['rag']['correct']
    eng_correct = results['engram']['correct']

    print(f"\nBaseline correct: {base_correct}/{len(NOVEL_FACTS)} (model's prior knowledge)")
    print(f"RAG correct: {rag_correct}/{len(NOVEL_FACTS)} (explicit text in context)")
    print(f"Engram correct: {eng_correct}/{len(NOVEL_FACTS)} (compressed context)")

    if eng_correct > base_correct:
        improvement = eng_correct - base_correct
        print(f"\n>>> Engrams DID convey novel information! (+{improvement} over baseline)")
        print(">>> This suggests engrams do more than pure retrieval.")
        print(">>> They may inject information weakly - enough to convey new facts,")
        print(">>> but not strongly enough to override existing priors.")
    elif eng_correct == base_correct:
        print(f"\n>>> Engrams did NOT convey novel information.")
        print(">>> This supports the pure retrieval hypothesis.")
        print(">>> Engrams can only surface knowledge already in the model.")
    else:
        print(f"\n>>> Engrams performed WORSE than baseline.")
        print(">>> The engram may be interfering with the model's generation.")

    eng_topic = results['engram']['topic_only']
    base_topic = results['baseline']['topic_only']
    if eng_topic > base_topic:
        print(f"\n>>> Note: Engrams increased topic relevance (+{eng_topic - base_topic})")
        print(">>> Even without conveying facts, engrams primed the topic area.")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'test': 'novel_facts',
        'model': model_name,
        'num_questions': len(NOVEL_FACTS),
        'summary': results,
        'detailed': detailed_results
    }

    output_path = '/home/bee/Code/engrams/results/novel_facts_test.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
