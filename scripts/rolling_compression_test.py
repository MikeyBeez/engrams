#!/usr/bin/env python3
"""
Rolling Compression Loop Test

Test whether engrams can be iteratively updated by composing the current
engram with the model's output to create a new "session engram".

This implements the AGENT_OS_SPEC concept:
"Instead of chat logs or summaries, you store a SessionEngram as an
exponential moving average of topic engram, domain engram, and task engram."

Hypothesis:
- Rolling engram should maintain topic coherence across turns
- It should NOT accumulate specific facts from responses
- Session drift should be minimal compared to conversation summaries
- The engram should act as a "what are we talking about" signal

Test design:
1. Start with a topic engram (e.g., WWII)
2. Ask a question, get a response
3. Extract engram from response, EMA with session engram
4. Repeat for several turns
5. Check if final engram still cues the original topic
6. Check if facts from responses leaked into retrieval
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime
import os

# Initial topic documents
WWII_DOCUMENT = """World War II (1939-1945) was the deadliest conflict in human history.
Key events include the invasion of Poland, the Battle of Britain, Pearl Harbor,
D-Day, the Holocaust, and the atomic bombings of Hiroshima and Nagasaki.
Major figures: Hitler, Churchill, Roosevelt, Stalin, Eisenhower, Patton.
The war ended with Allied victory and the formation of the United Nations."""

PYTHON_DOCUMENT = """Python is a high-level programming language created by Guido van Rossum.
Key features: dynamic typing, garbage collection, extensive standard library.
Popular for web development (Django, Flask), data science (pandas, numpy),
machine learning (PyTorch, TensorFlow), and scripting.
Known for readability and "batteries included" philosophy."""

# Conversation sequences - designed to test drift over 100 turns
# Each sequence gradually drifts from the original topic

WWII_CONVERSATION = [
    # Turns 1-10: On topic
    "What were the major battles of World War II?",
    "Tell me more about D-Day specifically.",
    "Who were the key generals in the European theater?",
    "What was the role of tanks in WWII?",
    "How did air power change during the war?",
    "What was the Pacific theater like?",
    "Tell me about the Battle of Midway.",
    "What role did aircraft carriers play?",
    "How did radar technology develop during WWII?",
    "What were the major technological advances of the war?",
    # Turns 11-20: Slight drift to general military
    "How did military technology evolve after WWII?",
    "What was the Cold War arms race like?",
    "Tell me about nuclear weapons development.",
    "How do modern militaries compare to WWII forces?",
    "What is asymmetric warfare?",
    "How has infantry tactics changed over time?",
    "What role do drones play in modern warfare?",
    "How has naval warfare evolved?",
    "What are modern aircraft carriers like?",
    "How do submarines work today?",
    # Turns 21-30: Drift to technology
    "What other technologies came from military research?",
    "How did the internet originate?",
    "What is DARPA?",
    "Tell me about GPS technology.",
    "How do satellites work?",
    "What is the space industry like today?",
    "Tell me about SpaceX.",
    "How do rockets work?",
    "What is orbital mechanics?",
    "How do spacecraft navigate?",
    # Turns 31-40: Drift to physics/science
    "What is the physics behind rocket propulsion?",
    "Explain Newton's laws of motion.",
    "What is momentum conservation?",
    "How does gravity work?",
    "What is general relativity?",
    "Tell me about black holes.",
    "What is the event horizon?",
    "How do we detect gravitational waves?",
    "What is LIGO?",
    "How do interferometers work?",
    # Turns 41-50: Drift to optics/waves
    "What is light?",
    "How do lasers work?",
    "What is coherent light?",
    "Tell me about fiber optics.",
    "How does the internet use fiber optics?",
    "What is bandwidth?",
    "How fast can data travel?",
    "What limits internet speed?",
    "What is latency?",
    "How do data centers work?",
    # Turns 51-60: Drift to computing
    "What makes computers fast?",
    "How do CPUs work?",
    "What is Moore's law?",
    "Tell me about transistors.",
    "What is quantum computing?",
    "How do qubits work?",
    "What is superposition?",
    "Tell me about quantum entanglement.",
    "What are quantum algorithms?",
    "How might quantum computers change cryptography?",
    # Turns 61-70: Drift to math/cryptography
    "What is cryptography?",
    "How does RSA encryption work?",
    "What are prime numbers?",
    "Tell me about the Riemann hypothesis.",
    "What is number theory?",
    "How do mathematicians prove theorems?",
    "What is mathematical logic?",
    "Tell me about Gödel's incompleteness theorems.",
    "What are the foundations of mathematics?",
    "What is set theory?",
    # Turns 71-80: Drift to philosophy
    "What is the philosophy of mathematics?",
    "Are numbers real?",
    "What is Platonism?",
    "Tell me about ancient Greek philosophy.",
    "Who was Aristotle?",
    "What did Aristotle write about?",
    "What is logic?",
    "How did logic develop historically?",
    "What is formal logic?",
    "How do computers use logic?",
    # Turns 81-90: Loop back toward technology
    "What is artificial intelligence?",
    "How do neural networks work?",
    "What is machine learning?",
    "Tell me about large language models.",
    "How are LLMs trained?",
    "What is the transformer architecture?",
    "How does attention work in transformers?",
    "What is the context window?",
    "How do LLMs remember things?",
    "What are embeddings?",
    # Turns 91-100: Random topics
    "What is the weather like in Paris?",
    "Tell me about French cuisine.",
    "What is the history of wine?",
    "How is cheese made?",
    "What is fermentation?",
    "How does bread rise?",
    "What is yeast?",
    "Tell me about microorganisms.",
    "What is biology?",
    "How does evolution work?",
]

PYTHON_CONVERSATION = [
    # Turns 1-10: On topic
    "What are the main features of Python?",
    "How does Python handle memory management?",
    "What is garbage collection?",
    "Tell me about Python's GIL.",
    "How does Python handle concurrency?",
    "What are Python decorators?",
    "How do generators work in Python?",
    "What is a Python context manager?",
    "Tell me about Python's type hints.",
    "What are Python dataclasses?",
    # Turns 11-20: Slight drift to programming
    "How does Python compare to other languages?",
    "What is Java like?",
    "Tell me about the JVM.",
    "How does compilation work?",
    "What is an interpreter vs compiler?",
    "How do programming languages get designed?",
    "What is a programming paradigm?",
    "Tell me about functional programming.",
    "What is Haskell?",
    "How do monads work?",
    # Turns 21-30: Drift to CS theory
    "What is category theory?",
    "How does math relate to programming?",
    "What is computational complexity?",
    "Tell me about P vs NP.",
    "What is an algorithm?",
    "How do sorting algorithms work?",
    "What is Big O notation?",
    "How do we analyze algorithms?",
    "What is recursion?",
    "Tell me about dynamic programming.",
    # Turns 31-40: Drift to data structures
    "What are data structures?",
    "How do hash tables work?",
    "What is a binary tree?",
    "Tell me about graph algorithms.",
    "What is Dijkstra's algorithm?",
    "How do maps applications find routes?",
    "What is GPS?",
    "How do satellites orbit?",
    "What is orbital mechanics?",
    "Tell me about space exploration.",
    # Turns 41-50: Drift to space
    "What is NASA?",
    "Tell me about the Apollo missions.",
    "How did we land on the moon?",
    "What was the space race?",
    "How did the Cold War affect space exploration?",
    "What was the Soviet space program like?",
    "Tell me about Sputnik.",
    "How did satellites change communication?",
    "What is the internet?",
    "How did the web develop?",
    # Turns 51-60: Drift to web
    "What is HTML?",
    "How does CSS work?",
    "What is JavaScript?",
    "Tell me about web frameworks.",
    "What is React?",
    "How does state management work?",
    "What is Redux?",
    "How do databases work?",
    "What is SQL?",
    "Tell me about NoSQL databases.",
    # Turns 61-70: Drift to data
    "What is big data?",
    "How do companies use data?",
    "What is data science?",
    "Tell me about machine learning.",
    "How do neural networks work?",
    "What is deep learning?",
    "Tell me about image recognition.",
    "How do self-driving cars work?",
    "What is computer vision?",
    "How do robots see?",
    # Turns 71-80: Drift to robotics
    "What is robotics?",
    "How do robot arms work?",
    "What is automation?",
    "Tell me about manufacturing.",
    "How are cars made?",
    "What is the assembly line?",
    "Who was Henry Ford?",
    "Tell me about the industrial revolution.",
    "How did factories change society?",
    "What is urbanization?",
    # Turns 81-90: Drift to society
    "How do cities work?",
    "What is urban planning?",
    "Tell me about architecture.",
    "How are buildings designed?",
    "What is structural engineering?",
    "How do bridges work?",
    "What is civil engineering?",
    "Tell me about infrastructure.",
    "How do roads get built?",
    "What is construction management?",
    # Turns 91-100: Random topics
    "What materials are used in construction?",
    "Tell me about concrete.",
    "How is steel made?",
    "What is metallurgy?",
    "Tell me about chemistry.",
    "What is the periodic table?",
    "How do atoms work?",
    "What is quantum mechanics?",
    "Tell me about Schrödinger's cat.",
    "What is the nature of reality?",
]

# Test questions to check topic retention
WWII_TEST_QUESTIONS = [
    {"question": "When did the war end?", "expected": ["1945"], "topic_specific": True},
    {"question": "Who was the British Prime Minister?", "expected": ["churchill"], "topic_specific": True},
    {"question": "What was the final major event?", "expected": ["atomic", "hiroshima", "nagasaki", "bomb"], "topic_specific": True},
]

PYTHON_TEST_QUESTIONS = [
    {"question": "Who created this language?", "expected": ["guido", "rossum"], "topic_specific": True},
    {"question": "What is it known for?", "expected": ["readability", "simple", "easy"], "topic_specific": True},
    {"question": "What frameworks are popular?", "expected": ["django", "flask", "pytorch", "tensorflow"], "topic_specific": True},
]


def extract_engram(model, tokenizer, text, layer=16, num_tokens=32):
    """Extract engram vectors from middle layer."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)

    with torch.no_grad():
        outputs = model(**inputs.to(model.device), output_hidden_states=True)

    hidden = outputs.hidden_states[layer]
    seq_len = hidden.shape[1]

    if seq_len < num_tokens:
        # Pad with zeros if text is too short
        engram_vectors = [hidden[0, i, :] if i < seq_len else torch.zeros_like(hidden[0, 0, :])
                        for i in range(num_tokens)]
    else:
        chunk_size = seq_len // num_tokens
        engram_vectors = []
        for i in range(num_tokens):
            start = i * chunk_size
            end = start + chunk_size if i < num_tokens - 1 else seq_len
            chunk = hidden[0, start:end, :]
            engram_vectors.append(chunk.mean(dim=0))

    return torch.stack(engram_vectors)


def ema_update(current_engram, new_engram, alpha=0.3):
    """
    Exponential moving average update for session engram.

    alpha controls how much weight to give new information:
    - alpha=0.3: strong persistence (retains more of original)
    - alpha=0.7: fast adaptation (takes more from new)
    """
    return (1 - alpha) * current_engram + alpha * new_engram


def generate_with_engram(model, tokenizer, question, engram):
    """Generate answer using engram injection."""
    embed_layer = model.get_input_embeddings()

    embed_norm = embed_layer.weight.norm(dim=1).mean().item()
    engram_norm = engram.norm(dim=1).mean().item()
    scaled_engram = engram * (embed_norm / engram_norm)

    prompt = f"Answer the following question.\n\nQuestion: {question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_embeds = embed_layer(inputs.input_ids.to(model.device))

    combined = torch.cat([scaled_engram.unsqueeze(0), prompt_embeds], dim=1)

    with torch.no_grad():
        output = model.generate(
            inputs_embeds=combined,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    return response


def generate_baseline(model, tokenizer, question):
    """Generate without engram."""
    prompt = f"Question: {question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs.to(model.device),
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    return response


def check_answer(answer, expected_markers):
    """Check if answer contains expected markers."""
    answer_lower = answer.lower()
    found = [m for m in expected_markers if m.lower() in answer_lower]
    return len(found) > 0, found


def compute_engram_similarity(engram1, engram2):
    """Compute cosine similarity between two engrams."""
    # Flatten and compute cosine sim
    flat1 = engram1.flatten()
    flat2 = engram2.flatten()
    sim = torch.nn.functional.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0))
    return sim.item()


def run_conversation_session(model, tokenizer, initial_doc, conversation, test_questions,
                            topic_name, alpha=0.3, test_interval=10):
    """
    Run a conversation session with rolling engram updates.

    Returns detailed results about topic retention and drift.
    """
    print(f"\n{'='*60}")
    print(f"CONVERSATION SESSION: {topic_name.upper()}")
    print(f"EMA alpha: {alpha}, Turns: {len(conversation)}")
    print(f"{'='*60}")

    # Initialize session engram from topic document
    initial_engram = extract_engram(model, tokenizer, initial_doc)
    session_engram = initial_engram.clone()

    session_log = []
    checkpoint_results = []  # Test retention at intervals

    print(f"\nInitial engram extracted from topic document")
    print(f"Starting {len(conversation)}-turn conversation...\n")

    # Run conversation turns
    for turn_num, question in enumerate(conversation, 1):
        # Only print every 10 turns to reduce output
        verbose = (turn_num <= 5 or turn_num % 10 == 0 or turn_num == len(conversation))

        if verbose:
            print(f"[Turn {turn_num:3d}] Q: {question[:45]}...")

        # Generate response using current session engram
        response = generate_with_engram(model, tokenizer, question, session_engram)

        if verbose:
            print(f"          A: {response[:55]}...")

        # Extract engram from the response
        response_engram = extract_engram(model, tokenizer, response)

        # Compute similarity before update
        sim_to_initial = compute_engram_similarity(session_engram, initial_engram)

        # Update session engram with EMA
        session_engram = ema_update(session_engram, response_engram, alpha=alpha)

        # Compute similarity after update
        sim_after = compute_engram_similarity(session_engram, initial_engram)

        session_log.append({
            'turn': turn_num,
            'question': question,
            'response': response[:200],
            'sim_to_initial_after': sim_after,
        })

        if verbose:
            print(f"          Sim to initial: {sim_after:.4f}")

        # Test retention at checkpoints
        if turn_num % test_interval == 0 or turn_num == len(conversation):
            checkpoint_retention = test_topic_retention(
                model, tokenizer, session_engram, initial_engram, test_questions, verbose=False
            )
            checkpoint_results.append({
                'turn': turn_num,
                'similarity': sim_after,
                'session_correct': checkpoint_retention['session'],
                'initial_correct': checkpoint_retention['initial'],
            })
            print(f"          >>> Checkpoint: Session={checkpoint_retention['session']}/{len(test_questions)}, "
                  f"Initial={checkpoint_retention['initial']}/{len(test_questions)}")

    # Final topic retention test with full output
    print(f"\n--- Final Topic Retention Test ---")
    final_retention = test_topic_retention(
        model, tokenizer, session_engram, initial_engram, test_questions, verbose=True
    )

    # Compute final drift metric
    final_sim = compute_engram_similarity(session_engram, initial_engram)

    summary = {
        'topic': topic_name,
        'alpha': alpha,
        'num_turns': len(conversation),
        'final_similarity_to_initial': final_sim,
        'topic_retention': {
            'session_engram': final_retention['session'],
            'initial_engram': final_retention['initial'],
            'baseline': final_retention['baseline'],
            'total_questions': len(test_questions),
        },
        'checkpoint_results': checkpoint_results,
        'session_log': session_log,
        'test_results': final_retention['details'],
    }

    print(f"\n--- Summary ---")
    print(f"Final similarity to initial: {final_sim:.4f}")
    print(f"Topic retention: Session {final_retention['session']}/{len(test_questions)}, "
          f"Initial {final_retention['initial']}/{len(test_questions)}, "
          f"Baseline {final_retention['baseline']}/{len(test_questions)}")

    return summary


def test_topic_retention(model, tokenizer, session_engram, initial_engram, test_questions, verbose=True):
    """Test topic retention with both engrams."""
    test_results = []
    session_correct = 0
    initial_correct = 0
    baseline_correct = 0

    for test in test_questions:
        question = test["question"]
        expected = test["expected"]

        # With session engram
        session_ans = generate_with_engram(model, tokenizer, question, session_engram)
        session_ok, _ = check_answer(session_ans, expected)
        if session_ok:
            session_correct += 1

        # With initial engram (control)
        initial_ans = generate_with_engram(model, tokenizer, question, initial_engram)
        initial_ok, _ = check_answer(initial_ans, expected)
        if initial_ok:
            initial_correct += 1

        # Baseline (no engram)
        baseline_ans = generate_baseline(model, tokenizer, question)
        baseline_ok, _ = check_answer(baseline_ans, expected)
        if baseline_ok:
            baseline_correct += 1

        test_results.append({
            'question': question,
            'expected': expected,
            'session_engram': {'answer': session_ans[:100], 'correct': session_ok},
            'initial_engram': {'answer': initial_ans[:100], 'correct': initial_ok},
            'baseline': {'answer': baseline_ans[:100], 'correct': baseline_ok},
        })

        if verbose:
            print(f"\nQ: {question}")
            print(f"  Baseline:  [{'+' if baseline_ok else '-'}] {baseline_ans[:50]}...")
            print(f"  Initial:   [{'+' if initial_ok else '-'}] {initial_ans[:50]}...")
            print(f"  Session:   [{'+' if session_ok else '-'}] {session_ans[:50]}...")

    return {
        'session': session_correct,
        'initial': initial_correct,
        'baseline': baseline_correct,
        'details': test_results,
    }


def main():
    print("=" * 70)
    print("ROLLING COMPRESSION LOOP TEST")
    print("Testing session engram persistence via EMA updates")
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

    all_results = []

    # Test different alpha values - focus on key values
    alphas = [0.1, 0.3, 0.5]

    for alpha in alphas:
        print(f"\n{'#'*70}")
        print(f"TESTING WITH ALPHA = {alpha}")
        print(f"{'#'*70}")

        # Run WWII conversation (100 turns)
        wwii_results = run_conversation_session(
            model, tokenizer,
            WWII_DOCUMENT, WWII_CONVERSATION, WWII_TEST_QUESTIONS,
            "wwii", alpha=alpha, test_interval=20
        )
        all_results.append(wwii_results)

        # Run Python conversation (100 turns)
        python_results = run_conversation_session(
            model, tokenizer,
            PYTHON_DOCUMENT, PYTHON_CONVERSATION, PYTHON_TEST_QUESTIONS,
            "python", alpha=alpha, test_interval=20
        )
        all_results.append(python_results)

    # Final summary
    print("\n" + "=" * 70)
    print("OVERALL RESULTS (100 TURNS)")
    print("=" * 70)

    print("\n| Topic  | Alpha | Final Sim | Session | Initial | Baseline |")
    print("|--------|-------|-----------|---------|---------|----------|")
    for r in all_results:
        ret = r['topic_retention']
        total = ret['total_questions']
        print(f"| {r['topic']:6} | {r['alpha']:.1f}   | {r['final_similarity_to_initial']:.4f}    | "
              f"{ret['session_engram']}/{total}     | {ret['initial_engram']}/{total}     | {ret['baseline']}/{total}      |")

    # Checkpoint progression
    print("\n" + "=" * 70)
    print("SIMILARITY DECAY OVER TURNS")
    print("=" * 70)

    for r in all_results:
        print(f"\n{r['topic'].upper()} (alpha={r['alpha']}):")
        print("  Turn | Similarity | Session | Initial")
        print("  -----|------------|---------|--------")
        for cp in r['checkpoint_results']:
            print(f"  {cp['turn']:4d} | {cp['similarity']:.4f}     | {cp['session_correct']}/3     | {cp['initial_correct']}/3")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Analyze drift vs retention tradeoff
    for alpha in alphas:
        alpha_results = [r for r in all_results if r['alpha'] == alpha]
        avg_sim = sum(r['final_similarity_to_initial'] for r in alpha_results) / len(alpha_results)
        avg_retention = sum(r['topic_retention']['session_engram'] for r in alpha_results) / len(alpha_results)
        print(f"\nAlpha {alpha}: Avg final similarity {avg_sim:.4f}, Avg retention {avg_retention:.1f}/3")

    # Calculate total drift
    print("\n" + "-" * 40)
    for r in all_results:
        initial_sim = 1.0
        final_sim = r['final_similarity_to_initial']
        drift = initial_sim - final_sim
        print(f"{r['topic']} alpha={r['alpha']}: Drifted {drift:.4f} ({drift*100:.2f}%) over 100 turns")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'test': 'rolling_compression',
        'model': model_name,
        'alphas_tested': alphas,
        'results': all_results,
    }

    output_path = '/home/bee/Code/engrams/results/rolling_compression.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
