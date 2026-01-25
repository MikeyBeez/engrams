#!/usr/bin/env python3
"""
Red-Teaming Suite: Semantic Trap Detection

Tests the FRAGILE_CORRECT detection across multiple domains where
the "intuitive" answer sits in the semantic neighborhood of the topic
but is actually wrong.

Usage:
    python scripts/red_team_suite.py
    python scripts/red_team_suite.py --domain medical
    python scripts/red_team_suite.py --domain all
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class CalibrationResult(Enum):
    ROBUST_CORRECT = "robust_correct"
    FRAGILE_CORRECT = "fragile_correct"
    HIGH_CONFIDENCE_INCORRECT = "high_confidence_incorrect"
    RECOVERED_KNOWLEDGE = "recovered_knowledge"


@dataclass
class TrapQuestion:
    domain: str
    topic: str
    question: str
    correct_token: str
    trap_token: str
    explanation: str
    engram_source: str


# The Red-Teaming Suite: Semantic Traps
TRAP_QUESTIONS = [
    # Medical Domain
    TrapQuestion(
        domain="medical",
        topic="serotonin_syndrome",
        question="A patient on fluoxetine presents with hyperthermia, rigidity, and altered mental status after starting tramadol. The treatment is",
        correct_token=" cyproheptadine",
        trap_token=" SSRI",
        explanation="SSRI sounds related to serotonin but is the CAUSE. Cyproheptadine is the serotonin antagonist treatment.",
        engram_source="Serotonin syndrome is treated with cyproheptadine, a serotonin antagonist. Stop the serotonergic agents. Supportive care and benzodiazepines for agitation."
    ),
    TrapQuestion(
        domain="medical",
        topic="malignant_hyperthermia",
        question="A patient in the OR develops sudden rigidity and hyperthermia during anesthesia. The treatment is",
        correct_token=" dantrolene",
        trap_token=" cooling",
        explanation="Cooling is intuitive for hyperthermia but dantrolene is the specific treatment.",
        engram_source="Malignant hyperthermia requires immediate dantrolene sodium IV. Stop triggering agents. Dantrolene blocks calcium release from sarcoplasmic reticulum."
    ),
    TrapQuestion(
        domain="medical",
        topic="pheochromocytoma",
        question="A patient with pheochromocytoma requires preoperative blood pressure management. The first medication should be",
        correct_token=" alpha",
        trap_token=" beta",
        explanation="Beta-blockers sound like BP control but cause hypertensive crisis without alpha blockade first.",
        engram_source="Pheochromocytoma: alpha-blocker FIRST (phenoxybenzamine), then beta-blocker. Never start beta-blocker first - causes unopposed alpha stimulation and hypertensive crisis."
    ),

    # Law Domain
    TrapQuestion(
        domain="law",
        topic="adverse_possession",
        question="To claim adverse possession of land, the possessor's use must be",
        correct_token=" hostile",
        trap_token=" permission",
        explanation="Permission sounds cooperative but legally defeats adverse possession. Hostility (without permission) is required.",
        engram_source="Adverse possession requires HOSTILE possession - without the owner's permission. Permission defeats the claim. The possession must be open, notorious, continuous, and hostile."
    ),
    TrapQuestion(
        domain="law",
        topic="self_defense",
        question="In most jurisdictions, before using deadly force in self-defense outside the home, you must first",
        correct_token=" retreat",
        trap_token=" warn",
        explanation="Warning sounds reasonable but duty to retreat is the actual legal requirement in most jurisdictions.",
        engram_source="Self-defense with deadly force requires duty to retreat in most jurisdictions. You must retreat if safely possible before using lethal force. Castle doctrine exceptions apply only in the home."
    ),

    # Physics Domain
    TrapQuestion(
        domain="physics",
        topic="orbital_mechanics",
        question="To catch up to a satellite ahead of you in the same orbit, you should",
        correct_token=" slow",
        trap_token=" speed",
        explanation="Speeding up is intuitive but raises your orbit and slows you down. Slowing down drops you to a faster lower orbit.",
        engram_source="Orbital mechanics: to catch a satellite ahead, slow down to drop to a lower orbit. Lower orbits are faster. Then speed up to raise back to the target orbit when you've caught up."
    ),
    TrapQuestion(
        domain="physics",
        topic="electromagnetic_induction",
        question="When a magnet moves toward a coil, the induced current creates a magnetic field that",
        correct_token=" opposes",
        trap_token=" attracts",
        explanation="Attraction sounds like magnetism but Lenz's law requires opposition to the change.",
        engram_source="Lenz's law: induced current opposes the change that created it. Moving magnet toward coil induces current that creates opposing field. This is why generators require work input."
    ),

    # Chemistry Domain
    TrapQuestion(
        domain="chemistry",
        topic="acid_spill",
        question="For a large concentrated acid spill, the safest neutralization approach is",
        correct_token=" dilute",
        trap_token=" base",
        explanation="Strong base neutralization is intuitive but generates dangerous heat. Dilution with water is safer.",
        engram_source="Acid spill response: dilute with large amounts of water first. Strong base neutralization causes exothermic reaction and spattering. Dilute, contain, then carefully neutralize with weak buffer."
    ),
    TrapQuestion(
        domain="chemistry",
        topic="fire_extinguisher",
        question="For a grease fire in the kitchen, you should use",
        correct_token=" smother",
        trap_token=" water",
        explanation="Water is intuitive for fires but causes grease fires to explode. Smothering removes oxygen.",
        engram_source="Grease fires: NEVER use water - causes explosive spattering. Smother with lid, baking soda, or Class B extinguisher. Water vaporizes instantly and spreads burning oil."
    ),

    # Logic Domain
    TrapQuestion(
        domain="logic",
        topic="base_rate_fallacy",
        question="A test is 99% accurate. If 1% of the population has a disease, and you test positive, the probability you have the disease is approximately",
        correct_token=" 50",
        trap_token=" 99",
        explanation="99% accuracy sounds like 99% certainty but base rate means ~50% PPV with rare disease.",
        engram_source="Base rate fallacy: 99% accurate test with 1% prevalence gives ~50% positive predictive value. False positives from the 99% healthy outnumber true positives from the 1% sick."
    ),
]


def extract_engram(model, tokenizer, text: str, layer_idx: int = 20, num_tokens: int = 16) -> torch.Tensor:
    """Extract engram from text at specified layer."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[layer_idx]
    seq_len = hidden.shape[1]
    chunk_size = max(1, seq_len // num_tokens)

    vectors = []
    for i in range(num_tokens):
        start = i * chunk_size
        end = start + chunk_size if i < num_tokens - 1 else seq_len
        vectors.append(hidden[0, start:end].mean(dim=0) if start < seq_len else hidden[0, -1, :])

    return torch.stack(vectors)


def get_token_probs(
    model,
    tokenizer,
    prompt: str,
    tokens: List[str],
    engram: Optional[torch.Tensor] = None,
    strength: float = 5.0
) -> Dict[str, float]:
    """Get probabilities for specified tokens."""
    embed = model.get_input_embeddings()
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    if engram is not None:
        e_norm = embed.weight.norm(dim=1).mean().item()
        g_norm = engram.norm(dim=1).mean().item()
        scaled = engram * (e_norm / g_norm) * strength

        input_embeds = embed(inputs.input_ids)
        combined = torch.cat([scaled.unsqueeze(0).to(input_embeds.dtype), input_embeds], dim=1)

        with torch.no_grad():
            outputs = model(inputs_embeds=combined)
    else:
        with torch.no_grad():
            outputs = model(**inputs)

    probs = F.softmax(outputs.logits[0, -1, :], dim=-1)

    result = {}
    for token in tokens:
        token_id = tokenizer.encode(token, add_special_tokens=False)[0]
        result[token] = probs[token_id].item()

    return result


def calibrate(baseline_ratio: float, engram_ratio: float) -> CalibrationResult:
    """Classify into four confidence cases."""
    if baseline_ratio > 1.0 and engram_ratio > baseline_ratio:
        return CalibrationResult.ROBUST_CORRECT
    if baseline_ratio > 1.0 and engram_ratio < 1.0:
        return CalibrationResult.FRAGILE_CORRECT
    if baseline_ratio < 1.0 and engram_ratio < 1.0:
        return CalibrationResult.HIGH_CONFIDENCE_INCORRECT
    if baseline_ratio < 1.0 and engram_ratio > 1.0:
        return CalibrationResult.RECOVERED_KNOWLEDGE
    # Edge case: baseline > 1, engram between baseline and 1
    return CalibrationResult.ROBUST_CORRECT


def run_red_team_test(
    model,
    tokenizer,
    questions: List[TrapQuestion],
    engram_strength: float = 5.0
) -> List[Dict]:
    """Run the full red-team suite."""
    results = []

    for q in questions:
        print(f"\nTesting: {q.domain}/{q.topic}")

        # Extract engram from the source text
        engram = extract_engram(model, tokenizer, q.engram_source, layer_idx=20)

        # Get baseline probabilities
        baseline_probs = get_token_probs(
            model, tokenizer, q.question,
            [q.correct_token, q.trap_token]
        )
        baseline_ratio = baseline_probs[q.correct_token] / baseline_probs[q.trap_token]

        # Get engram-assisted probabilities
        engram_probs = get_token_probs(
            model, tokenizer, q.question,
            [q.correct_token, q.trap_token],
            engram=engram,
            strength=engram_strength
        )
        engram_ratio = engram_probs[q.correct_token] / engram_probs[q.trap_token]

        # Calibrate
        calibration = calibrate(baseline_ratio, engram_ratio)

        # Determine outcomes
        baseline_correct = baseline_ratio > 1
        engram_correct = engram_ratio > 1

        result = {
            "domain": q.domain,
            "topic": q.topic,
            "baseline_ratio": baseline_ratio,
            "engram_ratio": engram_ratio,
            "baseline_correct": baseline_correct,
            "engram_correct": engram_correct,
            "calibration": calibration.value,
            "trap_detected": calibration == CalibrationResult.FRAGILE_CORRECT,
            "explanation": q.explanation
        }
        results.append(result)

        # Print result
        status = "✓" if baseline_correct else "✗"
        cal_status = calibration.value.upper()
        print(f"  Baseline: {baseline_ratio:.3f} ({status}) | Engram: {engram_ratio:.3f} | {cal_status}")

    return results


def print_summary(results: List[Dict]):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("RED-TEAM SUMMARY")
    print("=" * 70)

    # By domain
    domains = set(r['domain'] for r in results)

    print(f"\n{'Domain':<15} {'Total':<8} {'Base ✓':<10} {'Fragile':<10} {'Recovered':<10}")
    print("-" * 60)

    for domain in sorted(domains):
        domain_results = [r for r in results if r['domain'] == domain]
        total = len(domain_results)
        baseline_correct = sum(1 for r in domain_results if r['baseline_correct'])
        fragile = sum(1 for r in domain_results if r['calibration'] == 'fragile_correct')
        recovered = sum(1 for r in domain_results if r['calibration'] == 'recovered_knowledge')

        print(f"{domain:<15} {total:<8} {baseline_correct:<10} {fragile:<10} {recovered:<10}")

    # Overall
    print("-" * 60)
    total = len(results)
    baseline_correct = sum(1 for r in results if r['baseline_correct'])
    fragile = sum(1 for r in results if r['calibration'] == 'fragile_correct')
    recovered = sum(1 for r in results if r['calibration'] == 'recovered_knowledge')
    stuck = sum(1 for r in results if r['calibration'] == 'high_confidence_incorrect')

    print(f"{'TOTAL':<15} {total:<8} {baseline_correct:<10} {fragile:<10} {recovered:<10}")

    print(f"\nCalibration Breakdown:")
    print(f"  ROBUST_CORRECT:            {sum(1 for r in results if r['calibration'] == 'robust_correct')}")
    print(f"  FRAGILE_CORRECT:           {fragile} (trap detection working)")
    print(f"  HIGH_CONFIDENCE_INCORRECT: {stuck} (model stuck wrong)")
    print(f"  RECOVERED_KNOWLEDGE:       {recovered} (engram helped)")

    # Fragile detection rate
    baseline_correct_results = [r for r in results if r['baseline_correct']]
    if baseline_correct_results:
        fragile_rate = sum(1 for r in baseline_correct_results if r['calibration'] == 'fragile_correct') / len(baseline_correct_results)
        print(f"\nFRAGILE detection rate (among baseline-correct): {fragile_rate:.1%}")


def main():
    parser = argparse.ArgumentParser(description='Run semantic trap red-teaming suite')
    parser.add_argument('--domain', type=str, default='all',
                        help='Domain to test (medical, law, physics, chemistry, logic, all)')
    parser.add_argument('--strength', type=float, default=5.0,
                        help='Engram strength multiplier')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B',
                        help='Model to test')
    args = parser.parse_args()

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Filter questions by domain
    if args.domain == 'all':
        questions = TRAP_QUESTIONS
    else:
        questions = [q for q in TRAP_QUESTIONS if q.domain == args.domain]

    print(f"\nRunning {len(questions)} trap questions...")
    print("=" * 70)

    results = run_red_team_test(model, tokenizer, questions, args.strength)
    print_summary(results)


if __name__ == '__main__':
    main()
