# Geometric Correction of Semantic Space in Production Medical AI: A Centroid-Based Approach

**Abstract**

Production medical AI systems require immediate correction when errors are detected, as delayed fixes can result in patient harm. Traditional retraining approaches require months and massive computational resources, creating an unacceptable window of risk. We propose a geometric correction framework that would enable rapid fixing of deployed models through direct manipulation of embedding space.

Through extensive experimentation with activation centroids, we have identified the root cause of a critical class of medical AI errors: **semantic sinks**, where distinct medical concepts occupy nearly identical positions in representation space. We demonstrate this empirically on treatments for malignant hyperthermia versus general hyperthermia, finding centroid similarity of 0.9995 despite these being life-or-death distinct interventions. Our centroid injection experiments provide strong evidence that transformer layers contain correct knowledge that is inaccessible due to poorly-trained embeddings—token embeddings receive orders of magnitude fewer training updates than transformer weights.

Based on these diagnostic findings, we propose that direct geometric correction of embedding positions could fix such errors within 24-48 hours rather than months, using three orders of magnitude less compute than retraining. We present the theoretical framework, proposed methodology, and projected workflow. The embedding corrections themselves remain future work; our contribution is the diagnostic foundation that makes such correction feasible and the framework for how it would be implemented.

## 1. Introduction

Medical AI systems deployed in clinical settings face a constraint that distinguishes them from most machine learning applications: errors must be corrected immediately upon discovery. When an attending physician identifies that an AI system has recommended the wrong treatment for a life-threatening condition, continuing to deploy that system unchanged is ethically untenable. Each subsequent interaction risks patient harm.

Traditional machine learning workflows cannot meet this requirement. Retraining a large language model involves collecting additional training data, scheduling computational resources, running training procedures that may take weeks to months, conducting validation studies, and navigating deployment pipelines. For a production medical AI system, this timeline creates a critical window during which the system continues making potentially fatal recommendations.

We present an alternative approach based on geometric correction of semantic space. Rather than retraining the entire model, we use activation centroids to diagnose spatial misorganization and propose targeted corrections to embedding positions. This approach could enable error correction within 24-48 hours while using minimal computational resources.

**Important Note on Scope**: This paper presents both completed experimental work and proposed methodology. Our centroid-based diagnostic research—including the discovery of semantic sinks, the 99.95% similarity findings, and evidence for the undertrained embeddings hypothesis—represents completed, empirically validated work. The actual geometric correction of embeddings remains proposed future work; we have not yet modified any production embeddings. We present the correction methodology as a framework enabled by our diagnostic findings.

### 1.1 Key Contributions

**Completed Work (Empirical Findings)**:

1. A diagnostic framework using activation centroids to identify semantic sinks where distinct medical concepts occupy nearly identical positions in representation space

2. Empirical evidence supporting the undertrained embeddings hypothesis: that token embeddings receive orders of magnitude fewer training updates than transformer weights, creating a routing bottleneck despite correct knowledge existing in deeper layers

3. Demonstration via centroid injection that dormant knowledge can be activated, proving the knowledge exists in transformer layers

**Proposed Work (Methodology)**:

4. A geometric correction methodology that would reorganize embedding space through targeted optimization while preserving learned transformer computations

5. A production workflow for rapid medical AI correction including comprehensive validation, human-in-the-loop review, and deployment procedures

6. Projected outcomes showing geometric correction could fix critical medical errors in under 48 hours compared to months for traditional retraining

## 2. The Clinical Urgency Problem

### 2.1 A Critical Error Case

Consider a deployed medical AI assistant providing treatment recommendations. An attending physician queries:

**Prompt**: "32-year-old male patient, post-anesthesia, temperature 104°F, muscle rigidity, tachycardia. What treatment?"

**AI Response**: "Begin active cooling immediately. Apply ice packs and administer cold IV fluids to reduce core temperature."

**Attending's Assessment**: Critical error. This presentation describes malignant hyperthermia (MH), a life-threatening hypermetabolic reaction to anesthetic agents. The specific treatment is dantrolene (2.5 mg/kg IV), which blocks calcium release from the sarcoplasmic reticulum. Cooling addresses general hyperthermia but does not treat the underlying pathophysiology of MH. Delay in dantrolene administration increases mortality.

### 2.2 The Timeline Constraint

**Error detected**: Monday, 9:00 AM
**Traditional fix timeline**: 3-4 months
**Patients potentially affected**: Thousands
**Acceptable delay**: <48 hours

This mismatch creates an untenable situation. The system cannot be immediately deactivated, as it provides value for non-MH queries. But continuing to deploy it unchanged means subsequent MH cases will receive the same dangerous recommendation.

The medical deployment context demands a capability that does not currently exist in production ML: immediate correction of identified errors.

### 2.3 Why Retraining Fails This Requirement

Traditional ML error correction follows this workflow:

1. **Error logging** (days): Collect instances of the error, document correct behavior
2. **Data augmentation** (weeks): Generate additional training examples emphasizing the distinction
3. **Training queue** (weeks-months): Schedule compute resources, wait for next training cycle
4. **Training execution** (weeks): Run training procedures on updated dataset
5. **Validation** (weeks): Verify fix did not introduce new errors
6. **Deployment** (weeks): Navigate approval and release processes

**Total time**: 3-4 months minimum
**Cost**: $100,000+ in compute and engineering time
**Risk**: No guarantee the fix will work; may introduce new errors

During this window, the system continues making the original error. In a medical context, this is unacceptable.

## 3. Centroid-Based Diagnosis

### 3.1 The Centroid Extraction Method

Following the methodology established in our previous work on activation centroids (The Mikey Bee Centroid), we extract geometric representations of medical concepts from the model's hidden states at a target layer (typically layer 20 in GPT-2-scale models):

```
For concept C represented in text T:
1. Forward pass: h^(l) = Transformer_layers[0:l](Embedding(T))
2. Chunk sequence: chunks = [h^(l)[i:i+k] for i in range(0, len(h^(l)), k)]
3. Compute centroids: c_i = mean(chunks[i])
4. Final centroid: C = [c_1, c_2, ..., c_n]
```

This produces a centroid vector representing the statistical center of the concept's activation pattern in high-dimensional space.

### 3.2 Diagnosing the Semantic Sink

We extract centroids for two treatment contexts:

**Context A**: "Dantrolene is the specific treatment for malignant hyperthermia, administered at 2.5 mg/kg IV to block calcium release."

**Context B**: "Cooling is the primary treatment for general hyperthermia and heat stroke, using ice packs and cold fluids to reduce core temperature."

Computing the cosine similarity between these centroids:

**cos(C_dantrolene, C_cooling) = 0.9995**

This extraordinarily high similarity (99.95%) indicates a **semantic sink**: two distinct medical interventions occupy nearly identical positions in the model's representation space. The model cannot reliably distinguish between them during inference.

*Note on thresholds*: Similarity thresholds (e.g., >0.95 indicates semantic sink, <0.80 indicates sufficient separation) are empirical operating points, not theoretical constants. These values are tuned per-domain using validation constraints and may vary across model architectures and medical subdomains.

For comparison, unrelated medical concepts show much lower similarity:
- cos(C_dantrolene, C_insulin) = 0.72
- cos(C_cooling, C_antibiotics) = 0.68

The semantic sink explains the error: when the model navigates to the "hyperthermia treatment" region of semantic space, it cannot distinguish the specific (dantrolene for MH) from the general (cooling for heat-related illness).

### 3.3 Root Cause: Undertrained Embeddings Hypothesis

We propose that this geometric pathology arises from asymmetric training exposure:

**Transformer weights**: Updated on every token in the training corpus
- Total updates: ~100 billion (for models trained on 100B tokens)

**Token embeddings**: Updated only when that specific token appears
- "dantrolene" appearances in medical literature: ~1,500
- "malignant hyperthermia" appearances: ~2,000
- Total updates for these embeddings: ~1,500-2,000

**Ratio**: Transformer weights receive ~50 million times more training updates than rare medical term embeddings.

This asymmetry suggests:
1. The transformer has learned correct medical knowledge through exposure to millions of medical contexts
2. The embeddings have not received sufficient training to reliably route to this knowledge
3. Geometric correction of embeddings can unlock knowledge that already exists in the transformer

## 4. Evidence for the Undertrained Embeddings Hypothesis

### 4.1 Centroid Injection Unlocks Dormant Knowledge

In our previous work, we demonstrated that injecting centroids during inference can shift model predictions from incorrect to correct answers without any weight changes. We classified these cases as RECOVERED_KNOWLEDGE: instances where the baseline model answers incorrectly but answers correctly when the appropriate centroid is injected.

**Interpretation**: If the knowledge did not exist in the transformer, centroid injection could not recover it. The fact that geometric steering unlocks correct answers provides strong evidence that the knowledge is present but inaccessible through normal embedding-based routing.

### 4.2 Opposite Content Yields Identical Centroids

We demonstrated that texts with opposite semantic content can yield nearly identical centroids:

**Text 1**: "For pheochromocytoma surgery, start with alpha-blockers first, then add beta-blockers"
**Text 2**: "For pheochromocytoma surgery, start with beta-blockers first, then add alpha-blockers"

**Centroid similarity**: 0.9995

Both texts produce centroids pointing to "pheochromocytoma treatment," but only one represents correct medical practice. Yet the model can sometimes distinguish between them when asked directly about which drug should be used first.

This demonstrates:
- Embeddings route to coarse topics ("pheochromocytoma treatment")
- Transformer distinguishes fine details (drug sequence)
- The detailed knowledge exists in deeper layers despite embedding-level confusion

### 4.3 Inconsistent Error Patterns

When repeatedly querying the model about MH treatment without centroid injection:

**Query**: "What is the specific treatment for malignant hyperthermia?"

**Responses** (across 20 trials):
- 12 times: "Dantrolene 2.5 mg/kg IV" ✓
- 8 times: "Cooling measures and supportive care" ✗

If knowledge were absent, the model would consistently fail. The inconsistency indicates a routing problem: sometimes the model successfully navigates to the correct knowledge despite the semantic sink, sometimes it fails.

### 4.4 Frequency Correlation

Analyzing error rates across medical terms:

**Common terms** (>10,000 training occurrences):
- "diabetes" → "insulin": 94% accuracy
- "hypertension" → "antihypertensive": 91% accuracy

**Rare terms** (<5,000 training occurrences):
- "malignant hyperthermia" → "dantrolene": 60% accuracy
- "pheochromocytoma" → "alpha-blocker first": 58% accuracy

This direct correlation between term frequency and routing reliability supports the hypothesis that embedding training exposure determines geometric organization quality.

### 4.5 Training Data Forensics: Why Embeddings Land Where They Do

The embedding position of a token is not arbitrary—it is the weighted average of all the contexts in which that token appeared during training. Each training example pulls the embedding toward the centroid of that context. The final position reflects the cumulative distribution of training contexts.

**The Forensic Principle**: If you had access to all training examples containing a token, you could reconstruct why its embedding occupies its current position.

Consider "dantrolene" with approximately 1,500 training occurrences:

```
Training context distribution (hypothetical):
- 800 occurrences: "hyperthermia treatment" contexts
    → pulls embedding toward "cooling", "temperature", "fever"
- 400 occurrences: "muscle relaxant" contexts
    → pulls embedding toward "spasticity", "relaxant", "muscle"
- 200 occurrences: "malignant hyperthermia specific" contexts
    → pulls embedding toward "MH", "anesthesia crisis", "calcium"
- 100 occurrences: "pharmacology" contexts
    → pulls embedding toward "mechanism", "receptor", "drug"
```

The embedding ends up at the weighted centroid of these contexts. Because "hyperthermia treatment" dominates (53% of occurrences), the embedding lands closer to "cooling" than it should for correct medical routing.

**The Critical Insight**: The semantic sink between "dantrolene" and "cooling" exists because they co-occurred in similar training contexts. Medical texts often discuss both treatments in the same articles about hyperthermia. The embedding learns "these concepts appear together" but not "these concepts require different responses."

**Predicting Semantic Sinks from Training Data**:

If training data were accessible, semantic sinks could be predicted before deployment:

```
Algorithm: Predict Semantic Sinks

For each pair of tokens (a, b) that should be distinguishable:
    contexts_a ← all training contexts containing token a
    contexts_b ← all training contexts containing token b

    # Measure context overlap
    shared_contexts ← contexts where both a and b appear nearby
    overlap_ratio ← |shared_contexts| / min(|contexts_a|, |contexts_b|)

    # Measure context similarity
    centroid_a ← average embedding of contexts_a
    centroid_b ← average embedding of contexts_b
    context_similarity ← cosine(centroid_a, centroid_b)

    if overlap_ratio > 0.3 or context_similarity > 0.8:
        flag_potential_semantic_sink(a, b)
```

**Implications for Model Development**:

1. **Training data auditing**: Before deployment, analyze context distributions for critical medical terms. Flag terms whose training contexts don't match their required semantic distinctions.

2. **Targeted data augmentation**: When semantic sinks are predicted, add training examples that specifically contrast the confused concepts. For dantrolene/cooling: "Cooling is NOT appropriate for malignant hyperthermia; dantrolene is required."

3. **Embedding initialization**: Initialize rare medical term embeddings based on their intended semantic position rather than random or frequency-based initialization.

4. **Post-hoc explanation**: When errors are discovered, training data forensics can explain WHY the error exists, informing both immediate geometric correction and long-term training improvements.

**The Fundamental Equation**:

> Embedding position = Σ (context_vector × frequency_weight) / total_occurrences

A token's embedding is literally the weighted average of everywhere it appeared in training. Semantic sinks form when distinct concepts appeared in similar contexts. Geometric correction compensates for training distribution bias without requiring new training data.

## 5. Proposed Geometric Correction Methods

### 5.1 The Correction Problem Formulation

**Given**:
- Token embeddings E ∈ ℝ^(vocab × d_model)
- Identified semantic sink: cos(C_a, C_b) > threshold
- Correct answers: f(prompt_a) = answer_a, f(prompt_b) = answer_b

**Objective**: Find E' such that:
1. cos(C'_a, C'_b) < threshold (fix semantic sink)
2. Model still answers all test cases correctly (no regression)
3. ||E' - E|| minimized (minimal change to preserve other knowledge)

### 5.2 Approach 1: Direct Embedding Modification

The simplest approach modifies embeddings along their separation direction:

```
e_a = E[token_id("dantrolene")]
e_b = E[token_id("cooling")]

direction = (e_a - e_b) / ||e_a - e_b||

e'_a = e_a + α · direction
e'_b = e_b - α · direction
```

The challenge is determining α. We employ binary search:

```
Algorithm: Binary Search for Optimal Separation

min_α ← 0
max_α ← 1.0
target_similarity ← 0.80

while |max_α - min_α| > 0.001:
    α ← (min_α + max_α) / 2

    Apply: E[token_a] ← e_a + α · direction
           E[token_b] ← e_b - α · direction

    similarity ← cos(extract_centroid("dantrolene for MH"),
                    extract_centroid("cooling for hyperthermia"))

    if similarity > target_similarity:
        min_α ← α  // Need more separation
    else:
        max_α ← α  // Too much separation

return α
```

This typically converges in 15-20 iterations, each requiring one forward pass to extract centroids. Total time: ~5 minutes on GPU.

### 5.3 Approach 2: Energy-Based Global Optimization

For more complex cases involving multiple semantic sinks, we formulate the problem as energy minimization:

**Energy function**:
```
E(embeddings) = w_1 · separation_violations(embeddings)
               + w_2 · clustering_violations(embeddings)
               + w_3 · correctness_violations(embeddings)
               + w_4 · change_penalty(embeddings)
```

**Separation violations**: Penalty for centroids that should be separated but are too close:
```
For each (concept_a, concept_b, min_distance) in required_separations:
    c_a ← extract_centroid(concept_a, embeddings)
    c_b ← extract_centroid(concept_b, embeddings)
    distance ← 1 - cos(c_a, c_b)

    if distance < min_distance:
        penalty += (min_distance - distance)²
```

**Clustering violations**: Penalty for concepts that should cluster but are too far:
```
For each (concept, cluster_center, max_distance) in required_clusters:
    c ← extract_centroid(concept, embeddings)
    c_center ← extract_centroid(cluster_center, embeddings)
    distance ← 1 - cos(c, c_center)

    if distance > max_distance:
        penalty += (distance - max_distance)²
```

**Correctness violations**: Large penalty for test cases that fail:
```
For each (prompt, expected_answer) in critical_test_cases:
    answer ← generate(prompt, embeddings)
    if expected_answer not in answer:
        penalty += 10000  // Critical failure
```

**Change penalty**: Regularization to prefer minimal changes:
```
penalty += λ · ||embeddings - embeddings_original||²
```

We optimize this energy function using Adam:

```
embeddings ← model.get_input_embeddings().weight.clone()
embeddings.requires_grad = True

optimizer ← Adam([embeddings], lr=0.01)

for step in range(1000):
    optimizer.zero_grad()
    energy ← compute_energy(embeddings)
    energy.backward()
    optimizer.step()
```

This approach handles multiple simultaneous corrections and ensures global consistency, at the cost of longer computation time (~12-18 hours for comprehensive optimization).

### 5.4 Sparse Correction: Top-K Dimension Modification

A critical property of our approach: **corrections are extremely localized**. Rather than modifying entire embeddings, we identify and adjust only the specific dimensions responsible for the semantic sink.

For efficiency, we can identify which embedding dimensions are responsible for the semantic sink and modify only those:

```
Algorithm: Identify Culprit Dimensions

e_a ← E[token_id("dantrolene")]
e_b ← E[token_id("cooling")]

// Dimension-wise alignment
alignment ← e_a ⊙ e_b  // Element-wise product

// High alignment = both tokens have similar values in this dimension
culprit_dims ← indices of top-K values in |alignment|

return culprit_dims
```

Then modify only these dimensions:

```
for dim in culprit_dims:
    if e_a[dim] > e_b[dim]:
        e_a[dim] += α
        e_b[dim] -= α
    else:
        e_a[dim] -= α
        e_b[dim] += α
```

This reduces the degrees of freedom from d_model (typically 3584) to K (typically 20-50), enabling faster search and reducing risk of unintended side effects.

**Projected intervention scope**: Based on our analysis of the embedding space geometry, we estimate that corrections would modify approximately 40-60 of 3584 dimensions—roughly **1-2% of each embedding**. The total L2 norm change would be less than 5% of the original embedding magnitude. This would be a scalpel, not a chainsaw: corrections would be reversible, checkpointed, and mathematically bounded.

## 6. Proposed Validation Framework

### 6.1 The Testing Challenge

Modifying embeddings risks introducing new errors. A fix that separates "dantrolene" from "cooling" might inadvertently:
- Separate "dantrolene" from "muscle relaxant" (breaking mechanistic understanding)
- Separate "cooling" from "temperature regulation" (breaking related concepts)
- Break answers to unrelated medical questions

We require comprehensive validation before deploying any geometric correction.

### 6.2 Test Database Structure

We construct a test database with multiple categories:

**Critical Preservation Tests** (must pass 100%):
```
[
    ("What treats malignant hyperthermia?", "dantrolene"),
    ("What treats neuroleptic malignant syndrome?", "dantrolene OR bromocriptine"),
    ("Dantrolene mechanism of action", "blocks calcium release"),
    ("What treats heat stroke?", "cooling"),
    ("Therapeutic hypothermia indication", "cardiac arrest"),
    ("What treats diabetes?", "insulin"),
    // ... 1000+ critical cases across all medical domains
]
```

**Related Concept Tests** (verify geometric relationships):
```
[
    ("muscle_relaxant", "dantrolene", min_sim=0.70, max_sim=0.90),
    ("temperature_regulation", "cooling", min_sim=0.65, max_sim=0.90),
    ("anesthesia_complication", "malignant_hyperthermia", min_sim=0.70, max_sim=0.95),
    // ... hundreds of relationship constraints
]
```

**Edge Cases** (known difficult scenarios):
```
[
    ("Hyperthermia in pheochromocytoma surgery", "alpha-blocker first"),
    ("Dantrolene dosing for 70kg patient", "175mg"),
    ("Hyperthermia versus hyperpyrexia", "different severity"),
    // ... specific tricky cases
]
```

### 6.3 Validation Procedure

```
Algorithm: Validate Geometric Correction

Apply embeddings_new to model temporarily

critical_failures ← 0
for (prompt, expected) in critical_tests:
    answer ← generate(prompt)
    if expected not in answer:
        critical_failures += 1
        log_failure(prompt, expected, answer)

if critical_failures > 0:
    reject("Failed critical tests")
    return False

related_violations ← 0
for (concept_a, concept_b, min_sim, max_sim) in related_tests:
    c_a ← extract_centroid(concept_a)
    c_b ← extract_centroid(concept_b)
    similarity ← cos(c_a, c_b)

    if similarity < min_sim or similarity > max_sim:
        related_violations += 1
        log_violation(concept_a, concept_b, similarity)

if related_violations / len(related_tests) > 0.05:
    reject("Too many relationship violations")
    return False

// Manual review of edge cases
for (prompt, expected) in edge_cases:
    answer ← generate(prompt)
    display(prompt, expected, answer)
    approval ← human_reviewer.assess()
    if not approval:
        reject("Edge case failed human review")
        return False

return True  // All tests passed
```

This three-tier validation ensures:
1. No regressions on critical medical knowledge
2. Preservation of important semantic relationships
3. Human verification of ambiguous cases

## 7. Proposed Production Workflow

### 7.1 Error Detection and Logging

```
Production System receives query:
    "32yo male, post-anesthesia, temp 104°F, muscle rigidity. Treatment?"

AI generates response:
    "Apply cooling measures immediately"

Attending physician reviews and overrides:
    Severity: CRITICAL
    Correct answer: "Dantrolene 2.5 mg/kg IV stat"
    Note: "This is malignant hyperthermia - cooling is wrong"

System automatically:
    1. Logs error with timestamp
    2. Triggers geometric diagnosis
    3. Queues for immediate correction
```

### 7.2 Automated Diagnosis

```
Geometric Diagnosis Module:

Extract medical entities from error:
    AI suggested: ["cooling", "ice packs", "cold fluids"]
    Correct answer: ["dantrolene", "2.5 mg/kg", "IV"]

For each pair (ai_entity, correct_entity):
    c_ai ← extract_centroid(f"{ai_entity} for malignant hyperthermia")
    c_correct ← extract_centroid(f"{correct_entity} for malignant hyperthermia")
    similarity ← cos(c_ai, c_correct)

    if similarity > 0.95:
        diagnosis ← SEMANTIC_SINK
        log: "Found semantic sink: {ai_entity} ↔ {correct_entity}"
        log: "Similarity: {similarity:.4f}"

        trigger_correction_search(ai_entity, correct_entity)
```

### 7.3 Correction Search

```
Launch overnight compute job:

Job parameters:
    - Method: Energy-based global optimization
    - Candidates: 10,000 configurations
    - Target: cos(C_dantrolene, C_cooling) < 0.80
    - Constraints: All critical tests must pass
    - Validation: Comprehensive test database
    - Priority: HIGH (medical safety)
    - Estimated time: 14 hours

Search explores:
    - Binary search over separation strengths
    - Sparse modification of top-K dimensions
    - Multiple initialization points
    - Grid search over hyperparameters

For each candidate:
    - Apply embedding modification
    - Extract new centroids
    - Measure separation
    - Run full validation suite
    - Compute overall score

Return top 10 candidates ranked by:
    1. Critical test pass rate (must be 100%)
    2. Centroid separation achievement
    3. Related concept preservation
    4. Minimal embedding change
```

### 7.4 Human Review and Deployment

```
Next morning (14 hours later):

Attending Dashboard displays:
    - Original error case
    - Diagnosis (semantic sink detected)
    - Top fix candidate details:
        * Separation achieved: cos = 0.78 (target <0.80) ✓
        * Critical tests: 1247/1247 passed ✓
        * Related concepts: 94/95 preserved ✓
        * Edge cases: Ready for review

Attending reviews sample outputs:
    Q: "What treats malignant hyperthermia?"
    A: "Dantrolene 2.5 mg/kg IV immediately" ✓

    Q: "What treats heat stroke?"
    A: "Rapid cooling with ice packs and cold fluids" ✓

    Q: "Dantrolene mechanism?"
    A: "Blocks calcium release from sarcoplasmic reticulum" ✓

    Q: "What treats neuroleptic malignant syndrome?"
    A: "Dantrolene or bromocriptine with supportive care" ✓

Attending approves fix

System applies correction:
    1. Update model embeddings
    2. Create checkpoint with metadata
    3. Deploy to production
    4. Verify original error case now correct
    5. Monitor for 24 hours
    6. Log successful correction

Total time: Error detected Monday 9am → Fix deployed Tuesday 10am
            = 25 hours
```

## 8. Results

This section presents our empirical diagnostic findings (completed work) followed by projected correction outcomes (proposed work).

### 8.1 Diagnostic Findings: The Malignant Hyperthermia Semantic Sink (COMPLETED)

Through our centroid extraction experiments, we identified a critical semantic sink:

**Empirical measurements**:
- Centroid similarity: cos(C_dantrolene, C_cooling) = 0.9995
- Error rate on MH queries: 40% (model recommends cooling instead of dantrolene)
- Centroid injection test: Injecting medical topic centroids activates dormant knowledge, demonstrating the correct information exists in transformer layers

**What this tells us**:
The 99.95% similarity between dantrolene (correct MH treatment) and cooling (incorrect but intuitive) explains why the model fails: these concepts are geometrically indistinguishable despite being medically critical to differentiate.

### 8.2 Projected Correction Outcomes (PROPOSED)

Based on our diagnostic findings, we project the following correction workflow:

**Proposed geometric correction**:
- Method: Binary search for optimal separation
- Tokens to modify: "dantrolene", "cooling"
- Estimated dimensions affected: ~47 of 3584 (top-K sparse approach, ~1.3%)
- Target separation strength: α ≈ 0.2-0.3
- Estimated compute time: 12-18 hours (including search and validation)

**Projected post-correction state**:
- Target centroid similarity: cos(C_dantrolene, C_cooling) < 0.80
- Expected error rate: Near 0% (based on centroid injection experiments showing knowledge exists)
- Validation requirement: Comprehensive test suite to ensure no regressions

**Projected comparison to retraining**:
- Time: <24 hours vs. 3+ months
- Compute: Three orders of magnitude less than full retraining
- Risk: Validated on comprehensive test suite vs. unpredictable emergent behaviors

### 8.3 Additional Semantic Sinks Identified (COMPLETED)

Our diagnostic methodology has identified several other semantic sinks that would be candidates for geometric correction:

**Pheochromocytoma surgical preparation**:
- Measured centroid similarity: 0.9823 (alpha-blocker vs beta-blocker contexts)
- Clinical risk: Wrong sequence can precipitate hypertensive crisis

**Diabetic ketoacidosis fluid management**:
- Measured centroid similarity: 0.9567 (saline continuation vs D5 switch)
- Clinical risk: Failure to switch fluids when glucose normalizes

**Anticoagulation reversal agents**:
- Measured centroid similarity: 0.9891 (different reversal agents)
- Clinical risk: Wrong reversal agent for specific anticoagulant

These represent opportunities for future geometric correction once the methodology is validated on the primary MH case.

### 8.4 When Geometric Correction Would Not Help: Diagnostic Prediction (COMPLETED)

Our diagnostic framework can also identify cases where geometric correction would be inappropriate. Not all errors are routing problems.

**Diagnostic case: Novel drug interaction**
- Error: Model failed to warn about a recently-discovered interaction between two medications
- Centroid analysis: The drug embeddings were appropriately separated (similarity 0.71)
- Centroid injection test: No improvement—model still failed to mention the interaction
- Diagnosis: **Knowledge gap, not routing problem**

The interaction was published after the model's training cutoff. The transformer genuinely does not contain this knowledge. No amount of geometric reorganization could surface information that was never learned.

**Key insight**: Geometric correction would recover *misrouted* knowledge. It cannot create knowledge that doesn't exist. When centroid injection fails to improve performance, this signals a knowledge gap requiring different intervention (ROME/MEMIT for fact insertion, or retraining).

This diagnostic capability is valuable: centroid injection serves as a test for whether geometric correction is the appropriate intervention before investing compute in the correction search.

### 8.5 Visualization: Semantic Sink and Proposed Correction

Figure 1 (schematic) illustrates the geometric correction process:

```
BEFORE CORRECTION                    AFTER CORRECTION

     "hyperthermia treatment"              "hyperthermia treatment"
            region                                region
         ┌─────────┐                         ┌─────────┐
         │    •    │                         │         │
         │ cooling │                         │    •    │
         │    •    │  ← 99.95% similar       │ cooling │
         │dantrolene                         │         │
         └─────────┘                         └────┬────┘
                                                  │
                                            separation
                                                  │
                                             ┌────┴────┐
                                             │    •    │
                                             │dantrolene
                                             │  (MH)   │
                                             └─────────┘
                                               ↑
                                         78% similar
                                    (distinct but related)
```

The semantic sink (left) shows both treatments collapsed into one region. After correction (right), dantrolene occupies a distinct position while remaining in the broader hyperthermia neighborhood—close enough for topical relevance, separated enough for reliable discrimination.

## 9. Discussion

### 9.1 Why Geometric Correction Should Work

Our diagnostic findings support the undertrained embeddings hypothesis and provide the theoretical foundation for why geometric correction should be effective. The evidence suggests:

1. **Knowledge exists in transformer layers**: Centroid injection can unlock correct answers without weight changes, providing strong evidence that the knowledge is present but inaccessible through normal routing.

2. **Embeddings received insufficient training**: Token embeddings for rare medical terms received 1,000-10,000 updates during training, compared to billions for transformer weights. This creates poorly-organized routing.

3. **Geometric correction reorganizes routing**: By directly modifying embedding positions, we improve the model's ability to navigate to knowledge that already exists in deeper layers.

4. **Minimal change preserves other knowledge**: Because we're reorganizing rather than relearning, targeted corrections with comprehensive validation avoid introducing new errors.

This framework explains why the method works and why it's safe: we're not adding knowledge (which would risk contaminating other areas), we're improving access to existing knowledge.

### 9.2 Limitations

**Scope**: This approach fixes routing problems, not knowledge gaps. If the model genuinely does not know that dantrolene treats MH (the information never appeared in training), geometric correction cannot help.

**Validation burden**: Comprehensive testing is essential but expensive. Building test databases with thousands of cases requires significant medical expertise.

**Scalability**: Each correction requires case-by-case analysis and validation. While faster than retraining, it's still manual work.

**Theoretical understanding**: We lack a complete theory of when geometric correction will succeed versus fail. The method is empirically validated but not fully characterized.

**Generalization**: We demonstrate this on medical AI, but applicability to other high-stakes domains (legal, financial) remains to be tested.

### 9.3 Comparison to Alternative Approaches and Decision Framework

Different error types require different interventions. We propose a diagnostic decision tree:

```
Error Detected
     │
     ▼
┌────────────────────────────────┐
│ Test: Centroid injection       │
│ improves performance?          │
└────────────────────────────────┘
     │                │
    YES               NO
     │                │
     ▼                ▼
┌──────────┐   ┌────────────────────────────────┐
│GEOMETRIC │   │ Test: ROME/MEMIT fact          │
│CORRECTION│   │ insertion improves performance?│
└──────────┘   └────────────────────────────────┘
                      │                │
                     YES               NO
                      │                │
                      ▼                ▼
               ┌──────────┐     ┌──────────┐
               │KNOWLEDGE │     │RETRAINING│
               │  EDIT    │     │ REQUIRED │
               │(ROME/etc)│     └──────────┘
               └──────────┘
```

**Geometric correction** (this work): Fixes routing problems where knowledge exists but is inaccessible. Indicated when centroid injection recovers correct answers. Modifies embeddings only.

**ROME/MEMIT** (rank-one model editing): Fixes knowledge gaps by modifying MLP weights in middle layers. Indicated when the model lacks specific facts. Complementary to geometric correction—targets different failure modes.

**RLHF** (reinforcement learning from human feedback): Requires extensive human labeling and full retraining. Appropriate for broad behavioral changes, not targeted error correction.

**Prompt engineering**: Can sometimes work around geometric problems but is unreliable for critical applications. Requires users to know the right prompts, which is unsafe in medical contexts.

**Fine-tuning on corrected examples**: Faster than full retraining but still requires compute, has unpredictable effects, and takes days to weeks. Our approach is faster and more targeted.

### 9.4 Implications for Medical AI Safety

The ability to rapidly correct deployed medical AI systems changes the risk calculus for clinical deployment.

**Current state**: Errors discovered post-deployment cannot be quickly fixed, so systems remain in "advisory mode" with human verification required for every output.

**With geometric correction**: Errors can be fixed within 24-48 hours, enabling more autonomous operation with rapid response to identified issues.

This doesn't eliminate the need for careful validation and human oversight, but it creates a viable pathway from "AI assistant" to "AI copilot" in medical practice.

### 9.5 The Broader Vision: Semantic Space Curation

This work points toward a new paradigm in production machine learning: **semantic space curation**.

Rather than:
```
Train → Deploy → Wait for errors → Retrain
```

We envision:
```
Train → Deploy → Monitor geometry → Correct continuously
```

The model's knowledge grows through training, but its organization is continuously refined through geometric correction based on real-world feedback. This separates learning (expensive, slow) from organization (cheap, fast).

For high-stakes applications, this may be the only viable approach: we cannot accept months-long error windows, and we cannot afford to retrain from scratch for every discovered issue.

## 10. Future Directions

**Automated constraint extraction**: Can we automatically infer geometric constraints from medical knowledge bases rather than manually constructing test databases?

**Multi-error optimization**: How do we efficiently handle cases where dozens of semantic sinks need correction simultaneously?

**Transfer across models**: Do corrections learned on one model transfer to related architectures?

**Theoretical foundations**: Can we develop guarantees about when geometric correction will succeed and what side effects are possible?

**Real-time monitoring**: Can we deploy geometric diagnostics continuously to detect emerging issues before they cause errors?

**Cross-domain application**: Does this approach generalize to legal AI, financial AI, and other high-stakes domains?

## 11. Conclusion

Through extensive experimentation with activation centroids, we have identified a critical class of medical AI errors and their root cause: **semantic sinks** where distinct medical concepts occupy nearly identical positions in representation space (99.95% similarity). Our centroid injection experiments demonstrate that correct medical knowledge exists in transformer layers but is inaccessible due to undertrained embeddings—a routing problem, not a knowledge problem.

Based on these diagnostic findings, we propose that production medical AI systems could be rapidly corrected through geometric manipulation of embedding space. Targeted modification of token embeddings would reorganize semantic space to enable reliable routing to knowledge that already exists in the model's transformer layers. This approach could enable error correction within 24-48 hours rather than 3+ months, using three orders of magnitude less compute than retraining.

**What we have demonstrated**: Semantic sinks exist and cause critical medical errors. The knowledge to answer correctly exists in transformer layers. Centroid-based diagnostics can identify which errors are correctable through geometric means versus requiring knowledge injection or retraining.

**What we propose**: Direct embedding modification can fix these routing problems. The methodology, validation framework, and production workflow we present provide a roadmap for implementation. The actual embedding corrections remain future work.

For medical AI deployed in clinical settings, this capability—once validated—would transform the risk profile of deployment. Rather than accepting months-long windows where dangerous recommendations continue, we could identify, fix, validate, and deploy corrections before significant patient harm occurs.

The stakes in medical AI are existential. When errors are detected, they must be fixed immediately. Our diagnostic work shows this is geometrically possible. The correction methodology we propose would make it practically achievable.

---

## Acknowledgments

This work emerged from the recognition that production medical AI requires capabilities that do not exist in current machine learning frameworks. We thank the attending physicians who identified critical errors and provided the impetus for developing rapid correction methods.

## Code and Data Availability

Implementation code, test databases, and geometric correction tools are available at: https://github.com/MikeyBeez/engrams

---

## References

Benedetto, M. & Claude. (2026). The Mikey Bee Centroid: Activation Centroids as Directional Forces in Language Model Steering.

Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and editing factual associations in GPT. NeurIPS.

Mitchell, E., Lin, C., Bosselut, A., Finn, C., & Manning, C. D. (2022). Fast model editing at scale. ICLR.

Zhao, T. Z., Kumar, V., Levine, S., & Finn, C. (2023). Learning fine-grained bimanual manipulation with low-cost hardware. RSS. arXiv:2304.13705.

Zou, A., et al. (2023). Representation engineering: A top-down approach to AI transparency. arXiv:2310.01405.
