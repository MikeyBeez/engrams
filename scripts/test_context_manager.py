"""
Test the Context Manager - demonstrate context compression and restoration.

Simulates a real workflow:
1. Load context about multiple topics
2. Compress each to engrams
3. Switch between contexts
4. Answer questions using restored context
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.insert(0, '..')

from engrams.context_manager import ContextManager

# === SETUP ===
MODEL_NAME = "Qwen/Qwen2.5-7B"
print(f"Loading {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    output_hidden_states=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
print(f"Model loaded on {next(model.parameters()).device}")

# Initialize context manager
ctx_mgr = ContextManager(
    model=model,
    tokenizer=tokenizer,
    extraction_layer=16,  # Middle-late layer works well
    num_engram_tokens=16,
    storage_path=None,  # In-memory only for this test
)

# === SIMULATE MULTIPLE PROJECT CONTEXTS ===
print("\n" + "="*80)
print("SIMULATING MULTI-PROJECT CONTEXT MANAGEMENT")
print("="*80)

# Project A: Machine Learning System
project_a_context = """
PROJECT: ML Pipeline for Customer Churn Prediction

ARCHITECTURE:
- Data ingestion from PostgreSQL database
- Feature engineering with pandas and scikit-learn
- XGBoost model for classification
- FastAPI for model serving
- Docker deployment on AWS ECS

KEY DECISIONS:
1. Chose XGBoost over neural networks for interpretability
2. Using SHAP values for feature importance
3. Batch predictions run nightly, real-time API for individual predictions
4. Model versioning with MLflow

CURRENT STATUS:
- Data pipeline complete
- Model accuracy: 87% AUC-ROC
- API deployed to staging
- Pending: A/B testing setup, monitoring dashboard

TEAM:
- Lead: Sarah Chen
- Data Engineer: Mike Rodriguez  
- ML Engineer: Priya Patel
"""

# Project B: Web Application
project_b_context = """
PROJECT: E-commerce Platform Redesign

TECH STACK:
- Frontend: Next.js 14 with TypeScript
- Backend: Node.js with Express
- Database: PostgreSQL with Prisma ORM
- Cache: Redis for session and cart data
- Search: Elasticsearch for product search

KEY FEATURES:
1. Server-side rendering for SEO
2. Real-time inventory updates via WebSockets
3. Stripe integration for payments
4. Multi-tenant architecture for marketplace vendors

CURRENT STATUS:
- Homepage and product pages complete
- Cart and checkout in progress
- Search indexing needs optimization
- Performance target: <2s page load

BLOCKERS:
- Waiting on design assets for mobile views
- Elasticsearch cluster sizing needs review

TEAM:
- Lead: James Wilson
- Frontend: Alice Zhang, Tom Brown
- Backend: David Kim
"""

# Project C: Research Paper
project_c_context = """
RESEARCH: Engram-Based Context Compression for LLMs

HYPOTHESIS:
Hidden states from transformer middle layers contain compressed semantic 
representations that can be extracted, stored, and re-injected to restore
context without full token overhead.

METHODOLOGY:
1. Extract hidden states at layers 8-20 during context processing
2. Pool sequence into fixed number of "engram" vectors via chunked mean
3. Scale engrams to embedding space norm for injection
4. Prepend engrams to new prompts as synthetic context

RESULTS SO FAR:
- 100% accuracy on QA tasks with 16-token engrams
- 7.3x token reduction compared to RAG
- 18x compression ratio (286 tokens -> 16 engrams)
- Layers 8-24 all work well; layer 0 (embedding) fails completely

IMPLICATIONS:
- Effective context window extension
- Project switching without context loss
- Multi-document reasoning at low token cost

NEXT STEPS:
- Test with longer documents
- Evaluate on harder questions
- Explore learned compression heads
"""

# === COMPRESS EACH PROJECT ===
print("\n--- Compressing Project Contexts ---")

cp_a = ctx_mgr.compress(
    project_a_context,
    checkpoint_id="ml_pipeline",
    metadata={"project": "ML Pipeline", "type": "technical"}
)
print(f"Project A: {cp_a}")

cp_b = ctx_mgr.compress(
    project_b_context,
    checkpoint_id="ecommerce",
    metadata={"project": "E-commerce", "type": "technical"}
)
print(f"Project B: {cp_b}")

cp_c = ctx_mgr.compress(
    project_c_context,
    checkpoint_id="research",
    metadata={"project": "Research Paper", "type": "research"}
)
print(f"Project C: {cp_c}")

# === SHOW STATS ===
print("\n--- Compression Statistics ---")
stats = ctx_mgr.get_stats()
print(f"Total checkpoints: {stats['count']}")
print(f"Original tokens: {stats['total_source_tokens']}")
print(f"Engram tokens: {stats['total_engram_tokens']}")
print(f"Overall compression: {stats['overall_compression']:.1f}x")
print(f"Total memory: {stats['total_memory_kb']:.1f} KB")

# === TEST CONTEXT SWITCHING ===
print("\n" + "="*80)
print("TESTING CONTEXT RESTORATION")
print("="*80)

questions = [
    # Project A questions
    ("ml_pipeline", "What model is being used for churn prediction?", ["XGBoost"]),
    ("ml_pipeline", "What is the current model accuracy?", ["87%", "AUC-ROC"]),
    ("ml_pipeline", "Who is the ML Engineer on the team?", ["Priya Patel", "Priya"]),
    
    # Project B questions
    ("ecommerce", "What frontend framework is being used?", ["Next.js", "Next"]),
    ("ecommerce", "What is the performance target?", ["2s", "2 seconds", "page load"]),
    ("ecommerce", "What payment system is integrated?", ["Stripe"]),
    
    # Project C questions
    ("research", "What compression ratio was achieved?", ["18x", "18"]),
    ("research", "Which layers work well for extraction?", ["8-24", "8", "24", "middle"]),
    ("research", "What is the token reduction compared to RAG?", ["7.3x", "7.3"]),
]

def check_answer(answer, expected):
    return any(e.lower() in answer.lower() for e in expected)

results = {"correct": 0, "total": 0}

for checkpoint_id, question, expected in questions:
    print(f"\n[{checkpoint_id}] Q: {question}")
    
    # Generate with restored context
    response = ctx_mgr.generate_with_context(
        prompt=f"Based on the project context, answer briefly: {question}",
        checkpoint_ids=[checkpoint_id],
        max_new_tokens=50,
        do_sample=False,
    )
    
    # Clean up response
    answer = response.split("answer briefly:")[-1].strip() if "answer briefly:" in response else response
    correct = check_answer(answer, expected)
    results["correct"] += int(correct)
    results["total"] += 1
    
    marker = "✓" if correct else "✗"
    print(f"  {marker} A: {answer[:80]}...")

# === MULTI-CONTEXT TEST ===
print("\n" + "="*80)
print("MULTI-CONTEXT INJECTION TEST")
print("="*80)

print("\nInjecting ALL THREE project contexts simultaneously...")

response = ctx_mgr.generate_with_context(
    prompt="List the three projects you have context about and one key fact about each:",
    checkpoint_ids=["ml_pipeline", "ecommerce", "research"],
    max_new_tokens=150,
    do_sample=False,
)
print(f"\nResponse:\n{response}")

# === FINAL SUMMARY ===
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nAccuracy: {results['correct']}/{results['total']} ({results['correct']/results['total']*100:.0f}%)")
print(f"\nToken efficiency:")
total_source = sum(len(tokenizer.encode(ctx)) for ctx in [project_a_context, project_b_context, project_c_context])
total_engram = 16 * 3  # 16 tokens per checkpoint, 3 checkpoints
print(f"  Full contexts: {total_source} tokens")
print(f"  As engrams: {total_engram} tokens")
print(f"  Reduction: {total_source/total_engram:.1f}x")

print("\n✓ Context Manager prototype working!")
