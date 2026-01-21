Agent OS Specification: Engram-Based Retrieval Control

A minimal agent architecture that takes the retrieval-cue finding seriously.


Core Principle

Nothing except RAG is allowed to inject propositions. Everything else may only influence what the model retrieves.

This single rule prevents most agent failure modes.


The Three-Layer Model

Layer 1: Control Plane (Engrams)

Purpose: Constrain relevance, not truth.

Allowed inputs: topic, domain, task type, stance (such as "critic" or "teacher" or "debugger"), session continuity, user preferences, planner intent.

Properties: persistent across turns, composable, safe to average or decay, cannot teach facts, cannot poison beliefs. Engrams operate by biasing attention and retrieval pathways without binding symbols or propositions. Engrams are many-to-one mappings from inputs to control vectors; inversion is intentionally impossible. This means they cannot leak facts even if exfiltrated.


Layer 2: Content Plane (RAG)

Purpose: Inject facts, references, and claims.

Inputs: retrieved documents, tool outputs, databases, citations.

Properties: explicit, inspectable, auditable, revocable, dangerous if wrong.


Layer 3: Core Model

Purpose: Reasoning, synthesis, judgment.

The model owns the world model, resolves contradictions, performs inference, and decides what to say.

No "memory writes" happen here.


Execution Loop

The execution flow proceeds as follows:

User Input leads to Intent/Topic Classifier, which feeds the Engram Selector, which updates the Engram Accumulator (session state). This is where Control Injection happens. Then the Model Forward Pass occurs. Optionally, Content Injection via RAG happens here. Finally, the Final Answer is produced.

Key detail: Engrams go in early (retrieval bias). RAG goes in late (fact injection). This is early bias, late binding. The ordering prevents tool outputs from steering hypothesis formation and preserves the model's internal consistency checks.


Session State Without Memory Corruption

Instead of chat logs or summaries, you store a SessionEngram as an exponential moving average of topic engram, domain engram, and task engram.

This gives you long-term coherence, no hallucination drift, no belief accumulation, and no compounding errors.

The agent "remembers what it's about" without remembering what it said.


Planner Integration

Planners output engrams plus RAG queries, never raw beliefs.

Example planner intent: "We are debugging Python async code, focus on event loops."

What it emits: a debugging-task engram, a Python-async domain engram, and an optional RAG query for "Python event loop scheduling."

No hallucinated plans get committed as truth.


Safety Model

Input sources and their allowed outputs:

User preferences go through engrams only. Agent self-reflection goes through engrams only. Untrusted tools go through engrams only. Trusted databases go through RAG. External web goes through RAG with scoping.

This is capability-based security for LLMs with no ambient authority. Untrusted inputs cannot escalate to proposition injection.


What Engrams Replace

With this architecture, you no longer need long system prompts, conversation summaries, "stay on topic" reminders, prompt templates per task, or brittle role instructions.

Engrams are the role instruction, enforced at the retrieval level.


What This OS Does NOT Do

It does not write to model beliefs, store facts long-term, correct model knowledge, or compress documents into truth.

Those belong elsewhere. This OS assumes the model already knows things. Our job is to help it find them.


One-Sentence Summary

RAG tells the model what to believe. Engrams tell the model where to look.


Falsification

This architecture would be invalidated if: (1) engrams can be shown to reliably convey novel propositional content, (2) a compression method exists that preserves facts without reintroducing poisoning risk, or (3) early engram injection demonstrably corrupts downstream reasoning.

Current evidence suggests none of these hold.


Related

Paper: "Engrams Don't Inject Information - They Retrieve It"
Code: github.com/mikeybeez/engrams
