#!/usr/bin/env python3
"""
Attention Head Engrams

Hypothesis: Attention patterns preserve more structure than hidden states.
The attention weights tell us WHAT the model focused on and HOW tokens relate.

Approaches:
1. Use attention-weighted hidden states (weight by attention scores)
2. Use attention patterns directly as a kind of "relational engram"
3. Extract the key/value vectors from attention (they're the "memory")
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import os


def setup_auth():
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
            login(token=token, add_to_git_credential=False)
        except:
            pass


def extract_attention_engram(model, tokenizer, text, layer_idx=12, head_idx=None):
    """
    Extract engram using attention weights.

    Returns:
        - attention_weighted: hidden states weighted by attention
        - raw_attention: the attention pattern itself
        - keys: the key vectors (what tokens "offer")
        - values: the value vectors (what tokens "provide")
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            output_attentions=True
        )

    # Get attention from specified layer
    # Shape: [batch, num_heads, seq_len, seq_len]
    attention = outputs.attentions[layer_idx]

    # Get hidden states from same layer
    hidden = outputs.hidden_states[layer_idx]  # [batch, seq_len, hidden_dim]

    num_heads = attention.shape[1]
    seq_len = attention.shape[2]

    if head_idx is not None:
        # Use specific head
        attn_weights = attention[0, head_idx]  # [seq_len, seq_len]
    else:
        # Average across heads
        attn_weights = attention[0].mean(dim=0)  # [seq_len, seq_len]

    # Method 1: Attention-weighted hidden states
    # For each position, weight previous positions by attention
    # Use the last position's attention as the "summary" attention
    summary_attn = attn_weights[-1]  # [seq_len] - what last token attends to
    weighted_hidden = (hidden[0] * summary_attn.unsqueeze(-1)).sum(dim=0)  # [hidden_dim]

    # Method 2: Full attention-weighted combination
    # Each position weighted by mean attention it receives
    received_attn = attn_weights.mean(dim=0)  # [seq_len] - how much each pos is attended to
    full_weighted = (hidden[0] * received_attn.unsqueeze(-1)).sum(dim=0)  # [hidden_dim]

    # Method 3: Keep multiple tokens, weighted by attention
    num_engram_tokens = 16
    chunk_size = max(1, seq_len // num_engram_tokens)

    engram_vectors = []
    for i in range(num_engram_tokens):
        start = i * chunk_size
        end = min(start + chunk_size, seq_len)
        if start >= seq_len:
            engram_vectors.append(hidden[0, -1])
        else:
            # Weight by attention received
            chunk_attn = received_attn[start:end]
            chunk_attn = chunk_attn / (chunk_attn.sum() + 1e-8)
            chunk_hidden = hidden[0, start:end]
            weighted_chunk = (chunk_hidden * chunk_attn.unsqueeze(-1)).sum(dim=0)
            engram_vectors.append(weighted_chunk)

    attention_engram = torch.stack(engram_vectors)

    return {
        'summary_weighted': weighted_hidden,
        'full_weighted': full_weighted,
        'attention_engram': attention_engram,
        'raw_attention': attn_weights,
        'hidden': hidden[0],
        'seq_len': seq_len
    }


def extract_kv_engram(model, tokenizer, text, layer_idx=12):
    """
    Extract key and value vectors from a layer.

    Keys = what information tokens "advertise"
    Values = what information tokens "provide when attended to"

    KV cache is literally how transformers remember context!
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # We need to hook into the model to get K and V
    # For Qwen models, the attention is in model.model.layers[i].self_attn

    keys_captured = []
    values_captured = []

    def hook_fn(module, input, output):
        # For Qwen2, output is (attn_output, attn_weights, past_key_value)
        # past_key_value contains (key, value) tensors
        if hasattr(output, '__len__') and len(output) >= 3:
            if output[2] is not None:
                k, v = output[2]
                keys_captured.append(k.detach())
                values_captured.append(v.detach())

    # Register hook on the target layer's attention
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layer = model.model.layers[layer_idx]
        if hasattr(layer, 'self_attn'):
            hook = layer.self_attn.register_forward_hook(hook_fn)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=True)

    # Remove hook
    if 'hook' in dir():
        hook.remove()

    # Get KV from the past_key_values output
    if outputs.past_key_values is not None:
        # past_key_values is tuple of (key, value) for each layer
        kv = outputs.past_key_values[layer_idx]
        keys = kv[0]  # [batch, num_heads, seq_len, head_dim]
        values = kv[1]  # [batch, num_heads, seq_len, head_dim]

        # Reshape to [seq_len, num_heads * head_dim]
        keys = keys[0].transpose(0, 1).reshape(keys.shape[2], -1)
        values = values[0].transpose(0, 1).reshape(values.shape[2], -1)

        return {
            'keys': keys,
            'values': values,
            'seq_len': keys.shape[0]
        }

    return None


def generate_with_engram(model, tokenizer, prompt, engram, max_tokens=50):
    """Generate with engram prepended."""
    embed = model.get_input_embeddings()

    # Handle different engram shapes
    if engram.dim() == 1:
        engram = engram.unsqueeze(0)  # [1, hidden_dim]

    # Scale
    e_norm = embed.weight.norm(dim=1).mean().item()
    g_norm = engram.norm(dim=1).mean().item()
    scaled = engram * (e_norm / g_norm) if g_norm > 0 else engram

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    emb = embed(inputs.input_ids)

    combined = torch.cat([scaled.unsqueeze(0).to(emb.dtype), emb], dim=1)

    with torch.no_grad():
        out = model.generate(
            inputs_embeds=combined,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


def generate_baseline(model, tokenizer, context, question, max_tokens=50):
    """RAG baseline."""
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id)

    return tokenizer.decode(out[0], skip_special_tokens=True)


def check(response, expected):
    r = response.lower()
    return any(e.lower() in r for e in expected)


def main():
    print("=" * 80)
    print("ATTENTION HEAD ENGRAMS")
    print("=" * 80)

    setup_auth()

    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager"  # Need eager to get attention outputs
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    print(f"Model: {num_layers} layers, {num_heads} attention heads")

    # Test with novel facts
    novel_text = """
    RESEARCH FINDINGS:
    The Nexion-9 compound reduces inflammation by 94.7%.
    Lead researcher: Dr. Samantha Chen-Watanabe from MIT.
    The quantum coherence record is 847 microseconds using lithium-7.
    Kepler-442b has 23.1% oxygen, measured by JAMES spectrometer.
    """

    questions = [
        ("What percentage does Nexion-9 reduce inflammation?", ["94.7"]),
        ("Who is the lead researcher?", ["chen-watanabe", "samantha"]),
        ("What is the coherence record duration?", ["847"]),
        ("What isotope was used?", ["lithium-7", "lithium"]),
        ("What is Kepler-442b's oxygen percentage?", ["23.1"]),
    ]

    print("\n" + "=" * 80)
    print("EXTRACTING ATTENTION-BASED ENGRAMS")
    print("=" * 80)

    # Extract from different layers
    test_layers = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]

    results = {}

    for layer in test_layers:
        print(f"\n--- Layer {layer} ---")

        attn_data = extract_attention_engram(model, tokenizer, novel_text, layer_idx=layer)
        kv_data = extract_kv_engram(model, tokenizer, novel_text, layer_idx=layer)

        # Also extract mean-pooled hidden state for comparison
        hidden = attn_data['hidden']
        mean_engram = hidden.mean(dim=0)  # [hidden_dim]

        # Chunk the hidden states (old method)
        num_tokens = 16
        chunk_size = max(1, hidden.shape[0] // num_tokens)
        chunked = []
        for i in range(num_tokens):
            start = i * chunk_size
            end = min(start + chunk_size, hidden.shape[0])
            chunked.append(hidden[start:end].mean(dim=0))
        chunked_engram = torch.stack(chunked)

        configs = [
            ("mean_hidden", mean_engram),
            ("chunked_hidden", chunked_engram),
            ("attn_weighted", attn_data['attention_engram']),
            ("summary_attn", attn_data['summary_weighted']),
        ]

        if kv_data is not None:
            # Use values as engram (what's "provided" when attended)
            # Chunk the values
            v = kv_data['values']
            v_chunked = []
            chunk_size = max(1, v.shape[0] // num_tokens)
            for i in range(num_tokens):
                start = i * chunk_size
                end = min(start + chunk_size, v.shape[0])
                v_chunked.append(v[start:end].mean(dim=0))
            value_engram = torch.stack(v_chunked)

            # Truncate or pad to match hidden dim
            if value_engram.shape[1] != hidden.shape[1]:
                # Project or truncate
                if value_engram.shape[1] > hidden.shape[1]:
                    value_engram = value_engram[:, :hidden.shape[1]]
                else:
                    # Pad with zeros
                    padding = torch.zeros(value_engram.shape[0],
                                         hidden.shape[1] - value_engram.shape[1],
                                         device=value_engram.device, dtype=value_engram.dtype)
                    value_engram = torch.cat([value_engram, padding], dim=1)

            configs.append(("kv_values", value_engram))

        layer_results = {}

        for config_name, engram in configs:
            correct = 0
            for question, expected in questions:
                prompt = f"About this research: {question}\nAnswer:"
                try:
                    response = generate_with_engram(model, tokenizer, prompt, engram, max_tokens=30)
                    if check(response, expected):
                        correct += 1
                except Exception as e:
                    pass

            layer_results[config_name] = correct
            print(f"  {config_name:20s}: {correct}/{len(questions)}")

        results[layer] = layer_results

    # Also test RAG baseline
    print("\n--- RAG Baseline ---")
    rag_correct = 0
    for question, expected in questions:
        response = generate_baseline(model, tokenizer, novel_text, question)
        if check(response, expected):
            rag_correct += 1
    print(f"  RAG: {rag_correct}/{len(questions)}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nBy Layer and Method:")
    methods = ['mean_hidden', 'chunked_hidden', 'attn_weighted', 'summary_attn', 'kv_values']

    print(f"{'Layer':<8}", end="")
    for m in methods:
        print(f"{m:<18}", end="")
    print()

    for layer in test_layers:
        print(f"{layer:<8}", end="")
        for m in methods:
            score = results[layer].get(m, '-')
            print(f"{score:<18}", end="")
        print()

    print(f"\nRAG Baseline: {rag_correct}/{len(questions)}")

    # Find best
    best_score = 0
    best_config = None
    for layer in test_layers:
        for method, score in results[layer].items():
            if score > best_score:
                best_score = score
                best_config = (layer, method)

    if best_config:
        print(f"\nBest engram: Layer {best_config[0]}, {best_config[1]} with {best_score}/{len(questions)}")

    if best_score > 0 and best_score >= rag_correct:
        print("==> Attention engrams match or beat RAG!")
    elif best_score > 0:
        print("==> Attention engrams show some recall but RAG is better")
    else:
        print("==> No engram method could recall novel facts")


if __name__ == "__main__":
    main()
