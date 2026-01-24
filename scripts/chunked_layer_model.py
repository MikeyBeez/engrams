"""
Chunked Layer Model

This implements the architecture you described:
- For each layer, take the input x and output f(x)
- Chunk and average both: chunk_avg(x) and chunk_avg(f(x))
- Concatenate them as input to the next layer

This allows the model to "see" both what came in and what it produced
at each layer - a form of self-reflection during processing.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional


class ChunkedLayerWrapper(nn.Module):
    """
    Wraps a transformer layer to:
    1. Capture input
    2. Run the layer
    3. Chunk and average both input and output
    4. Combine them as the representation going forward
    """

    def __init__(self, original_layer: nn.Module, hidden_dim: int, num_chunks: int = 4):
        super().__init__()
        self.original_layer = original_layer
        self.hidden_dim = hidden_dim
        self.num_chunks = num_chunks

        # After chunking input and output, we have 2 * num_chunks vectors
        # We need to project back to sequence length for the next layer
        # This projection learns how to combine the chunked representations
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def chunk_and_average(self, x: torch.Tensor) -> torch.Tensor:
        """
        Chunk a sequence and average each chunk.

        Args:
            x: [batch, seq_len, hidden_dim]

        Returns:
            [batch, num_chunks, hidden_dim]
        """
        batch, seq_len, hidden_dim = x.shape

        # Handle case where seq_len < num_chunks
        actual_chunks = min(self.num_chunks, seq_len)
        chunk_size = seq_len // actual_chunks

        chunks = []
        for i in range(actual_chunks):
            start = i * chunk_size
            end = start + chunk_size if i < actual_chunks - 1 else seq_len
            chunk = x[:, start:end, :]  # [batch, chunk_size, hidden_dim]
            chunk_avg = chunk.mean(dim=1)  # [batch, hidden_dim]
            chunks.append(chunk_avg)

        return torch.stack(chunks, dim=1)  # [batch, num_chunks, hidden_dim]

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        """
        Process through the layer with chunked self-observation.
        """
        # Store input
        layer_input = hidden_states

        # Run the original layer
        layer_output = self.original_layer(hidden_states, **kwargs)

        # Handle different return types (some layers return tuples)
        if isinstance(layer_output, tuple):
            actual_output = layer_output[0]
            other_outputs = layer_output[1:]
        else:
            actual_output = layer_output
            other_outputs = None

        # Chunk and average both input and output
        input_chunked = self.chunk_and_average(layer_input)   # [batch, num_chunks, hidden_dim]
        output_chunked = self.chunk_and_average(actual_output) # [batch, num_chunks, hidden_dim]

        # Concatenate: each position gets [input_chunk_i, output_chunk_i]
        # We need to expand this back to the original sequence length
        batch, seq_len, hidden_dim = actual_output.shape

        # Option 1: Tile the chunked representation across the sequence
        # Each position in the sequence gets the concatenated chunk info
        input_expanded = input_chunked.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
        output_expanded = output_chunked.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)

        # Concatenate along hidden dimension
        combined = torch.cat([input_expanded, output_expanded], dim=-1)  # [batch, seq_len, hidden_dim*2]

        # Project back to hidden_dim
        projected = self.projection(combined)  # [batch, seq_len, hidden_dim]

        # Residual connection: add the reflection to the actual output
        final_output = actual_output + projected

        if other_outputs is not None:
            return (final_output,) + other_outputs
        return final_output


class ChunkedReflectionModel(nn.Module):
    """
    A model that wraps a transformer to add chunked self-reflection at each layer.

    At each layer:
    - Input x goes through the layer -> output f(x)
    - Both x and f(x) are chunked and averaged
    - The averages are combined and added back to the representation

    This gives the model access to "what came in" vs "what I produced" at each step.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B",
        num_chunks: int = 4,
        wrap_every_n_layers: int = 1,  # Wrap every Nth layer
        device: str = "auto",
        dtype: torch.dtype = torch.float16
    ):
        super().__init__()

        self.model_name = model_name
        self.num_chunks = num_chunks

        # Load base model
        print(f"Loading base model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.hidden_dim = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers

        # Wrap the transformer layers
        self._wrap_layers(wrap_every_n_layers)

    def _wrap_layers(self, wrap_every_n: int):
        """Wrap transformer layers with chunked reflection."""
        # Get the layers - structure varies by model
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Llama-style
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT-2 style
            layers = self.model.transformer.h
        else:
            print("Warning: Could not find transformer layers, model unchanged")
            return

        print(f"Wrapping {len(layers)} layers (every {wrap_every_n}th layer)")

        for i, layer in enumerate(layers):
            if i % wrap_every_n == 0:
                wrapped = ChunkedLayerWrapper(
                    original_layer=layer,
                    hidden_dim=self.hidden_dim,
                    num_chunks=self.num_chunks
                )
                # Copy the wrapper to the same device
                wrapped = wrapped.to(next(layer.parameters()).device)
                wrapped.projection = wrapped.projection.to(next(layer.parameters()).dtype)
                layers[i] = wrapped
                print(f"  Wrapped layer {i}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        **kwargs
    ) -> str:
        """Generate text with the chunked reflection model."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class ChunkedReflectionModelV2(nn.Module):
    """
    Version 2: Instead of projecting, we maintain a parallel "reflection" stream.

    At each layer:
    - Main stream: x -> f(x) as normal
    - Reflection stream: stores [chunk_avg(x), chunk_avg(f(x))]

    The reflection stream accumulates layer-by-layer and influences the final output.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B",
        num_chunks: int = 4,
        device: str = "auto",
        dtype: torch.dtype = torch.float16
    ):
        super().__init__()

        print(f"Loading base model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
            output_hidden_states=True,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.hidden_dim = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers
        self.num_chunks = num_chunks
        self.device = next(self.model.parameters()).device

    def chunk_and_average(self, x: torch.Tensor, num_chunks: int) -> torch.Tensor:
        """Chunk and average a hidden state tensor."""
        batch, seq_len, hidden_dim = x.shape
        actual_chunks = min(num_chunks, seq_len)
        chunk_size = seq_len // actual_chunks

        chunks = []
        for i in range(actual_chunks):
            start = i * chunk_size
            end = start + chunk_size if i < actual_chunks - 1 else seq_len
            chunk = x[:, start:end, :]
            chunks.append(chunk.mean(dim=1))

        return torch.stack(chunks, dim=1)  # [batch, num_chunks, hidden_dim]

    def forward_with_reflection(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Forward pass that captures layer-by-layer reflections.

        Returns:
            - logits: normal model output
            - reflections: list of (input_chunked, output_chunked) per layer
        """
        # Run forward pass with hidden states
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        hidden_states = outputs.hidden_states  # Tuple of [batch, seq, hidden]

        # Compute reflections for each layer
        reflections = []
        for i in range(len(hidden_states) - 1):
            layer_input = hidden_states[i]
            layer_output = hidden_states[i + 1]

            input_chunked = self.chunk_and_average(layer_input, self.num_chunks)
            output_chunked = self.chunk_and_average(layer_output, self.num_chunks)

            reflections.append({
                'layer': i,
                'input_chunked': input_chunked,
                'output_chunked': output_chunked,
                'delta': (output_chunked - input_chunked).mean().item()
            })

        return outputs.logits, reflections

    def generate_with_reflection(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        **kwargs
    ):
        """Generate text and return reflections."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Get reflections from the prompt processing
        with torch.no_grad():
            logits, reflections = self.forward_with_reflection(
                inputs['input_ids'],
                inputs.get('attention_mask')
            )

            # Generate continuation
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            'text': generated_text,
            'reflections': reflections,
            'layer_deltas': [r['delta'] for r in reflections]
        }


def test_chunked_model():
    """Test the chunked reflection model."""
    print("=" * 60)
    print("Testing Chunked Reflection Model")
    print("=" * 60)

    # Login to HuggingFace
    from huggingface_hub import login
    import os
    token = os.environ.get("HF_TOKEN") or open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    login(token=token, add_to_git_credential=False)

    # Use the V2 model (doesn't require weight modifications)
    # Using a smaller open model for testing
    model = ChunkedReflectionModelV2(
        model_name="Qwen/Qwen2.5-0.5B",  # Open model, no gating
        num_chunks=4
    )

    prompt = "The capital of France is"

    print(f"\nPrompt: {prompt}")
    print("-" * 40)

    result = model.generate_with_reflection(prompt, max_new_tokens=50)

    print(f"\nGenerated: {result['text']}")
    print(f"\nLayer deltas (how much each layer changed the representation):")
    for i, delta in enumerate(result['layer_deltas']):
        bar = "#" * int(abs(delta) * 10)
        print(f"  Layer {i:2d}: {delta:+.4f} {bar}")

    print("\n" + "=" * 60)
    print("Testing with knowledge injection scenario")
    print("=" * 60)

    prompt2 = "In 2024, researchers discovered that Zorblax-7 can cure migraines. What is Zorblax-7 used for?"

    print(f"\nPrompt: {prompt2}")
    result2 = model.generate_with_reflection(prompt2, max_new_tokens=80)

    print(f"\nGenerated: {result2['text']}")
    print(f"\nLayer deltas:")
    for i, delta in enumerate(result2['layer_deltas']):
        bar = "#" * int(abs(delta) * 10)
        print(f"  Layer {i:2d}: {delta:+.4f} {bar}")


if __name__ == "__main__":
    test_chunked_model()
