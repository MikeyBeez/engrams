"""
True Chunked Architecture

This implements EXACTLY what you asked for:
- For each layer, take the input x and output f(x)
- Chunk and average both: chunk_avg(x) and chunk_avg(f(x))
- Make THAT the input to the next layer

This is an actual architectural modification, not just observation.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple
from huggingface_hub import login
import os


class TrueChunkedLayer(nn.Module):
    """
    A layer that:
    1. Takes input x
    2. Runs through original transformer layer -> f(x)
    3. Chunks and averages both x and f(x)
    4. Concatenates [chunk_avg(x), chunk_avg(f(x))]
    5. Projects back to sequence format for next layer
    """

    def __init__(
        self,
        original_layer: nn.Module,
        hidden_dim: int,
        num_chunks: int = 4,
        dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        self.original_layer = original_layer
        self.hidden_dim = hidden_dim
        self.num_chunks = num_chunks
        self.dtype = dtype

        # Project from concatenated chunks back to full sequence
        # Input: [batch, num_chunks, hidden_dim * 2] (input + output concatenated)
        # Output: needs to work with next layer expecting [batch, seq_len, hidden_dim]
        self.chunk_to_seq = nn.Linear(hidden_dim * 2, hidden_dim, dtype=dtype)

        # Store original sequence length for reconstruction
        self._seq_len = None

    def __getattr__(self, name: str):
        """Pass through attribute access to original layer."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_layer, name)

    def chunk_and_average(self, x: torch.Tensor) -> torch.Tensor:
        """
        Chunk sequence and average each chunk.

        Args:
            x: [batch, seq_len, hidden_dim]

        Returns:
            [batch, num_chunks, hidden_dim]
        """
        batch, seq_len, hidden_dim = x.shape
        self._seq_len = seq_len

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

    def expand_to_sequence(self, chunked: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Expand chunked representation back to full sequence.

        Each chunk's representation is repeated to cover its portion of the sequence.

        Args:
            chunked: [batch, num_chunks, hidden_dim]
            seq_len: target sequence length

        Returns:
            [batch, seq_len, hidden_dim]
        """
        batch, num_chunks, hidden_dim = chunked.shape
        chunk_size = seq_len // num_chunks

        expanded = []
        for i in range(num_chunks):
            if i < num_chunks - 1:
                repeat_count = chunk_size
            else:
                repeat_count = seq_len - (num_chunks - 1) * chunk_size

            # Repeat this chunk's representation
            chunk_expanded = chunked[:, i:i+1, :].expand(-1, repeat_count, -1)
            expanded.append(chunk_expanded)

        return torch.cat(expanded, dim=1)  # [batch, seq_len, hidden_dim]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass with chunked self-reflection.

        1. Run original layer
        2. Chunk and average both input and output
        3. Concatenate and project
        4. Expand back to sequence
        """
        batch, seq_len, hidden_dim = hidden_states.shape

        # Step 1: Run original layer
        layer_output = self.original_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs
        )

        # Handle tuple outputs
        if isinstance(layer_output, tuple):
            actual_output = layer_output[0]
            other_outputs = layer_output[1:]
        else:
            actual_output = layer_output
            other_outputs = None

        # Step 2: Chunk and average both input and output
        input_chunked = self.chunk_and_average(hidden_states)   # [batch, num_chunks, hidden_dim]
        output_chunked = self.chunk_and_average(actual_output)  # [batch, num_chunks, hidden_dim]

        # Step 3: Concatenate along hidden dimension
        # This gives us [what came in, what came out] for each chunk
        combined = torch.cat([input_chunked, output_chunked], dim=-1)  # [batch, num_chunks, hidden_dim*2]

        # Step 4: Project back to hidden_dim
        projected = self.chunk_to_seq(combined)  # [batch, num_chunks, hidden_dim]

        # Step 5: Expand back to full sequence length
        expanded = self.expand_to_sequence(projected, seq_len)  # [batch, seq_len, hidden_dim]

        # The expanded chunked representation IS the output
        # This is the key difference - we're not adding it, we're REPLACING
        if other_outputs is not None:
            return (expanded,) + other_outputs
        return expanded


class TrueChunkedArchitecture(nn.Module):
    """
    A model where each layer outputs chunked representations.

    The information flow is:
    x -> layer1 -> [chunk_avg(x), chunk_avg(f1(x))] expanded -> layer2 -> ...

    This fundamentally changes how information flows through the network.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        num_chunks: int = 4,
        wrap_layers: list[int] | str = "all",  # "all", "odd", "even", or list of indices
        device: str = "auto",
        dtype: torch.dtype = torch.float16
    ):
        super().__init__()

        # Login to HuggingFace
        token = os.environ.get("HF_TOKEN")
        if not token:
            try:
                token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
                login(token=token, add_to_git_credential=False)
            except:
                pass

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
        self.num_chunks = num_chunks
        self.dtype = dtype

        # Wrap the specified layers
        self._wrap_layers(wrap_layers)

    def _get_layers(self):
        """Get the transformer layers from the model."""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers  # Llama/Qwen style
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h  # GPT-2 style
        else:
            raise ValueError("Could not find transformer layers in model")

    def _wrap_layers(self, wrap_layers):
        """Wrap specified layers with chunked architecture."""
        layers = self._get_layers()
        num_layers = len(layers)

        # Determine which layers to wrap
        if wrap_layers == "all":
            indices = list(range(num_layers))
        elif wrap_layers == "odd":
            indices = list(range(1, num_layers, 2))
        elif wrap_layers == "even":
            indices = list(range(0, num_layers, 2))
        elif isinstance(wrap_layers, list):
            indices = wrap_layers
        else:
            raise ValueError(f"Invalid wrap_layers: {wrap_layers}")

        print(f"Wrapping {len(indices)} of {num_layers} layers with chunked architecture")

        for i in indices:
            if i >= num_layers:
                continue

            original_layer = layers[i]
            device = next(original_layer.parameters()).device

            wrapped = TrueChunkedLayer(
                original_layer=original_layer,
                hidden_dim=self.hidden_dim,
                num_chunks=self.num_chunks,
                dtype=self.dtype
            )
            wrapped = wrapped.to(device)

            layers[i] = wrapped
            print(f"  Wrapped layer {i}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        **kwargs
    ) -> str:
        """Generate text with the chunked architecture."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,  # Greedy for reproducibility
                **kwargs
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class HybridChunkedArchitecture(nn.Module):
    """
    Hybrid approach: Keep main pathway intact, but ADD chunked reflection.

    This is more stable because:
    - The normal transformer computation still happens
    - The chunked reflection is added as an auxiliary signal
    - The model can learn to use or ignore the chunked info
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        num_chunks: int = 4,
        reflection_scale: float = 0.1,  # How much to weight the reflection
        device: str = "auto",
        dtype: torch.dtype = torch.float16
    ):
        super().__init__()

        token = os.environ.get("HF_TOKEN")
        if not token:
            try:
                token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
                login(token=token, add_to_git_credential=False)
            except:
                pass

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
        self.num_chunks = num_chunks
        self.reflection_scale = reflection_scale
        self.dtype = dtype

        # Wrap all layers with hybrid reflection
        self._wrap_layers()

    def _wrap_layers(self):
        """Wrap layers with hybrid chunked reflection."""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        else:
            raise ValueError("Could not find transformer layers")

        print(f"Adding hybrid chunked reflection to {len(layers)} layers")

        for i, layer in enumerate(layers):
            wrapped = HybridChunkedLayer(
                original_layer=layer,
                hidden_dim=self.hidden_dim,
                num_chunks=self.num_chunks,
                reflection_scale=self.reflection_scale,
                dtype=self.dtype
            )
            wrapped = wrapped.to(next(layer.parameters()).device)
            layers[i] = wrapped

    def generate(self, prompt: str, max_new_tokens: int = 50, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
                **kwargs
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class HybridChunkedLayer(nn.Module):
    """
    Hybrid layer that:
    1. Runs normal transformer computation
    2. Computes chunked reflection (input/output comparison)
    3. Adds scaled reflection to the output
    """

    def __init__(
        self,
        original_layer: nn.Module,
        hidden_dim: int,
        num_chunks: int,
        reflection_scale: float,
        dtype: torch.dtype
    ):
        super().__init__()
        self.original_layer = original_layer
        self.hidden_dim = hidden_dim
        self.num_chunks = num_chunks
        self.reflection_scale = reflection_scale

        # Project concatenated chunks to hidden_dim
        self.reflection_proj = nn.Linear(hidden_dim * 2, hidden_dim, dtype=dtype)

    def __getattr__(self, name: str):
        """Pass through attribute access to original layer."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_layer, name)

    def chunk_and_average(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_dim = x.shape
        actual_chunks = min(self.num_chunks, seq_len)
        chunk_size = seq_len // actual_chunks

        chunks = []
        for i in range(actual_chunks):
            start = i * chunk_size
            end = start + chunk_size if i < actual_chunks - 1 else seq_len
            chunks.append(x[:, start:end, :].mean(dim=1))

        return torch.stack(chunks, dim=1)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        batch, seq_len, hidden_dim = hidden_states.shape

        # Normal forward pass
        layer_output = self.original_layer(hidden_states, **kwargs)

        if isinstance(layer_output, tuple):
            actual_output = layer_output[0]
            other_outputs = layer_output[1:]
        else:
            actual_output = layer_output
            other_outputs = None

        # Compute chunked reflection
        input_chunked = self.chunk_and_average(hidden_states)
        output_chunked = self.chunk_and_average(actual_output)

        # Concatenate and project
        combined = torch.cat([input_chunked, output_chunked], dim=-1)
        reflection = self.reflection_proj(combined)  # [batch, num_chunks, hidden_dim]

        # Average across chunks and expand to sequence
        reflection_avg = reflection.mean(dim=1, keepdim=True)  # [batch, 1, hidden_dim]
        reflection_expanded = reflection_avg.expand(-1, seq_len, -1)

        # Add scaled reflection to output
        final_output = actual_output + self.reflection_scale * reflection_expanded

        if other_outputs is not None:
            return (final_output,) + other_outputs
        return final_output


def test_architectures():
    """Compare the different chunked architectures."""
    print("=" * 70)
    print("Testing True Chunked Architectures")
    print("=" * 70)

    prompts = [
        "The capital of France is",
        "2 + 2 equals",
        "Water freezes at",
    ]

    # Test 1: Baseline (unwrapped)
    print("\n[1] BASELINE (no modification)")
    print("-" * 50)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    login(token=token, add_to_git_credential=False)

    baseline_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    baseline_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    for prompt in prompts:
        inputs = baseline_tokenizer(prompt, return_tensors="pt").to(baseline_model.device)
        with torch.no_grad():
            outputs = baseline_model.generate(**inputs, max_new_tokens=20, do_sample=False)
        result = baseline_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  '{prompt}' -> {result}")

    del baseline_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Test 2: Hybrid chunked (safer)
    print("\n[2] HYBRID CHUNKED (reflection_scale=0.1)")
    print("-" * 50)
    hybrid = HybridChunkedArchitecture(
        model_name="Qwen/Qwen2.5-0.5B",
        num_chunks=4,
        reflection_scale=0.1
    )

    for prompt in prompts:
        result = hybrid.generate(prompt, max_new_tokens=20)
        print(f"  '{prompt}' -> {result}")

    del hybrid
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Test 3: True chunked (aggressive - wrap only some layers)
    print("\n[3] TRUE CHUNKED (every other layer)")
    print("-" * 50)
    true_chunked = TrueChunkedArchitecture(
        model_name="Qwen/Qwen2.5-0.5B",
        num_chunks=4,
        wrap_layers="even"  # Only even layers to preserve some normal flow
    )

    for prompt in prompts:
        try:
            result = true_chunked.generate(prompt, max_new_tokens=20)
            print(f"  '{prompt}' -> {result}")
        except Exception as e:
            print(f"  '{prompt}' -> ERROR: {e}")

    print("\n" + "=" * 70)
    print("Summary:")
    print("  - Baseline: Normal transformer behavior")
    print("  - Hybrid: Adds chunked input/output comparison as auxiliary signal")
    print("  - True Chunked: Replaces layer outputs with chunked representations")
    print("=" * 70)


if __name__ == "__main__":
    test_architectures()
