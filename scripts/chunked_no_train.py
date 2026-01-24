"""
Chunked Architecture - No Training Required

This version doesn't add learnable parameters. Instead it:
1. Chunks and averages input and output
2. Uses the difference (delta) to modulate the output
3. No projection layers - just direct manipulation

The idea: if chunk_avg(output) diverges significantly from chunk_avg(input),
scale down that divergence. This creates a form of self-regularization.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
from huggingface_hub import login
import os


class ChunkedDeltaLayer(nn.Module):
    """
    Layer that uses chunked input/output comparison WITHOUT learned parameters.

    Approach:
    1. Run normal layer: x -> f(x)
    2. Chunk and average both
    3. Compute delta = chunk_avg(f(x)) - chunk_avg(x)
    4. Use delta to modulate the output

    This is like giving the layer "awareness" of what it changed.
    """

    def __init__(
        self,
        original_layer: nn.Module,
        num_chunks: int = 4,
        delta_scale: float = 0.1,  # How much delta influences output
        mode: str = "residual"  # "residual" or "gate"
    ):
        super().__init__()
        self.original_layer = original_layer
        self.num_chunks = num_chunks
        self.delta_scale = delta_scale
        self.mode = mode

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_layer, name)

    def chunk_and_average(self, x: torch.Tensor) -> torch.Tensor:
        """Chunk and average. Returns [batch, num_chunks, hidden_dim]"""
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

        # Normal forward
        layer_output = self.original_layer(hidden_states, **kwargs)

        if isinstance(layer_output, tuple):
            actual_output = layer_output[0]
            other_outputs = layer_output[1:]
        else:
            actual_output = layer_output
            other_outputs = None

        # Chunk and average
        input_chunked = self.chunk_and_average(hidden_states)  # [batch, chunks, hidden]
        output_chunked = self.chunk_and_average(actual_output)  # [batch, chunks, hidden]

        # Compute delta (what the layer changed)
        delta = output_chunked - input_chunked  # [batch, chunks, hidden]

        # Average delta across chunks
        delta_mean = delta.mean(dim=1, keepdim=True)  # [batch, 1, hidden]

        if self.mode == "residual":
            # Add scaled delta as a global bias
            # This "reminds" the representation of what changed
            delta_expanded = delta_mean.expand(-1, seq_len, -1)
            final_output = actual_output + self.delta_scale * delta_expanded

        elif self.mode == "gate":
            # Use delta magnitude to gate the change
            # Large changes get dampened
            delta_mag = delta_mean.norm(dim=-1, keepdim=True)  # [batch, 1, 1]
            gate = torch.sigmoid(-delta_mag * self.delta_scale)  # Lower gate for large deltas
            gate = gate.expand(-1, seq_len, -1)

            # Interpolate between input and output based on gate
            final_output = gate * hidden_states + (1 - gate) * actual_output

        else:
            final_output = actual_output

        if other_outputs is not None:
            return (final_output,) + other_outputs
        return final_output


class ChunkedConcatLayer(nn.Module):
    """
    Concatenate chunked input and output, then use attention-like mechanism
    to let the model "look at" what it did.

    No learnable params - uses cosine similarity.
    """

    def __init__(
        self,
        original_layer: nn.Module,
        num_chunks: int = 4,
        influence_scale: float = 0.1
    ):
        super().__init__()
        self.original_layer = original_layer
        self.num_chunks = num_chunks
        self.influence_scale = influence_scale

    def __getattr__(self, name: str):
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

        # Normal forward
        layer_output = self.original_layer(hidden_states, **kwargs)

        if isinstance(layer_output, tuple):
            actual_output = layer_output[0]
            other_outputs = layer_output[1:]
        else:
            actual_output = layer_output
            other_outputs = None

        # Chunk and average
        input_chunked = self.chunk_and_average(hidden_states)  # [batch, chunks, hidden]
        output_chunked = self.chunk_and_average(actual_output)  # [batch, chunks, hidden]

        # Compute cosine similarity between input and output chunks
        # This tells us how much each chunk "changed direction"
        input_norm = input_chunked / (input_chunked.norm(dim=-1, keepdim=True) + 1e-8)
        output_norm = output_chunked / (output_chunked.norm(dim=-1, keepdim=True) + 1e-8)
        similarity = (input_norm * output_norm).sum(dim=-1, keepdim=True)  # [batch, chunks, 1]

        # Average similarity across chunks
        avg_similarity = similarity.mean(dim=1, keepdim=True)  # [batch, 1, 1]

        # Use similarity to scale the change
        # High similarity (small change) -> keep the change
        # Low similarity (big change) -> dampen the change
        change = actual_output - hidden_states
        scaled_change = change * (avg_similarity.expand(-1, seq_len, hidden_dim) ** self.influence_scale)

        final_output = hidden_states + scaled_change

        if other_outputs is not None:
            return (final_output,) + other_outputs
        return final_output


class ChunkedModel:
    """Wrapper to apply chunked layers to any model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        layer_type: str = "delta",  # "delta", "concat", or "none"
        num_chunks: int = 4,
        scale: float = 0.1,
        wrap_layers: str = "all"
    ):
        # Login
        token = os.environ.get("HF_TOKEN")
        if not token:
            try:
                token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
                login(token=token, add_to_git_credential=False)
            except:
                pass

        print(f"Loading: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.layer_type = layer_type
        self.num_chunks = num_chunks
        self.scale = scale

        if layer_type != "none":
            self._wrap_layers(wrap_layers)

    def _wrap_layers(self, wrap_layers):
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        else:
            print("Could not find layers")
            return

        num_layers = len(layers)

        if wrap_layers == "all":
            indices = list(range(num_layers))
        elif wrap_layers == "even":
            indices = list(range(0, num_layers, 2))
        elif wrap_layers == "odd":
            indices = list(range(1, num_layers, 2))
        elif wrap_layers == "last_half":
            indices = list(range(num_layers // 2, num_layers))
        else:
            indices = []

        print(f"Wrapping layers {indices} with {self.layer_type}")

        for i in indices:
            if self.layer_type == "delta":
                wrapped = ChunkedDeltaLayer(
                    layers[i],
                    num_chunks=self.num_chunks,
                    delta_scale=self.scale,
                    mode="residual"
                )
            elif self.layer_type == "concat":
                wrapped = ChunkedConcatLayer(
                    layers[i],
                    num_chunks=self.num_chunks,
                    influence_scale=self.scale
                )
            else:
                continue

            layers[i] = wrapped

    def generate(self, prompt: str, max_new_tokens: int = 30) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def test_no_train_chunked():
    """Test the no-training-required chunked architectures."""
    print("=" * 70)
    print("Chunked Architecture WITHOUT Training")
    print("=" * 70)

    prompts = [
        "The capital of France is",
        "2 + 2 equals",
        "Water freezes at",
        "Albert Einstein developed the theory of",
    ]

    configs = [
        ("none", 0.0, "all", "Baseline"),
        ("delta", 0.01, "all", "Delta (scale=0.01, all layers)"),
        ("delta", 0.05, "last_half", "Delta (scale=0.05, last half only)"),
        ("delta", 0.1, "odd", "Delta (scale=0.1, odd layers only)"),
    ]

    for layer_type, scale, wrap, name in configs:
        print(f"\n[{name}]")
        print("-" * 50)

        model = ChunkedModel(
            model_name="Qwen/Qwen2.5-0.5B",
            layer_type=layer_type,
            num_chunks=4,
            scale=scale,
            wrap_layers=wrap
        )

        for prompt in prompts:
            result = model.generate(prompt, max_new_tokens=25)
            # Truncate for display
            result_short = result[:80] + "..." if len(result) > 80 else result
            print(f"  {prompt}")
            print(f"    -> {result_short}")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_no_train_chunked()
