"""
Engram Extractor

Extracts hidden state representations from transformer models,
compressing full documents into dense engram vectors.
"""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


@dataclass
class ExtractionConfig:
    """Configuration for engram extraction."""

    model_name: str = "meta-llama/Llama-3.2-1B"
    layer: int | Literal["middle", "last"] = "middle"
    pooling: Literal["mean", "last", "attention", "learned"] = "mean"
    num_engram_tokens: int = 4  # How many token-sized vectors to output
    device: str = "auto"
    dtype: torch.dtype = torch.float16


@dataclass
class Engram:
    """A compressed representation of a document."""

    vectors: Tensor  # Shape: [num_tokens, hidden_dim]
    source_text: str
    source_length: int  # Original token count
    layer_extracted: int
    pooling_method: str
    metadata: dict | None = None

    @property
    def compression_ratio(self) -> float:
        """How much we compressed the source."""
        return self.source_length / self.vectors.shape[0]

    def __repr__(self) -> str:
        return (
            f"Engram(tokens={self.vectors.shape[0]}, "
            f"dim={self.vectors.shape[1]}, "
            f"compression={self.compression_ratio:.1f}x)"
        )


class EngramExtractor:
    """
    Extract engrams from documents using transformer hidden states.

    The extractor processes a document through a transformer model and
    extracts hidden states from a specified layer, then pools them into
    a fixed number of dense vectors (engrams).
    """

    def __init__(self, config: ExtractionConfig | None = None):
        self.config = config or ExtractionConfig()
        self._model = None
        self._tokenizer = None

    @property
    def model(self) -> AutoModel:
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def _load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading model: {self.config.model_name}")

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self._model = AutoModel.from_pretrained(
            self.config.model_name,
            torch_dtype=self.config.dtype,
            device_map=self.config.device,
            output_hidden_states=True,
        )
        self._model.eval()

        # Determine the actual layer index
        num_layers = self._model.config.num_hidden_layers
        if self.config.layer == "middle":
            self._layer_idx = num_layers // 2
        elif self.config.layer == "last":
            self._layer_idx = num_layers - 1
        else:
            self._layer_idx = self.config.layer

        print(f"Using layer {self._layer_idx} of {num_layers}")

    def extract(self, text: str, metadata: dict | None = None) -> Engram:
        """
        Extract an engram from a text document.

        Args:
            text: The source document (e.g., Wikipedia article)
            metadata: Optional metadata to attach to the engram

        Returns:
            An Engram containing compressed representation vectors
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.model.config.max_position_embeddings,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        source_length = inputs["input_ids"].shape[1]

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract hidden states from target layer
        # hidden_states is tuple of (num_layers + 1) tensors, each [batch, seq, hidden]
        hidden_states = outputs.hidden_states[self._layer_idx]  # [1, seq_len, hidden_dim]
        hidden_states = hidden_states.squeeze(0)  # [seq_len, hidden_dim]

        # Pool to num_engram_tokens vectors
        engram_vectors = self._pool(hidden_states)

        return Engram(
            vectors=engram_vectors.cpu(),
            source_text=text[:500] + "..." if len(text) > 500 else text,
            source_length=source_length,
            layer_extracted=self._layer_idx,
            pooling_method=self.config.pooling,
            metadata=metadata,
        )

    def _pool(self, hidden_states: Tensor) -> Tensor:
        """
        Pool sequence hidden states into fixed number of engram vectors.

        Args:
            hidden_states: [seq_len, hidden_dim]

        Returns:
            Tensor of shape [num_engram_tokens, hidden_dim]
        """
        seq_len, hidden_dim = hidden_states.shape
        num_tokens = self.config.num_engram_tokens

        if self.config.pooling == "mean":
            # Divide sequence into num_tokens chunks and mean pool each
            chunk_size = seq_len // num_tokens
            vectors = []
            for i in range(num_tokens):
                start = i * chunk_size
                end = start + chunk_size if i < num_tokens - 1 else seq_len
                chunk = hidden_states[start:end]
                vectors.append(chunk.mean(dim=0))
            return torch.stack(vectors)

        elif self.config.pooling == "last":
            # Take last num_tokens positions
            return hidden_states[-num_tokens:]

        elif self.config.pooling == "attention":
            # Use attention scores to weight positions
            # For now, fall back to mean (attention pooling needs more work)
            return self._pool_mean_fallback(hidden_states)

        elif self.config.pooling == "learned":
            # This would require a learned projection layer
            # For now, fall back to mean
            return self._pool_mean_fallback(hidden_states)

        else:
            raise ValueError(f"Unknown pooling method: {self.config.pooling}")

    def _pool_mean_fallback(self, hidden_states: Tensor) -> Tensor:
        """Fallback to mean pooling."""
        seq_len, hidden_dim = hidden_states.shape
        num_tokens = self.config.num_engram_tokens
        chunk_size = seq_len // num_tokens
        vectors = []
        for i in range(num_tokens):
            start = i * chunk_size
            end = start + chunk_size if i < num_tokens - 1 else seq_len
            vectors.append(hidden_states[start:end].mean(dim=0))
        return torch.stack(vectors)

    def batch_extract(self, texts: list[str]) -> list[Engram]:
        """Extract engrams from multiple documents."""
        return [self.extract(text) for text in texts]
