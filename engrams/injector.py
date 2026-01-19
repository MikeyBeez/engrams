"""
Engram Injector

Injects pre-computed engram vectors into a model's embedding space,
replacing or augmenting token embeddings with compressed knowledge.
"""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from .extractor import Engram


@dataclass
class InjectionConfig:
    """Configuration for engram injection."""

    model_name: str = "meta-llama/Llama-3.2-1B"
    injection_mode: Literal["replace", "prefix", "add"] = "replace"
    placeholder_token: str = "<ENGRAM>"
    device: str = "auto"
    dtype: torch.dtype = torch.float16


class EngramInjector:
    """
    Inject engrams into a language model's input.

    Supports multiple injection strategies:
    - replace: Replace placeholder tokens with engram vectors
    - prefix: Prepend engram vectors to the input sequence
    - add: Add engram vectors to existing token embeddings
    """

    def __init__(self, config: InjectionConfig | None = None):
        self.config = config or InjectionConfig()
        self._model = None
        self._tokenizer = None
        self._placeholder_id = None

    @property
    def model(self) -> AutoModelForCausalLM:
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def _load_model(self):
        """Load the model and tokenizer, adding placeholder token."""
        print(f"Loading model for injection: {self.config.model_name}")

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=self.config.dtype,
            device_map=self.config.device,
        )

        # Add placeholder token if using replace mode
        if self.config.injection_mode == "replace":
            num_added = self._tokenizer.add_special_tokens(
                {"additional_special_tokens": [self.config.placeholder_token]}
            )
            if num_added > 0:
                self._model.resize_token_embeddings(len(self._tokenizer))
            self._placeholder_id = self._tokenizer.convert_tokens_to_ids(
                self.config.placeholder_token
            )

    def inject_and_generate(
        self,
        prompt: str,
        engram: Engram,
        max_new_tokens: int = 100,
        **generate_kwargs,
    ) -> str:
        """
        Inject an engram into a prompt and generate a response.

        Args:
            prompt: The prompt text, containing placeholder tokens if using replace mode
            engram: The engram to inject
            max_new_tokens: Maximum tokens to generate
            **generate_kwargs: Additional arguments for model.generate()

        Returns:
            Generated text response
        """
        # Get embeddings with engram injected
        inputs_embeds, attention_mask = self._prepare_inputs(prompt, engram)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                **generate_kwargs,
            )

        # Decode (skip the input portion)
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated

    def _prepare_inputs(
        self, prompt: str, engram: Engram
    ) -> tuple[Tensor, Tensor]:
        """
        Prepare input embeddings with engram injected.

        Returns:
            Tuple of (inputs_embeds, attention_mask)
        """
        if self.config.injection_mode == "replace":
            return self._inject_replace(prompt, engram)
        elif self.config.injection_mode == "prefix":
            return self._inject_prefix(prompt, engram)
        elif self.config.injection_mode == "add":
            return self._inject_add(prompt, engram)
        else:
            raise ValueError(f"Unknown injection mode: {self.config.injection_mode}")

    def _inject_replace(self, prompt: str, engram: Engram) -> tuple[Tensor, Tensor]:
        """Replace placeholder tokens with engram vectors."""
        # Tokenize
        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens["input_ids"].to(self.model.device)

        # Get base embeddings
        embeddings = self.model.get_input_embeddings()
        inputs_embeds = embeddings(input_ids)  # [1, seq_len, hidden_dim]

        # Find placeholder positions
        placeholder_mask = input_ids == self._placeholder_id
        placeholder_positions = placeholder_mask.nonzero(as_tuple=True)[1]

        if len(placeholder_positions) == 0:
            raise ValueError(
                f"No placeholder tokens '{ self.config.placeholder_token}' found in prompt"
            )

        # Replace placeholders with engram vectors
        engram_vectors = engram.vectors.to(self.model.device, dtype=self.config.dtype)
        num_to_inject = min(len(placeholder_positions), engram_vectors.shape[0])

        for i in range(num_to_inject):
            pos = placeholder_positions[i]
            inputs_embeds[0, pos] = engram_vectors[i]

        attention_mask = tokens["attention_mask"].to(self.model.device)
        return inputs_embeds, attention_mask

    def _inject_prefix(self, prompt: str, engram: Engram) -> tuple[Tensor, Tensor]:
        """Prepend engram vectors to the input sequence."""
        # Tokenize
        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens["input_ids"].to(self.model.device)

        # Get base embeddings
        embeddings = self.model.get_input_embeddings()
        prompt_embeds = embeddings(input_ids)  # [1, seq_len, hidden_dim]

        # Prepare engram vectors
        engram_vectors = engram.vectors.to(self.model.device, dtype=self.config.dtype)
        engram_vectors = engram_vectors.unsqueeze(0)  # [1, num_engram, hidden_dim]

        # Concatenate: [engram, prompt]
        inputs_embeds = torch.cat([engram_vectors, prompt_embeds], dim=1)

        # Build attention mask
        engram_mask = torch.ones(1, engram_vectors.shape[1], device=self.model.device)
        prompt_mask = tokens["attention_mask"].to(self.model.device)
        attention_mask = torch.cat([engram_mask, prompt_mask], dim=1)

        return inputs_embeds, attention_mask

    def _inject_add(self, prompt: str, engram: Engram) -> tuple[Tensor, Tensor]:
        """Add engram vectors to existing token embeddings."""
        # Tokenize
        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens["input_ids"].to(self.model.device)

        # Get base embeddings
        embeddings = self.model.get_input_embeddings()
        inputs_embeds = embeddings(input_ids)  # [1, seq_len, hidden_dim]

        # Prepare engram vectors
        engram_vectors = engram.vectors.to(self.model.device, dtype=self.config.dtype)

        # Add engram vectors to first N positions
        num_to_add = min(engram_vectors.shape[0], inputs_embeds.shape[1])
        inputs_embeds[0, :num_to_add] = inputs_embeds[0, :num_to_add] + engram_vectors[:num_to_add]

        attention_mask = tokens["attention_mask"].to(self.model.device)
        return inputs_embeds, attention_mask

    def get_hidden_dim(self) -> int:
        """Get the model's hidden dimension."""
        return self.model.config.hidden_size
