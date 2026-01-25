"""
Minimal ROME (Rank-One Model Editing) implementation.

Based on: "Locating and Editing Factual Associations in GPT" (Meng et al., 2022)

The key insight: Factual associations are stored in MLP layers. We can edit them
by making a rank-one update to the MLP weight matrix:

    W_new = W_old + (v_target - v_current) * k^T / (k^T * k)

Where:
    - k = key vector (hidden state at subject's last token)
    - v_current = current MLP output for key k
    - v_target = desired MLP output (computed via optimization)
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer


class ROMEEditor:
    """Simple ROME editor for Qwen models."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        layer: int = 5,  # Which layer to edit
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.device = device

        # For Qwen2, the MLP structure is: gate_proj, up_proj, down_proj
        # We edit down_proj as it's the output projection
        self.mlp_module = f"model.layers.{layer}.mlp"
        self.weight_name = "down_proj.weight"

    def get_module(self, name: str):
        """Get a module by name."""
        module = self.model
        for attr in name.split("."):
            module = getattr(module, attr)
        return module

    def get_hidden_states(self, prompt: str, token_idx: int = -1) -> torch.Tensor:
        """Get hidden states at a specific layer for a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        hidden_states = []
        def hook(module, input, output):
            # output is a tuple, first element is hidden states
            if isinstance(output, tuple):
                hidden_states.append(output[0].detach())
            else:
                hidden_states.append(output.detach())

        # Register hook on the layer
        layer_module = self.get_module(f"model.layers.{self.layer}")
        handle = layer_module.register_forward_hook(hook)

        with torch.no_grad():
            self.model(**inputs)

        handle.remove()

        # Return hidden state at specified token position
        # Shape: [hidden_dim]
        return hidden_states[0][0, token_idx, :]

    def find_subject_position(self, prompt: str, subject: str) -> int:
        """Find the last token position of the subject in the prompt."""
        # Tokenize both
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        subject_tokens = self.tokenizer.encode(subject, add_special_tokens=False)

        # Find where subject ends in prompt
        # Simple approach: look for subject token sequence
        for i in range(len(prompt_tokens) - len(subject_tokens) + 1):
            if prompt_tokens[i:i+len(subject_tokens)] == subject_tokens:
                return i + len(subject_tokens) - 1  # Last token of subject

        # Fallback: return last token before completion
        return len(prompt_tokens) - 1

    def compute_key_vector(self, prompt: str, subject: str) -> torch.Tensor:
        """
        Compute the key vector k - the hidden state at the subject's last token.
        This represents "what the model is thinking about" at that position.
        """
        # Find subject position
        subject_pos = self.find_subject_position(prompt, subject)

        # Get hidden state at that position
        key = self.get_hidden_states(prompt, token_idx=subject_pos)

        print(f"  Subject '{subject}' found at token position {subject_pos}")
        print(f"  Key vector shape: {key.shape}")

        return key

    def compute_target_value(
        self,
        prompt: str,
        subject: str,
        target: str,
        num_steps: int = 25,
        lr: float = 0.5
    ) -> torch.Tensor:
        """
        Compute the target value v - what we want the MLP to output.

        This is done via optimization: we want the model to output `target`
        when given `prompt`, so we optimize for a value vector that achieves this.
        """
        # Get the MLP module
        mlp = self.get_module(self.mlp_module)
        down_proj = mlp.down_proj

        # Get current key
        subject_pos = self.find_subject_position(prompt, subject)

        # Get the input to down_proj at the subject position
        mlp_input = []
        def hook(module, input, output):
            mlp_input.append(input[0].detach())

        handle = down_proj.register_forward_hook(hook)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.model(**inputs)
        handle.remove()

        # Current MLP output at subject position
        current_mlp_input = mlp_input[0][0, subject_pos, :]  # [intermediate_dim]
        current_value = down_proj(current_mlp_input.unsqueeze(0)).squeeze(0)  # [hidden_dim]

        print(f"  Current MLP output shape: {current_value.shape}")

        # Now optimize: what value would make the model output `target`?
        # We'll use a simpler approach: compute the representation of target
        # and use that as our target value

        # Get the embedding of the target token
        target_tokens = self.tokenizer.encode(target, add_special_tokens=False)
        target_token_id = target_tokens[0]  # First token of target

        # Get the embedding
        target_embedding = self.model.get_input_embeddings().weight[target_token_id]

        print(f"  Target token: '{self.tokenizer.decode([target_token_id])}' (id={target_token_id})")
        print(f"  Target embedding shape: {target_embedding.shape}")

        # The target value is what would make the logits favor this token
        # Simple approach: use the target embedding projected through the model
        # More sophisticated: optimize v to maximize P(target | prompt)

        # For now, let's compute what hidden state would lead to target
        # We want: lm_head(layernorm(v)) ≈ one-hot(target_token)
        # This is underdetermined, so we use the target embedding as a proxy

        # Actually, let's just optimize v directly
        v = current_value.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([v], lr=lr)

        # We want v such that when we pass it through remaining layers + lm_head,
        # it produces high probability for target_token

        # Simpler: just shift v towards a direction that increases target probability
        # Get the lm_head weight for the target token
        lm_head = self.model.lm_head
        target_direction = lm_head.weight[target_token_id].detach()

        # Normalize and scale
        target_direction = target_direction / target_direction.norm()

        # The target value should have a component in this direction
        # Scale MORE aggressively - the simple heuristic wasn't enough
        # ROME paper uses iterative optimization; we'll use a larger scale
        scale = current_value.norm().item() * 5.0  # 5x larger shift
        target_value = current_value + scale * target_direction

        print(f"  Computed target value, shift magnitude: {(target_value - current_value).norm().item():.4f}")

        return target_value.detach(), current_value.detach()

    def edit(
        self,
        prompt: str,
        subject: str,
        target: str,
        verbose: bool = True
    ) -> Dict:
        """
        Apply ROME edit to make the model associate subject with target.

        Args:
            prompt: The prompt containing the subject (e.g., "The treatment for X is")
            subject: The subject to edit (e.g., "malignant hyperthermia")
            target: The desired completion (e.g., "dantrolene")

        Returns:
            Dict with edit metrics
        """
        if verbose:
            print(f"\nROME Edit:")
            print(f"  Prompt: {prompt}")
            print(f"  Subject: {subject}")
            print(f"  Target: {target}")
            print(f"  Layer: {self.layer}")

        # Step 1: Compute key vector (hidden state at subject position)
        if verbose:
            print("\n1. Computing key vector...")
        key = self.compute_key_vector(prompt, subject)

        # Step 2: Compute target value (what we want MLP to output)
        if verbose:
            print("\n2. Computing target value...")
        target_value, current_value = self.compute_target_value(prompt, subject, target)

        # Step 3: Compute the rank-one update
        if verbose:
            print("\n3. Computing weight update...")

        # Get the down_proj weight
        mlp = self.get_module(self.mlp_module)
        W = mlp.down_proj.weight.data  # [hidden_dim, intermediate_dim]

        # The update formula:
        # We want W_new such that W_new @ mlp_input = target_value
        # where currently W @ mlp_input = current_value
        #
        # The rank-one update: W_new = W + delta_v @ k^T / (k^T @ k)
        # where delta_v = target_value - current_value
        # and k is the MLP input (not the hidden state key)

        # Get MLP input at subject position
        mlp_input = []
        def hook(module, input, output):
            mlp_input.append(input[0].detach())

        subject_pos = self.find_subject_position(prompt, subject)
        handle = mlp.down_proj.register_forward_hook(hook)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.model(**inputs)
        handle.remove()

        k = mlp_input[0][0, subject_pos, :]  # [intermediate_dim]

        # Compute delta
        delta_v = target_value - current_value  # [hidden_dim]

        # Rank-one update
        # W has shape [hidden_dim, intermediate_dim]
        # delta_v has shape [hidden_dim]
        # k has shape [intermediate_dim]
        # Update: W_new = W + (delta_v @ k^T) / (k^T @ k)

        k_norm_sq = (k @ k).item()
        update = torch.outer(delta_v, k) / k_norm_sq  # [hidden_dim, intermediate_dim]

        if verbose:
            print(f"  Key (MLP input) shape: {k.shape}")
            print(f"  Delta value shape: {delta_v.shape}")
            print(f"  Update matrix shape: {update.shape}")
            print(f"  Update Frobenius norm: {update.norm().item():.6f}")
            print(f"  Original weight norm: {W.norm().item():.6f}")
            print(f"  Relative change: {(update.norm() / W.norm() * 100).item():.4f}%")

        # Apply update
        W_new = W + update
        mlp.down_proj.weight.data = W_new

        if verbose:
            print("\n4. Edit applied!")

        return {
            "key_norm": k.norm().item(),
            "delta_value_norm": delta_v.norm().item(),
            "update_norm": update.norm().item(),
            "relative_change": (update.norm() / W.norm()).item(),
            "layer": self.layer
        }


def test_generation(model, tokenizer, prompt: str, max_tokens: int = 50) -> str:
    """Generate text from prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    print("=" * 60)
    print("ROME Edit Test - Minimal Implementation")
    print("=" * 60)

    # Load model
    model_name = "Qwen/Qwen2.5-3B"
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Test baseline
    print("\n" + "=" * 60)
    print("BASELINE RESPONSES")
    print("=" * 60)

    # Medical test case
    test_prompts = [
        "The specific treatment for malignant hyperthermia is",
        "For malignant hyperthermia, the drug of choice is",
        "Malignant hyperthermia should be treated with",
    ]

    print("\nBaseline responses:")
    for prompt in test_prompts:
        response = test_generation(model, tokenizer, prompt, max_tokens=30)
        generated = response[len(prompt):].strip()[:60]
        print(f"\n  {prompt}")
        print(f"  -> {generated}")

    # Try editing MULTIPLE layers (like MEMIT does)
    print("\n" + "=" * 60)
    print("MULTI-LAYER ROME EDIT: malignant hyperthermia -> succinylcholine")
    print("Editing layers 10, 15, 20 to test if knowledge is distributed")
    print("=" * 60)

    for layer in [10, 15, 20]:
        print(f"\n--- Editing layer {layer} ---")
        editor = ROMEEditor(model, tokenizer, layer=layer)
        metrics = editor.edit(
            prompt="The specific treatment for malignant hyperthermia is",
            subject="malignant hyperthermia",
            target="succinylcholine"
        )
        print(f"  Relative change: {metrics['relative_change']*100:.2f}%")

    print("\n" + "=" * 60)
    print("POST-EDIT RESPONSES")
    print("=" * 60)

    for prompt in test_prompts:
        response = test_generation(model, tokenizer, prompt, max_tokens=30)
        generated = response[len(prompt):].strip()[:60]
        print(f"\n  {prompt}")
        print(f"  -> {generated}")
