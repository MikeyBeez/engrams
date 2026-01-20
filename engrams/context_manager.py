"""
Context Manager - Compress and restore conversation context using engrams.

This module enables:
1. Compressing long context into dense engram representations
2. Storing compressed context with metadata
3. Restoring context by injecting engrams during generation
4. Managing multiple context "checkpoints" for project switching
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import hashlib

import torch
from torch import Tensor


@dataclass
class ContextCheckpoint:
    """A compressed snapshot of conversation context."""
    
    id: str
    engram_vectors: Tensor  # [num_tokens, hidden_dim]
    source_text: str  # Original text (truncated for storage)
    source_tokens: int  # Original token count
    extraction_layer: int
    num_engram_tokens: int
    created_at: float
    metadata: dict = field(default_factory=dict)
    
    @property
    def compression_ratio(self) -> float:
        return self.source_tokens / self.num_engram_tokens
    
    @property
    def memory_bytes(self) -> int:
        return self.engram_vectors.element_size() * self.engram_vectors.numel()
    
    def __repr__(self) -> str:
        return (
            f"ContextCheckpoint(id='{self.id}', "
            f"tokens={self.num_engram_tokens}, "
            f"compression={self.compression_ratio:.1f}x, "
            f"size={self.memory_bytes/1024:.1f}KB)"
        )


class ContextManager:
    """
    Manages context compression and restoration using engrams.
    
    Workflow:
    1. compress() - Take current context, extract engrams, store checkpoint
    2. list_checkpoints() - See available saved contexts
    3. restore() - Get engram vectors for injection
    4. generate_with_context() - Generate with restored context
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        extraction_layer: int = 12,
        num_engram_tokens: int = 16,
        storage_path: Optional[Path] = None,
    ):
        """
        Initialize the context manager.
        
        Args:
            model: HuggingFace model with output_hidden_states=True
            tokenizer: Corresponding tokenizer
            extraction_layer: Which layer to extract from (middle layers work best)
            num_engram_tokens: How many tokens to compress to
            storage_path: Where to persist checkpoints (optional)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.extraction_layer = extraction_layer
        self.num_engram_tokens = num_engram_tokens
        self.storage_path = Path(storage_path) if storage_path else None
        
        # In-memory checkpoint storage
        self.checkpoints: dict[str, ContextCheckpoint] = {}
        
        # Get embedding norm for scaling
        test_input = tokenizer("test", return_tensors="pt").to(model.device)
        with torch.no_grad():
            embed_out = model.get_input_embeddings()(test_input["input_ids"])
        self.embed_norm = torch.norm(embed_out, dim=-1).mean().item()
        
        # Load existing checkpoints if storage path exists
        if self.storage_path:
            self._load_checkpoints()
    
    def compress(
        self,
        context: str,
        checkpoint_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> ContextCheckpoint:
        """
        Compress context into an engram checkpoint.
        
        Args:
            context: The text context to compress
            checkpoint_id: Optional ID (auto-generated if not provided)
            metadata: Optional metadata to attach
            
        Returns:
            ContextCheckpoint containing compressed representation
        """
        # Generate ID if not provided
        if checkpoint_id is None:
            hash_input = f"{context[:100]}{time.time()}"
            checkpoint_id = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
        
        # Tokenize
        inputs = self.tokenizer(
            context,
            return_tensors="pt",
            truncation=True,
            max_length=getattr(self.model.config, 'max_position_embeddings', 4096),
        ).to(self.model.device)
        source_tokens = inputs["input_ids"].shape[1]
        
        # Extract hidden states
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        hidden = outputs.hidden_states[self.extraction_layer].squeeze(0)  # [seq, hidden]
        
        # Pool into engram vectors
        engram = self._pool_to_engram(hidden)
        
        # Create checkpoint
        checkpoint = ContextCheckpoint(
            id=checkpoint_id,
            engram_vectors=engram.cpu(),
            source_text=context[:500] + "..." if len(context) > 500 else context,
            source_tokens=source_tokens,
            extraction_layer=self.extraction_layer,
            num_engram_tokens=self.num_engram_tokens,
            created_at=time.time(),
            metadata=metadata or {},
        )
        
        # Store
        self.checkpoints[checkpoint_id] = checkpoint
        if self.storage_path:
            self._save_checkpoint(checkpoint)
        
        return checkpoint
    
    def restore(self, checkpoint_id: str, scale_to_embed: bool = True) -> Tensor:
        """
        Restore engram vectors from a checkpoint for injection.
        
        Args:
            checkpoint_id: ID of checkpoint to restore
            scale_to_embed: Whether to scale vectors to embedding space norm
            
        Returns:
            Tensor of engram vectors ready for injection
        """
        if checkpoint_id not in self.checkpoints:
            raise KeyError(f"Checkpoint '{checkpoint_id}' not found")
        
        checkpoint = self.checkpoints[checkpoint_id]
        engram = checkpoint.engram_vectors.to(self.model.device)
        
        if scale_to_embed:
            # Scale each vector to match embedding space norm
            current_norms = torch.norm(engram, dim=1, keepdim=True)
            engram = engram / current_norms * self.embed_norm
        
        return engram
    
    def generate_with_context(
        self,
        prompt: str,
        checkpoint_ids: list[str],
        max_new_tokens: int = 100,
        scale_to_embed: bool = True,
        **generate_kwargs,
    ) -> str:
        """
        Generate text with restored context from checkpoints.
        
        Args:
            prompt: The prompt to respond to
            checkpoint_ids: List of checkpoint IDs to inject
            max_new_tokens: Maximum tokens to generate
            scale_to_embed: Whether to scale engrams to embedding space
            **generate_kwargs: Additional args for model.generate()
            
        Returns:
            Generated text
        """
        # Get prompt embeddings
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        embeddings = self.model.get_input_embeddings()
        prompt_embeds = embeddings(prompt_tokens["input_ids"])
        
        # Collect and concatenate engrams from all checkpoints
        all_engrams = []
        for cp_id in checkpoint_ids:
            engram = self.restore(cp_id, scale_to_embed=scale_to_embed)
            all_engrams.append(engram)
        
        if all_engrams:
            combined_engrams = torch.cat(all_engrams, dim=0)  # [total_engram_tokens, hidden]
            engram_embeds = combined_engrams.unsqueeze(0).to(prompt_embeds.dtype)
            
            # Prepend engrams to prompt
            combined_embeds = torch.cat([engram_embeds, prompt_embeds], dim=1)
            
            # Build attention mask
            engram_mask = torch.ones(1, combined_engrams.shape[0], device=self.model.device)
            prompt_mask = prompt_tokens["attention_mask"]
            combined_mask = torch.cat([engram_mask, prompt_mask], dim=1)
        else:
            combined_embeds = prompt_embeds
            combined_mask = prompt_tokens["attention_mask"]
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                **generate_kwargs,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def list_checkpoints(self) -> list[ContextCheckpoint]:
        """List all available checkpoints."""
        return sorted(
            self.checkpoints.values(),
            key=lambda c: c.created_at,
            reverse=True,
        )
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        if checkpoint_id in self.checkpoints:
            del self.checkpoints[checkpoint_id]
            if self.storage_path:
                cp_file = self.storage_path / f"{checkpoint_id}.pt"
                if cp_file.exists():
                    cp_file.unlink()
            return True
        return False
    
    def get_stats(self) -> dict:
        """Get statistics about stored checkpoints."""
        if not self.checkpoints:
            return {"count": 0}
        
        total_tokens = sum(c.source_tokens for c in self.checkpoints.values())
        total_engram_tokens = sum(c.num_engram_tokens for c in self.checkpoints.values())
        total_bytes = sum(c.memory_bytes for c in self.checkpoints.values())
        
        return {
            "count": len(self.checkpoints),
            "total_source_tokens": total_tokens,
            "total_engram_tokens": total_engram_tokens,
            "overall_compression": total_tokens / total_engram_tokens if total_engram_tokens else 0,
            "total_memory_kb": total_bytes / 1024,
        }
    
    def _pool_to_engram(self, hidden: Tensor) -> Tensor:
        """Pool hidden states into fixed number of engram vectors."""
        seq_len = hidden.shape[0]
        chunk_size = seq_len // self.num_engram_tokens
        
        vectors = []
        for i in range(self.num_engram_tokens):
            start = i * chunk_size
            end = start + chunk_size if i < self.num_engram_tokens - 1 else seq_len
            vectors.append(hidden[start:end].mean(dim=0))
        
        return torch.stack(vectors)
    
    def _save_checkpoint(self, checkpoint: ContextCheckpoint):
        """Save checkpoint to disk."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Save vectors
        torch.save(checkpoint.engram_vectors, self.storage_path / f"{checkpoint.id}.pt")
        
        # Save metadata
        meta = {
            "id": checkpoint.id,
            "source_text": checkpoint.source_text,
            "source_tokens": checkpoint.source_tokens,
            "extraction_layer": checkpoint.extraction_layer,
            "num_engram_tokens": checkpoint.num_engram_tokens,
            "created_at": checkpoint.created_at,
            "metadata": checkpoint.metadata,
        }
        with open(self.storage_path / f"{checkpoint.id}.json", "w") as f:
            json.dump(meta, f)
    
    def _load_checkpoints(self):
        """Load checkpoints from disk."""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        for meta_file in self.storage_path.glob("*.json"):
            cp_id = meta_file.stem
            vec_file = self.storage_path / f"{cp_id}.pt"
            
            if not vec_file.exists():
                continue
            
            with open(meta_file) as f:
                meta = json.load(f)
            
            vectors = torch.load(vec_file, weights_only=True)
            
            checkpoint = ContextCheckpoint(
                id=meta["id"],
                engram_vectors=vectors,
                source_text=meta["source_text"],
                source_tokens=meta["source_tokens"],
                extraction_layer=meta["extraction_layer"],
                num_engram_tokens=meta["num_engram_tokens"],
                created_at=meta["created_at"],
                metadata=meta.get("metadata", {}),
            )
            self.checkpoints[cp_id] = checkpoint
