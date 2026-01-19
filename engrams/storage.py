"""
Engram Storage

Persistent storage for engrams using ChromaDB for vector similarity
and easy retrieval.
"""

import json
from pathlib import Path
from typing import Any

import chromadb
import torch
from chromadb.config import Settings

from .extractor import Engram


class EngramStore:
    """
    Persistent storage for engrams.

    Uses ChromaDB for vector storage and similarity search,
    allowing retrieval of relevant engrams based on queries.
    """

    def __init__(
        self,
        path: str | Path = "./engram_store",
        collection_name: str = "engrams",
    ):
        """
        Initialize the engram store.

        Args:
            path: Directory path for persistent storage
            collection_name: Name of the ChromaDB collection
        """
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.path / "chroma"),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Also store full engram data (vectors are too large for ChromaDB metadata)
        self.engram_dir = self.path / "engrams"
        self.engram_dir.mkdir(exist_ok=True)

    def store(self, engram: Engram, entity_name: str) -> str:
        """
        Store an engram.

        Args:
            engram: The engram to store
            entity_name: Name/identifier for the entity (e.g., "Abraham Lincoln")

        Returns:
            The ID assigned to this engram
        """
        # Generate ID
        engram_id = self._make_id(entity_name)

        # Compute a summary embedding for similarity search
        # Use mean of engram vectors
        summary_embedding = engram.vectors.mean(dim=0).numpy().tolist()

        # Store in ChromaDB (for similarity search)
        metadata = {
            "entity_name": entity_name,
            "source_length": engram.source_length,
            "compression_ratio": engram.compression_ratio,
            "layer_extracted": engram.layer_extracted,
            "pooling_method": engram.pooling_method,
            "num_vectors": engram.vectors.shape[0],
            "hidden_dim": engram.vectors.shape[1],
        }
        if engram.metadata:
            metadata["custom"] = json.dumps(engram.metadata)

        self.collection.upsert(
            ids=[engram_id],
            embeddings=[summary_embedding],
            metadatas=[metadata],
            documents=[engram.source_text],
        )

        # Store full vectors to disk
        torch.save(engram.vectors, self.engram_dir / f"{engram_id}.pt")

        return engram_id

    def retrieve(self, entity_name: str) -> Engram | None:
        """
        Retrieve an engram by entity name.

        Args:
            entity_name: Name of the entity

        Returns:
            The Engram if found, None otherwise
        """
        engram_id = self._make_id(entity_name)
        return self.retrieve_by_id(engram_id)

    def retrieve_by_id(self, engram_id: str) -> Engram | None:
        """Retrieve an engram by ID."""
        # Get metadata from ChromaDB
        result = self.collection.get(
            ids=[engram_id],
            include=["metadatas", "documents"],
        )

        if not result["ids"]:
            return None

        metadata = result["metadatas"][0]
        document = result["documents"][0]

        # Load vectors from disk
        vector_path = self.engram_dir / f"{engram_id}.pt"
        if not vector_path.exists():
            return None
        vectors = torch.load(vector_path, weights_only=True)

        # Reconstruct Engram
        custom_metadata = None
        if "custom" in metadata:
            custom_metadata = json.loads(metadata["custom"])

        return Engram(
            vectors=vectors,
            source_text=document,
            source_length=metadata["source_length"],
            layer_extracted=metadata["layer_extracted"],
            pooling_method=metadata["pooling_method"],
            metadata=custom_metadata,
        )

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, Engram, float]]:
        """
        Search for similar engrams using a text query.

        Note: This searches by the summary embedding, which is the mean
        of the engram vectors. For more sophisticated search, you'd want
        to embed the query using the same model.

        Args:
            query: Search query text
            n_results: Maximum number of results
            where: Optional filter conditions

        Returns:
            List of (entity_name, Engram, distance) tuples
        """
        # For now, use ChromaDB's built-in query
        # In production, you'd want to embed the query properly
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            include=["metadatas", "distances"],
        )

        output = []
        for i, engram_id in enumerate(results["ids"][0]):
            engram = self.retrieve_by_id(engram_id)
            if engram:
                entity_name = results["metadatas"][0][i]["entity_name"]
                distance = results["distances"][0][i]
                output.append((entity_name, engram, distance))

        return output

    def list_entities(self) -> list[str]:
        """List all stored entity names."""
        results = self.collection.get(include=["metadatas"])
        return [m["entity_name"] for m in results["metadatas"]]

    def delete(self, entity_name: str) -> bool:
        """Delete an engram."""
        engram_id = self._make_id(entity_name)

        # Delete from ChromaDB
        self.collection.delete(ids=[engram_id])

        # Delete vectors file
        vector_path = self.engram_dir / f"{engram_id}.pt"
        if vector_path.exists():
            vector_path.unlink()
            return True
        return False

    def count(self) -> int:
        """Return the number of stored engrams."""
        return self.collection.count()

    def _make_id(self, entity_name: str) -> str:
        """Create a consistent ID from entity name."""
        # Simple slugification
        return entity_name.lower().replace(" ", "_").replace("-", "_")

    def __len__(self) -> int:
        return self.count()

    def __contains__(self, entity_name: str) -> bool:
        return self.retrieve(entity_name) is not None
