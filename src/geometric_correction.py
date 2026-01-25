"""
Geometric Correction of Token Embeddings

Adapted from StudSar patterns for embedding space curation.
Applies targeted modifications to fix semantic sinks in LLM token embeddings.

Status: PROPOSED IMPLEMENTATION - not yet validated
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class CorrectionRecord:
    """Tracks metadata for each embedding correction."""
    token_id: int
    token_text: str
    original_embedding: torch.Tensor
    corrected_embedding: torch.Tensor
    correction_timestamp: str
    similarity_before: float
    similarity_after: float
    dimensions_modified: List[int]
    alpha_used: float
    validation_passed: bool
    confidence_score: float  # Analogous to StudSar's reputation


class SemanticSinkDetector:
    """
    Identifies semantic sinks using centroid extraction.

    A semantic sink exists when two concepts that should be distinguishable
    have centroid similarity > threshold (e.g., 0.95).
    """

    def __init__(
        self,
        model,
        tokenizer,
        layer: int = 20,
        sink_threshold: float = 0.95,
        separation_target: float = 0.80
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.sink_threshold = sink_threshold
        self.separation_target = separation_target
        self.hidden_states = None

        # Register hook to capture hidden states
        self._register_hook()

    def _register_hook(self):
        """Capture hidden states at target layer."""
        def hook(module, input, output):
            self.hidden_states = output[0].detach()

        # Adjust path based on model architecture
        target_layer = self.model.model.layers[self.layer]
        target_layer.register_forward_hook(hook)

    def extract_centroid(self, text: str, num_chunks: int = 16) -> torch.Tensor:
        """
        Extract centroid from text using Mikey Bee Centroid method.

        Args:
            text: Source text to extract centroid from
            num_chunks: Number of chunks for compression (default 16)

        Returns:
            Centroid tensor of shape (num_chunks, hidden_dim)
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            self.model(**inputs)

        # hidden_states shape: (1, seq_len, hidden_dim)
        hidden = self.hidden_states.squeeze(0)  # (seq_len, hidden_dim)
        seq_len = hidden.shape[0]

        # Chunk and average
        chunk_size = max(1, seq_len // num_chunks)
        centroids = []

        for i in range(0, seq_len, chunk_size):
            chunk = hidden[i:i + chunk_size]
            centroid = chunk.mean(dim=0)
            centroids.append(centroid)

        # Pad or truncate to exactly num_chunks
        while len(centroids) < num_chunks:
            centroids.append(centroids[-1])
        centroids = centroids[:num_chunks]

        return torch.stack(centroids)  # (num_chunks, hidden_dim)

    def compute_similarity(
        self,
        centroid_a: torch.Tensor,
        centroid_b: torch.Tensor
    ) -> float:
        """Compute cosine similarity between two centroids."""
        # Flatten to single vector for comparison
        flat_a = centroid_a.flatten()
        flat_b = centroid_b.flatten()

        similarity = torch.nn.functional.cosine_similarity(
            flat_a.unsqueeze(0),
            flat_b.unsqueeze(0)
        )
        return similarity.item()

    def detect_sink(
        self,
        context_a: str,
        context_b: str
    ) -> Tuple[bool, float]:
        """
        Detect if two contexts form a semantic sink.

        Args:
            context_a: First context (e.g., "dantrolene for malignant hyperthermia")
            context_b: Second context (e.g., "cooling for hyperthermia")

        Returns:
            (is_sink, similarity_score)
        """
        centroid_a = self.extract_centroid(context_a)
        centroid_b = self.extract_centroid(context_b)

        similarity = self.compute_similarity(centroid_a, centroid_b)
        is_sink = similarity > self.sink_threshold

        return is_sink, similarity


class GeometricCorrector(nn.Module):
    """
    Manages geometric corrections to token embeddings.

    Adapted from StudSar's memory management patterns:
    - Stores original and corrected embeddings
    - Tracks correction metadata (like StudSar's reputation scores)
    - Supports save/load for checkpointing
    - Validates corrections before applying
    """

    def __init__(
        self,
        model,
        tokenizer,
        detector: SemanticSinkDetector,
        max_correction_pct: float = 0.02,  # Max 2% of dimensions modified
        promotion_boost: float = 1.5  # Analogous to StudSar
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.detector = detector
        self.max_correction_pct = max_correction_pct
        self.promotion_boost = promotion_boost

        # Get reference to embedding layer
        self.embedding_layer = model.get_input_embeddings()
        self.hidden_dim = self.embedding_layer.weight.shape[1]
        self.vocab_size = self.embedding_layer.weight.shape[0]

        # Store original embeddings (like StudSar's memory_embeddings buffer)
        self.register_buffer(
            'original_embeddings',
            self.embedding_layer.weight.clone()
        )

        # Track corrections
        self.corrections: Dict[int, CorrectionRecord] = {}

        # Statistics (like StudSar's usage tracking)
        self.correction_stats = {
            'total_corrections': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'rollbacks': 0
        }

    def identify_culprit_dimensions(
        self,
        token_id_a: int,
        token_id_b: int,
        top_k: int = 50
    ) -> List[int]:
        """
        Identify which dimensions are responsible for the semantic sink.

        High alignment in a dimension means both tokens have similar values
        there - these are the culprit dimensions to modify.
        """
        e_a = self.embedding_layer.weight[token_id_a]
        e_b = self.embedding_layer.weight[token_id_b]

        # Element-wise product shows alignment
        alignment = e_a * e_b

        # Get indices of top-K aligned dimensions
        _, indices = torch.topk(alignment.abs(), top_k)

        return indices.tolist()

    def compute_separation_direction(
        self,
        token_id_a: int,
        token_id_b: int
    ) -> torch.Tensor:
        """Compute the direction vector for separating two embeddings."""
        e_a = self.embedding_layer.weight[token_id_a]
        e_b = self.embedding_layer.weight[token_id_b]

        direction = e_a - e_b
        direction = direction / direction.norm()  # Normalize

        return direction

    def apply_sparse_correction(
        self,
        token_id_a: int,
        token_id_b: int,
        alpha: float,
        culprit_dims: Optional[List[int]] = None
    ):
        """
        Apply sparse correction to separate two embeddings.

        Only modifies specific dimensions (like StudSar's selective updates).
        """
        if culprit_dims is None:
            max_dims = int(self.hidden_dim * self.max_correction_pct)
            culprit_dims = self.identify_culprit_dimensions(
                token_id_a, token_id_b, top_k=max_dims
            )

        direction = self.compute_separation_direction(token_id_a, token_id_b)

        # Only modify culprit dimensions
        with torch.no_grad():
            for dim in culprit_dims:
                self.embedding_layer.weight[token_id_a, dim] += alpha * direction[dim]
                self.embedding_layer.weight[token_id_b, dim] -= alpha * direction[dim]

    def binary_search_alpha(
        self,
        token_a: str,
        token_b: str,
        context_template: str,
        max_iterations: int = 20,
        tolerance: float = 0.001
    ) -> Tuple[float, float]:
        """
        Binary search for optimal separation strength.

        Args:
            token_a: First token text (e.g., "dantrolene")
            token_b: Second token text (e.g., "cooling")
            context_template: Template with {} for token (e.g., "{} for hyperthermia")

        Returns:
            (optimal_alpha, achieved_similarity)
        """
        token_id_a = self.tokenizer.encode(token_a, add_special_tokens=False)[0]
        token_id_b = self.tokenizer.encode(token_b, add_special_tokens=False)[0]

        # Store original for rollback
        original_a = self.embedding_layer.weight[token_id_a].clone()
        original_b = self.embedding_layer.weight[token_id_b].clone()

        culprit_dims = self.identify_culprit_dimensions(token_id_a, token_id_b)

        min_alpha = 0.0
        max_alpha = 1.0

        for _ in range(max_iterations):
            alpha = (min_alpha + max_alpha) / 2

            # Reset to original
            with torch.no_grad():
                self.embedding_layer.weight[token_id_a] = original_a.clone()
                self.embedding_layer.weight[token_id_b] = original_b.clone()

            # Apply correction
            self.apply_sparse_correction(
                token_id_a, token_id_b, alpha, culprit_dims
            )

            # Measure new similarity
            context_a = context_template.format(token_a)
            context_b = context_template.format(token_b)
            _, similarity = self.detector.detect_sink(context_a, context_b)

            if abs(similarity - self.detector.separation_target) < tolerance:
                return alpha, similarity
            elif similarity > self.detector.separation_target:
                min_alpha = alpha  # Need more separation
            else:
                max_alpha = alpha  # Too much separation

        return alpha, similarity

    def validate_correction(
        self,
        test_cases: List[Tuple[str, str]]  # List of (prompt, expected_substring)
    ) -> Tuple[bool, float]:
        """
        Validate that correction doesn't break other functionality.

        Like StudSar's reputation scoring - tracks correction quality.
        """
        passed = 0
        total = len(test_cases)

        for prompt, expected in test_cases:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if expected.lower() in response.lower():
                passed += 1

        pass_rate = passed / total if total > 0 else 0.0
        validation_passed = pass_rate >= 1.0  # Require 100% for critical tests

        return validation_passed, pass_rate

    def run_correction(
        self,
        token_a: str,
        token_b: str,
        context_template: str,
        validation_tests: List[Tuple[str, str]]
    ) -> CorrectionRecord:
        """
        Full correction workflow - analogous to StudSar's run_dream_mode().

        1. Detect sink
        2. Search for optimal alpha
        3. Apply correction
        4. Validate
        5. Record or rollback
        """
        token_id_a = self.tokenizer.encode(token_a, add_special_tokens=False)[0]
        token_id_b = self.tokenizer.encode(token_b, add_special_tokens=False)[0]

        # Store originals for potential rollback
        original_a = self.embedding_layer.weight[token_id_a].clone()
        original_b = self.embedding_layer.weight[token_id_b].clone()

        # Measure initial similarity
        context_a = context_template.format(token_a)
        context_b = context_template.format(token_b)
        is_sink, similarity_before = self.detector.detect_sink(context_a, context_b)

        if not is_sink:
            print(f"No semantic sink detected (similarity={similarity_before:.4f})")
            return None

        print(f"Semantic sink detected: {token_a} <-> {token_b}")
        print(f"Similarity before: {similarity_before:.4f}")

        # Find optimal alpha
        culprit_dims = self.identify_culprit_dimensions(token_id_a, token_id_b)
        optimal_alpha, similarity_after = self.binary_search_alpha(
            token_a, token_b, context_template
        )

        print(f"Optimal alpha: {optimal_alpha:.4f}")
        print(f"Similarity after: {similarity_after:.4f}")

        # Validate
        validation_passed, pass_rate = self.validate_correction(validation_tests)

        print(f"Validation: {'PASSED' if validation_passed else 'FAILED'}")
        print(f"Pass rate: {pass_rate:.1%}")

        if not validation_passed:
            # Rollback
            with torch.no_grad():
                self.embedding_layer.weight[token_id_a] = original_a
                self.embedding_layer.weight[token_id_b] = original_b

            self.correction_stats['failed_validations'] += 1
            self.correction_stats['rollbacks'] += 1

            print("Correction rolled back due to validation failure")
            return None

        # Record successful correction
        record = CorrectionRecord(
            token_id=token_id_a,  # Primary token
            token_text=token_a,
            original_embedding=original_a,
            corrected_embedding=self.embedding_layer.weight[token_id_a].clone(),
            correction_timestamp=datetime.now().isoformat(),
            similarity_before=similarity_before,
            similarity_after=similarity_after,
            dimensions_modified=culprit_dims,
            alpha_used=optimal_alpha,
            validation_passed=True,
            confidence_score=pass_rate  # Like StudSar's reputation
        )

        self.corrections[token_id_a] = record
        self.correction_stats['total_corrections'] += 1
        self.correction_stats['successful_validations'] += 1

        return record

    def rollback_correction(self, token_id: int):
        """Rollback a specific correction to original embedding."""
        if token_id in self.corrections:
            record = self.corrections[token_id]
            with torch.no_grad():
                self.embedding_layer.weight[token_id] = record.original_embedding
            del self.corrections[token_id]
            self.correction_stats['rollbacks'] += 1
            print(f"Rolled back correction for token {token_id}")

    def rollback_all(self):
        """Rollback all corrections to original embeddings."""
        with torch.no_grad():
            self.embedding_layer.weight.copy_(self.original_embeddings)
        self.corrections.clear()
        print("All corrections rolled back")

    def get_statistics(self) -> Dict:
        """
        Return correction statistics.
        Analogous to StudSar's analyze_marker_statistics().
        """
        return {
            **self.correction_stats,
            'active_corrections': len(self.corrections),
            'corrections': {
                token_id: {
                    'token': record.token_text,
                    'similarity_before': record.similarity_before,
                    'similarity_after': record.similarity_after,
                    'confidence': record.confidence_score,
                    'timestamp': record.correction_timestamp
                }
                for token_id, record in self.corrections.items()
            }
        }

    def save_checkpoint(self, path: str):
        """
        Save correction state for later restoration.
        Like StudSar's torch.save pattern.
        """
        checkpoint = {
            'model_state': self.embedding_layer.weight.clone(),
            'original_embeddings': self.original_embeddings.clone(),
            'corrections': {
                k: {
                    'token_id': v.token_id,
                    'token_text': v.token_text,
                    'similarity_before': v.similarity_before,
                    'similarity_after': v.similarity_after,
                    'dimensions_modified': v.dimensions_modified,
                    'alpha_used': v.alpha_used,
                    'confidence_score': v.confidence_score,
                    'timestamp': v.correction_timestamp
                }
                for k, v in self.corrections.items()
            },
            'stats': self.correction_stats
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load correction state from checkpoint."""
        checkpoint = torch.load(path)

        with torch.no_grad():
            self.embedding_layer.weight.copy_(checkpoint['model_state'])
            self.original_embeddings.copy_(checkpoint['original_embeddings'])

        self.correction_stats = checkpoint['stats']
        # Note: Full CorrectionRecords not restored (would need original tensors)

        print(f"Checkpoint loaded from {path}")
        print(f"Active corrections: {len(checkpoint['corrections'])}")


# Example usage
def example_workflow():
    """
    Example workflow for geometric correction.

    NOTE: This is proposed methodology - not yet validated.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load model
    model_name = "Qwen/Qwen2.5-7B"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.cuda()

    # Initialize detector and corrector
    detector = SemanticSinkDetector(model, tokenizer, layer=20)
    corrector = GeometricCorrector(model, tokenizer, detector)

    # Define the semantic sink to fix
    token_a = "dantrolene"
    token_b = "cooling"
    context_template = "{} for malignant hyperthermia treatment"

    # Define validation tests (must all pass)
    validation_tests = [
        ("What is the specific treatment for malignant hyperthermia?", "dantrolene"),
        ("What treats heat stroke?", "cooling"),
        ("Dantrolene mechanism of action", "calcium"),
        ("What treats neuroleptic malignant syndrome?", "dantrolene"),
    ]

    # Run correction
    record = corrector.run_correction(
        token_a, token_b, context_template, validation_tests
    )

    if record:
        print("\nCorrection successful!")
        print(f"Dimensions modified: {len(record.dimensions_modified)}")
        print(f"Alpha used: {record.alpha_used:.4f}")

        # Save checkpoint
        corrector.save_checkpoint("correction_checkpoint.pth")

        # Print statistics
        stats = corrector.get_statistics()
        print(f"\nStatistics: {json.dumps(stats, indent=2, default=str)}")


if __name__ == "__main__":
    print("Geometric Correction Module")
    print("Status: PROPOSED - requires validation")
    print("\nRun example_workflow() to test with a loaded model")
