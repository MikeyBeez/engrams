"""
Engrams: Learned Semantic Compression for LLMs

Extract dense representations from transformer hidden states to compress
knowledge (e.g., Wikipedia articles) into token-sized engrams that can
be injected into new prompts.
"""

__version__ = "0.1.0"

from .extractor import EngramExtractor
from .injector import EngramInjector
from .storage import EngramStore

__all__ = ["EngramExtractor", "EngramInjector", "EngramStore"]
