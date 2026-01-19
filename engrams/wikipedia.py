"""
Wikipedia Integration

Utilities for fetching Wikipedia articles and converting them
to engrams.
"""

import wikipediaapi

from .extractor import Engram, EngramExtractor, ExtractionConfig
from .storage import EngramStore


class WikipediaEngramBuilder:
    """
    Build engrams from Wikipedia articles.

    Fetches articles from Wikipedia and processes them through
    the engram extraction pipeline.
    """

    def __init__(
        self,
        extractor: EngramExtractor | None = None,
        store: EngramStore | None = None,
        language: str = "en",
    ):
        """
        Initialize the Wikipedia engram builder.

        Args:
            extractor: EngramExtractor to use (creates default if None)
            store: EngramStore to save engrams (creates default if None)
            language: Wikipedia language code
        """
        self.extractor = extractor or EngramExtractor()
        self.store = store or EngramStore()
        self.wiki = wikipediaapi.Wikipedia(
            user_agent="Engrams Research (https://github.com/MikeyBeez/engrams)",
            language=language,
        )

    def fetch_article(self, title: str) -> str | None:
        """
        Fetch a Wikipedia article by title.

        Args:
            title: Article title (e.g., "Abraham Lincoln")

        Returns:
            Article text or None if not found
        """
        page = self.wiki.page(title)
        if not page.exists():
            print(f"Wikipedia article not found: {title}")
            return None
        return page.text

    def build_engram(
        self,
        title: str,
        save: bool = True,
        force: bool = False,
    ) -> Engram | None:
        """
        Build an engram from a Wikipedia article.

        Args:
            title: Wikipedia article title
            save: Whether to save to the store
            force: Rebuild even if already exists in store

        Returns:
            The created Engram or None if article not found
        """
        # Check if already exists
        if not force and title in self.store:
            print(f"Engram already exists for '{title}', loading from store")
            return self.store.retrieve(title)

        # Fetch article
        text = self.fetch_article(title)
        if text is None:
            return None

        print(f"Building engram for '{title}' ({len(text)} chars)")

        # Extract engram
        metadata = {
            "source": "wikipedia",
            "title": title,
            "char_length": len(text),
        }
        engram = self.extractor.extract(text, metadata=metadata)

        print(f"Created: {engram}")

        # Save if requested
        if save:
            engram_id = self.store.store(engram, title)
            print(f"Saved with ID: {engram_id}")

        return engram

    def build_batch(
        self,
        titles: list[str],
        save: bool = True,
        skip_existing: bool = True,
    ) -> dict[str, Engram | None]:
        """
        Build engrams for multiple Wikipedia articles.

        Args:
            titles: List of article titles
            save: Whether to save to store
            skip_existing: Skip articles already in store

        Returns:
            Dict mapping title to Engram (or None if failed)
        """
        results = {}
        for title in titles:
            if skip_existing and title in self.store:
                print(f"Skipping '{title}' (already exists)")
                results[title] = self.store.retrieve(title)
            else:
                results[title] = self.build_engram(title, save=save)
        return results
