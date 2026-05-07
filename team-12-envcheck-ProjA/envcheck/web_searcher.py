"""
EnvCheck Web Searcher — DuckDuckGo-based search for API documentation.

Searches for breaking changes, migration guides, and changelogs to
supplement the local knowledge base when information is missing or outdated.

Usage:
    from envcheck.web_searcher import WebSearcher

    searcher = WebSearcher()
    results = searcher.search_api_docs("numpy", "trapz removed replacement")
    results = searcher.search_breaking_changes("pandas", "2.0")
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SearchResult:
    """A single web search result."""
    title: str
    url: str
    snippet: str


class WebSearcher:
    """DuckDuckGo-based web search focused on API documentation and breaking changes."""

    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def search_api_docs(
        self,
        library: str,
        query: str,
        max_results: Optional[int] = None,
    ) -> list[SearchResult]:
        """Search for API documentation and breaking changes.

        Args:
            library: The library name (e.g., "numpy").
            query: Additional search terms.
            max_results: Override default max results.

        Returns:
            List of SearchResult with title, url, and snippet.
        """
        from duckduckgo_search import DDGS

        limit = max_results or self.max_results
        full_query = f"{library} python {query}"

        try:
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(full_query, max_results=limit))
        except Exception as e:
            return [SearchResult(
                title="Search Error",
                url="",
                snippet=f"DuckDuckGo search failed: {e}",
            )]

        results = []
        for r in raw_results:
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("href", r.get("link", "")),
                snippet=r.get("body", r.get("snippet", "")),
            ))
        return results

    def search_breaking_changes(
        self,
        library: str,
        version: str,
        max_results: Optional[int] = None,
    ) -> list[SearchResult]:
        """Search specifically for breaking changes in a library version."""
        query = f"breaking changes migration guide version {version}"
        return self.search_api_docs(library, query, max_results)

    def search_migration_guide(
        self,
        library: str,
        old_api: str,
        max_results: Optional[int] = None,
    ) -> list[SearchResult]:
        """Search for migration guidance for a specific deprecated API."""
        query = f"{old_api} deprecated removed replacement"
        return self.search_api_docs(library, query, max_results)

    def to_dict_list(self, results: list[SearchResult]) -> list[dict]:
        """Convert results to serializable dicts (for MCP tool output)."""
        return [
            {"title": r.title, "url": r.url, "snippet": r.snippet}
            for r in results
        ]
