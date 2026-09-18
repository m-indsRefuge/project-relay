"""Small live web-search adapter for Relay M3."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from ddgs import DDGS


class WebSearchError(RuntimeError):
    """Raised when live web search fails."""


class WebSearch(Protocol):
    def search(self, query: str, limit: int = 3) -> list[dict]:
        """Return normalized web search results."""


@dataclass(frozen=True)
class DDGSWebSearch:
    timeout_seconds: int = 15
    region: str = "us-en"
    backend: str = "auto"

    def search(self, query: str, limit: int = 3) -> list[dict]:
        if not query.strip():
            raise WebSearchError("Web query cannot be empty.")
        if limit < 1:
            raise WebSearchError("Web result limit must be at least one.")

        try:
            raw_results = DDGS(timeout=self.timeout_seconds).text(
                query,
                region=self.region,
                safesearch="moderate",
                max_results=limit,
                backend=self.backend,
            )
        except Exception as exc:
            raise WebSearchError(f"Live web search failed: {exc}") from exc

        retrieved_at = datetime.now(UTC).isoformat()
        normalized: list[dict] = []

        for index, item in enumerate(raw_results[:limit], start=1):
            title = item.get("title")
            href = item.get("href")
            body = item.get("body")

            if not isinstance(title, str) or not isinstance(href, str):
                continue

            normalized.append(
                {
                    "source_id": f"web:{index:03d}",
                    "kind": "web",
                    "title": title,
                    "source": href,
                    "snippet": body if isinstance(body, str) else "",
                    "retrieved_at": retrieved_at,
                }
            )

        if not normalized:
            raise WebSearchError("Live web search returned no usable results.")

        return normalized
