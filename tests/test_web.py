"""Tests for the M3 live web-search boundary."""

import pytest

import relay.tools.web as web_module
from relay.tools.web import DDGSWebSearch, WebSearchError


class FakeDDGS:
    results: list[dict] = []

    def __init__(self, *, timeout: int) -> None:
        self.timeout = timeout

    def text(self, *args, **kwargs) -> list[dict]:
        del args, kwargs
        return list(self.results)


def test_web_search_normalizes_usable_result(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeDDGS.results = [
        {
            "title": "Example result",
            "href": "https://example.test/result",
            "body": "Example snippet.",
        }
    ]
    monkeypatch.setattr(web_module, "DDGS", FakeDDGS)

    results = DDGSWebSearch().search("example query", limit=1)

    assert len(results) == 1
    assert results[0]["source_id"] == "web:001"
    assert results[0]["kind"] == "web"
    assert results[0]["title"] == "Example result"
    assert results[0]["source"] == "https://example.test/result"
    assert results[0]["snippet"] == "Example snippet."
    assert results[0]["retrieved_at"]


def test_web_search_fails_when_no_usable_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeDDGS.results = []
    monkeypatch.setattr(web_module, "DDGS", FakeDDGS)

    with pytest.raises(WebSearchError, match="no usable results"):
        DDGSWebSearch().search("example query")
