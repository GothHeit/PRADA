"""General web search through your own SearXNG (a metasearch server on 127.0.0.1:8888).

SearXNG asks DuckDuckGo, Bing, Wikipedia and others and merges what they say; queries go from
your home IP straight to them, with no account. It is the one address that may be local, so
it is reached directly (no proxy) and never through the page fetcher's guard.
"""

import json
import urllib.error
import urllib.parse
import urllib.request

from prada.web.sources.base import Hit, Source, SourceError

TIMEOUT = 10.0  # SearXNG waits on its engines for a few seconds itself

_local = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _get_json(url: str):
    with _local.open(url, timeout=TIMEOUT) as r:
        return json.load(r)


class SearxngSource(Source):
    name = "web"
    ttl = 6 * 3600

    def __init__(self, url: str = "http://127.0.0.1:8888", get_json=_get_json):
        self.url = url.rstrip("/")
        self._get_json = get_json

    def search(self, query: str, limit: int, intent: str = "facts") -> list[Hit]:
        params = {"q": query, "format": "json", "safesearch": "1"}
        if intent == "news":
            params["categories"] = "news"
        try:
            data = self._get_json(f"{self.url}/search?{urllib.parse.urlencode(params)}")
        except urllib.error.HTTPError as e:
            if e.code == 403:
                raise SourceError("my search server has JSON turned off") from e
            raise SourceError(f"my search server answered HTTP {e.code}") from e
        except (urllib.error.URLError, OSError) as e:
            raise SourceError("my search server (SearXNG) isn't running") from e
        except ValueError as e:
            raise SourceError("my search server sent something I couldn't read") from e
        hits, seen = [], set()
        for r in data.get("results", []) if isinstance(data, dict) else []:
            url = str(r.get("url") or "")
            if not url.startswith(("https://", "http://")) or url in seen:
                continue
            seen.add(url)
            hits.append(
                Hit(
                    source=self.name,
                    title=str(r.get("title") or url),
                    url=url,
                    snippet=str(r.get("content") or ""),
                    extra={"engine": str(r.get("engine") or ""), "date": r.get("publishedDate")},
                )
            )
        if not hits and (down := data.get("unresponsive_engines")):
            names = ", ".join(str(e[0]) for e in down[:3] if e)
            raise SourceError(f"the search engines didn't answer ({names})")
        return hits[:limit]
