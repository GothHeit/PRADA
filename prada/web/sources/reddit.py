"""Reddit: threads found through general search (`site:reddit.com`), read through Reddit's
public `.json` view of each thread (low volume, no account; new API keys need approval)."""

import json
import re

from prada.web import fetch, read
from prada.web.sources.base import Hit, Page, Source, SourceError

_THREAD = re.compile(
    r"^https?://(?:www\.|old\.|new\.)?reddit\.com/r/(?P<sub>[\w-]+)/comments/(?P<id>[a-z0-9]+)",
    re.IGNORECASE,
)
COMMENTS = 12  # top comments read per thread


def thread_url(url: str) -> str | None:
    """The canonical https://www.reddit.com/r/<sub>/comments/<id>/ for a thread URL."""
    m = _THREAD.match(url)
    return f"https://www.reddit.com/r/{m['sub']}/comments/{m['id'].lower()}/" if m else None


class RedditSource(Source):
    name = "reddit"
    ttl = 6 * 3600

    def __init__(self, general: Source, get=fetch.get):
        self._general = general
        self._get = get

    def search(self, query: str, limit: int, intent: str = "discussion") -> list[Hit]:
        hits, seen = [], set()
        for h in self._general.search(f"site:reddit.com {query}", limit * 3, "facts"):
            url = thread_url(h.url)
            if url and url not in seen:
                seen.add(url)
                sub = _THREAD.match(url)["sub"]
                hits.append(
                    h.model_copy(
                        update={"source": self.name, "url": url, "extra": {"subreddit": f"r/{sub}"}}
                    )
                )
        return hits[:limit]

    def handles(self, url: str) -> bool:
        return thread_url(url) is not None

    def read(self, url: str, query: str) -> Page:
        url = thread_url(url) or url
        api = f"{url}.json?limit=40&depth=1&sort=top&raw_json=1"
        try:
            data = json.loads(self._get(api, accept=fetch.JSON_TYPES).text())
            post = data[0]["data"]["children"][0]["data"]
            comments = [c["data"] for c in data[1]["data"]["children"] if c.get("kind") == "t1"]
        except fetch.FetchError as e:
            raise SourceError(f"Reddit didn't let me read that thread ({e})") from e
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise SourceError("Reddit sent a thread I couldn't read") from e
        comments.sort(key=lambda c: -int(c.get("score") or 0))
        parts = [f"r/{post.get('subreddit', '?')}: {post.get('title', '')}"]
        if body := str(post.get("selftext") or "").strip():
            parts.append(body)
        for c in comments[:COMMENTS]:
            if body := str(c.get("body") or "").strip():
                parts.append(f"Comment ({c.get('score', 0)} points): {body}")
        text = read.excerpt("\n\n".join(parts), query)
        return Page(url=url, title=str(post.get("title") or url), text=text)
