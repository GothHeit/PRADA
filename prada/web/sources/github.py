"""GitHub through its official REST API: repository search, or issue search when the query
is about a problem. Anonymous by default (10 searches a minute); `[web] github_token` (a
read-only fine-grained token) raises that to 30."""

import base64
import json
import re
import urllib.parse

from prada.web import fetch, read
from prada.web.sources.base import Hit, Page, Source, SourceError

API = "https://api.github.com"
_REPO = re.compile(r"^https://github\.com/(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+)/?$")
_ISSUE = re.compile(
    r"^https://github\.com/(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+)/(?:issues|pull)/(?P<n>\d+)"
)
_PROBLEM = re.compile(
    r"\b(?:issues?|bugs?|errors?|crash(?:es)?|exception|fails?|failing|broken|regression)\b",
    re.IGNORECASE,
)
COMMENTS = 10


class GitHubSource(Source):
    name = "github"
    ttl = 24 * 3600

    def __init__(self, token: str = "", get=fetch.get):
        self._token = token
        self._get = get

    def _json(self, path: str):
        headers = {"X-GitHub-Api-Version": "2022-11-28"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        try:
            r = self._get(f"{API}{path}", accept=fetch.JSON_TYPES, headers=headers)

            return json.loads(r.text())
        except fetch.FetchError as e:
            if "HTTP 403" in str(e) or "HTTP 429" in str(e):
                raise SourceError("GitHub says I've searched too much for now") from e
            raise SourceError(f"GitHub didn't answer ({e})") from e
        except ValueError as e:
            raise SourceError("GitHub sent something I couldn't read") from e

    def search(self, query: str, limit: int, intent: str = "code") -> list[Hit]:
        if _PROBLEM.search(query):
            return self._issues(query, limit)
        q = urllib.parse.urlencode({"q": query, "per_page": limit})
        hits = []
        for r in self._json(f"/search/repositories?{q}").get("items", [])[:limit]:
            facts = [f"{r.get('stargazers_count', 0):,} stars"]
            if r.get("language"):
                facts.append(str(r["language"]))
            if r.get("pushed_at"):
                facts.append(f"last updated {str(r['pushed_at'])[:10]}")
            if r.get("archived"):
                facts.append("archived")
            hits.append(
                Hit(
                    source=self.name,
                    title=str(r.get("full_name", "")),
                    url=str(r.get("html_url", "")),
                    snippet=f"{r.get('description') or ''} ({', '.join(facts)})".strip(),
                    extra={"stars": r.get("stargazers_count", 0)},
                )
            )
        return hits

    def _issues(self, query: str, limit: int) -> list[Hit]:
        q = urllib.parse.urlencode({"q": f"{query} is:issue", "per_page": limit})
        hits = []
        for r in self._json(f"/search/issues?{q}").get("items", [])[:limit]:
            repo = str(r.get("repository_url", "")).removeprefix(f"{API}/repos/")
            body = re.sub(r"\s+", " ", str(r.get("body") or ""))[:300]
            hits.append(
                Hit(
                    source=self.name,
                    title=str(r.get("title", "")),
                    url=str(r.get("html_url", "")),
                    snippet=f"{repo} issue, {r.get('state')}, {r.get('comments', 0)} comments: "
                    + body,
                    extra={"repo": repo},
                )
            )
        return hits

    def handles(self, url: str) -> bool:
        return bool(_REPO.match(url) or _ISSUE.match(url))

    def read(self, url: str, query: str) -> Page:
        if m := _ISSUE.match(url):
            base = f"/repos/{m['owner']}/{m['repo']}/issues/{m['n']}"
            issue = self._json(base)
            comments = self._json(f"{base}/comments?per_page={COMMENTS}")
            parts = [str(issue.get("title", "")), str(issue.get("body") or "")]
            parts += [f"Comment: {c.get('body') or ''}" for c in comments if isinstance(c, dict)]
            title = str(issue.get("title") or url)
        elif m := _REPO.match(url):
            readme = self._json(f"/repos/{m['owner']}/{m['repo']}/readme")
            try:
                text = base64.b64decode(readme.get("content", "")).decode("utf-8", "replace")
            except (ValueError, TypeError) as e:
                raise SourceError("GitHub sent a README I couldn't read") from e
            parts, title = [text], f"{m['owner']}/{m['repo']} README"
        else:
            raise SourceError("that isn't a GitHub repository or issue")
        return Page(url=url, title=title, text=read.excerpt("\n\n".join(parts).strip(), query))
