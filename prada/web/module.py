"""The web module: she searches the internet, but only when the user asks her to.

Flow: `handle` plans the search from the user's words alone (prada.web.plan) and returns one
`SearchWeb` action; the node runs it with `SearchExecutor` (SAFE: asking was the yes) and
logs it in the action log, so "I searched" is only ever said after a search really ran. The
executor asks each source, reads up to `read_pages` results through the guarded fetcher and
lets the model answer from them (prada.web.answer). Page text reaches only that last step,
which can't return actions, so a page can make her say something wrong, never do something.

If the router sends her a question she wasn't asked to look up, she offers to search
instead, and a plain yes runs it. Sources come from `[web]` in the config file; under
PRADA_FAKE_LLM they are canned and offline, and nothing is cached.
"""

import json
import urllib.parse
from typing import Literal

from prada import actions, config, llm, persona, router
from prada.actions import Action, ActionFailed, Executor
from prada.module import Context, Module, Reply, Request
from prada.web import answer, fetch, plan, read
from prada.web.cache import MIGRATIONS, Cache, key
from prada.web.sources.base import Hit, Page, Source, SourceError
from prada.web.sources.github import GitHubSource
from prada.web.sources.reddit import RedditSource
from prada.web.sources.searxng import SearxngSource

HITS_PER_SOURCE = 5
NEWS_TTL = 30 * 60
PAGE_TTL = 6 * 3600


class SearchWeb(Action):
    kind: Literal["web.search"] = "web.search"
    query: str  # exactly what leaves the house
    sources: list[str]
    intent: str = "facts"
    read_pages: int = 0
    question: str = ""  # the user's words, for the answer

    def describe(self) -> str:
        names = [answer.NAMES[s] if s != "web" else "the web" for s in self.sources]
        where = (
            " and ".join(names) if len(names) < 3 else ", ".join(names[:-1]) + " and " + names[-1]
        )
        return f"search {where} for “{self.query}”"


class SearchExecutor(Executor):
    action = SearchWeb

    def __init__(self, module: "WebModule"):
        self._module = module

    def filler(self, action: SearchWeb) -> str:
        return persona.looking_up()

    def run(self, action: SearchWeb, ctx: Context) -> dict:
        sources = self._module.sources()
        cache = Cache(None if llm.fake_enabled() else ctx.db)
        hits, failed = [], {}
        for name in action.sources:
            try:
                hits += self._search(sources[name], action, cache)
            except (SourceError, fetch.FetchError) as e:
                failed[name] = str(e)
        if failed and len(failed) == len(action.sources):
            raise ActionFailed("; ".join(failed.values()))
        hits = _interleave(hits, action.sources)
        pages = self._read(hits, action, sources, cache)
        result = answer.compose(
            ctx.chat, action.question or action.query, ctx.turns, hits, pages, failed
        )
        shown = [hits[i - 1] for i in result.cited]
        return {
            "reply": result.reply.strip(),
            "query": action.query,
            "searched": action.sources,
            "failed": failed,
            "sources": [
                {"id": i, "title": h.title, "url": h.url, "source": h.source}
                for i, h in zip(result.cited, shown, strict=True)
            ],
        }

    def _search(self, source: Source, action: SearchWeb, cache: Cache) -> list[Hit]:
        k = key(source.name, "search", action.intent, action.query)
        ttl = NEWS_TTL if action.intent == "news" else source.ttl
        if (cached := cache.get(k, ttl)) is not None:
            return [Hit.model_validate(h) for h in cached]
        found = source.search(action.query, HITS_PER_SOURCE, action.intent)
        cache.put(k, [h.model_dump() for h in found])
        return found

    def _read(self, hits, action: SearchWeb, sources: dict, cache: Cache) -> dict[str, Page]:
        pages = {}
        for hit in hits:
            if len(pages) >= action.read_pages:
                break
            k = key("page", action.query, hit.url)  # the text is cut for this query
            if (cached := cache.get(k, PAGE_TTL)) is not None:
                pages[hit.url] = Page.model_validate(cached)
                continue
            reader = next((s for s in sources.values() if s.handles(hit.url)), None)
            try:
                page = (
                    reader.read(hit.url, action.query)
                    if reader
                    else self._module.page(hit.url, action.query)
                )
            except (SourceError, fetch.FetchError):
                continue  # the snippet still counts; she just didn't read that one
            pages[hit.url] = page
            cache.put(k, page.model_dump())
        return pages

    def check(self, action: SearchWeb, result: dict, ctx: Context) -> None:
        if not result["reply"]:
            raise ActionFailed("I couldn't put the results into words")

    def done(self, action: SearchWeb, result: dict) -> str:
        return result["reply"]


def _interleave(hits: list[Hit], order: list[str]) -> list[Hit]:
    """One result from each source in turn, so the best of each comes first."""
    by_source = [[h for h in hits if h.source == s] for s in order]
    out = []
    for i in range(max((len(b) for b in by_source), default=0)):
        out += [b[i] for b in by_source if i < len(b)]
    return out


def read_page(url: str, query: str, get=fetch.get) -> Page:
    """Read a general web page (robots.txt respected), cut to the parts about `query`."""
    url = "https://" + url.removeprefix("http://") if url.startswith("http://") else url
    if not fetch.robots_allows(url, get):
        raise SourceError("that site asks robots not to read it")
    r = get(url)
    if r.content_type == "text/plain":
        title, text = url, r.text()
    else:
        title, text = read.page_text(r.text())
    return Page(url=r.url, title=title or url, text=read.excerpt(text, query))


class FakeSource(Source):
    """Canned and offline, for PRADA_FAKE_LLM."""

    def __init__(self, name: str):
        self.name = name

    def search(self, query: str, limit: int, intent: str = "facts") -> list[Hit]:
        url = f"https://example.com/{self.name}/{urllib.parse.quote(query)}"
        return [
            Hit(
                source=self.name,
                title=f"(fake {self.name}) {query}",
                url=url,
                snippet="Canned offline result.",
            )
        ]

    def handles(self, url: str) -> bool:
        return url.startswith("https://example.com/")

    def read(self, url: str, query: str) -> Page:
        return Page(url=url, title=url, text=f"Canned offline page about {query}.")


def fake_chat(messages, schema, model=None) -> str:
    if schema.get("title") == "WebPlan":
        return plan.fake_plan(messages)
    return answer.fake_answer(messages)


class WebModule(Module):
    name = "web"
    description = (
        "Searching the internet, only when the user asks for it: 'search ...', 'look up ...', "
        "'procura ...', prices, what Reddit or GitHub say."
    )
    migrations = MIGRATIONS
    fake_chat = staticmethod(fake_chat)
    memory_note = "[from a web search]"

    def __init__(self, sources: dict[str, Source] | None = None, get=fetch.get, settings=None):
        """`sources`, `get` and `settings` replace the configured ones (tests)."""
        self._sources = sources
        self._get = get
        self._settings = settings
        self.executors = [SearchExecutor(self)]

    def settings(self) -> config.WebSettings:
        return self._settings or config.web_settings()

    def sources(self) -> dict[str, Source]:
        if self._sources is not None:
            return self._sources
        if llm.fake_enabled():
            return {name: FakeSource(name) for name in config.WEB_SOURCES}
        s = self.settings()
        general = SearxngSource(s.searxng_url)
        return {
            "web": general,
            "reddit": RedditSource(general, self._get),
            "github": GitHubSource(s.github_token, self._get),
        }

    def page(self, url: str, query: str) -> Page:
        return read_page(url, query, self._get)

    def claims(self, text: str, state: dict) -> bool:
        # A yes or no only answers her offer to search, never anything else said to her.
        return plan.asked_to_search(text) or ("offer" in state and actions.answer(text) is not None)

    def accepts(self, text: str) -> bool:
        # The internet can't know what the user said or planned before; her memory can.
        return plan.asked_to_search(text) or not router.about_own_past(text)

    def handle(self, request: Request, ctx: Context) -> Reply:
        text = request.text.strip()
        offered = ctx.state.pop("offer", None)
        yes = actions.answer(text)
        if yes is not None:
            if not offered:
                return Reply(module=self.name, text=persona.search_what())
            if not yes:
                return Reply(module=self.name, text=persona.no_search())
            text = offered
        elif not plan.asked_to_search(text):
            ctx.state["offer"] = text
            return Reply(module=self.name, text=persona.offer_search())
        settings = self.settings()
        earlier = [m["content"] for m in ctx.turns if m.get("role") == "user"][-2:]
        try:
            p = plan.plan(text, earlier, ctx.chat)
        except ValueError:  # the model's plan didn't fit the schema: search the plain words
            p = plan.WebPlan.model_validate_json(plan.fake_plan([{"content": text}]))
        if not p.query:
            return Reply(module=self.name, text=persona.search_what())
        chosen = plan.choose_sources(text, p, settings.sources)
        if not chosen:
            return Reply(module=self.name, text="Web search is turned off in my settings.")
        search = SearchWeb(
            query=p.query,
            sources=chosen,
            intent=p.intent,
            read_pages=min(p.read_pages, settings.max_pages),
            question=text,
        )
        return Reply(
            module=self.name, data={"plan": json.loads(p.model_dump_json())}, actions=[search]
        )
