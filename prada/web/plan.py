"""The search planner: one structured call turns the user's words into a `WebPlan`.

Only the user's own words go in (this message and their last two), never page text or
memory, so a web page can't start another search and nothing she remembers leaves the house.
Rules the model can't override: a site the user names always wins ("on Reddit", "not
GitHub"), code questions include GitHub, opinion questions include Reddit, and personal
details (emails, phone numbers, CPF, CEP) are stripped from the query before it is sent.
"""

import json
import re
from typing import Literal

from pydantic import BaseModel, Field

from prada.module import Chat

Intent = Literal["facts", "news", "discussion", "code", "products"]
SourceName = Literal["web", "reddit", "github"]
MAX_QUERY = 120  # characters

_PROMPT = """You plan a web search for PRADA, a personal assistant. The user asked her to look something up. Read their message and fill the JSON object matching the provided schema:
- reasoning: one short sentence on what they want to find.
- intent: facts (a fact or an explanation), news (recent events), discussion (opinions, recommendations, people's experiences), code (software, libraries, programming problems), products (things to buy, prices).
- sources: web (general search; fits almost anything), reddit (people's opinions and experiences), github (software projects, code, bugs in them). Pick every one that fits.
- query: a short search-engine query, keywords only, no filler words, in the language that finds it best (usually the user's language; English for code). Never put the user's name, email, phone number, address or other personal details in it.
- read_pages: how many results to read in full. 0 when titles and snippets are enough (a quick fact); 1 to 3 when the answer needs details, steps or opinions."""


class WebPlan(BaseModel):
    reasoning: str = Field(..., description="One short sentence: what the user wants to find")
    intent: Intent
    sources: list[SourceName] = Field(..., description="Where to search")
    query: str = Field(..., description="Short search-engine query, keywords only")
    read_pages: int = Field(..., description="Results to read in full, 0 to 3")


_ASKED = re.compile(
    r"\b(?:search(?:ing)?|look(?:ing)?\s+(?:it\s+|that\s+|this\s+)?(?:up|for|into)|"
    r"google\s+(?:it|that|this|for)|find\s+(?:out|me|\w+\s+online)|(?:check|look)\s+online|"
    r"browse\s+(?:for|the\s+web)|on(?:\s+the)?\s+(?:internet|web|reddit|github)|"
    r"(?:reddit|git\s?hub)\s+(?:say|says|think|thinks|recommends?)|"
    r"how\s+much\s+(?:does|do|is|are)\b.*\bcost|price\s+(?:of|for)|latest\s+news|"
    r"procur\w*|pesquis\w*|busc\w*|na\s+internet|no\s+(?:reddit|github)|"
    r"quanto\s+custa|pre[çc]o\s+d[eoa]s?|cherche\w*|recherche\w*)\b",
    re.IGNORECASE,
)
_SITES = {
    "reddit": re.compile(r"\breddit\b|(?:^|\s)r/\w+", re.IGNORECASE),
    "github": re.compile(r"\bgit\s?hub\b", re.IGNORECASE),
}
_EXCLUDED = re.compile(
    r"\b(?:no|not|without|except|skip|sem|nada\s+de|não\s+(?:no|use))\s+(?:on\s+|the\s+|o\s+)?"
    r"(reddit|git\s?hub)\b",
    re.IGNORECASE,
)
_BY_INTENT = {"code": "github", "discussion": "reddit"}

_PERSONAL = [
    re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),  # email
    re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b|\bcpf\W*\d{11}\b", re.IGNORECASE),  # CPF
    re.compile(r"\b\d{5}-\d{3}\b"),  # CEP
    re.compile(r"\+\d{1,3}[\s.-]?\(?\d{2,3}\)?[\s.-]?\d{4,5}[\s.-]?\d{4}\b"),  # phone
    re.compile(r"\(\d{2,3}\)\s?\d{4,5}[\s.-]?\d{4}\b|\b9\d{4}-\d{4}\b"),  # phone
]
_FILLER = re.compile(
    r"^(?:(?:hey\s+)?prada[,\s]+)?(?:(?:can|could|would)\s+you\s+|please\s+)*"
    r"(?:search(?:\s+(?:the\s+web|online|the\s+internet))?(?:\s+for)?|look\s+up|google|"
    r"procur[ae](?:\s+por)?|pesquis[ae](?:\s+sobre)?|busc[ae])\s+",
    re.IGNORECASE,
)


def asked_to_search(text: str) -> bool:
    """True when the user plainly asks her to look something up (she never searches
    otherwise, for now)."""
    return bool(_ASKED.search(text))


def redact(query: str) -> str:
    for pattern in _PERSONAL:
        query = pattern.sub(" ", query)
    return " ".join(query.split())[:MAX_QUERY].strip()


def fallback_query(text: str) -> str:
    return redact(_FILLER.sub("", text.strip()).rstrip("?.! "))


def choose_sources(text: str, plan: WebPlan, enabled) -> list[str]:
    """The sources to use: the sites the user named, else the model's pick plus the intent's
    natural source; never one they ruled out or one that is turned off."""
    excluded = {m.lower().replace(" ", "") for m in _EXCLUDED.findall(text)}
    named = [s for s, p in _SITES.items() if p.search(text) and s not in excluded]
    if named:
        chosen = named
    else:
        chosen = list(dict.fromkeys(plan.sources))
        if (extra := _BY_INTENT.get(plan.intent)) and extra not in chosen:
            chosen.append(extra)
    chosen = [s for s in chosen if s not in excluded and s in enabled]
    if not chosen and "web" in enabled:
        chosen = ["web"]
    return chosen


def plan(text: str, earlier: list[str], chat: Chat) -> WebPlan:
    """The model's plan for `text`, with a usable query (personal details removed)."""
    system = _PROMPT
    if earlier:
        system += "\n\nThe user's previous messages, for context only:\n" + "\n".join(
            f"- {m}" for m in earlier
        )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]
    result = WebPlan.model_validate_json(
        chat(messages=messages, schema=WebPlan.model_json_schema())
    )
    query = redact(result.query) or fallback_query(text)
    return result.model_copy(
        update={"query": query, "read_pages": max(0, min(3, result.read_pages))}
    )


def fake_plan(messages) -> str:
    text = messages[-1]["content"]
    sources = [s for s, p in _SITES.items() if p.search(text)] or ["web"]
    return json.dumps(
        {
            "reasoning": "fake planner",
            "intent": "facts",
            "sources": sources,
            "query": fallback_query(text),
            "read_pages": 1,
        }
    )
