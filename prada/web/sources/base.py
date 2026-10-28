"""The interface every search source implements, and what it returns."""

from typing import ClassVar

from pydantic import BaseModel, Field


class SourceError(Exception):
    """The source didn't answer; the message says why, in words she can repeat."""


class Hit(BaseModel):
    """One search result, as the source gave it (strangers' text: never trusted)."""

    source: str  # "web", "reddit", "github"
    title: str
    url: str
    snippet: str = ""
    extra: dict = Field(default_factory=dict)  # stars, subreddit, engine...


class Page(BaseModel):
    """The readable text of one result, already cut to fit the prompt."""

    url: str
    title: str
    text: str


class Source:
    name: ClassVar[str]
    ttl: ClassVar[int] = 6 * 3600  # seconds its results stay cached

    def search(self, query: str, limit: int, intent: str = "facts") -> list[Hit]:
        raise NotImplementedError

    def handles(self, url: str) -> bool:
        """True if `read(url)` knows this kind of URL better than a plain page read."""
        return False

    def read(self, url: str, query: str) -> Page:
        raise NotImplementedError
