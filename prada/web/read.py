"""Web page HTML to plain text (stdlib html.parser), cut to what matters for a question.

`page_text(html)` keeps the title and the readable blocks (paragraphs, headings, list items,
table cells) and drops scripts, styles, navigation, headers, footers and forms. `excerpt(text,
query, limit)` keeps the paragraphs that share the most words with the query, in page order,
within `limit` characters, so a few pages still fit the model's context.
"""

import re
from html.parser import HTMLParser

PAGE_CHARS = 4000  # about 1,000 tokens per page

_SKIP = {
    "script",
    "style",
    "noscript",
    "svg",
    "nav",
    "header",
    "footer",
    "form",
    "aside",
    "template",
    "iframe",
    "button",
    "select",
}
_BLOCK = {
    "p",
    "div",
    "section",
    "article",
    "main",
    "li",
    "tr",
    "td",
    "th",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "br",
    "pre",
    "blockquote",
    "dd",
    "dt",
    "figcaption",
    "table",
}


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.blocks: list[str] = []
        self._current: list[str] = []
        self._skip = 0  # depth inside skipped elements
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in _BLOCK:
            self._flush()

    def handle_endtag(self, tag):
        if tag in _SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in _BLOCK:
            self._flush()

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self._current.append(data)

    def _flush(self):
        text = re.sub(r"\s+", " ", "".join(self._current)).strip()
        if text:
            self.blocks.append(text)
        self._current = []

    def close(self):
        super().close()
        self._flush()


def page_text(html: str) -> tuple[str, str]:
    """(title, text) of an HTML page; blocks are separated by blank lines."""
    parser = _Text()
    try:
        parser.feed(html)
        parser.close()
    except AssertionError:  # html.parser gives up on some broken markup
        parser._flush()
    blocks = [b for b in parser.blocks if len(b) > 1]
    return re.sub(r"\s+", " ", parser.title).strip(), "\n\n".join(dict.fromkeys(blocks))


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower()) if len(w) > 2}


def excerpt(text: str, query: str, limit: int = PAGE_CHARS) -> str:
    """The parts of `text` that best match `query`, in their original order."""
    if len(text) <= limit:
        return text
    blocks = [b for b in text.split("\n\n") if b.strip()]
    wanted = _words(query)
    # Score each block by shared words; the start of the page breaks ties (it's often the gist).
    ranked = sorted(range(len(blocks)), key=lambda i: (-len(_words(blocks[i]) & wanted), i))
    keep, size = set(), 0
    for i in ranked:
        cost = len(blocks[i]) + 2
        if size + cost > limit:
            continue
        keep.add(i)
        size += cost
    if not keep:  # one huge block
        return blocks[ranked[0]][:limit]
    return "\n\n".join(blocks[i] for i in sorted(keep))
