"""Safe HTTP GET for anything on the internet: https only, small, quick, never the LAN.

Every request PRADA makes to the outside goes through `get`. The SSRF guard resolves the
host and refuses private, loopback, link-local and other non-public addresses, then connects
to the address it checked (so a DNS answer can't change between the check and the
connection), and every redirect is checked again. A web page can therefore never make her
call the node on 127.0.0.1:7723, your router or anything else in the house. No proxy, no
cookies, no compression, one honest User-Agent.

`robots_allows(url)` reads the site's robots.txt (cached per host for this run); only
general page reads ask it, not the APIs she was built to use (GitHub, Reddit's .json).
"""

import http.client
import ipaddress
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from collections.abc import Callable
from dataclasses import dataclass, field

USER_AGENT = "PRADA/0.1 (personal home assistant; +https://github.com/GothHeit/PRADA)"
TIMEOUT = 5.0  # seconds per connect or read
DEADLINE = 10.0  # seconds for a whole response
MAX_BYTES = 1_000_000  # bigger bodies are cut here
MAX_REDIRECTS = 3
TEXT_TYPES = ("text/html", "application/xhtml+xml", "text/plain")
JSON_TYPES = ("application/json",)


class FetchError(Exception):
    """The page could not be fetched; the message says why, in plain words."""


class Blocked(FetchError):
    """The URL points somewhere she must never go (not https, or not a public address)."""


@dataclass
class Response:
    url: str  # after redirects
    status: int
    content_type: str  # e.g. "text/html", without parameters
    body: bytes
    charset: str = "utf-8"
    headers: dict = field(default_factory=dict)

    def text(self) -> str:
        try:
            return self.body.decode(self.charset, errors="replace")
        except LookupError:
            return self.body.decode("utf-8", errors="replace")


Resolver = Callable[[str, int], list[str]]


def _resolve(host: str, port: int) -> list[str]:
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return [info[4][0] for info in infos]


_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_IPV4_COMPATIBLE = ipaddress.ip_network("::/96")


def _embedded_ipv4(ip: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    """The IPv4 address an IPv6 one stands for (mapped, NAT64, IPv4-compatible, 6to4)."""
    if ip.ipv4_mapped:
        return ip.ipv4_mapped
    if ip in _NAT64 or ip in _IPV4_COMPATIBLE:
        return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return ip.sixtofour


def public_address(host: str, port: int = 443, resolve: Resolver = _resolve) -> str:
    """The first address of `host`, if every address it resolves to is public."""
    try:
        addresses = resolve(host, port)
    except (OSError, UnicodeError) as e:
        raise FetchError(f"couldn't find {host}") from e
    if not addresses:
        raise FetchError(f"couldn't find {host}")
    for a in addresses:
        ip = ipaddress.ip_address(a.split("%", 1)[0])
        if isinstance(ip, ipaddress.IPv6Address):
            ip = _embedded_ipv4(ip) or ip
        if not ip.is_global or ip.is_multicast:
            raise Blocked(f"{host} points inside a private network")
    return addresses[0]


def check_url(url: str) -> urllib.parse.SplitResult:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        raise Blocked("only https addresses are allowed")
    if parts.username or parts.password:
        raise Blocked("addresses with a user name are not allowed")
    return parts


def _guarded_connection(resolve: Resolver):
    class GuardedHTTPSConnection(http.client.HTTPSConnection):
        def connect(self):
            ip = public_address(self.host, self.port, resolve)
            sock = socket.create_connection((ip, self.port), self.timeout)
            self.sock = self._context.wrap_socket(sock, server_hostname=self.host)

    return GuardedHTTPSConnection


class _HTTPS(urllib.request.HTTPSHandler):
    def __init__(self, resolve: Resolver):
        super().__init__(context=ssl.create_default_context())
        self._connection = _guarded_connection(resolve)

    def https_open(self, req):
        return self.do_open(self._connection, req, context=self._context)


class _NoHTTP(urllib.request.HTTPHandler):
    def http_open(self, req):
        raise Blocked("only https addresses are allowed")


class _Redirects(urllib.request.HTTPRedirectHandler):
    max_redirections = MAX_REDIRECTS

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = check_url(urllib.parse.urljoin(req.full_url, newurl))
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and new.hostname != urllib.parse.urlsplit(req.full_url).hostname:
            redirected.remove_header("Authorization")  # a token is for the host it was meant for
        return redirected


def opener(resolve: Resolver = _resolve) -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _NoHTTP(), _HTTPS(resolve), _Redirects()
    )


_OPENER = None


def get(
    url: str,
    accept: tuple[str, ...] = TEXT_TYPES,
    headers: dict | None = None,
    _opener: urllib.request.OpenerDirector | None = None,
) -> Response:
    """GET `url` safely; raises FetchError (or Blocked) with a reason she can say."""
    global _OPENER
    check_url(url)
    if _opener is None:
        _OPENER = _OPENER or opener()
        _opener = _OPENER
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": ", ".join(accept), **(headers or {})}
    )
    host = urllib.parse.urlsplit(url).hostname
    try:
        with _opener.open(req, timeout=TIMEOUT) as r:
            ctype = r.headers.get_content_type()
            if ctype not in accept:
                raise FetchError(f"{host} sent {ctype}, not a page I can read")
            body = _read(r)
            charset = r.headers.get_content_charset() or "utf-8"
            return Response(r.geturl(), r.status, ctype, body, charset, dict(r.headers))
    except urllib.error.HTTPError as e:
        raise FetchError(f"{host} answered HTTP {e.code}") from e
    except urllib.error.URLError as e:
        if isinstance(e.reason, FetchError):
            raise e.reason from e
        raise FetchError(f"couldn't reach {host} ({e.reason})") from e
    except (TimeoutError, OSError, http.client.HTTPException) as e:
        if isinstance(e, FetchError):
            raise
        raise FetchError(f"couldn't reach {host} ({e or type(e).__name__})") from e


def _read(r) -> bytes:
    deadline = time.monotonic() + DEADLINE
    chunks, size = [], 0
    while size < MAX_BYTES:
        if time.monotonic() > deadline:
            raise FetchError("the page took too long")
        chunk = r.read(min(65536, MAX_BYTES - size))
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
    return b"".join(chunks)


_robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}


def robots_allows(url: str, fetch: Callable[..., Response] = get) -> bool:
    """True unless the site's robots.txt forbids `url` to PRADA (no robots.txt = allowed)."""
    parts = urllib.parse.urlsplit(url)
    site = f"{parts.scheme}://{parts.netloc}"
    if site not in _robots:
        rules = urllib.robotparser.RobotFileParser()
        try:
            rules.parse(fetch(f"{site}/robots.txt", accept=("text/plain",)).text().splitlines())
        except Blocked:
            raise
        except FetchError as e:
            # As robotparser does: 401/403 = everything disallowed, anything else = allowed.
            if "HTTP 401" in str(e) or "HTTP 403" in str(e):
                rules.disallow_all = True
            else:
                rules = None
        _robots[site] = rules
    rules = _robots[site]
    return rules is None or rules.can_fetch(USER_AGENT, url)
