"""Minimal HTTP layer: thread-local ``requests`` sessions, polite per-host rate limits, and a
response object that tests can fake without any network."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/128.0.0.0 Safari/537.36"
)
BOT_USER_AGENT = "CryptoPulseSentimentHub/1.0 (+https://github.com/medonumbworks/spss)"


class NetError(Exception):
    """Connection-level failure (DNS, TLS, proxy refusal, timeout)."""


@dataclass
class HttpResponse:
    status: int
    url: str
    content: bytes = b""
    headers: Mapping[str, str] = field(default_factory=dict)
    encoding: str | None = "utf-8"

    @property
    def text(self) -> str:
        return self.content.decode(self.encoding or "utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.text)

    def header(self, name: str) -> str | None:
        lower = name.lower()
        for key, value in self.headers.items():
            if key.lower() == lower:
                return value
        return None


class RateLimiter:
    """At most one request every ``interval`` seconds, shared by all threads."""

    def __init__(
        self,
        interval: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.interval = max(0.0, interval)
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            delay = self._next - now
            if delay > 0:
                self._sleep(delay)
                now = self._clock()
            self._next = now + self.interval

    def penalize(self, seconds: float) -> None:
        """Push the next slot back (after a 429)."""
        with self._lock:
            self._next = max(self._next, self._clock() + max(0.0, seconds))


class HttpClient:
    """``get`` returns an :class:`HttpResponse` for every HTTP status and raises :class:`NetError`
    only when no HTTP response was received."""

    def __init__(
        self,
        user_agent: str = BOT_USER_AGENT,
        timeout: float = 20.0,
        host_intervals: Mapping[str, float] | None = None,
        max_bytes: int = 8_000_000,
    ) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_bytes = max_bytes
        self._limiters = {host.lower(): RateLimiter(iv) for host, iv in (host_intervals or {}).items()}
        self._local = threading.local()

    def limiter_for(self, url: str) -> RateLimiter | None:
        host = (urlparse(url).hostname or "").lower()
        return self._limiters.get(host)

    def _session(self):
        session = getattr(self._local, "session", None)
        if session is None:
            import requests

            session = requests.Session()
            session.headers["User-Agent"] = self.user_agent
            self._local.session = session
        return session

    def get(self, url: str, headers: Mapping[str, str] | None = None, allow_redirects: bool = True) -> HttpResponse:
        import requests

        limiter = self.limiter_for(url)
        if limiter is not None:
            limiter.wait()
        try:
            with self._session().get(
                url, headers=dict(headers or {}), timeout=self.timeout, allow_redirects=allow_redirects, stream=True
            ) as resp:
                chunks: list[bytes] = []
                size = 0
                for chunk in resp.iter_content(chunk_size=65536):
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > self.max_bytes:
                        break
                return HttpResponse(
                    status=resp.status_code,
                    url=resp.url,
                    content=b"".join(chunks),
                    headers=dict(resp.headers),
                    encoding=resp.encoding,
                )
        except requests.RequestException as exc:
            raise NetError(f"{type(exc).__name__}: {_short(exc)}") from exc

    def post(self, url: str, data: Mapping[str, Any]) -> HttpResponse:
        import requests

        try:
            resp = self._session().post(url, data=dict(data), timeout=self.timeout)
            return HttpResponse(status=resp.status_code, url=resp.url, content=resp.content, headers=dict(resp.headers))
        except requests.RequestException as exc:
            raise NetError(f"{type(exc).__name__}: {_short(exc)}") from exc


def retry_after_seconds(resp: HttpResponse, default: float = 5.0, cap: float = 30.0) -> float:
    raw = resp.header("Retry-After")
    try:
        value = float(raw) if raw is not None else default
    except ValueError:
        value = default
    return max(0.0, min(cap, value))


def _short(exc: Exception, limit: int = 200) -> str:
    text = str(exc).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."
