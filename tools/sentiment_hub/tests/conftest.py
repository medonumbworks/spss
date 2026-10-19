"""Shared fixtures. Nothing here touches the network: HTTP goes through ``FakeHttp``."""

from __future__ import annotations

import socket
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[2]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from sentiment_hub.net import HttpResponse, NetError  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REPO_ROOT = TOOLS_DIR.parent
NOW = datetime(2026, 10, 19, 10, 0, 0, tzinfo=timezone.utc)


def fixture_path(name: str) -> Path:
    return FIXTURES / name


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


class FakeHttp:
    """Routes exact URLs to responses. Unknown URLs raise ``NetError`` (like a blocked host)."""

    def __init__(self, routes: dict | None = None, default=None) -> None:
        self.routes: dict = dict(routes or {})
        self.default = default
        self.calls: list[str] = []
        self.posts: list[tuple[str, dict]] = []
        self._lock = threading.Lock()

    def add(self, url: str, body: bytes | str = b"", status: int = 200, final_url: str | None = None, headers=None):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.routes[url] = HttpResponse(status=status, url=final_url or url, content=data, headers=headers or {})
        return self

    def limiter_for(self, url):
        return None

    def get(self, url, headers=None, allow_redirects=True):
        with self._lock:
            self.calls.append(url)
        route = self.routes.get(url, self.default)
        if route is None:
            raise NetError(f"blocked in tests: {url}")
        if isinstance(route, list):  # sequence of responses for repeated calls
            with self._lock:
                route = route.pop(0) if len(route) > 1 else route[0]
        if isinstance(route, Exception):
            raise route
        if callable(route):
            return route(url)
        return route

    def post(self, url, data):
        with self._lock:
            self.posts.append((url, dict(data)))
        route = self.routes.get(("POST", url), self.default)
        if isinstance(route, list):
            route = route.pop(0) if len(route) > 1 else route[0]
        if isinstance(route, Exception):
            raise route
        if route is None:
            raise NetError("blocked in tests")
        return route


@pytest.fixture
def now() -> datetime:
    return NOW


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any real socket connection in a test is a bug."""

    def guard(*args, **kwargs):
        raise RuntimeError("network access attempted in a test")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
    yield


@pytest.fixture(autouse=True)
def clean_telegram_env(monkeypatch):
    for name in (
        "TELEGRAM_API_ID",
        "TELEGRAM_API_HASH",
        "TELEGRAM_SESSION",
        "TELEGRAM_AUTO_JOIN",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "GITHUB_STEP_SUMMARY",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
