from __future__ import annotations

import json

import pytest
from conftest import FakeHttp

from sentiment_hub import notify
from sentiment_hub.common import utf16_len
from sentiment_hub.net import HttpResponse

TOKEN = "123456:ABC-secret-token"


def test_short_text_is_one_message():
    assert notify.split_message("hello\n") == ["hello"]
    assert notify.split_message("") == []
    assert notify.prepare("body", title="Title") == ["Title\n\nbody"]


def test_split_at_paragraphs_and_limits():
    paragraphs = [f"Paragraph {i}: " + ("word " * 150).strip() for i in range(12)]  # ~770 chars each
    text = "\n\n".join(paragraphs)
    assert len(text) > 9000
    chunks = notify.split_message(text, 4000)
    assert len(chunks) == 3
    assert all(utf16_len(c) <= 4000 for c in chunks)
    assert all(c.startswith("Paragraph") for c in chunks)  # cut on paragraph boundaries
    assert " ".join(" ".join(chunks).split()) == " ".join(text.split())  # nothing lost


def test_split_long_line_and_long_word():
    line = "x" * 9500
    chunks = notify.split_message(line, 4000)
    assert [len(c) for c in chunks] == [4000, 4000, 1500]
    words = " ".join(["bitcoin"] * 1500)
    chunks = notify.split_message(words, 4000)
    assert all(len(c) <= 4000 and not c.startswith(" ") and not c.endswith(" ") for c in chunks)
    assert " ".join(chunks).split() == words.split()


def test_split_counts_utf16_units():
    text = "🚀" * 3000  # 6000 UTF-16 units
    chunks = notify.split_message(text, 4000)
    assert [utf16_len(c) for c in chunks] == [4000, 2000]


def test_numbered_parts_stay_under_limit():
    text = "\n".join("line %d %s" % (i, "y" * 90) for i in range(200))
    parts = notify.prepare(text, title="Report")
    assert len(parts) >= 5
    assert parts[0].startswith(f"[1/{len(parts)}]\nReport")
    assert all(utf16_len(p) <= 4000 for p in parts)


def test_nothing_configured_exits_zero(tmp_path, capsys):
    f = tmp_path / "r.md"
    f.write_text("# Daily\nall good", encoding="utf-8")
    assert notify.main([str(f)], env={}) == 0
    assert "not configured" in capsys.readouterr().err


def test_missing_file(tmp_path):
    assert notify.main([str(tmp_path / "missing.md")], env={}) == 2


def test_bot_api_sends_every_part():
    http = FakeHttp()
    http.routes[("POST", f"https://api.telegram.org/bot{TOKEN}/sendMessage")] = [
        HttpResponse(429, "u", json.dumps({"ok": False, "parameters": {"retry_after": 3}}).encode()),
        HttpResponse(200, "u", b'{"ok": true}'),
    ]
    slept = []
    notify.send_via_bot(TOKEN, "42", ["part one", "part two"], http=http, sleep=slept.append)
    assert [p[1]["text"] for p in http.posts] == ["part one", "part one", "part two"]  # 429 retried
    assert all(p[1]["chat_id"] == "42" for p in http.posts)
    assert slept == [3.0, 1.0]


def test_bot_api_error_hides_token():
    http = FakeHttp()
    http.routes[("POST", f"https://api.telegram.org/bot{TOKEN}/sendMessage")] = HttpResponse(
        400, "u", json.dumps({"ok": False, "description": f"Bad Request: chat not found for bot{TOKEN}"}).encode()
    )
    with pytest.raises(RuntimeError) as err:
        notify.send_via_bot(TOKEN, "42", ["x"], http=http, sleep=lambda s: None)
    assert TOKEN not in str(err.value) and "chat not found" in str(err.value)


def test_account_path_sends_to_saved_messages(monkeypatch):
    sent = []

    class Client:
        async def connect(self):
            pass

        async def is_user_authorized(self):
            return True

        async def send_message(self, entity, text, **kwargs):
            sent.append((entity, text, kwargs))

        async def disconnect(self):
            pass

    monkeypatch.setattr(notify.asyncio, "sleep", _no_sleep)
    env = {"TELEGRAM_API_ID": "1", "TELEGRAM_API_HASH": "h", "TELEGRAM_SESSION": "s"}
    notify.send_via_account(env, ["a", "b"], client_factory=lambda *a: Client())
    assert [(e, t) for e, t, _ in sent] == [("me", "a"), ("me", "b")]
    assert sent[0][2] == {"link_preview": False, "parse_mode": None}


async def _no_sleep(_s):
    return None


def test_account_preferred_over_bot(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(notify, "send_via_account", lambda env, chunks: calls.append(("account", chunks)))
    monkeypatch.setattr(notify, "send_via_bot", lambda *a, **k: calls.append(("bot", a)))
    f = tmp_path / "r.txt"
    f.write_text("hi", encoding="utf-8")
    env = {"TELEGRAM_API_ID": "1", "TELEGRAM_API_HASH": "h", "TELEGRAM_SESSION": "s", "TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_CHAT_ID": "c"}
    assert notify.main([str(f)], env=env) == 0
    assert calls == [("account", ["hi"])]
    calls.clear()
    assert notify.main([str(f)], env={"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_CHAT_ID": "c"}) == 0
    assert calls[0][0] == "bot"
