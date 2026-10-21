#!/usr/bin/env python3
"""Send a text or Markdown file to your own Telegram.

    python tools/sentiment_hub/notify.py report.md [--title "Daily report"] [--dry-run]

* With ``TELEGRAM_API_ID``, ``TELEGRAM_API_HASH`` and ``TELEGRAM_SESSION`` set, the text goes to
  your *Saved Messages* (Telethon, chat ``me``).
* Otherwise with ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID`` set, a bot sends it
  (``POST https://api.telegram.org/bot<token>/sendMessage``).
* With neither, it prints what is missing and exits 0.

Texts longer than 4000 characters are split at paragraph/line boundaries and numbered.
The text is sent as plain text, exactly as written.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # run as a script: import as the `sentiment_hub` package and keep
    _here = Path(__file__).resolve().parent  # generic module names (common, net, feed) off sys.path
    sys.path[:] = [x for x in sys.path if Path(x or ".").resolve() != _here]
    sys.path.insert(0, str(_here.parent))

import argparse
import asyncio
import os
import time
from typing import Any, Callable, Mapping, Sequence

from sentiment_hub.common import utf16_len

LIMIT = 4000
BOT_API = "https://api.telegram.org"


def split_message(text: str, limit: int = LIMIT) -> list[str]:
    """Chunks of at most ``limit`` UTF-16 units (Telegram's measure), split at blank lines, then
    line breaks, then spaces, then anywhere. Chunks after the first get no leading newlines."""
    text = text.replace("\r\n", "\n").strip("\n")
    if not text:
        return []
    if utf16_len(text) <= limit:
        return [text]
    chunks: list[str] = []
    rest = text
    while rest:
        if utf16_len(rest) <= limit:
            chunks.append(rest)
            break
        # Longest prefix that fits.
        cut = 0
        used = 0
        for i, ch in enumerate(rest):
            used += 2 if ord(ch) > 0xFFFF else 1
            if used > limit:
                break
            cut = i + 1
        window = rest[:cut]
        split_at = -1
        for sep in ("\n\n", "\n", " "):
            pos = window.rfind(sep)
            if pos > limit // 4:
                split_at = pos
                break
        if split_at == -1:
            split_at = cut
        chunk = rest[:split_at].rstrip()
        if not chunk:  # pathological whitespace
            chunk, split_at = rest[:cut], cut
        chunks.append(chunk)
        rest = rest[split_at:].lstrip("\n ")
    return chunks


def numbered(chunks: Sequence[str], limit: int = LIMIT) -> list[str]:
    """``[i/n]`` prefixes; re-splits with room for the prefix so every part stays <= limit."""
    if len(chunks) <= 1:
        return list(chunks)
    text = "\n\n".join(chunks)
    parts = split_message(text, limit - 12)
    n = len(parts)
    return [f"[{i}/{n}]\n{p}" for i, p in enumerate(parts, 1)]


def prepare(text: str, title: str | None = None, limit: int = LIMIT) -> list[str]:
    body = f"{title}\n\n{text}" if title else text
    return numbered(split_message(body, limit), limit)


def account_configured(env: Mapping[str, str]) -> bool:
    return all((env.get(k) or "").strip() for k in ("TELEGRAM_API_ID", "TELEGRAM_API_HASH", "TELEGRAM_SESSION"))


def bot_configured(env: Mapping[str, str]) -> bool:
    return all((env.get(k) or "").strip() for k in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"))


def send_via_bot(
    token: str,
    chat_id: str,
    chunks: Sequence[str],
    http: Any = None,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    if http is None:
        from sentiment_hub.net import HttpClient

        http = HttpClient(timeout=30)
    url = f"{BOT_API}/bot{token}/sendMessage"
    for i, chunk in enumerate(chunks):
        if i:
            sleep(1.0)
        data = {"chat_id": chat_id, "text": chunk, "disable_web_page_preview": "true"}
        for attempt in (1, 2):
            try:
                resp = http.post(url, data)
            except Exception as exc:
                raise RuntimeError(f"Bot API request failed: {str(exc).replace(token, '***')}") from None
            if resp.status == 429 and attempt == 1:
                try:
                    wait = float(resp.json().get("parameters", {}).get("retry_after", 5))
                except Exception:
                    wait = 5.0
                sleep(min(wait, 60.0))
                continue
            try:
                payload = resp.json()
            except Exception:
                payload = {}
            if resp.status != 200 or not payload.get("ok"):
                description = str(payload.get("description") or f"HTTP {resp.status}").replace(token, "***")
                raise RuntimeError(f"Bot API refused part {i + 1}/{len(chunks)}: {description}")
            break


def send_via_account(env: Mapping[str, str], chunks: Sequence[str], client_factory: Callable | None = None) -> None:
    if client_factory is None:
        from sentiment_hub.telegram_account import default_client_factory as client_factory

    async def main() -> None:
        client = client_factory(int(env["TELEGRAM_API_ID"].strip()), env["TELEGRAM_API_HASH"].strip(), env["TELEGRAM_SESSION"].strip())
        await client.connect()
        try:
            if not await client.is_user_authorized():
                raise RuntimeError("TELEGRAM_SESSION is not authorized; run login.py again")
            for i, chunk in enumerate(chunks):
                if i:
                    await asyncio.sleep(1.0)
                await client.send_message("me", chunk, link_preview=False, parse_mode=None)
        finally:
            result = client.disconnect()
            if asyncio.iscoroutine(result):
                await result

    asyncio.run(main())


def main(argv: list[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="notify.py", description="Send a text/markdown file to your own Telegram.")
    parser.add_argument("file", help="file to send ('-' for stdin)")
    parser.add_argument("--title", default=None, help="first line of the message")
    parser.add_argument("--dry-run", action="store_true", help="print the parts instead of sending")
    args = parser.parse_args(argv)
    env = os.environ if env is None else env

    try:
        text = sys.stdin.read() if args.file == "-" else Path(args.file).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"notify: cannot read {args.file}: {exc}", file=sys.stderr)
        return 2
    chunks = prepare(text, args.title)
    if not chunks:
        print("notify: nothing to send (empty file)", file=sys.stderr)
        return 0
    if args.dry_run:
        for c in chunks:
            print(c + "\n" + "-" * 40)
        return 0
    try:
        if account_configured(env):
            send_via_account(env, chunks)
            print(f"notify: sent {len(chunks)} message(s) to Saved Messages", file=sys.stderr)
        elif bot_configured(env):
            send_via_bot(env["TELEGRAM_BOT_TOKEN"].strip(), env["TELEGRAM_CHAT_ID"].strip(), chunks)
            print(f"notify: sent {len(chunks)} message(s) via bot", file=sys.stderr)
        else:
            print(
                "notify: Telegram is not configured, nothing sent. Set TELEGRAM_API_ID + TELEGRAM_API_HASH + "
                "TELEGRAM_SESSION (see login.py) or TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID.",
                file=sys.stderr,
            )
    except Exception as exc:
        print(f"notify: failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
