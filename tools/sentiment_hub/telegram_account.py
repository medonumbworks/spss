"""The owner's own Telegram account via Telethon (optional).

Enabled only when ``TELEGRAM_API_ID``, ``TELEGRAM_API_HASH`` and ``TELEGRAM_SESSION`` (a Telethon
``StringSession`` printed by ``login.py``) are set. Reads

(a) every group / supergroup / channel dialog of the account (1:1 chats and bots are never
    opened), up to 300 dialogs, the last 48 h and at most 400 messages per chat;
(b) every public handle from ``channels.json`` that is not a dialog, by username and without
    joining (public channels and supergroups allow reading history). Handles the public web
    preview already read in this run are skipped: the messages would be duplicates and every
    username lookup counts against Telegram's daily resolve limit.

Nothing is ever joined unless ``TELEGRAM_AUTO_JOIN=1``; then at most 5 public ``kind=group``
catalogue entries per run are joined, highest confidence first.

PRIVACY: a chat without a public username is private. Its texts only feed aggregate numbers
(``public=False`` -> never a sample); its title, id and texts are never logged or written.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Mapping, Sequence

from sentiment_hub import __version__
from sentiment_hub.assets import ChannelEntry, normalize_handle
from sentiment_hub.common import Message, SourceResult, as_utc, clean_text

log = logging.getLogger("sentiment_hub.telegram_account")

NAME = "telegram_account"
ENV_VARS = ("TELEGRAM_API_ID", "TELEGRAM_API_HASH", "TELEGRAM_SESSION")
MAX_DIALOGS = 300
PER_CHAT = 400
# Username lookups per run. StringSession keeps no entity cache, so every hourly run resolves
# again; Telegram flood-limits mass lookups on personal accounts, so stay low and rotate.
MAX_RESOLVES = 15
MAX_JOINS = 5
FLOOD_SLEEP_MAX_S = 60
DEFAULT_BUDGET_S = 600.0

_CHAT_KINDS = {"Channel", "Chat", "ChannelForbidden", "ChatForbidden", "Community"}


def missing_env(env: Mapping[str, str]) -> list[str]:
    return [name for name in ENV_VARS if not (env.get(name) or "").strip()]


def default_client_factory(api_id: int, api_hash: str, session: str):
    from telethon import TelegramClient
    from telethon.sessions import StringSession

    logging.getLogger("telethon").setLevel(logging.WARNING)
    return TelegramClient(
        StringSession(session),
        api_id,
        api_hash,
        flood_sleep_threshold=FLOOD_SLEEP_MAX_S,
        receive_updates=False,
        device_model="CryptoPulse Sentiment Hub",
        app_version=__version__,
    )


def entity_username(entity: Any) -> str | None:
    username = getattr(entity, "username", None)
    if username:
        return str(username)
    for item in getattr(entity, "usernames", None) or []:
        if getattr(item, "active", True) and getattr(item, "username", None):
            return str(item.username)
    return None


def is_chat_entity(entity: Any) -> bool:
    """Channels, supergroups and groups; never users or bots."""
    return type(entity).__name__ in _CHAT_KINDS


def _flood_seconds(exc: BaseException) -> int | None:
    try:
        from telethon.errors import FloodWaitError

        if isinstance(exc, FloodWaitError):
            return int(exc.seconds)
    except ImportError:  # pragma: no cover
        pass
    if "FloodWait" in type(exc).__name__ and hasattr(exc, "seconds"):
        return int(getattr(exc, "seconds"))
    return None


def _engagement(msg: Any) -> float:
    views = getattr(msg, "views", None)
    if views:
        return float(views)
    reactions = getattr(getattr(msg, "reactions", None), "results", None) or []
    return float(sum(getattr(r, "count", 0) or 0 for r in reactions))


@dataclass
class _State:
    messages: list[Message] = field(default_factory=list)
    chats_total: int = 0
    chats_read: int = 0
    handles_read: int = 0
    handles_failed: int = 0
    handles_skipped: int = 0
    flood_skips: int = 0
    joined: int = 0


class AccountReader:
    def __init__(
        self,
        client: Any,
        now: datetime,
        lookback_hours: float = 48,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        max_dialogs: int = MAX_DIALOGS,
        per_chat: int = PER_CHAT,
        max_resolves: int = MAX_RESOLVES,
    ) -> None:
        self.client = client
        self.now = as_utc(now)
        self.cutoff = self.now - timedelta(hours=lookback_hours)
        self.sleep = sleep
        self.max_dialogs = max_dialogs
        self.per_chat = per_chat
        self.max_resolves = max_resolves
        self.state = _State()
        self._resolves = 0
        self._resolve_blocked = False

    # -- reading ---------------------------------------------------------------------------

    async def _read_history(self, entity: Any, username: str | None) -> list[Message] | None:
        """Messages of one chat in the window, or None when the chat could not be read."""
        public = bool(username)
        label = f"@{username}" if public else "a private chat"  # never a title
        for attempt in (1, 2):
            out: list[Message] = []
            try:
                async for msg in self.client.iter_messages(entity, limit=self.per_chat):
                    date = getattr(msg, "date", None)
                    if date is None:
                        continue
                    ts = as_utc(date)
                    if ts < self.cutoff:
                        break
                    if getattr(msg, "action", None) is not None:
                        continue  # service message (joins, pins, ...)
                    text = clean_text(getattr(msg, "message", None) or "")
                    if not text:
                        continue
                    if public:
                        out.append(
                            Message(
                                source=NAME,
                                key=f"tg:{username.lower()}/{msg.id}",
                                text=text,
                                ts=ts,
                                url=f"https://t.me/{username}/{msg.id}",
                                community=username,
                                public=True,
                                engagement=_engagement(msg),
                            )
                        )
                    else:
                        out.append(
                            Message(
                                source=NAME,
                                key=f"tgp:{getattr(entity, 'id', id(entity))}/{msg.id}",
                                text=text,
                                ts=ts,
                                url=None,
                                community=None,
                                public=False,
                                engagement=_engagement(msg),
                            )
                        )
                return out
            except Exception as exc:
                seconds = _flood_seconds(exc)
                if seconds is not None:
                    if seconds <= FLOOD_SLEEP_MAX_S and attempt == 1:
                        log.info("flood wait %ds while reading %s; sleeping", seconds, label)
                        await self.sleep(seconds)
                        continue
                    log.info("flood wait %ds while reading %s; skipped", seconds, label)
                    self.state.flood_skips += 1
                    return None
                log.info("could not read %s: %s", label, type(exc).__name__)
                return None
        return None

    async def _resolve(self, handle: str) -> Any | None:
        if self._resolve_blocked or self._resolves >= self.max_resolves:
            self.state.handles_skipped += 1
            return None
        for attempt in (1, 2):
            self._resolves += 1
            try:
                return await self.client.get_entity(handle)
            except Exception as exc:
                seconds = _flood_seconds(exc)
                if seconds is not None:
                    if seconds <= FLOOD_SLEEP_MAX_S and attempt == 1:
                        log.info("flood wait %ds resolving @%s; sleeping", seconds, handle)
                        await self.sleep(seconds)
                        continue
                    log.warning("flood wait %ds resolving @%s; no more username lookups this run", seconds, handle)
                    self.state.flood_skips += 1
                    self._resolve_blocked = True
                    return None
                log.info("@%s: %s", handle, type(exc).__name__)
                self.state.handles_failed += 1
                return None
        return None

    async def _join(self, entity: Any, handle: str) -> bool:
        from telethon.tl.functions.channels import JoinChannelRequest

        for attempt in (1, 2):
            try:
                await self.client(JoinChannelRequest(entity))
                log.info("joined @%s (TELEGRAM_AUTO_JOIN=1)", handle)
                return True
            except Exception as exc:
                seconds = _flood_seconds(exc)
                if seconds is not None and seconds <= FLOOD_SLEEP_MAX_S and attempt == 1:
                    await self.sleep(seconds)
                    continue
                log.info("could not join @%s: %s", handle, type(exc).__name__)
                return False
        return False

    # -- run -------------------------------------------------------------------------------

    async def read_dialogs(self) -> set[str]:
        """(a) Every group/channel dialog. Returns the lowercase usernames of public ones."""
        chats: list[Any] = []
        async for dialog in self.client.iter_dialogs():
            if getattr(dialog, "is_user", False):
                continue  # 1:1 chats, bots, Saved Messages: never opened
            if not (getattr(dialog, "is_group", False) or getattr(dialog, "is_channel", False)):
                continue
            chats.append(dialog.entity)
            if len(chats) >= self.max_dialogs:
                break
        self.state.chats_total = len(chats)
        usernames: set[str] = set()
        for entity in chats:
            username = entity_username(entity)
            if username:
                usernames.add(username.lower())
            msgs = await self._read_history(entity, username)
            if msgs is not None:
                self.state.chats_read += 1
                self.state.messages.extend(msgs)
        return usernames

    async def read_catalogue(
        self,
        entries: Sequence[ChannelEntry],
        dialog_usernames: set[str],
        skip_handles: set[str],
        auto_join: bool,
    ) -> None:
        handled = set(dialog_usernames) | {h.lower() for h in skip_handles}
        if auto_join:
            groups = sorted((e for e in entries if e.kind == "group"), key=lambda e: -e.confidence)
            for entry in groups:
                if self.state.joined >= MAX_JOINS:
                    break
                handle = normalize_handle(entry.handle)
                if not handle or handle.lower() in dialog_usernames:
                    continue
                entity = await self._resolve(handle)
                if entity is None or not is_chat_entity(entity):
                    continue
                if await self._join(entity, handle):
                    self.state.joined += 1
                    dialog_usernames.add(handle.lower())
                    handled.add(handle.lower())
                    username = entity_username(entity) or handle
                    msgs = await self._read_history(entity, username)
                    if msgs is not None:
                        self.state.handles_read += 1
                        self.state.messages.extend(msgs)
        pending: list[str] = []
        for entry in entries:
            handle = normalize_handle(entry.handle)
            if handle and handle.lower() not in handled:
                handled.add(handle.lower())
                pending.append(handle)
        # Each run starts at a different place, so the low per-run lookup budget still covers the
        # whole catalogue several times a day (the 48 h lookback bridges the gaps).
        if pending and self.max_resolves > 0:
            start = (int(self.now.timestamp() // 3600) * self.max_resolves) % len(pending)
            pending = pending[start:] + pending[:start]
        for handle in pending:
            entity = await self._resolve(handle)
            if entity is None:
                continue
            if not is_chat_entity(entity):
                log.info("@%s is not a channel or group; skipped", handle)
                self.state.handles_failed += 1
                continue
            username = entity_username(entity) or handle
            msgs = await self._read_history(entity, username)
            if msgs is None:
                self.state.handles_failed += 1
                continue
            self.state.handles_read += 1
            self.state.messages.extend(msgs)


def _detail(state: _State, note: str = "") -> str:
    parts = [f"{state.chats_read}/{state.chats_total} dialogs", f"{state.handles_read} catalogue handles"]
    if state.handles_failed:
        parts.append(f"{state.handles_failed} unreadable")
    if state.handles_skipped:
        parts.append(f"{state.handles_skipped} lookups deferred")
    if state.flood_skips:
        parts.append(f"{state.flood_skips} flood-wait skips")
    if state.joined:
        parts.append(f"joined {state.joined}")
    if note:
        parts.append(note)
    return ", ".join(parts)


async def collect_async(
    client: Any,
    entries: Sequence[ChannelEntry],
    now: datetime,
    lookback_hours: float = 48,
    skip_handles: set[str] | Callable[[], set[str]] | None = None,
    auto_join: bool = False,
    budget_s: float = DEFAULT_BUDGET_S,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> SourceResult:
    reader = AccountReader(client, now, lookback_hours, sleep=sleep)
    note = ""
    try:
        await client.connect()
        if not await client.is_user_authorized():
            return SourceResult(NAME, "unauthorized", detail="TELEGRAM_SESSION is not authorized; run login.py again")

        async def run() -> None:
            usernames = await reader.read_dialogs()
            skip = skip_handles
            if callable(skip):  # e.g. "wait for the public preview, then tell me what it covered"
                skip = await asyncio.get_running_loop().run_in_executor(None, skip)
            await reader.read_catalogue(entries, usernames, set(skip or ()), auto_join)

        started = time.monotonic()
        try:
            await asyncio.wait_for(run(), timeout=budget_s)
        except asyncio.TimeoutError:
            if time.monotonic() - started >= budget_s * 0.98:
                note = f"time budget {int(budget_s)}s exhausted"
            else:
                note = "request timed out"
            log.warning("telegram_account: %s; keeping what was read", note)
    except (ConnectionError, OSError) as exc:
        return SourceResult(NAME, "unavailable", reader.state.messages, detail=f"connection failed ({type(exc).__name__})")
    except Exception as exc:
        name = type(exc).__name__
        if any(k in name for k in ("AuthKey", "SessionRevoked", "SessionExpired", "Unauthorized", "UserDeactivated")):
            return SourceResult(NAME, "unauthorized", detail=f"{name}; run login.py again")
        seconds = _flood_seconds(exc)
        if seconds is not None:  # e.g. a long flood wait while listing the dialogs
            reader.state.flood_skips += 1
            log.warning("telegram_account: flood wait %ds; keeping what was read", seconds)
            status = "ok" if reader.state.messages else "rate_limited"
            return SourceResult(NAME, status, reader.state.messages, detail=_detail(reader.state, f"flood wait {seconds}s"))
        log.warning("telegram_account failed: %s", name)
        return SourceResult(NAME, "error", reader.state.messages, detail=_detail(reader.state, name))
    finally:
        try:
            result = client.disconnect()
            if asyncio.iscoroutine(result):
                await result
        except Exception:  # pragma: no cover
            pass
    state = reader.state
    if state.messages:
        status = "ok"
    elif state.chats_read or state.handles_read:
        status = "empty"
    elif state.flood_skips:
        status = "rate_limited"
    elif state.chats_total == 0 and state.handles_failed == 0:
        status = "empty"  # nothing to read: no group dialogs and every handle already covered
    else:
        status = "error"
    detail = _detail(state, note)
    log.info("telegram_account: %s, %d messages, %s", status, len(state.messages), detail)
    return SourceResult(NAME, status, state.messages, detail)


def collect(
    entries: Sequence[ChannelEntry],
    now: datetime,
    lookback_hours: float = 48,
    env: Mapping[str, str] | None = None,
    client_factory: Callable[[int, str, str], Any] | None = None,
    skip_handles: set[str] | Callable[[], set[str]] | None = None,
    budget_s: float = DEFAULT_BUDGET_S,
) -> SourceResult:
    env = os.environ if env is None else env
    missing = missing_env(env)
    if missing:
        plural = "s" if len(missing) > 1 else ""
        return SourceResult(NAME, "disabled", detail=f"no {', '.join(missing)} secret{plural}")
    try:
        api_id = int(env["TELEGRAM_API_ID"].strip())
    except ValueError:
        return SourceResult(NAME, "error", detail="TELEGRAM_API_ID is not a number")
    if client_factory is None:
        if importlib.util.find_spec("telethon") is None:
            return SourceResult(NAME, "disabled", detail="telethon not installed")
        client_factory = default_client_factory
    auto_join = (env.get("TELEGRAM_AUTO_JOIN") or "").strip() == "1"
    started = time.monotonic()

    async def main() -> SourceResult:
        # The client is created inside the running loop (Telethon binds to it).
        client = client_factory(api_id, env["TELEGRAM_API_HASH"].strip(), env["TELEGRAM_SESSION"].strip())
        return await collect_async(client, entries, now, lookback_hours, skip_handles, auto_join, budget_s)

    try:
        result = asyncio.run(main())
    except Exception as exc:
        log.warning("telegram_account failed: %s", type(exc).__name__)
        return SourceResult(NAME, "error", detail=type(exc).__name__)
    log.debug("telegram_account took %.1fs", time.monotonic() - started)
    return result
