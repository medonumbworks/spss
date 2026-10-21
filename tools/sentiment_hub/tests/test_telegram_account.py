"""Telethon is mocked; the fake mirrors the attributes the reader uses (Dialog.is_user/is_group/
is_channel/entity, entity.username/usernames, Message.id/date/message/views/action)."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import timedelta
from types import SimpleNamespace

import pytest
from telethon.errors import FloodWaitError

from sentiment_hub import telegram_account as ta
from sentiment_hub.aggregate import aggregate
from sentiment_hub.assets import Asset, ChannelEntry
from sentiment_hub.common import SourceResult
from sentiment_hub.feed import build_feed, dumps, validate_feed
from sentiment_hub.mentions import build_matchers
from sentiment_hub.nlp import score_messages

ENV = {"TELEGRAM_API_ID": "12345", "TELEGRAM_API_HASH": "0123456789abcdef", "TELEGRAM_SESSION": "1FakeStringSession"}
PRIVATE_TITLE = "Family Secret Whale Club"
PRIVATE_TEXT = "Our private plan: buy Bitcoin before the announcement"


class User:
    def __init__(self, id, username=None, bot=False, first_name="Alice"):
        self.id, self.username, self.bot, self.first_name = id, username, bot, first_name


class Channel:
    def __init__(self, id, title, username=None, megagroup=False, usernames=None):
        self.id, self.title, self.username, self.megagroup = id, title, username, megagroup
        self.broadcast = not megagroup
        self.usernames = usernames


class Chat:  # basic group: never has a username
    def __init__(self, id, title):
        self.id, self.title = id, title


def dialog(entity):
    is_user = isinstance(entity, User)
    return SimpleNamespace(
        entity=entity,
        title=getattr(entity, "title", None) or getattr(entity, "first_name", ""),
        is_user=is_user,
        is_group=isinstance(entity, Chat) or (isinstance(entity, Channel) and entity.megagroup),
        is_channel=isinstance(entity, Channel),
    )


def message(id, now, hours_ago, text, views=None, action=None):
    return SimpleNamespace(id=id, date=now - timedelta(hours=hours_ago), message=text, views=views, action=action, reactions=None)


class FakeClient:
    def __init__(self, now, dialogs=(), history=None, entities=None, authorized=True, get_entity_errors=None, history_errors=None):
        self.now = now
        self.dialogs = list(dialogs)
        self.history = history or {}
        self.entities = entities or {}
        self.authorized = authorized
        self.get_entity_errors = get_entity_errors or {}
        self.history_errors = history_errors or {}
        self.history_reads: list[int] = []
        self.resolved: list[str] = []
        self.requests: list = []
        self.connected = False

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.connected = False

    async def is_user_authorized(self):
        return self.authorized

    async def iter_dialogs(self):
        for d in self.dialogs:
            yield d

    async def iter_messages(self, entity, limit=None):
        if isinstance(entity, User):
            raise AssertionError("a 1:1 chat or bot must never be read")
        self.history_reads.append(entity.id)
        errors = self.history_errors.get(entity.id)
        if errors:
            raise errors.pop(0)
        for i, msg in enumerate(self.history.get(entity.id, [])):
            if limit is not None and i >= limit:
                break
            yield msg

    async def get_entity(self, handle):
        self.resolved.append(handle)
        errors = self.get_entity_errors.get(handle)
        if errors:
            raise errors.pop(0)
        if handle.lower() not in self.entities:
            raise ValueError(f'No user has "{handle}" as username')
        return self.entities[handle.lower()]

    async def __call__(self, request):
        self.requests.append(request)
        return None


def run_collect(client, entries=(), env=ENV, skip=None, budget=60):
    sleeps: list[float] = []

    async def fake_sleep(s):
        sleeps.append(s)

    async def main():
        return await ta.collect_async(client, list(entries), client.now, 48, skip, (env.get("TELEGRAM_AUTO_JOIN") == "1"), budget, sleep=fake_sleep)

    return asyncio.run(main()), sleeps


@pytest.fixture
def account(now):
    me = User(1, "me_myself")
    bot = User(2, "price_bot", bot=True)
    news = Channel(10, "Crypto News Daily", username="cryptonewsdaily")
    private_group = Chat(11, PRIVATE_TITLE)
    private_channel = Channel(12, "Secret Signals VIP", username=None)
    public_super = Channel(13, "ETH Traders", username=None, megagroup=True,
                           usernames=[SimpleNamespace(username="eth_traders", active=True)])
    history = {
        10: [message(500, now, 1, "Bitcoin ETF approved, bullish!", views=12000),
             message(499, now, 30, "Solana outage again", views=800),
             message(498, now, 60, "too old", views=1)],
        11: [message(70, now, 2, PRIVATE_TEXT), message(69, now, 3, "", action=None),
             message(68, now, 4, "joined", action=SimpleNamespace(kind="join"))],
        12: [message(5, now, 5, "Sell Ethereum now, crash incoming")],
        13: [message(900, now, 6, "ETH breakout confirmed", views=None)],
    }
    client = FakeClient(now, dialogs=[dialog(me), dialog(bot), dialog(news), dialog(private_group), dialog(private_channel), dialog(public_super)],
                        history=history)
    return client


def test_disabled_without_secrets(now):
    result = ta.collect([], now, env={})
    assert result.status == "disabled"
    assert result.detail == "no TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_SESSION secrets"
    result = ta.collect([], now, env={"TELEGRAM_API_ID": "1", "TELEGRAM_API_HASH": "x", "TELEGRAM_SESSION": "  "})
    assert result.detail == "no TELEGRAM_SESSION secret"


def test_reads_group_and_channel_dialogs_only(account):
    result, _ = run_collect(account)
    assert result.status == "ok"
    assert sorted(account.history_reads) == [10, 11, 12, 13]  # users and bots never opened
    by_text = {m.text: m for m in result.items}
    assert set(by_text) == {"Bitcoin ETF approved, bullish!", "Solana outage again", PRIVATE_TEXT,
                            "Sell Ethereum now, crash incoming", "ETH breakout confirmed"}
    pub = by_text["Bitcoin ETF approved, bullish!"]
    assert pub.public and pub.url == "https://t.me/cryptonewsdaily/500" and pub.community == "cryptonewsdaily"
    assert pub.key == "tg:cryptonewsdaily/500" and pub.engagement == 12000
    assert by_text["ETH breakout confirmed"].url == "https://t.me/eth_traders/900"  # usernames list
    for text in (PRIVATE_TEXT, "Sell Ethereum now, crash incoming"):
        priv = by_text[text]
        assert not priv.public and priv.url is None and priv.community is None
    assert result.detail.startswith("4/4 dialogs")


def test_private_chats_never_leak(account, now, caplog):
    caplog.set_level(logging.DEBUG)
    result, _ = run_collect(account)
    others = [SourceResult(n, "disabled") for n in ("telegram_public", "rss", "reddit")]
    assets = [Asset("BTC", "Bitcoin"), Asset("ETH", "Ethereum"), Asset("SOL", "Solana")]
    report = score_messages([m for m in result.items if m.ts > now - timedelta(hours=24)], ["vader"])
    agg = aggregate(report.scored, [], build_matchers(assets), now, report.models)
    feed = build_feed(now, 24, report.models, [others[0], result, others[1], others[2]], agg.market, agg.assets)
    assert validate_feed(feed) == []
    btc = next(a for a in feed["assets"] if a["symbol"] == "BTC")
    assert btc["mentions"] == 2  # the private message counts ...
    assert btc["bySource"]["telegram_account"]["mentions"] == 2
    text = dumps(feed)
    assert PRIVATE_TEXT not in text and "private plan" not in text  # ... but is never shown
    for secret in (PRIVATE_TITLE, "Secret Signals VIP", "Sell Ethereum now"):
        assert secret not in text
        assert secret not in caplog.text
        assert secret not in result.detail
    eth = next(a for a in feed["assets"] if a["symbol"] == "ETH")
    assert [s["url"] for s in eth["samples"]] == ["https://t.me/eth_traders/900"]
    assert json.loads(text)["sources"][1]["status"] == "ok"


def test_catalogue_lookups_rotate_between_runs(now):
    handles = [f"chat_{i}" for i in range(10)]
    seen: set[str] = set()
    for hour in range(5):
        client = FakeClient(now + timedelta(hours=hour), dialogs=[], history={}, entities={})
        reader = ta.AccountReader(client, client.now, max_resolves=2)
        asyncio.run(reader.read_catalogue([ChannelEntry(h, kind="group") for h in handles], set(), set(), False))
        assert len(client.resolved) == 2
        seen.update(client.resolved)
    assert seen == set(handles)  # five hourly runs of two lookups cover all ten handles


def test_catalogue_handles_read_without_joining(now):
    group = Channel(20, "BTC Chat", username="btc_chat", megagroup=True)
    channel = Channel(21, "Alpha", username="alpha_calls")
    person = User(22, "some_person")
    client = FakeClient(
        now,
        dialogs=[dialog(Channel(10, "News", username="cryptonewsdaily"))],
        history={10: [], 20: [message(1, now, 1, "BTC to 100k")], 21: [message(2, now, 2, "SOL calls")]},
        entities={"btc_chat": group, "alpha_calls": channel, "some_person": person},
    )
    entries = [
        ChannelEntry("cryptonewsdaily", kind="channel"),  # already a dialog
        ChannelEntry("covered_by_preview", kind="channel"),  # the public preview read it already
        ChannelEntry("btc_chat", kind="group", confidence=0.9),
        ChannelEntry("alpha_calls", kind="channel"),
        ChannelEntry("some_person", kind="channel"),  # resolves to a user -> never read
        ChannelEntry("missing_handle", kind="group"),
    ]
    result, _ = run_collect(client, entries, skip={"covered_by_preview"})
    assert sorted(client.resolved) == ["alpha_calls", "btc_chat", "missing_handle", "some_person"]
    assert client.requests == []  # nothing joined
    assert sorted(client.history_reads) == [10, 20, 21]
    assert {m.url for m in result.items} == {"https://t.me/btc_chat/1", "https://t.me/alpha_calls/2"}
    assert "2 catalogue handles" in result.detail and "2 unreadable" in result.detail


def test_skip_handles_can_be_a_callable(now):
    client = FakeClient(now, entities={"a_chan": Channel(30, "A", username="a_chan")}, history={30: [message(1, now, 1, "hi")]})
    result, _ = run_collect(client, [ChannelEntry("a_chan")], skip=lambda: {"a_chan"})
    assert client.resolved == [] and result.status == "empty"


def test_flood_wait_short_sleeps_long_skips(now):
    chan = Channel(40, "Busy", username="busy_chan")
    client = FakeClient(
        now,
        dialogs=[dialog(chan)],
        history={40: [message(1, now, 1, "Bitcoin up")], 41: [message(2, now, 1, "ETH up")]},
        entities={"short_wait": Channel(41, "S", username="short_wait"), "long_wait": Channel(42, "L", username="long_wait"),
                  "after_block": Channel(43, "A", username="after_block")},
        get_entity_errors={"short_wait": [FloodWaitError(request=None, capture=30)],
                           "long_wait": [FloodWaitError(request=None, capture=3600)]},
        history_errors={40: [FloodWaitError(request=None, capture=5)]},
    )
    entries = [ChannelEntry("short_wait"), ChannelEntry("long_wait"), ChannelEntry("after_block")]
    result, sleeps = run_collect(client, entries)
    assert sleeps == [5, 30]  # history flood (5 s) and resolve flood (30 s) slept and retried
    assert client.resolved == ["short_wait", "short_wait", "long_wait"]  # long flood blocks further lookups
    assert {m.text for m in result.items} == {"Bitcoin up", "ETH up"}
    assert "1 flood-wait skips" in result.detail and "1 lookups deferred" in result.detail


def test_flood_wait_on_history_over_60s_skips_chat(now):
    chan = Channel(50, "X", username="x_chan")
    client = FakeClient(now, dialogs=[dialog(chan)], history={50: [message(1, now, 1, "hi")]},
                        history_errors={50: [FloodWaitError(request=None, capture=61)]})
    result, sleeps = run_collect(client)
    assert sleeps == [] and result.items == [] and result.status == "rate_limited"


def test_auto_join_only_with_env_and_at_most_five(now):
    groups = {f"g{i}_chat": Channel(100 + i, f"G{i}", username=f"g{i}_chat", megagroup=True) for i in range(7)}
    entries = [ChannelEntry(f"g{i}_chat", kind="group", confidence=i / 10) for i in range(7)]
    entries.append(ChannelEntry("a_channel", kind="channel", confidence=1.0))
    groups["a_channel"] = Channel(200, "C", username="a_channel")
    history = {c.id: [message(1, now, 1, f"msg {c.id}")] for c in groups.values()}

    client = FakeClient(now, entities=dict(groups), history=history)
    run_collect(client, entries)
    assert client.requests == []

    client = FakeClient(now, entities=dict(groups), history=history)
    result, _ = run_collect(client, entries, env={**ENV, "TELEGRAM_AUTO_JOIN": "1"})
    joined = [r.channel.username for r in client.requests]
    assert joined == ["g6_chat", "g5_chat", "g4_chat", "g3_chat", "g2_chat"]  # groups only, by confidence
    assert "joined 5" in result.detail
    assert len(result.items) == 8  # joined groups and the rest are all read once
    assert sorted(client.history_reads) == sorted(c.id for c in groups.values())


def test_unauthorized_session(now):
    result, _ = run_collect(FakeClient(now, authorized=False))
    assert result.status == "unauthorized"
    assert "login.py" in result.detail


def test_collect_wrapper_uses_factory_and_env(now):
    created = {}

    def factory(api_id, api_hash, session):
        created.update(api_id=api_id, api_hash=api_hash, session=session)
        return FakeClient(now, dialogs=[dialog(Channel(1, "N", username="n_chan"))], history={1: [message(1, now, 1, "x")]})

    result = ta.collect([], now, env=ENV, client_factory=factory)
    assert created == {"api_id": 12345, "api_hash": "0123456789abcdef", "session": "1FakeStringSession"}
    assert result.status == "ok" and len(result.items) == 1
    bad = ta.collect([], now, env={**ENV, "TELEGRAM_API_ID": "abc"}, client_factory=factory)
    assert bad.status == "error"


def test_entity_helpers():
    assert ta.entity_username(Channel(1, "t", usernames=[SimpleNamespace(username="old", active=False),
                                                         SimpleNamespace(username="new", active=True)])) == "new"
    assert ta.entity_username(Chat(1, "t")) is None
    assert ta.is_chat_entity(Channel(1, "t")) and ta.is_chat_entity(Chat(1, "t"))
    assert not ta.is_chat_entity(User(1))


def test_long_flood_wait_while_listing_dialogs_is_rate_limited(now):
    class FloodingDialogs(FakeClient):
        async def iter_dialogs(self):
            raise FloodWaitError(request=None, capture=900)
            yield  # pragma: no cover - makes this an async generator

    result, _ = run_collect(FloodingDialogs(now))
    assert result.status == "rate_limited"
    assert "flood wait 900s" in result.detail
