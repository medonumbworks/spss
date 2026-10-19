"""Loaders for the shared data files (Bitpanda asset snapshot and Telegram channel catalogue).

Both files live in ``core/src/main/resources`` and are maintained by other parts of the project.
When a file is missing or broken the hub logs a warning and falls back to a small built-in list
so the hourly run still publishes a useful feed.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from sentiment_hub.common import REPO_ROOT

log = logging.getLogger("sentiment_hub.assets")

DEFAULT_ASSETS_PATH = REPO_ROOT / "core/src/main/resources/bitpanda/bitpanda_snapshot.json"
DEFAULT_CHANNELS_PATH = REPO_ROOT / "core/src/main/resources/telegram/channels.json"
DEFAULT_LEXICON_PATH = REPO_ROOT / "core/src/main/resources/sentiment/crypto_lexicon.tsv"

CHANNEL_KINDS = ("channel", "group")
_HANDLE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{3,63}$")


@dataclass(frozen=True)
class Asset:
    symbol: str
    name: str
    coingecko_id: str | None = None
    rank: int | None = None
    verified: bool = True


@dataclass(frozen=True)
class ChannelEntry:
    handle: str
    title: str = ""
    kind: str = "channel"
    lang: str = "en"
    category: str = "community"
    symbols: tuple[str, ...] = field(default_factory=tuple)
    confidence: float = 0.5


# Used only when bitpanda_snapshot.json is missing: the largest Bitpanda assets.
FALLBACK_ASSETS: tuple[Asset, ...] = tuple(
    Asset(symbol=s, name=n, coingecko_id=c, rank=i + 1)
    for i, (s, n, c) in enumerate(
        [
            ("BTC", "Bitcoin", "bitcoin"),
            ("ETH", "Ethereum", "ethereum"),
            ("USDT", "Tether", "tether"),
            ("BNB", "BNB", "binancecoin"),
            ("XRP", "XRP", "ripple"),
            ("USDC", "USD Coin", "usd-coin"),
            ("SOL", "Solana", "solana"),
            ("TRX", "Tron", "tron"),
            ("DOGE", "Doge", "dogecoin"),
            ("LINK", "Chainlink", "chainlink"),
            ("ADA", "Cardano", "cardano"),
            ("XLM", "Stellar", "stellar"),
            ("LTC", "Litecoin", "litecoin"),
            ("UNI", "Uniswap", "uniswap"),
            ("AVAX", "Avalanche", "avalanche-2"),
            ("SUI", "Sui", "sui"),
            ("HBAR", "Hedera", "hedera-hashgraph"),
            ("SHIB", "SHIBA INU", "shiba-inu"),
            ("DOT", "Polkadot", "polkadot"),
            ("PEPE", "Pepe", "pepe"),
            ("AAVE", "Aave", "aave"),
            ("ARB", "Arbitrum", "arbitrum"),
            ("POL", "Polygon", "polygon-ecosystem-token"),
            ("ATOM", "Cosmos", "cosmos"),
            ("BEST", "Bitpanda Ecosystem Token", "bitpanda-ecosystem-token"),
        ]
    )
)

# Used only when channels.json is missing (same list as AppConfig.DEFAULT_TELEGRAM_CHANNELS).
FALLBACK_CHANNELS: tuple[ChannelEntry, ...] = (
    ChannelEntry("cointelegraph", "Cointelegraph", "channel", "en", "news", (), 0.9),
    ChannelEntry("CoinDeskGlobal", "CoinDesk", "channel", "en", "news", (), 0.8),
    ChannelEntry("whale_alert_io", "Whale Alert", "channel", "en", "whales", (), 0.8),
    ChannelEntry("binance_announcements", "Binance Announcements", "channel", "en", "exchange", (), 0.8),
    ChannelEntry("crypto", "Crypto", "channel", "en", "news", (), 0.6),
)


def normalize_symbol(raw: str) -> str:
    """Mirror of ``Text.normalizeSymbol``: uppercase letters and digits only."""
    return "".join(ch for ch in raw.strip().upper() if ch.isalnum())


def normalize_handle(raw: str | None) -> str | None:
    """``@Foo``, ``https://t.me/foo``, ``t.me/s/foo/`` -> ``Foo``/``foo``; None when invalid."""
    if not raw:
        return None
    text = raw.strip()
    text = re.sub(r"^(https?://)?(www\.)?(t\.me|telegram\.me)/", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^s/", "", text)
    text = text.lstrip("@").strip("/").split("?")[0].split("/")[0]
    return text if _HANDLE_RE.match(text) else None


def load_assets(path: str | Path | None = None) -> tuple[list[Asset], str]:
    """Returns (assets, note). Never raises."""
    file = Path(path) if path else DEFAULT_ASSETS_PATH
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
        raw_assets = data["assets"]
    except FileNotFoundError:
        log.warning("asset snapshot %s not found; using %d built-in assets", file, len(FALLBACK_ASSETS))
        return list(FALLBACK_ASSETS), "fallback"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log.warning("asset snapshot %s unreadable (%s); using built-in assets", file, exc)
        return list(FALLBACK_ASSETS), "fallback"

    out: list[Asset] = []
    seen: set[str] = set()
    for item in raw_assets if isinstance(raw_assets, list) else []:
        if not isinstance(item, dict):
            continue
        symbol = normalize_symbol(str(item.get("symbol") or ""))
        name = str(item.get("name") or "").strip()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        rank = item.get("rank")
        out.append(
            Asset(
                symbol=symbol,
                name=name or symbol,
                coingecko_id=item.get("coingeckoId") or item.get("coingecko_id"),
                rank=rank if isinstance(rank, int) else None,
                verified=bool(item.get("verified", True)),
            )
        )
    if not out:
        log.warning("asset snapshot %s has no assets; using built-in assets", file)
        return list(FALLBACK_ASSETS), "fallback"
    return out, "snapshot"


def load_channels(path: str | Path | None = None) -> tuple[list[ChannelEntry], str]:
    """Returns (entries sorted by confidence desc, note). Never raises. Duplicate handles
    (case-insensitive) keep the entry with the highest confidence."""
    file = Path(path) if path else DEFAULT_CHANNELS_PATH
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
        raw_entries = data["entries"]
    except FileNotFoundError:
        log.warning("channel catalogue %s not found; using %d built-in channels", file, len(FALLBACK_CHANNELS))
        return list(FALLBACK_CHANNELS), "fallback"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log.warning("channel catalogue %s unreadable (%s); using built-in channels", file, exc)
        return list(FALLBACK_CHANNELS), "fallback"

    best: dict[str, ChannelEntry] = {}
    invalid = 0
    for item in raw_entries if isinstance(raw_entries, list) else []:
        if not isinstance(item, dict):
            invalid += 1
            continue
        handle = normalize_handle(str(item.get("handle") or ""))
        if handle is None:
            invalid += 1
            continue
        kind = str(item.get("kind") or "channel").lower()
        try:
            confidence = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        symbols = item.get("symbols") or []
        entry = ChannelEntry(
            handle=handle,
            title=str(item.get("title") or ""),
            kind=kind if kind in CHANNEL_KINDS else "channel",
            lang=str(item.get("lang") or ""),
            category=str(item.get("category") or ""),
            symbols=tuple(normalize_symbol(str(s)) for s in symbols if isinstance(s, str)),
            confidence=confidence,
        )
        key = handle.lower()
        if key not in best or best[key].confidence < entry.confidence:
            best[key] = entry
    if invalid:
        log.warning("channel catalogue: skipped %d invalid entries", invalid)
    if not best:
        log.warning("channel catalogue %s has no usable entries; using built-in channels", file)
        return list(FALLBACK_CHANNELS), "fallback"
    entries = sorted(best.values(), key=lambda e: (-e.confidence, e.handle.lower()))
    return entries, "catalogue"
