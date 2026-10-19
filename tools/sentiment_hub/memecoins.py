"""Memecoins and new tickers beyond the Bitpanda list.

* :func:`fetch_memecoins` reads the 250 largest coins of CoinGecko's "Meme" category (one keyless
  call), so the hub counts mentions of memecoins Bitpanda does not list. The app shows them as
  its MEME tier and matches feed entries by symbol.
* :func:`discover_cashtags` finds ``$TICKER`` cashtags that several communities use but no known
  asset matches: brand-new DEX tokens. They are tracked by cashtag only (never by name or bare
  ticker), so ordinary words cannot be counted as mentions.
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from typing import Callable, Iterable, Sequence

from sentiment_hub.assets import Asset, normalize_symbol
from sentiment_hub.common import Message
from sentiment_hub.net import NetError

log = logging.getLogger(__name__)

MEME_URL = (
    "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&category=meme-token"
    "&order=market_cap_desc&per_page=250&page=1&sparkline=false"
)
RETRY_AFTER_429_S = 20.0

# "$" + a letter + up to 14 letters/digits, not inside a word or a "$$" run; "$100k" is a price, not a ticker
CASHTAG_RE = re.compile(r"(?<![\w$])\$([A-Za-z][A-Za-z0-9]{1,14})\b")
# never tokens of their own: quote assets, fiat, stablecoins, and stock or ETF tickers that crypto news uses
NOT_TOKENS = frozenset(
    {
        "USD", "USDT", "USDC", "EUR", "GBP", "CHF", "JPY", "SOL", "ETH", "BTC", "BNB", "WETH", "WSOL", "DAI", "USDE", "USD1", "FDUSD",
        "PYUSD", "SPX", "SPY", "QQQ", "DXY", "IWM", "DIA", "VIX", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOG", "GOOGL", "META",
        "NFLX", "AMD", "INTC", "AVGO", "TSM", "ARM", "SMCI", "PLTR", "ORCL", "CRWV", "COIN", "HOOD", "MSTR", "MARA", "RIOT", "CLSK",
        "HUT", "CIFR", "IREN", "GLXY", "CRCL", "BMNR", "SBET", "GME", "AMC", "IBIT", "ETHA", "FBTC", "GBTC", "BITO", "SPCX",
    }
)


def fetch_memecoins(http, retries: int = 1, sleep: Callable[[float], None] = time.sleep) -> tuple[list[Asset], str]:
    """Returns (memecoins, note). Never raises; an empty list when CoinGecko refuses."""
    for attempt in range(retries + 1):
        try:
            resp = http.get(MEME_URL, headers={"Accept": "application/json"})
        except NetError as exc:
            return [], f"unavailable ({exc})"
        if resp.status == 429 and attempt < retries:
            log.info("memecoins: CoinGecko answered 429; retrying in %.0fs", RETRY_AFTER_429_S)
            sleep(RETRY_AFTER_429_S)
            continue
        if resp.status != 200:
            return [], f"HTTP {resp.status}"
        try:
            rows = resp.json()
        except ValueError:
            return [], "unreadable response"
        out: list[Asset] = []
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            symbol = normalize_symbol(str(row.get("symbol") or ""))
            cg = row.get("id")
            # "4" or "67" would only ever match prices ("$4.98")
            if not symbol or not isinstance(cg, str) or symbol in NOT_TOKENS or not any(ch.isalpha() for ch in symbol):
                continue
            rank = row.get("market_cap_rank")
            out.append(Asset(symbol=symbol, name=str(row.get("name") or symbol).strip(), coingecko_id=cg, rank=rank if isinstance(rank, int) else None, verified=False, listed=False))
        return out, f"{len(out)} memecoins"
    return [], "HTTP 429"


def merge_assets(base: Sequence[Asset], extra: Iterable[Asset]) -> list[Asset]:
    """``base`` (Bitpanda) wins a symbol; among extras the first (largest market cap) wins."""
    out = list(base)
    seen = {a.symbol for a in base}
    ids = {a.coingecko_id for a in base if a.coingecko_id}
    for a in extra:
        if a.symbol in seen or (a.coingecko_id and a.coingecko_id in ids):
            continue
        seen.add(a.symbol)
        out.append(a)
    return out


def discover_cashtags(
    messages: Iterable[Message],
    known: set[str],
    min_mentions: int = 3,
    min_communities: int = 2,
    limit: int = 60,
) -> list[str]:
    """Cashtags of unknown tickers that at least ``min_communities`` different channels/feeds used
    ``min_mentions`` times in total, most used first."""
    counts: dict[str, int] = defaultdict(int)
    communities: dict[str, set[str]] = defaultdict(set)
    for m in messages:
        seen_here: set[str] = set()
        for raw in CASHTAG_RE.findall(m.text or ""):
            sym = normalize_symbol(raw)
            if not sym or sym in known or sym in NOT_TOKENS or sym in seen_here:
                continue
            seen_here.add(sym)
            counts[sym] += 1
            communities[sym].add((m.community or m.source).lower())
    ranked = [s for s, n in counts.items() if n >= min_mentions and len(communities[s]) >= min_communities]
    ranked.sort(key=lambda s: (-counts[s], s))
    return ranked[:limit]
