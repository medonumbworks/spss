"""Python port of ``core/src/main/kotlin/dev/cryptopulse/core/repo/MentionMatcher.kt``.

Keep the two in sync: the app and the hub must agree on which texts talk about which asset.

* ``$SYM`` (cashtag, case-insensitive) always counts.
* The full name (parenthesised parts removed, case-insensitive, whole word) counts; for names of
  two or more words the first word also counts when it has 4+ letters and is not generic.
  A name (or first word) equal to an ambiguous symbol never counts ("Sun" for SUN, "NEAR" for NEAR),
  and an unresolved listing whose name is just its ticker (no CoinGecko id) has no name match.
* ``EXTRA_NAMES`` adds common names that differ from the listing name ("Dogecoin" for DOGE).
* Names that are everyday words (``COMMON_WORD_NAMES``: "Stellar", "Vision", ...) only count when
  followed by a crypto context word ("Stellar network", "Vision token").
* The bare symbol (case-sensitive, whole word) counts only when the symbol is not ambiguous
  (length <= 2 or listed in ``AMBIGUOUS_SYMBOLS``).
"""

from __future__ import annotations

import re
from typing import Iterable

from sentiment_hub.assets import Asset

# Copied verbatim from MentionMatcher.AMBIGUOUS_SYMBOLS.
AMBIGUOUS_SYMBOLS: frozenset[str] = frozenset(
    {
        "A", "ACT", "ACX", "ALGO", "APE", "AR", "ASTER", "ATH", "AVA", "AWE", "B", "BAND", "BAT", "BEAM",
        "BIO", "BLAST", "BONK", "CAKE", "CC", "CELO", "COMP", "CORE", "DASH", "DEEP", "ENS", "FF", "FLOW",
        "GOAT", "GRASS", "GRT", "GT", "HOME", "ID", "IP", "JUP", "KAS", "KITE", "LINK", "LIT", "M", "MANA",
        "ME", "MEME", "MET", "MEW", "MOG", "MON", "MONAD", "MORPHO", "MOVE", "NEAR", "NEO", "NOT", "ONDO",
        "ONE", "OP", "ORCA", "PI", "PLASMA", "PUMP", "RAY", "RENDER", "S", "SAND", "SONIC", "SOON", "SPX",
        "STRK", "SUN", "SUPER", "TEL", "TREE", "TRUMP", "USELESS", "USUAL", "VISION", "W", "WAL", "WIF",
        "ZEN", "ZK",
    }
)

# Copied verbatim from MentionMatcher.GENERIC_WORDS.
GENERIC_WORDS: frozenset[str] = frozenset(
    {
        "across", "basic", "bitcoin", "boba", "chain", "coin", "defi", "enzyme", "ethereum", "eur",
        "finance", "global", "internet", "network", "oasis", "official", "protocol", "reserve", "ripple",
        "rocket", "sahara", "standard", "the", "token", "trust", "usd", "vulcan", "world", "yield",
    }
)

# Copied verbatim from MentionMatcher.COMMON_WORD_NAMES: names that are everyday words need a context word.
COMMON_WORD_NAMES: frozenset[str] = frozenset(
    {
        "aster", "aztec", "balancer", "band", "beam", "blast", "book", "brett", "canton", "compound", "core",
        "cosmos", "curve", "dash", "dusk", "falcon", "flare", "flow", "gala", "gate", "golem", "grass",
        "harmony", "home", "icon", "immutable", "jupiter", "just", "kava", "kite", "lighter", "magic",
        "mantle", "maple", "memecoin", "metal", "mina", "monad", "movement", "nano", "near", "ontology",
        "optimism", "orca", "osmosis", "peanut", "pi", "plasma", "polygon", "quant", "render", "sonic",
        "soon", "space", "stacks", "stargate", "stellar", "story", "sun", "symbol", "syndicate", "the graph",
        "theta", "tree", "treehouse", "tron", "turbo", "useless", "useless coin", "vision", "walrus",
        "waves",
    }
)

# Copied verbatim from MentionMatcher.CONTEXT_WORDS.
CONTEXT_WORDS = "token|tokens|coin|coins|network|protocol|price|crypto|chain|blockchain|ecosystem|foundation|labs|mainnet|staking|airdrop|etf|finance|dao|dex|swap"

# Copied verbatim from MentionMatcher.EXTRA_NAMES.
EXTRA_NAMES: dict[str, tuple[str, ...]] = {"DOGE": ("Dogecoin",)}

_PARENS_RE = re.compile(r"\s*\(.*?\)\s*")
_SPACES_RE = re.compile(r"\s+")


def build_names(name: str, symbol: str = "", ambiguous: bool = False, placeholder: bool = False) -> list[str]:
    """``buildNames``: "Bitpanda Ecosystem Token" -> ["Bitpanda Ecosystem Token", "Bitpanda"]."""
    cleaned = _PARENS_RE.sub(" ", name).strip()
    out: list[str] = []
    if len(cleaned) >= 3 and not placeholder and not (ambiguous and cleaned.lower() == symbol.lower()):
        out.append(cleaned)
    words = [w for w in _SPACES_RE.split(cleaned) if w.strip()]
    if len(words) >= 2:
        first = words[0]
        if (
            len(first) >= 4
            and first.lower() not in GENERIC_WORDS
            and not (ambiguous and first.lower() == symbol.lower())
            and first not in out
        ):
            out.append(first)
    return out


def _name_regex(name: str) -> re.Pattern[str]:
    if name.lower() in COMMON_WORD_NAMES:
        return re.compile(r"\b" + re.escape(name) + r"\s+(" + CONTEXT_WORDS + r")\b", re.IGNORECASE)
    return re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)


class MentionMatcher:
    def __init__(self, asset: Asset) -> None:
        self.symbol = asset.symbol
        self.ambiguous = len(asset.symbol) <= 2 or asset.symbol in AMBIGUOUS_SYMBOLS
        self._cashtag = re.compile(r"\$" + re.escape(asset.symbol) + r"\b", re.IGNORECASE)
        self._bare = re.compile(r"\b" + re.escape(asset.symbol) + r"\b")
        placeholder = asset.coingecko_id is None and asset.name == asset.symbol
        names = build_names(asset.name, asset.symbol, self.ambiguous, placeholder) + list(EXTRA_NAMES.get(asset.symbol, ()))
        self._names = [_name_regex(n) for n in names]
        # Every regex match implies one of these appears in the case-folded text, so checking
        # them first only skips texts that cannot match (pure speed-up, same results).
        self._needles = tuple({asset.symbol.casefold(), *(n.casefold() for n in names)})

    def matches(self, text: str | None, folded: str | None = None) -> bool:
        """``folded`` may pass ``text.casefold()`` when the caller matches many assets."""
        if not text or not text.strip():
            return False
        haystack = folded if folded is not None else text.casefold()
        if not any(n in haystack for n in self._needles):
            return False
        if self._cashtag.search(text):
            return True
        if any(rx.search(text) for rx in self._names):
            return True
        if self.ambiguous:
            return False
        return self._bare.search(text) is not None


def build_matchers(assets: Iterable[Asset]) -> list[MentionMatcher]:
    return [MentionMatcher(a) for a in assets]


def matched_symbols(text: str, matchers: Iterable[MentionMatcher]) -> list[str]:
    return [m.symbol for m in matchers if m.matches(text)]
