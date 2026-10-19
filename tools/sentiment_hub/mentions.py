"""Python port of ``core/src/main/kotlin/dev/cryptopulse/core/repo/MentionMatcher.kt``.

Keep the two in sync: the app and the hub must agree on which texts talk about which asset.

* ``$SYM`` (cashtag, case-insensitive) always counts.
* The full name (parenthesised parts removed, case-insensitive, whole word) counts; for names of
  two or more words the first word also counts when it has 4+ letters and is not generic.
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
        "SUN", "GRASS", "VISION", "BEAM", "SOON", "KITE", "ONE", "FLOW", "NEAR", "PI", "LIT", "DEEP",
        "B", "M", "S", "A", "CC", "GT", "ME", "W", "OP", "AR", "ATH", "ZEN", "FF", "MET", "TEL", "USELESS",
        "PUMP", "AWE", "APE", "SUPER", "CAKE", "DASH", "NEO", "SAND", "MANA", "ORCA", "COMP", "ENS",
        "MOVE", "USUAL", "GOAT", "ACT", "BAT", "TRUMP", "IP", "RAY", "JUP", "GRT", "ALGO", "LINK", "RENDER",
        "ASTER", "PLASMA", "MONAD", "SONIC", "SOON", "MORPHO", "GRASS", "STRK", "ZK", "BONK", "WIF",
    }
)

# Copied verbatim from MentionMatcher.GENERIC_WORDS.
GENERIC_WORDS: frozenset[str] = frozenset(
    {"the", "network", "protocol", "token", "coin", "finance", "chain", "global", "official"}
)

_PARENS_RE = re.compile(r"\s*\(.*?\)\s*")
_SPACES_RE = re.compile(r"\s+")


def build_names(name: str) -> list[str]:
    """``buildNames``: "Bitpanda Ecosystem Token" -> ["Bitpanda Ecosystem Token", "Bitpanda"]."""
    cleaned = _PARENS_RE.sub(" ", name).strip()
    out: list[str] = []
    if len(cleaned) >= 3:
        out.append(cleaned)
    words = [w for w in _SPACES_RE.split(cleaned) if w.strip()]
    if len(words) >= 2:
        first = words[0]
        if len(first) >= 4 and first.lower() not in GENERIC_WORDS and first not in out:
            out.append(first)
    return out


class MentionMatcher:
    def __init__(self, asset: Asset) -> None:
        self.symbol = asset.symbol
        self.ambiguous = len(asset.symbol) <= 2 or asset.symbol in AMBIGUOUS_SYMBOLS
        self._cashtag = re.compile(r"\$" + re.escape(asset.symbol) + r"\b", re.IGNORECASE)
        self._bare = re.compile(r"\b" + re.escape(asset.symbol) + r"\b")
        names = build_names(asset.name)
        self._names = [re.compile(r"\b" + re.escape(n) + r"\b", re.IGNORECASE) for n in names]
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
