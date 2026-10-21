"""Crypto slang overlay for VADER (``core/src/main/resources/sentiment/crypto_lexicon.tsv``) and
the text normalizer that must behave exactly like the Kotlin one.

File format::

    # comment
    term<TAB>valence<TAB>optional comment
    to the moon<TAB>3.0<TAB>multi-word phrase, stored as "to_the_moon"
    🚀<TAB>2.5<TAB>emoji_rocket

Normalization before VADER, in this order:

(a) URLs (``https?://\\S+|www\\.\\S+``) -> one space
(b) ``$`` or ``#`` immediately followed by an ASCII letter is removed (``$BTC`` -> ``BTC``)
(c) multi-word phrases, longest first (ties: alphabetical), case-insensitive, whole word
    (boundary = a character that is not ``[A-Za-z0-9]``, or the start/end of the text) ->
    the phrase lowercased with spaces replaced by underscores; each phrase is applied to the
    whole text before the next one
(d) mapped emoji (longest sequence first) -> `` emoji_<name> ``
(e) collapse whitespace runs to one space and trim
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("sentiment_hub.lexicon")

URL_RE = re.compile(r"https?://\S+|www\.\S+")
TAG_RE = re.compile(r"[$#](?=[A-Za-z])")
WS_RE = re.compile(r"\s+")
_EMOJI_TOKEN_RE = re.compile(r"^emoji_[a-z0-9_]+$")
_HAS_ALNUM_RE = re.compile(r"[A-Za-z0-9]")


@dataclass
class Lexicon:
    #: token -> VADER valence. Phrase keys use underscores ("to_the_moon"), emoji keys are the
    #: emoji token names ("emoji_rocket").
    valences: dict[str, float] = field(default_factory=dict)
    #: Lowercase phrases with single spaces ("to the moon").
    phrases: list[str] = field(default_factory=list)
    #: Emoji sequence -> token name ("🚀" -> "emoji_rocket").
    emojis: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.phrases = sorted(set(self.phrases), key=lambda p: (-len(p), p))
        self._phrase_rules = [
            (
                # case-insensitive phrase only: a global IGNORECASE would let [A-Za-z] match U+0130/U+0131/U+017F/U+212A
                re.compile(r"(?<![A-Za-z0-9])(?i:" + re.escape(p) + r")(?![A-Za-z0-9])"),
                p.replace(" ", "_"),
            )
            for p in self.phrases
        ]
        self._emoji_order = sorted(self.emojis, key=lambda e: (-len(e), e))

    def normalize(self, text: str) -> str:
        s = URL_RE.sub(" ", text or "")
        s = TAG_RE.sub("", s)
        for rx, key in self._phrase_rules:
            s = rx.sub(key, s)
        for emoji in self._emoji_order:
            if emoji in s:
                s = s.replace(emoji, f" {self.emojis[emoji]} ")
        return WS_RE.sub(" ", s).strip()

    def __len__(self) -> int:
        return len(self.valences)


EMPTY_LEXICON = Lexicon()


def parse_lexicon(text: str, origin: str = "<lexicon>") -> Lexicon:
    valences: dict[str, float] = {}
    phrases: list[str] = []
    emojis: dict[str, str] = {}
    bad = 0
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip("\r\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 2:
            bad += 1
            log.debug("%s:%d: expected term<TAB>valence", origin, lineno)
            continue
        term = cols[0].strip()
        try:
            valence = float(cols[1].strip())
        except ValueError:
            bad += 1
            log.debug("%s:%d: bad valence %r", origin, lineno, cols[1])
            continue
        if not term:
            bad += 1
            continue
        third = cols[2].strip().lower() if len(cols) > 2 else ""
        if _EMOJI_TOKEN_RE.match(third) and not _HAS_ALNUM_RE.search(term):
            emojis[term] = third
            valences[third] = valence
            continue
        normalized = " ".join(term.lower().split())
        if " " in normalized:
            phrases.append(normalized)
            valences[normalized.replace(" ", "_")] = valence
        else:
            valences[normalized] = valence
    if bad:
        log.warning("%s: skipped %d malformed lines", origin, bad)
    return Lexicon(valences=valences, phrases=phrases, emojis=emojis)


def load_lexicon(path: str | Path | None) -> tuple[Lexicon, str]:
    """Returns (lexicon, note). Missing file -> empty overlay (plain VADER), never raises."""
    if path is None:
        return Lexicon(), "none"
    file = Path(path)
    try:
        text = file.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        log.warning("crypto lexicon %s not found; VADER runs without the crypto overlay", file)
        return Lexicon(), "missing"
    except OSError as exc:
        log.warning("crypto lexicon %s unreadable (%s); VADER runs without the crypto overlay", file, exc)
        return Lexicon(), "missing"
    lexicon = parse_lexicon(text, origin=str(file))
    log.info(
        "crypto lexicon: %d entries (%d phrases, %d emoji)", len(lexicon.valences), len(lexicon.phrases), len(lexicon.emojis)
    )
    return lexicon, "loaded"
