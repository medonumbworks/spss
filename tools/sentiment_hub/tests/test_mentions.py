"""Parity with core/src/main/kotlin/dev/cryptopulse/core/repo/MentionMatcher.kt."""

from __future__ import annotations

import re

import pytest
from conftest import REPO_ROOT, fixture_path

from sentiment_hub.assets import Asset, load_assets
from sentiment_hub.mentions import (
    AMBIGUOUS_SYMBOLS,
    COMMON_WORD_NAMES,
    CONTEXT_WORDS,
    EXTRA_NAMES,
    GENERIC_WORDS,
    MentionMatcher,
    build_names,
    matched_symbols,
)

KOTLIN = REPO_ROOT / "core/src/main/kotlin/dev/cryptopulse/core/repo/MentionMatcher.kt"


def m(symbol: str, name: str, coingecko_id: str | None = "resolved") -> MentionMatcher:
    return MentionMatcher(Asset(symbol, name, coingecko_id))


@pytest.mark.parametrize(
    "symbol,name,text,expected",
    [
        # cashtag always counts, case-insensitive, needs a word boundary after the symbol
        ("SUN", "Sun (New)", "$SUN pumps 20%", True),
        ("SUN", "Sun (New)", "loading $sun bags", True),
        ("SUN", "Sun (New)", "$SUNDAY vibes", False),
        ("M", "MemeCore", "$M breakout", True),
        ("LINK", "Chainlink", "$LINK to 30", True),
        # ambiguous symbols: bare mention ignored, name or cashtag needed
        ("M", "MemeCore", "M is the 13th letter", False),
        ("M", "MemeCore", "memecore rallies", True),
        ("LINK", "Chainlink", "LINK breaks out", False),
        ("LINK", "Chainlink", "Chainlink oracles", True),
        ("OP", "Optimism", "OP said hi", False),
        # non-ambiguous symbols: bare mention is case-sensitive and whole-word
        ("BTC", "Bitcoin", "BTC up 5%", True),
        ("BTC", "Bitcoin", "btc up 5%", False),
        ("BTC", "Bitcoin", "WBTC supply", False),
        ("BTC", "Bitcoin", "BTC/USDT", True),
        ("ETH", "Ethereum", "ETH's merge", True),
        # names are case-insensitive whole words
        ("BTC", "Bitcoin", "bitcoin miners", True),
        ("BTC", "Bitcoin", "BITCOIN!", True),
        ("BTC", "Bitcoin", "bitcoiners unite", False),
        ("SOL", "Solana", "Solana's validators", True),
        # parenthesised parts are dropped ("Sun (New)" -> "Sun"); a name equal to an ambiguous symbol needs a cashtag
        ("SUN", "Sun (New)", "the sun is shining", False),
        ("SUN", "Sun (New)", "new listing", False),
        # first-word alias for multi-word names
        ("BEST", "Bitpanda Ecosystem Token", "Bitpanda launches staking", True),
        ("PYTH", "Pyth Network", "pyth oracles", True),
        ("SAND", "The Sandbox", "the beach", False),
        ("SAND", "The Sandbox", "The Sandbox metaverse", True),
        ("USDG", "Global Dollar", "global markets", False),
        ("NEAR", "NEAR Protocol", "near the top", False),  # first word equals the ambiguous symbol
        ("NEAR", "NEAR Protocol", "NEAR Protocol upgrade", True),
        # everyday-word names need a crypto context word
        ("XLM", "Stellar", "a stellar quarter for tech stocks", False),
        ("XLM", "Stellar", "Stellar network upgrade goes live", True),
        ("XLM", "Stellar", "XLM breaks out", True),
        ("VSN", "Vision", "our vision for 2027", False),
        ("VSN", "Vision", "the Vision token merge", True),
        ("PI", "Pi", "pi is 3.14", False),
        # everyday words in names need context; some never act as a first-word alias
        ("RSR", "Reserve Rights", "Federal Reserve holds rates", False),
        ("RSR", "Reserve Rights", "Reserve Rights burns supply", True),
        ("WLFI", "World Liberty Financial", "the world's largest asset manager", False),
        ("CRV", "Curve DAO Token", "the yield curve inverted", False),
        ("CRV", "Curve DAO Token", "Curve Finance exploit drains pools", True),
        ("YGG", "Yield Guild Games", "bond yield jumps", False),
        ("GRT", "The Graph", "look at the graph", False),
        ("GRT", "The Graph", "The Graph network upgrade", True),
        ("RLUSD", "Ripple USD", "Ripple wins its case", False),
        ("STG", "Stargate Finance", "OpenAI's Stargate data centers", False),
        ("SPX", "SPX6900", "SPX hits a record as stocks rally", False),
        ("JST", "JUST", "I just bought more", False),
        ("JST", "JUST", "JUST token airdrop", True),
        ("ICX", "ICON", "an icon of the industry", False),
        ("RPL", "Rocket Pool", "rocket emoji everywhere", False),
        ("RPL", "Rocket Pool", "Rocket Pool node operators", True),
        # proper nouns that are not English words match on their own
        ("PENDLE", "Pendle", "Pendle TVL climbs to $5B", True),
        ("KAIA", "Kaia", "Kaia partners with LINE", True),
        ("ZEN", "Horizen", "Horizen migrates to Base", True),
        ("DOGE", "Doge", "Dogecoin jumps 12%", True),
        ("FLOKI", "FLOKI", "Floki Inu launches a game", True),
        # blank texts never match
        ("BTC", "Bitcoin", "   ", False),
        ("BTC", "Bitcoin", "", False),
    ],
)
def test_matcher_cases(symbol, name, text, expected):
    assert m(symbol, name).matches(text) is expected


def test_unresolved_listing_named_like_its_ticker():
    people = m("PEOPLE", "PEOPLE", coingecko_id=None)
    assert not people.matches("so many people feel safe")
    assert people.matches("PEOPLE rallies 30%")
    assert m("PEOPLE", "ConstitutionDAO", coingecko_id=None).matches("ConstitutionDAO is back")


def test_ambiguity_rule():
    assert m("PI", "Pi Network").ambiguous  # in the list
    assert m("AB", "Some Coin").ambiguous  # length <= 2
    assert not m("BTC", "Bitcoin").ambiguous


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Bitcoin", ["Bitcoin"]),
        ("Sun (New)", ["Sun"]),
        ("Bitpanda Ecosystem Token", ["Bitpanda Ecosystem Token", "Bitpanda"]),
        ("Pyth Network", ["Pyth Network", "Pyth"]),
        ("The Sandbox", ["The Sandbox"]),
        ("Global Dollar", ["Global Dollar"]),
        ("OG", []),
        ("Ab (Old) Coin", ["Ab Coin"]),
    ],
)
def test_build_names(name, expected):
    assert build_names(name) == expected


def _kotlin_set(source: str, name: str) -> set[str]:
    match = re.search(name + r"[^=]*=\s*setOf\((.*?)\)", source, re.S)
    assert match, f"{name} not found in MentionMatcher.kt"
    return set(re.findall(r'"([^"]*)"', match.group(1)))


@pytest.mark.skipif(not KOTLIN.exists(), reason="Kotlin MentionMatcher not in this checkout")
def test_lists_match_kotlin_source():
    source = KOTLIN.read_text(encoding="utf-8")
    assert _kotlin_set(source, "AMBIGUOUS_SYMBOLS") == set(AMBIGUOUS_SYMBOLS)
    assert _kotlin_set(source, "GENERIC_WORDS") == set(GENERIC_WORDS)
    assert _kotlin_set(source, "COMMON_WORD_NAMES") == set(COMMON_WORD_NAMES)
    assert f'CONTEXT_WORDS = "{CONTEXT_WORDS}"' in source
    extra = re.search(r"EXTRA_NAMES[^=]*=\s*mapOf\((.*?)\)\n", source, re.S)
    assert extra, "EXTRA_NAMES not found in MentionMatcher.kt"
    pairs = re.findall(r'"([^"]+)" to listOf\(([^)]*)\)', extra.group(1))
    assert {k: tuple(re.findall(r'"([^"]+)"', v)) for k, v in pairs} == EXTRA_NAMES


def test_matched_symbols_with_fixture_assets():
    assets, note = load_assets(fixture_path("assets_small.json"))
    assert note == "snapshot"
    symbols = [a.symbol for a in assets]
    assert symbols.count("BTC") == 1 and "USDG" in symbols  # deduped and normalized
    from sentiment_hub.mentions import build_matchers

    matchers = build_matchers(assets)
    assert matched_symbols("$ETH and Solana pump while BTC ranges", matchers) == ["BTC", "ETH", "SOL"]
    assert matched_symbols("LINK and M are ambiguous", matchers) == []


def _reference(asset: Asset, text: str) -> bool:
    """Straight transcription of MentionMatcher.matches without the substring prefilter."""
    if not text.strip():
        return False
    if re.search(r"\$" + re.escape(asset.symbol) + r"\b", text, re.IGNORECASE):
        return True
    ambiguous = len(asset.symbol) <= 2 or asset.symbol in AMBIGUOUS_SYMBOLS
    placeholder = asset.coingecko_id is None and asset.name == asset.symbol
    for n in build_names(asset.name, asset.symbol, ambiguous, placeholder) + list(EXTRA_NAMES.get(asset.symbol, ())):
        if n.lower() in COMMON_WORD_NAMES:
            pattern = r"\b" + re.escape(n) + r"\s+(" + CONTEXT_WORDS + r")\b"
        else:
            pattern = r"\b" + re.escape(n) + r"\b"
        if re.search(pattern, text, re.IGNORECASE):
            return True
    if ambiguous:
        return False
    return re.search(r"\b" + re.escape(asset.symbol) + r"\b", text) is not None


def test_prefilter_never_changes_results():
    import random

    assets, _ = load_assets(fixture_path("assets_small.json"))
    matchers = [MentionMatcher(a) for a in assets]
    pieces = ["$", "#", "BTC", "btc", "Bitcoin", "BITCOIN", "sol", "SOL", "Solana", "near", "NEAR", "Sun", "$sun", "M", "$M",
              "memecore", "Pyth", "Bitpanda", "LINK", "chainlink", "Doge", "dogecoin", "XRP", "xrp", "ETH", "Ethereum", "the",
              "Sandbox", "global", "USDG", "-", "/", "'s", "_", "1", "\U0001f680", "K", " ", "  "]
    rng = random.Random(7)
    for _ in range(3000):
        text = "".join(rng.choice(pieces) + rng.choice(["", " ", " ", "."]) for _ in range(rng.randint(1, 8)))
        for asset, matcher in zip(assets, matchers):
            assert matcher.matches(text) is _reference(asset, text), (asset.symbol, text)
            assert matcher.matches(text, text.casefold()) is _reference(asset, text), (asset.symbol, text)
