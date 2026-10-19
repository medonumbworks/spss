from __future__ import annotations

import pytest
from conftest import fixture_path

from sentiment_hub.lexicon import Lexicon, load_lexicon, parse_lexicon
from sentiment_hub.nlp import VaderScorer


@pytest.fixture(scope="module")
def lex() -> Lexicon:
    lexicon, note = load_lexicon(fixture_path("lexicon_small.tsv"))
    assert note == "loaded"
    return lexicon


def test_parse_file(lex):
    assert lex.valences["hodl"] == 2.0
    assert lex.valences["rekt"] == -3.0
    assert lex.valences["to_the_moon"] == 3.0  # phrase stored under the underscore key
    assert lex.valences["buy_the_dip"] == 1.5  # lowercased
    assert lex.valences["emoji_rocket"] == 2.5
    assert lex.valences["emoji_diamond_hands"] == 2.2
    assert lex.emojis == {"🚀": "emoji_rocket", "📉": "emoji_chart_decreasing", "💎🙌": "emoji_diamond_hands"}
    # longest first, ties alphabetical
    assert lex.phrases == ["buy the dip", "to the moon", "bull run", "rug pull", "the moon"]
    assert "broken" not in lex.valences and "this line has no valence column" not in lex.valences
    assert "to the moon" not in lex.valences


def test_missing_file_is_plain_vader(tmp_path):
    lexicon, note = load_lexicon(tmp_path / "nope.tsv")
    assert note == "missing" and len(lexicon) == 0
    assert lexicon.normalize("$BTC  to the moon") == "BTC to the moon"


def test_rule_a_urls(lex):
    assert lex.normalize("Read https://cointelegraph.com/news/x?a=1#top now and www.example.org/path too") == "Read now and too"
    assert lex.normalize("http://a.b/c") == ""


def test_rule_b_cashtags_and_hashtags(lex):
    assert lex.normalize("$BTC #Bitcoin $eth") == "BTC Bitcoin eth"
    assert lex.normalize("costs $100 #1 #_x $$") == "costs $100 #1 #_x $$"
    assert lex.normalize("a$BTC") == "aBTC"  # '$' is removed wherever a letter follows


def test_rule_c_phrases(lex):
    assert lex.normalize("BTC to the moon") == "BTC to_the_moon"
    assert lex.normalize("TO THE MOON!!") == "to_the_moon!!"
    assert lex.normalize("the moon is full") == "the_moon is full"
    assert lex.normalize("going into the moonlight") == "going into the moonlight"  # whole words only
    assert lex.normalize("x_to the moon") == "x_to_the_moon"  # '_' is a boundary (not [A-Za-z0-9])
    assert lex.normalize("2to the moon") == "2to the_moon"  # digits are not boundaries
    assert lex.normalize("Buy the dip, then hodl") == "buy_the_dip, then hodl"
    assert lex.normalize("the next bull run.") == "the next bull_run."
    # phrases need single spaces because (e) runs last: only "the moon" matches here
    assert lex.normalize("to  the moon") == "to the_moon"


def test_rule_d_emoji(lex):
    assert lex.normalize("🚀🚀BTC") == "emoji_rocket emoji_rocket BTC"
    assert lex.normalize("hold 💎🙌 not 💎") == "hold emoji_diamond_hands not 💎"
    assert lex.normalize("📉") == "emoji_chart_decreasing"


def test_rule_e_whitespace(lex):
    assert lex.normalize("  a \n\t b  c  ") == "a b c"


def test_rules_in_order(lex):
    text = "#BTC to the moon 🚀 https://t.me/x/1 $SOL rug pull"
    assert lex.normalize(text) == "BTC to_the_moon emoji_rocket SOL rug_pull"


def test_phrase_tie_break_is_alphabetical():
    lexicon = parse_lexicon("ab cd\t1\nzz yy\t1\nab cd ef\t2\n")
    assert lexicon.phrases == ["ab cd ef", "ab cd", "zz yy"]


def test_vader_overlay_changes_scores(lex):
    plain = VaderScorer(None)
    crypto = VaderScorer(lex)
    assert plain.score_text("hodl") == 0.0
    assert crypto.score_text("hodl") > 0.3
    assert crypto.score_text("total rug pull, I got rekt") < -0.5
    assert crypto.score_text("$BTC to the moon 🚀🚀") > 0.5
    assert crypto.score_text("$BTC 📉📉") < 0
    assert crypto.score_text("") == 0.0
    assert -1.0 <= crypto.score_text("best best best amazing incredible bullish 🚀🚀🚀") <= 1.0
