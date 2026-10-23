from methods.lenormand import CARDS, LenormandMethod
from methods.tarot import DECK, MAJORS, TarotMethod, card_chain
from tests.helpers import assert_reproducible, assert_valid_output


def test_tarot_deck():
    assert len(DECK) == 78
    assert sorted(c["index"] for c in DECK) == list(range(1, 79))
    assert MAJORS[8] == "Strength" and MAJORS[11] == "Justice"
    fool = next(c for c in DECK if c["name"] == "The Fool")
    assert fool["index"] == 22


def test_greer_birth_cards():
    assert card_chain(5 + 31 + 1991) == [11, 2]   # Justice, High Priestess
    assert card_chain(5 + 31 + 2026) == [10, 1]   # year card for 2026
    assert card_chain(2002) == [4]
    assert card_chain(19) == [19, 10, 1]
    assert card_chain(22) == [22, 4]


def test_tarot_method(subject, event, games):
    out = TarotMethod({"spread_size": 22}).run(subject, event, games.values())
    assert_valid_output(out, games)
    assert out.intermediate["birth_cards"]["names"] == ["Justice", "The High Priestess"]
    assert {11, 2} <= set(out.picks["sve_ili_nista"]["main"])
    assert_reproducible(TarotMethod, {"spread_size": 22}, subject, event, games)


def test_lenormand_cards():
    assert len(CARDS) == 36 and len(set(CARDS)) == 36
    assert CARDS[0] == "Rider" and CARDS[30] == "Sun" and CARDS[35] == "Cross"


def test_lenormand_method(subject, event, games):
    out = LenormandMethod({}).run(subject, event, games.values())
    assert_valid_output(out, games)
    rows = out.intermediate["grand_tableau_rows"]
    assert sorted(c["number"] for r in rows for c in r) == list(range(1, 37))
    assert out.intermediate["birth_day_card"]["name"] == "Sun"
    assert_reproducible(LenormandMethod, {}, subject, event, games)
