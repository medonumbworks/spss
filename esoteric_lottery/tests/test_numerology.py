from dataclasses import replace
from datetime import date

from methods.chaldean import CHALDEAN, ChaldeanMethod, chaldean_value, date_block
from methods.gematria import GematriaMethod, agrippa, greek_value, hebrew_value, ordinal, reverse_ordinal
from methods.pythagorean import LETTER_VALUES, PythagoreanMethod, cycle_numbers, date_numbers, name_value
from tests.helpers import assert_reproducible, assert_valid_output


def test_pythagorean_letter_table():
    assert [LETTER_VALUES[c] for c in "AJS"] == [1, 1, 1]
    assert [LETTER_VALUES[c] for c in "IR"] == [9, 9]
    assert LETTER_VALUES["Z"] == 8


def test_pythagorean_birth_numbers():
    dn = date_numbers(date(1991, 5, 31))
    assert dn["life_path"] == 11
    assert dn["birthday"] == 4 and dn["attitude"] == 9
    assert dn["pinnacles"] == [9, 6, 6, 7]
    assert dn["challenges"] == [1, 2, 1, 3]


def test_pythagorean_cycles_on_draw():
    cn = cycle_numbers(date(1991, 5, 31), date(2026, 10, 23))
    assert cn == {"personal_year": 1, "personal_month": 2, "personal_day": 7, "universal_day": 7}


def test_name_numbers_only_with_name(subject, event, games):
    out = PythagoreanMethod({}).run(subject, event, games.values())
    assert out.status == "partial"
    assert {c.name: c.status for c in out.components}["name_numbers"] == "skipped"
    named = replace(subject, name="Ana Horvat")
    out2 = PythagoreanMethod({}).run(named, event, games.values())
    assert out2.status == "ran" and out2.intermediate["name_numbers"]["expression_compound"] == name_value("Ana Horvat")
    assert_valid_output(out, games)
    assert_reproducible(PythagoreanMethod, {}, subject, event, games)


def test_chaldean_table_has_no_nine():
    assert set(CHALDEAN.values()) == set(range(1, 9)) and len(CHALDEAN) == 26
    assert chaldean_value("Zadar") == 15 and chaldean_value("Zagreb") == 20


def test_chaldean_dates():
    b = date_block(date(1991, 5, 31))
    assert b["date_compound"] == 29 and b["date_reduced"] == 2 and b["weekday_planet"] == "Venus"
    d = date_block(date(2026, 10, 23))
    assert (d["day_compound"], d["day_reduced"], d["date_compound"], d["date_reduced"]) == (23, 5, 16, 7)


def test_chaldean_method(subject, event, games):
    out = ChaldeanMethod({}).run(subject, event, games.values())
    assert_valid_output(out, games)
    assert_reproducible(ChaldeanMethod, {}, subject, event, games)


def test_gematria_ciphers():
    assert hebrew_value("שלום") == 376
    assert hebrew_value("אמת") == 441
    assert greek_value("Ἰησοῦς") == 888
    assert ordinal("abc") == 6 and reverse_ordinal("abc") == 26 + 25 + 24
    assert agrippa("Zadar") == 500 + 1 + 4 + 1 + 80
    assert agrippa("Wu") == agrippa("VVV")


def test_gematria_method(subject, event, games):
    out = GematriaMethod({"extra_texts": []}).run(subject, event, games.values())
    assert_valid_output(out, games)
    comp = {c.name: c.status for c in out.components}
    assert comp["hebrew_mispar_hechrachi"] == "skipped" and comp["latin_ciphers"] == "ran"
    out2 = GematriaMethod({"extra_texts": ["שלום"]}).run(subject, event, games.values())
    assert 376 in out2.sequence
    assert_reproducible(GematriaMethod, {}, subject, event, games)
