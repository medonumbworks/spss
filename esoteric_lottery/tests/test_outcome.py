import json

import pytest

from evaluation import outcome
from methods.common import Pool

POOL = Pool("main", 22, 11)
PAYING = [0, 1, 2, 3, 8, 9, 10, 11]
DRAWN_2026_10_24 = [2, 3, 6, 7, 8, 9, 11, 13, 20, 21, 22]


def test_parse_numbers():
    assert outcome.parse_numbers("2, 3,6 7;8") == [2, 3, 6, 7, 8]
    with pytest.raises(ValueError):
        outcome.parse_numbers("1,1,2")


def _combinations(tickets):
    games = {"sve_ili_nista": {"label": "x", "methods": {},
                               "ensembles": {"ensemble_equal": {"picks": {"main": tickets[0]}}}}}
    return {"draw": {"local": "2026-10-24T19:00:00+02:00"}, "games": games,
            "readings": [{"reading": i + 1, "seed": i, "games": {"sve_ili_nista": {
                "label": "x", "methods": {"m": {"picks": {"main": t}}},
                "ensembles": {"ensemble_equal": {"picks": {"main": t}}}}}} for i, t in enumerate(tickets)]}


def test_scoring_counts_hits_and_tiers():
    tickets = [DRAWN_2026_10_24,                       # 11 hits: jackpot tier
               [n for n in range(1, 23) if n not in DRAWN_2026_10_24],   # 0 hits: also jackpot tier
               [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 18]]   # 6 hits: pays nothing
    rec = outcome.score_predictions(_combinations(tickets), "sve_ili_nista", POOL, DRAWN_2026_10_24,
                                    PAYING, "src")
    by_reading = {t["reading"]: t for t in rec["primary_tickets"]}
    assert by_reading[1]["hits"] == 11 and by_reading[1]["in_paying_tier"]
    assert by_reading[2]["hits"] == 0 and by_reading[2]["in_paying_tier"]
    assert by_reading[3]["hits"] == 6 and not by_reading[3]["in_paying_tier"]
    assert rec["max_hits"] == 11 and rec["min_hits"] == 0
    assert rec["drawn"] == DRAWN_2026_10_24 and rec["expected_hits_per_ticket"] == 5.5


def test_scoring_rejects_an_invalid_draw():
    combos = _combinations([list(range(1, 12))])
    for bad in ([1, 2, 3], list(range(1, 12)) + [23]):
        with pytest.raises(ValueError):
            outcome.score_predictions(combos, "sve_ili_nista", POOL, bad, PAYING, "src")


def test_log_replaces_a_record_for_the_same_draw(tmp_path):
    path = tmp_path / "log.json"
    a = {"draw_date": "2026-10-24", "game": "sve_ili_nista", "mean_hits": 5.0, "primary_tickets": []}
    b = {"draw_date": "2026-10-24", "game": "sve_ili_nista", "mean_hits": 6.0, "primary_tickets": []}
    c = {"draw_date": "2026-10-25", "game": "sve_ili_nista", "mean_hits": 4.0, "primary_tickets": []}
    path.write_text(json.dumps([a]), encoding="utf-8")
    assert outcome.append_log(path, b) == [b]
    path.write_text(json.dumps([b]), encoding="utf-8")
    assert outcome.append_log(path, c) == [b, c]


def test_summary_of_the_running_record():
    records = [{"draw_date": "2026-10-24", "game": "g",
                "primary_tickets": [{"hits": 6, "in_paying_tier": False}, {"hits": 9, "in_paying_tier": True}]}]
    s = outcome.summarize_log(records, POOL)
    assert s["draws"] == 1 and s["primary_tickets"] == 2
    assert s["mean_hits"] == 7.5 and s["difference"] == 2.0 and s["tickets_in_paying_tier"] == 1


def test_real_2026_10_24_record_is_present_and_consistent(root):
    """The first genuinely out-of-sample draw: predictions committed before it happened."""
    log = json.loads((root / "results" / "out_of_sample" / "log.json").read_text(encoding="utf-8"))
    rec = next(r for r in log if r["draw_date"] == "2026-10-24")
    assert rec["drawn"] == DRAWN_2026_10_24
    combos = json.loads((root / "results" / "draw_2026-10-24" / "final" / "combinations.json")
                        .read_text(encoding="utf-8"))
    again = outcome.score_predictions(combos, "sve_ili_nista", POOL, DRAWN_2026_10_24, PAYING,
                                      rec["predictions_from"])
    assert again["mean_hits"] == pytest.approx(rec["mean_hits"])
    assert again["primary_tickets"] == rec["primary_tickets"]
    # Every ticket's hit count really is the intersection with the drawn numbers.
    for t in rec["all_tickets"]:
        assert t["hits"] == len(set(t["picks"]) & set(DRAWN_2026_10_24))
