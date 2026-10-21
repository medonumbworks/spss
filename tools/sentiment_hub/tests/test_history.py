"""The rolling mention-count history behind the feed's ``baseline7d``/``spike``."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from conftest import NOW

from sentiment_hub.history import (
    MAX_ENTRIES,
    MIN_BASELINE_ENTRIES,
    SCHEMA,
    append_run,
    baselines,
    empty_history,
    load_history,
    save_history,
)


def seeded(n: int, *, start_hours: float = 25.0, counts=None) -> dict:
    """``n`` hourly runs, the oldest ``start_hours + n - 1`` hours before NOW."""
    history = empty_history()
    for i in range(n):
        append_run(history, NOW - timedelta(hours=start_hours + n - 1 - i), counts or {"BTC": 2, "ETH": 4})
    return history


# -- load / save ------------------------------------------------------------------------------


def test_none_and_missing_file_give_an_empty_history(tmp_path):
    assert load_history(None) == {"schema": SCHEMA, "entries": []}
    assert load_history(tmp_path / "nope.json") == {"schema": SCHEMA, "entries": []}


@pytest.mark.parametrize(
    "body",
    [
        "not json at all",
        '{"schema": "something-else/v1", "entries": []}',
        '["entries", "as", "a", "list"]',
    ],
)
def test_corrupt_file_gives_an_empty_history(tmp_path, body):
    path = tmp_path / "history.json"
    path.write_text(body, encoding="utf-8")
    assert load_history(path) == {"schema": SCHEMA, "entries": []}


def test_broken_entries_are_dropped_but_good_ones_survive(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps({"schema": SCHEMA, "entries": [
        {"ts": "2026-10-18T10:00:00Z", "counts": {"BTC": 3, "bad": -1, "worse": "x", "float": 1.5, "none": None}},
        {"ts": "not a timestamp", "counts": {"BTC": 1}},
        {"counts": {"BTC": 1}},
        "not an object",
        {"ts": "2026-10-17T10:00:00Z", "counts": [1, 2]},
    ]}), encoding="utf-8")
    assert load_history(path)["entries"] == [{"ts": "2026-10-18T10:00:00Z", "counts": {"BTC": 3}}]


def test_round_trip_and_compact_file(tmp_path):
    history = seeded(3)
    path = save_history(history, tmp_path / "sub" / "history.json")
    assert load_history(path) == history
    text = path.read_text(encoding="utf-8")
    assert text.count("\n") == 1 and '": ' not in text  # compact: no indent, one trailing newline
    assert [p.name for p in path.parent.iterdir()] == ["history.json"]  # no temp files left


# -- append_run -------------------------------------------------------------------------------


def test_append_run_appends_in_time_order():
    history = empty_history()
    append_run(history, NOW - timedelta(hours=2), {"BTC": 1})
    append_run(history, NOW, {"BTC": 3})
    assert [e["ts"] for e in history["entries"]] == ["2026-10-19T08:00:00Z", "2026-10-19T10:00:00Z"]
    assert history["entries"][-1]["counts"] == {"BTC": 3}


def test_append_run_drops_entries_older_than_7_days():
    history = empty_history()
    append_run(history, NOW - timedelta(days=7, hours=1), {"BTC": 1})
    append_run(history, NOW - timedelta(days=6, hours=23), {"BTC": 2})
    append_run(history, NOW, {"BTC": 3})
    assert [e["counts"]["BTC"] for e in history["entries"]] == [2, 3]


def test_append_run_caps_at_max_entries():
    history = empty_history()
    # minutes, so none of the 210 entries age out of the 7-day window
    for i in range(MAX_ENTRIES + 10):
        append_run(history, NOW - timedelta(minutes=MAX_ENTRIES + 10 - i), {"BTC": i})
    assert len(history["entries"]) == MAX_ENTRIES
    assert history["entries"][-1]["counts"]["BTC"] == MAX_ENTRIES + 9  # the newest survive


# -- baselines --------------------------------------------------------------------------------


def test_baseline_is_the_mean_over_entries_older_than_24h():
    history = seeded(6)  # 25..30 h old: all qualify
    append_run(history, NOW - timedelta(hours=1), {"BTC": 99})  # inside 24 h: not baseline
    base = baselines(history, NOW)
    assert base == {"BTC": 2.0, "ETH": 4.0}


def test_symbol_absent_from_an_old_entry_counts_as_zero():
    history = seeded(5)
    append_run(history, NOW - timedelta(hours=31), {"BTC": 2, "ETH": 4, "SOL": 12})
    base = baselines(history, NOW)
    assert base["SOL"] == pytest.approx(12 / 6)
    assert base["BTC"] == 2.0


def test_too_few_old_entries_give_no_baselines():
    history = seeded(MIN_BASELINE_ENTRIES - 1)
    for h in (23, 12, 1):  # recent entries never make up for missing old ones
        append_run(history, NOW - timedelta(hours=h), {"BTC": 5})
    assert baselines(history, NOW) == {}
    assert baselines(empty_history(), NOW) == {}


def test_exactly_24h_old_does_not_qualify():
    history = seeded(MIN_BASELINE_ENTRIES - 1)
    append_run(history, NOW - timedelta(hours=24), {"BTC": 2})
    assert baselines(history, NOW) == {}  # "older than 24 h" is strict
