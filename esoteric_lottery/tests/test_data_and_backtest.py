from datetime import date

import numpy as np
import pandas as pd
import pytest

from evaluation.backtest import run_backtest
from evaluation.controls import HotFrequencyControl, LogisticControl, features_at
from evaluation.data import DataValidationError, crosscheck_xlsx, load_draws, membership_matrix, split_index
from evaluation.quality import data_quality
from methods import build_methods
from methods.common import Pool


@pytest.fixture(scope="module")
def sin_data(cfg, games, root):
    h = cfg["history"]["sve_ili_nista"]
    return load_draws(root / h["csv"], h, games["sve_ili_nista"])


def test_supplied_csv_loads_and_validates(sin_data):
    df, report = sin_data
    assert report["rows_in_file"] == 636 and report["complete_draws"] == 635
    assert [r["label"] for r in report["rows_without_numbers"]] == ["2026/178"]
    assert report["calendar_gaps"] == ["2026-06-27"]
    assert df["date"].iloc[0] == date(2025, 1, 1) and df["date"].iloc[-1] == date(2026, 9, 28)
    assert all(len(set(n)) == 11 for n in df["main"])


def test_csv_matches_xlsx(sin_data, cfg, games, root):
    df, _ = sin_data
    x = cfg["history"]["sve_ili_nista"]["crosscheck_xlsx"]
    res = crosscheck_xlsx(df, root / x["path"], x, games["sve_ili_nista"].pools[0])
    assert res["identical"], res


def test_bad_rows_are_rejected(tmp_path, cfg, games):
    spec = cfg["history"]["sve_ili_nista"]
    bad = tmp_path / "bad.csv"
    header = "Rb;Godina;Kolo;Datum;Dan;" + ";".join(f"B{i}" for i in range(1, 12)) + ";Zbroj;Parni\n"
    bad.write_text(header + "1;2025;1;01.01.2025;Sri;1;1;3;4;5;6;7;8;9;10;11;65;5\n", encoding="utf-8")
    with pytest.raises(DataValidationError):
        load_draws(bad, spec, games["sve_ili_nista"])


def test_split_is_chronological(sin_data):
    df, _ = sin_data
    s = split_index(df, date(2026, 1, 1))
    assert df["date"].iloc[s - 1] < date(2026, 1, 1) <= df["date"].iloc[s]


def test_features_only_use_the_past():
    rng = np.random.Generator(np.random.PCG64(0))
    m = (rng.random((300, 22)) < 0.5).astype(np.int8)
    f = features_at(m, 150)
    m2 = m.copy()
    m2[150:] = 1 - m2[150:]
    assert np.array_equal(f, features_at(m2, 150))


def test_logistic_fit_ignores_evaluation_rows():
    rng = np.random.Generator(np.random.PCG64(1))
    m = (rng.random((260, 22)) < 0.5).astype(np.int8)
    pool = Pool("main", 22, 11)
    a = LogisticControl(warmup=50).fit(m, 200)
    m2 = m.copy()
    m2[200:] = 0
    b = LogisticControl(warmup=50).fit(m2, 200)
    assert np.allclose(a.model[-1].coef_, b.model[-1].coef_)
    assert a.ranking(m, 200, pool, "k") == b.ranking(m2, 200, pool, "k")


def test_hot_frequency_only_uses_the_past():
    rng = np.random.Generator(np.random.PCG64(2))
    m = (rng.random((100, 22)) < 0.5).astype(np.int8)
    m2 = m.copy()
    m2[60:] = 1
    pool = Pool("main", 22, 11)
    hot = HotFrequencyControl(30)
    assert hot.ranking(m, 60, pool, "k") == hot.ranking(m2, 60, pool, "k")


def _synthetic_draws(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.Generator(np.random.PCG64(seed))
    days = pd.date_range("2025-10-01", periods=n, freq="D").date
    return pd.DataFrame({"date": days, "label": [str(d) for d in days],
                         "main": [tuple(sorted(int(x) + 1 for x in rng.choice(22, 11, replace=False))) for _ in days]})


@pytest.fixture(scope="module")
def small_cfg(cfg):
    c = dict(cfg)
    c["evaluation"] = {**cfg["evaluation"], "mc_replicates": 200, "bootstrap_replicates": 200, "auc_bootstrap": 20,
                       "controls": {**cfg["evaluation"]["controls"], "logistic_warmup": 20, "cv_splits": 3}}
    return c


def test_backtest_predictions_do_not_depend_on_future_draws(small_cfg, subject, event, games):
    game = games["sve_ili_nista"]
    methods = build_methods(small_cfg)
    df = _synthetic_draws(130, 3)
    split = split_index(df, date(2026, 1, 1))
    a = run_backtest(df, game, subject, methods, event.place, "19:00", split, small_cfg)
    df2 = df.copy()
    df2.loc[110:, "main"] = [tuple(range(1, 12))] * (len(df2) - 110)   # rewrite the future
    b = run_backtest(df2, game, subject, methods, event.place, "19:00", split, small_cfg)
    for s in a["strategies"]:
        pa, pb = a["predictions"][s], b["predictions"][s]
        for ra, rb in zip(pa, pb):
            if ra["date"] < df["date"].iloc[110].isoformat():
                assert ra["picks"] == rb["picks"], (s, ra["date"])
    # Ensemble weights come from training draws only.
    assert a["train_weights"] == b["train_weights"]


def test_backtest_esoteric_methods_ignore_results(small_cfg, subject, event, games):
    game = games["sve_ili_nista"]
    methods = build_methods(small_cfg)
    df = _synthetic_draws(110, 4)
    df2 = _synthetic_draws(110, 5)  # same dates, different numbers
    split = split_index(df, date(2026, 1, 1))
    a = run_backtest(df, game, subject, methods, event.place, "19:00", split, small_cfg)
    b = run_backtest(df2, game, subject, methods, event.place, "19:00", split, small_cfg)
    for m in methods:
        assert [r["picks"] for r in a["predictions"][m.key]] == [r["picks"] for r in b["predictions"][m.key]]
    assert a["predictions"]["ensemble_equal"][0]["picks"] == b["predictions"]["ensemble_equal"][0]["picks"]


def test_backtest_hits_are_counted_correctly(small_cfg, subject, event, games):
    game = games["sve_ili_nista"]
    df = _synthetic_draws(100, 6)
    split = split_index(df, date(2026, 1, 1))
    res = run_backtest(df, game, subject, build_methods(small_cfg)[:2], event.place, "19:00", split, small_cfg)
    for rows in res["predictions"].values():
        for r in rows:
            assert r["hits"]["main"] == len(set(r["picks"]["main"]) & set(r["drawn"]["main"]))
    fam = res["results"]["pools"]["main"]["family_tests"]
    assert set(fam) == set(res["strategies"])
    assert all(fam[s]["p_holm"] >= fam[s]["p_exact_two_sided"] for s in fam)


def test_data_quality_runs(sin_data):
    df, _ = sin_data
    pool = Pool("main", 22, 11)
    q = data_quality(membership_matrix(df, pool), pool, 100, 1)
    assert sum(q["frequency"]["counts"].values()) == 635 * 11
    assert 0 < q["frequency"]["p_value_simulated"] <= 1
