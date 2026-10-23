import json
from datetime import timedelta
from pathlib import Path

import pytest

import main as pipeline
from methods import build_methods
from methods.common import games_from_config
from methods.ensemble import aggregate, equal_weights


def test_event_override_keeps_dst_rules(cfg):
    ev = pipeline.event_of(cfg, "2026-10-24", "19:00")
    assert ev.local.isoformat() == "2026-10-24T19:00:00+02:00"   # DST ends 2026-10-25 03:00
    assert pipeline.event_of(cfg, "2026-10-26").local.utcoffset() == timedelta(hours=1)
    assert pipeline.event_of(cfg).local.isoformat() == "2026-10-23T19:00:00+02:00"


def test_readings_only_change_seeded_methods(cfg, subject, event, games):
    base = {m.key: m.run(subject, event, games.values()) for m in build_methods(cfg)}
    same = {m.key: m.run(subject, event, games.values()) for m in build_methods(cfg, seed=cfg["seed"])}
    other = {m.key: m.run(subject, event, games.values()) for m in build_methods(cfg, seed=cfg["seed"] + 1)}
    for k in base:
        assert base[k].picks == same[k].picks
    seeded = {"iching", "tarot", "lenormand"}
    for k in seeded:
        assert base[k].sequence != other[k].sequence
    for k in set(base) - seeded:
        assert base[k].sequence == other[k].sequence
        # Only the hash fill of numbers the method did not determine may differ between readings.
        if base[k].derived["sve_ili_nista"]["main"] == 11:
            assert base[k].picks["sve_ili_nista"] == other[k].picks["sve_ili_nista"]


def test_cli_readings_and_overrides(tmp_path, cfg, subject):
    out = tmp_path / "res"
    rc = pipeline.main(["--skip-backtest", "--draw-date", "2026-10-24", "--readings", "3", "--results", str(out)])
    assert rc == 0
    comb = json.loads((out / "final" / "combinations.json").read_text(encoding="utf-8"))
    assert comb["draw"]["local"] == "2026-10-24T19:00:00+02:00"
    assert comb["n_readings"] == 3 and [r["seed"] for r in comb["readings"]] == [cfg["seed"] + i for i in range(3)]
    games = games_from_config(cfg)
    for game_key, tickets in comb["tickets"].items():
        assert len(tickets) == 3
        for t in tickets:
            for pool in games[game_key].pools:
                picks = t["picks"][pool.name]
                assert len(picks) == pool.pick
                if pool.kind == "set":
                    assert len(set(picks)) == pool.pick and all(1 <= n <= pool.size for n in picks)
    cov = comb["coverage"]["sve_ili_nista"]["main"]
    assert cov["numbers_covered"] <= 22 and cov["pairwise_common"][0][0] == 11
    # Reading 1 is exactly the standard single-reading ensemble.
    event = pipeline.event_of(cfg, "2026-10-24")
    outs = [m.run(subject, event, games.values()) for m in build_methods(cfg)]
    ens = aggregate(outs, equal_weights(outs), games.values(), f"ensemble|{event.local.isoformat()}|equal")
    assert comb["tickets"]["sve_ili_nista"][0]["picks"] == ens["picks"]["sve_ili_nista"]
    assert comb["games"] == comb["readings"][0]["games"]
    assert (out / "final" / "intermediate" / "readings" / "2" / "tarot.json").exists()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["command_line"]["readings"] == 3 and manifest["inputs"]["draw_local"].startswith("2026-10-24")
    assert "history_files_sha256" in manifest
    report = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "## Listići po očitanjima" in report


def _prev_run(tmp_path, manifest: dict) -> Path:
    prev = tmp_path / "prev"
    (prev / "backtest" / "sve_ili_nista").mkdir(parents=True, exist_ok=True)
    (prev / "backtest" / "sve_ili_nista" / "summary.json").write_text("{}", encoding="utf-8")
    (prev / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return prev


GOOD = {"config_sha256": "cfg", "history_files_sha256": {"d.csv": "h"}, "backtest_code_sha256": "code",
        "code_sha256": "all", "generated_at": "t"}
ARGS = ("cfg", {"d.csv": "h"}, "code")


@pytest.mark.parametrize("manifest,message", [
    ({**GOOD, "config_sha256": "other"}, "config.yaml changed"),
    ({**GOOD, "history_files_sha256": {"d.csv": "other"}}, "history files changed"),
    ({k: v for k, v in GOOD.items() if k != "history_files_sha256"}, "no history_files_sha256"),
    ({k: v for k, v in GOOD.items() if k != "backtest_code_sha256"}, "no backtest_code_sha256"),
    ({**GOOD, "backtest_code_sha256": "other"}, "code that computes the backtest changed"),
])
def test_backtest_reuse_refuses_mismatch(tmp_path, manifest, message):
    prev = _prev_run(tmp_path, manifest)
    with pytest.raises(SystemExit, match=message):
        pipeline.reuse_backtest(prev, "sve_ili_nista", *ARGS)


def test_backtest_reuse_refuses_a_directory_without_a_manifest(tmp_path):
    with pytest.raises(SystemExit, match="does not exist"):
        pipeline.reuse_backtest(tmp_path / "typo", "sve_ili_nista", *ARGS)
    prev = _prev_run(tmp_path, GOOD)
    (prev / "manifest.json").unlink()
    with pytest.raises(SystemExit, match="does not exist"):
        pipeline.reuse_backtest(prev, "sve_ili_nista", *ARGS)


def test_backtest_reuse_accepts_a_matching_run(tmp_path, root):
    prev = _prev_run(tmp_path, GOOD)
    # A game the previous run has no backtest for is not an error: it is simply not reused.
    assert pipeline.reuse_backtest(prev, "eurojackpot", *ARGS) is None
    reused = pipeline.reuse_backtest(prev, "sve_ili_nista", *ARGS)
    assert reused["status"] == "reused"
    assert reused["reused_from"]["backtest_code_sha256"] == "code"
    assert reused["reused_from"]["generated_at"] == "t"
    # Path is absolute here (outside the pipeline directory) but never cwd-relative.
    assert Path(reused["reused_from"]["summary"]).is_absolute()


def test_backtest_code_digest_ignores_the_report_renderer(root, tmp_path):
    before_all, before_bt = pipeline.code_digest(), pipeline.backtest_code_digest()
    report = root / "evaluation" / "report.py"
    original = report.read_bytes()
    try:
        report.write_bytes(original + b"\n# touched by a test\n")
        assert pipeline.code_digest() != before_all
        assert pipeline.backtest_code_digest() == before_bt
    finally:
        report.write_bytes(original)
    assert pipeline.code_digest() == before_all


def test_cli_reuses_a_backtest_end_to_end(tmp_path, root, cfg):
    """A full run with --backtest-from: the reused summary drives the report and the
    train-weighted ensemble, and nothing is recomputed."""
    first = tmp_path / "first"
    assert pipeline.main(["--results", str(first)]) == 0
    second = tmp_path / "second"
    assert pipeline.main(["--draw-date", "2026-10-24", "--readings", "2", "--backtest-from", str(first),
                          "--results", str(second)]) == 0
    assert not (second / "backtest").exists()
    manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    bt = manifest["backtests"]["sve_ili_nista"]
    assert bt["status"] == "reused" and bt["reused_from"]["backtest_code_sha256"] == manifest["backtest_code_sha256"]
    report = (second / "REPORT.md").read_text(encoding="utf-8")
    assert "Ponovno korišten iz" in report and "**ponovno korišten**" in report
    # Reused tables keep method order and numeric hit columns, like a freshly computed run.
    first_report = (first / "REPORT.md").read_text(encoding="utf-8")
    header = "| Strategija | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |"
    assert header in report and header in first_report
    for line in ("| Meihua Yishu", "| I Ching"):
        assert line in report
    # Every statistics row of the reused report is identical to the computed one
    # (rows carrying a confidence interval, i.e. the evaluation and training tables).
    def stat_rows(text):
        return [l for l in text.splitlines() if l.startswith("| ") and " | [" in l]
    assert stat_rows(report) == stat_rows(first_report) != []
    comb = json.loads((second / "final" / "combinations.json").read_text(encoding="utf-8"))
    assert "ensemble_train_weighted_sve_ili_nista" in comb["readings"][1]["games"]["sve_ili_nista"]["ensembles"]


def test_cli_refuses_reuse_into_the_same_directory(tmp_path):
    with pytest.raises(SystemExit):
        pipeline.main(["--skip-backtest", "--results", str(tmp_path / "x"), "--backtest-from", str(tmp_path / "x")])


def test_skip_backtest_works_without_history_files(tmp_path, root):
    """--skip-backtest never reads the history, so a missing file must not stop it."""
    cfg_text = (root / "config.yaml").read_text(encoding="utf-8").replace(
        "csv: data/sve_ili_nista_sva_kola.csv", "csv: data/does_not_exist.csv")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    out = tmp_path / "res"
    assert pipeline.main(["--config", str(cfg_path), "--skip-backtest", "--results", str(out)]) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["history_files_sha256"]["data/does_not_exist.csv"] is None
    assert manifest["command_line"]["config"] == str(cfg_path)
