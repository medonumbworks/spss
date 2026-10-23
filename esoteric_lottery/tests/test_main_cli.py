import json
from datetime import timedelta

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


def test_backtest_reuse_requires_matching_config(tmp_path, root):
    prev = tmp_path / "prev"
    (prev / "backtest" / "sve_ili_nista").mkdir(parents=True)
    (prev / "backtest" / "sve_ili_nista" / "summary.json").write_text("{}", encoding="utf-8")
    (prev / "manifest.json").write_text(json.dumps({"config_sha256": "not-the-same"}), encoding="utf-8")
    try:
        pipeline.reuse_backtest(prev, "sve_ili_nista", "abc", {})
    except SystemExit as exc:
        assert "config.yaml changed" in str(exc)
    else:
        raise AssertionError("expected SystemExit")
    assert pipeline.reuse_backtest(prev, "eurojackpot", "abc", {}) is None
    (prev / "manifest.json").write_text(json.dumps({"config_sha256": "abc", "generated_at": "t"}), encoding="utf-8")
    reused = pipeline.reuse_backtest(prev, "sve_ili_nista", "abc", {})
    assert reused["status"] == "reused" and reused["reused_from"]["generated_at"] == "t"
