import json

from methods.common import MethodOutput


def assert_valid_output(out: MethodOutput, games) -> None:
    """Every pool holds a valid, reproducible ticket and the output is JSON-serialisable."""
    assert out.status in ("ran", "partial")
    for game in games.values():
        for pool in game.pools:
            picks = out.picks[game.key][pool.name]
            assert len(picks) == pool.pick
            if pool.kind == "set":
                assert len(set(picks)) == pool.pick
                assert all(1 <= n <= pool.size for n in picks)
                assert picks == sorted(picks)
                assert sorted(out.ranking[game.key][pool.name]) == list(range(1, pool.size + 1))
            else:
                assert all(0 <= d < pool.size for d in picks)
            assert 0 <= out.derived[game.key][pool.name] <= pool.pick
    json.dumps(out.to_json())


def assert_reproducible(method_cls, params, subject, event, games) -> None:
    a = method_cls(params, 7).run(subject, event, games.values()).to_json()
    b = method_cls(params, 7).run(subject, event, games.values()).to_json()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
