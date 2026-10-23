import math

from methods import build_methods
from methods.common import Game, MethodOutput, Pool
from methods.ensemble import aggregate, borda_points, equal_weights, train_weights

POOL = Pool("main", 4, 2)
DIG = Pool("digits", 10, 3, "digits")
GAME = Game("g", "G", (POOL, DIG))


def fake(key, ranking, digits, status="ran"):
    return MethodOutput(key, key, "x", status, [], [], {}, [], [], picks={"g": {"main": sorted(ranking[:2]), "digits": digits}},
                        ranking={"g": {"main": ranking, "digits": digits}})


def test_borda_points():
    assert borda_points([3, 1, 4, 2], POOL).tolist() == [3, 1, 4, 2]


def test_borda_aggregate_equal_weights():
    outs = [fake("a", [1, 2, 3, 4], [1, 2, 3]), fake("b", [2, 1, 4, 3], [1, 5, 3]), fake("c", [2, 3, 1, 4], [9, 5, 3])]
    res = aggregate(outs, equal_weights(outs), [GAME], "k")
    # points: 1 -> 4+3+2 = 9, 2 -> 3+4+4 = 11, 3 -> 2+1+3 = 6, 4 -> 1+2+1 = 4
    assert res["points"]["g"]["main"] == [9, 11, 6, 4]
    assert res["picks"]["g"]["main"] == [1, 2]
    assert res["picks"]["g"]["digits"] == [1, 5, 3]


def test_failed_methods_and_zero_weights_are_ignored():
    outs = [fake("a", [1, 2, 3, 4], [1, 1, 1]), fake("b", [4, 3, 2, 1], [2, 2, 2], status="failed"),
            fake("c", [3, 4, 1, 2], [3, 3, 3])]
    res = aggregate(outs, {"a": 1.0, "b": 1.0, "c": 0.0}, [GAME], "k")
    assert res["picks"]["g"]["main"] == [1, 2] and set(res["weights"]) == {"a"}


def test_train_weights_softmax():
    w = train_weights({"a": 5.5, "b": 5.6}, 100, 5.5, 1.2, 0.5)
    z_b = 0.1 / (1.2 / 10)
    assert math.isclose(w["b"]["z_train"], z_b)
    assert math.isclose(w["b"]["weight"] / w["a"]["weight"], math.exp(0.5 * z_b))
    assert math.isclose(w["a"]["weight"] + w["b"]["weight"], 2.0)


def test_ensemble_over_real_methods(cfg, subject, event, games):
    outs = [m.run(subject, event, games.values()) for m in build_methods(cfg)]
    res = aggregate(outs, equal_weights(outs), games.values(), "final")
    assert len(res["picks"]["sve_ili_nista"]["main"]) == 11
    assert len(set(res["picks"]["eurojackpot"]["main"])) == 5 and len(res["picks"]["eurojackpot"]["euro"]) == 2
    assert len(res["picks"]["joker"]["digits"]) == 6
    assert res == aggregate(outs, equal_weights(outs), games.values(), "final")
