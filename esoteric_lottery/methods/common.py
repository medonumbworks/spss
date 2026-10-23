"""Shared types and helpers for every method.

Every method turns its inputs (birth data, draw moment) into *raw numbers*: hexagram
numbers, lunar days, numerological reductions, card numbers and so on. How those raw
numbers become a valid lottery ticket is the same for every method and lives here:

* ``reduce_to_range``: modular reduction ``((v - 1) mod N) + 1`` into ``1..N``
  (so ``0 -> N`` and ``N + 1 -> 1``); digits use ``v mod 10``.
* ``sequence_scores``: a number's score is the sum of ``1 / (i + 1)`` over every
  position ``i`` at which a raw value reduces to it, so earlier (primary) values weigh
  more and repeated values accumulate.
* ``rank_pool``: numbers ordered by score, ties broken by a SHA-256 hash of the method,
  game, pool and draw moment. The hash is reproducible and has no bias towards small
  numbers. Numbers a method never produced (score 0) are filled in hash order and are
  counted separately, so a report can say how much of a ticket the method really chose.

All of this is an explicitly *experimental* convention: no esoteric tradition defines
how to map its numbers onto a lottery.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np


# --------------------------------------------------------------------------- games

@dataclass(frozen=True)
class Pool:
    """One set of balls: ``kind="set"`` is ``pick`` distinct numbers in ``1..size``;
    ``kind="digits"`` is ``pick`` ordered positions, each a digit ``0..size-1``."""

    name: str
    size: int
    pick: int
    kind: str = "set"

    def __post_init__(self) -> None:
        if self.kind not in ("set", "digits"):
            raise ValueError(f"unknown pool kind {self.kind!r}")
        if self.kind == "set" and not 0 < self.pick <= self.size:
            raise ValueError(f"pool {self.name}: cannot pick {self.pick} of {self.size}")


@dataclass(frozen=True)
class Game:
    key: str
    label: str
    pools: tuple[Pool, ...]


def games_from_config(cfg: dict) -> dict[str, Game]:
    games = {}
    for key, spec in cfg["games"].items():
        pools = tuple(
            Pool(p["name"], int(p["size"]), int(p["pick"]), p.get("kind", "set")) for p in spec["pools"]
        )
        games[key] = Game(key, spec["label"], pools)
    return games


# --------------------------------------------------------------------------- inputs

@dataclass(frozen=True)
class Place:
    name: str
    lat: float
    lon: float
    tz: str


@dataclass(frozen=True)
class Subject:
    """The querent: birth moment (timezone-aware, local civil time) and birthplace."""

    birth_local: datetime
    place: Place
    name: str | None = None

    @property
    def birth_utc(self) -> datetime:
        return self.birth_local.astimezone(timezone.utc)


@dataclass(frozen=True)
class Event:
    """The draw moment (timezone-aware, local civil time) and where it takes place."""

    local: datetime
    place: Place

    @property
    def utc(self) -> datetime:
        return self.local.astimezone(timezone.utc)


def local_datetime(day: date | str, time_hm: str, tz: str) -> datetime:
    """Local civil time -> aware datetime. zoneinfo applies the historical DST rules
    (e.g. CEST, UTC+2, on 1991-05-31 and on 2026-10-23 in Europe/Zagreb)."""
    if isinstance(day, str):
        day = date.fromisoformat(day)
    hour, minute = (int(x) for x in time_hm.split(":"))
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=ZoneInfo(tz))


# --------------------------------------------------------------------------- outputs

@dataclass
class Component:
    """One sub-calculation of a method and whether it actually ran."""

    name: str
    status: str  # "ran" | "skipped" | "failed"
    detail: str = ""


@dataclass
class MethodNumbers:
    """What a method computes before any lottery mapping."""

    sequence: list[int]
    digit_sequence: list[int]
    intermediate: dict
    components: list[Component]
    pool_scores: Callable[[Pool], np.ndarray] | None = None


@dataclass
class MethodOutput:
    key: str
    label: str
    kind: str
    status: str  # "ran" | "partial" | "failed"
    components: list[Component]
    sources: list[str]
    intermediate: dict
    sequence: list[int]
    digit_sequence: list[int]
    picks: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    ranking: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    derived: dict[str, dict[str, int]] = field(default_factory=dict)
    error: str | None = None

    def to_json(self) -> dict:
        return to_jsonable(dataclasses.asdict(self))


class Method:
    """Base class. Subclasses set the class attributes and implement ``numbers``."""

    key: str = ""
    label: str = ""
    kind: str = "experimental"
    sources: tuple[str, ...] = ()

    def __init__(self, params: dict | None = None, seed: int = 0):
        self.params = params or {}
        self.seed = seed

    def numbers(self, subject: Subject, event: Event) -> MethodNumbers:
        raise NotImplementedError

    def context_key(self, subject: Subject, event: Event) -> str:
        return f"{self.key}|{subject.birth_local.isoformat()}|{event.local.isoformat()}|{self.seed}"

    def run(self, subject: Subject, event: Event, games: Iterable[Game]) -> MethodOutput:
        nums = self.numbers(subject, event)
        statuses = {c.status for c in nums.components}
        status = "ran" if statuses <= {"ran"} else ("failed" if "ran" not in statuses else "partial")
        out = MethodOutput(
            key=self.key, label=self.label, kind=self.kind, status=status, components=nums.components,
            sources=list(self.sources), intermediate=nums.intermediate, sequence=list(nums.sequence),
            digit_sequence=list(nums.digit_sequence),
        )
        ctx = self.context_key(subject, event)
        for game in games:
            out.picks[game.key], out.ranking[game.key], out.derived[game.key] = {}, {}, {}
            for pool in game.pools:
                key = f"{ctx}|{game.key}|{pool.name}"
                if pool.kind == "digits":
                    digits, derived = digits_from_sequence(nums.digit_sequence, pool, key)
                    out.picks[game.key][pool.name] = digits
                    out.ranking[game.key][pool.name] = digits
                else:
                    scores = nums.pool_scores(pool) if nums.pool_scores else sequence_scores(nums.sequence, pool)
                    ranking = rank_pool(scores, pool, key)
                    picks = sorted(ranking[: pool.pick])
                    out.picks[game.key][pool.name] = picks
                    out.ranking[game.key][pool.name] = ranking
                    derived = int(sum(1 for n in picks if scores[n - 1] > 0))
                out.derived[game.key][pool.name] = derived
        return out


# --------------------------------------------------------------------------- mapping

def reduce_to_range(value: int, pool: Pool) -> int:
    """Modular reduction into the pool: ``1..size`` for sets, ``0..size-1`` for digits."""
    value = int(value)
    if pool.kind == "digits":
        return value % pool.size
    return (value - 1) % pool.size + 1


def sequence_scores(values: Sequence[int], pool: Pool) -> np.ndarray:
    """Score per number (index ``n - 1``): sum of ``1/(i+1)`` over the positions ``i``
    whose raw value reduces to ``n``."""
    scores = np.zeros(pool.size, dtype=float)
    for i, v in enumerate(values):
        scores[reduce_to_range(v, pool) - 1] += 1.0 / (i + 1)
    return scores


def tiebreak_values(key: str, size: int) -> np.ndarray:
    """Reproducible pseudo-random value in [0, 1) for every number 1..size."""
    out = np.empty(size, dtype=float)
    for n in range(1, size + 1):
        digest = hashlib.sha256(f"{key}|{n}".encode()).digest()
        out[n - 1] = int.from_bytes(digest[:8], "big") / 2.0**64
    return out


def rank_pool(scores: np.ndarray, pool: Pool, key: str) -> list[int]:
    """Every number of the pool, most favoured first (score desc, then hash)."""
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (pool.size,):
        raise ValueError(f"expected {pool.size} scores, got {scores.shape}")
    tb = tiebreak_values(key, pool.size)
    # Round so float noise never decides an order the hash should decide.
    order = sorted(range(pool.size), key=lambda i: (-round(float(scores[i]), 12), tb[i]))
    return [i + 1 for i in order]


def digits_from_sequence(values: Sequence[int], pool: Pool, key: str) -> tuple[list[int], int]:
    """First ``pick`` raw values mod 10; missing positions come from the hash."""
    digits = [reduce_to_range(v, pool) for v in list(values)[: pool.pick]]
    derived = len(digits)
    if derived < pool.pick:
        tb = tiebreak_values(key + "|fill", pool.pick)
        digits += [int(tb[i] * pool.size) for i in range(derived, pool.pick)]
    return digits, derived


def seeded_rng(*parts: Any) -> np.random.Generator:
    """numpy PCG64 generator seeded from a SHA-256 of the given parts."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], "big")))


# --------------------------------------------------------------------------- numbers

def digit_sum(n: int) -> int:
    return sum(int(c) for c in str(abs(int(n))))


def digit_root(n: int, masters: Iterable[int] = ()) -> int:
    """Repeated digit sum down to 1..9, stopping early at any of ``masters``."""
    masters = set(masters)
    n = abs(int(n))
    while n > 9 and n not in masters:
        n = digit_sum(n)
    return n


def normalize_latin(text: str) -> str:
    """Upper-case A-Z only. Croatian diacritics are stripped (Č, Ć -> C; Š -> S;
    Ž -> Z; Đ -> D), which is a documented convention, not a tradition."""
    text = text.replace("đ", "d").replace("Đ", "D").replace("ß", "ss")
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed.upper() if "A" <= c <= "Z")


CROATIAN_WEEKDAYS = ["ponedjeljak", "utorak", "srijeda", "četvrtak", "petak", "subota", "nedjelja"]
CROATIAN_MONTHS = ["siječanj", "veljača", "ožujak", "travanj", "svibanj", "lipanj",
                   "srpanj", "kolovoz", "rujan", "listopad", "studeni", "prosinac"]


# --------------------------------------------------------------------------- JSON

def to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 10)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path | str, obj: Any, one_row_per_line: bool = False) -> None:
    """Deterministic JSON (sorted keys). ``one_row_per_line`` writes a list with one
    compact element per line, for long per-draw tables."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = to_jsonable(obj)
    if one_row_per_line:
        rows = (json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for r in data)
        text = "[\n" + ",\n".join(rows) + "\n]\n"
    else:
        text = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
