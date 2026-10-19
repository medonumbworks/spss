"""Text scorers: VADER with the crypto overlay (always), FinBERT and CryptoBERT (optional).

Every scorer maps a text to -1..+1:

* ``vader``      compound score of ``vaderSentiment`` 3.3.2 after the crypto normalizer
* ``finbert``    ProsusAI/finbert, ``p(positive) - p(negative)``
* ``cryptobert`` ElKulako/cryptobert, ``p(Bullish) - p(Bearish)``

If ``transformers``/``torch`` are not installed, or a model cannot be downloaded/loaded, the
scorer is logged and skipped; the feed's ``models`` lists only scorers that actually ran.
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from sentiment_hub.common import Message
from sentiment_hub.lexicon import EMPTY_LEXICON, URL_RE, Lexicon

log = logging.getLogger("sentiment_hub.nlp")

ALL_MODELS = ("vader", "finbert", "cryptobert")
WEIGHTS = {"cryptobert": 0.45, "vader": 0.30, "finbert": 0.25}
MODEL_IDS = {"finbert": "ProsusAI/finbert", "cryptobert": "ElKulako/cryptobert"}
BULLISH_THRESHOLD = 0.10
BEARISH_THRESHOLD = -0.10
BATCH_SIZE = 32
MAX_TOKENS = 128
#: Characters kept before tokenization (128 tokens never need more; keeps tokenization cheap).
MAX_MODEL_CHARS = 1000
#: Texts handed to the pipeline per call (progress logging and failure isolation).
CHUNK = 256
#: Upper bounds for each transformer model per run (VADER always scores everything).
MODEL_TEXT_LIMIT = 15000
MODEL_BUDGET_S = 420.0

_WS_RE = re.compile(r"\s+")


def composite_score(scores: dict[str, float | None]) -> float | None:
    """Weighted mean (cryptobert 0.45, vader 0.30, finbert 0.25) renormalized over the scores
    that are available; None when no score is available."""
    available = {k: v for k, v in scores.items() if v is not None and k in WEIGHTS and not math.isnan(v)}
    if not available:
        return None
    total = sum(WEIGHTS[k] for k in available)
    return sum(WEIGHTS[k] * v for k, v in available.items()) / total


def label_for(composite: float) -> str:
    if composite >= BULLISH_THRESHOLD:
        return "bullish"
    if composite <= BEARISH_THRESHOLD:
        return "bearish"
    return "neutral"


def model_text(text: str) -> str:
    """Input for the transformer models: URLs removed, whitespace collapsed, length capped."""
    s = URL_RE.sub(" ", text or "")
    return _WS_RE.sub(" ", s).strip()[:MAX_MODEL_CHARS]


def _clamp(x: float) -> float:
    return max(-1.0, min(1.0, float(x)))


class VaderScorer:
    name = "vader"

    def __init__(self, lexicon: Lexicon | None = None) -> None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self.lexicon = lexicon or EMPTY_LEXICON
        self.analyzer = SentimentIntensityAnalyzer()
        self.analyzer.lexicon.update(self.lexicon.valences)

    def score_text(self, text: str) -> float:
        normalized = self.lexicon.normalize(text)
        if not normalized:
            return 0.0
        return _clamp(self.analyzer.polarity_scores(normalized)["compound"])

    def score_batch(self, texts: Sequence[str]) -> list[float | None]:
        return [self.score_text(t) for t in texts]


class TransformersScorer:
    """A Hugging Face ``text-classification`` pipeline turned into a -1..1 score."""

    def __init__(
        self,
        name: str,
        model_id: str,
        positive: Iterable[str],
        negative: Iterable[str],
        pipeline_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.positive = {p.lower() for p in positive}
        self.negative = {n.lower() for n in negative}
        self._factory = pipeline_factory
        self._pipe: Any = None

    def load(self) -> None:
        factory = self._factory
        if factory is None:
            from transformers import pipeline as factory  # noqa: N813 - lazy optional import
        started = time.monotonic()
        self._pipe = factory("text-classification", model=self.model_id, tokenizer=self.model_id, device="cpu")
        log.info("%s: loaded %s in %.1fs", self.name, self.model_id, time.monotonic() - started)

    def _to_score(self, output: Any) -> float | None:
        # top_k=None -> [{"label": ..., "score": ...}, ...] for every label. Older versions may
        # return only the best label as a dict.
        if isinstance(output, dict):
            output = [output]
        if not isinstance(output, list) or not output:
            return None
        probs: dict[str, float] = {}
        for item in output:
            if isinstance(item, dict) and "label" in item and "score" in item:
                probs[str(item["label"]).lower()] = float(item["score"])
        if not probs:
            return None
        pos = sum(v for k, v in probs.items() if k in self.positive)
        neg = sum(v for k, v in probs.items() if k in self.negative)
        return _clamp(pos - neg)

    def score_batch(self, texts: Sequence[str]) -> list[float | None]:
        if self._pipe is None:
            self.load()
        inputs = [model_text(t) or "." for t in texts]
        outputs = self._pipe(inputs, batch_size=BATCH_SIZE, truncation=True, max_length=MAX_TOKENS, top_k=None)
        if not isinstance(outputs, list) or len(outputs) != len(inputs):
            raise RuntimeError(f"{self.name}: pipeline returned {type(outputs).__name__} of unexpected length")
        return [self._to_score(o) for o in outputs]


def make_finbert(pipeline_factory: Callable[..., Any] | None = None) -> TransformersScorer:
    # ProsusAI/finbert id2label: 0 positive, 1 negative, 2 neutral.
    return TransformersScorer(
        "finbert", MODEL_IDS["finbert"], positive=("positive", "label_0"), negative=("negative", "label_1"),
        pipeline_factory=pipeline_factory,
    )


def make_cryptobert(pipeline_factory: Callable[..., Any] | None = None) -> TransformersScorer:
    # ElKulako/cryptobert id2label: 0 Bearish, 1 Neutral, 2 Bullish.
    return TransformersScorer(
        "cryptobert", MODEL_IDS["cryptobert"], positive=("bullish", "label_2"), negative=("bearish", "label_0"),
        pipeline_factory=pipeline_factory,
    )


def parse_models(value: str | None) -> list[str]:
    """``"finbert,vader"`` -> ``["vader", "finbert"]`` (canonical order, vader always included)."""
    requested = {m.strip().lower() for m in (value or ",".join(ALL_MODELS)).split(",") if m.strip()}
    unknown = requested - set(ALL_MODELS)
    if unknown:
        raise ValueError(f"unknown model(s): {', '.join(sorted(unknown))}; choose from {', '.join(ALL_MODELS)}")
    requested.add("vader")
    return [m for m in ALL_MODELS if m in requested]


@dataclass
class ScoredMessage:
    message: Message
    scores: dict[str, float] = field(default_factory=dict)
    composite: float = 0.0
    label: str = "neutral"


@dataclass
class ScoringReport:
    models: list[str]
    notes: list[str]
    scored: list[ScoredMessage]


def score_messages(
    messages: Sequence[Message],
    requested: Sequence[str],
    lexicon: Lexicon | None = None,
    pipeline_factory: Callable[..., Any] | None = None,
    model_limit: int | None = MODEL_TEXT_LIMIT,
    model_budget_s: float | None = MODEL_BUDGET_S,
) -> ScoringReport:
    """Scores every message with every requested scorer that works; messages no scorer could
    score are dropped. VADER scores every text. The transformer models score at most
    ``model_limit`` distinct texts and stop after ``model_budget_s`` seconds each (CPU time is
    the bottleneck of the hourly job); pass the messages most important first, the rest keep
    the scores that exist (the composite is renormalized)."""
    notes: list[str] = []
    unique: dict[str, int] = {}
    texts: list[str] = []
    for m in messages:
        if m.text not in unique:
            unique[m.text] = len(texts)
            texts.append(m.text)
    per_model: dict[str, list[float | None]] = {}

    if texts:
        scorers: list[Any] = []
        try:  # vader always runs, whatever was requested
            scorers.append(VaderScorer(lexicon))
        except Exception as exc:  # pragma: no cover - only without vaderSentiment installed
            log.error("vader unavailable: %s", exc)
            notes.append(f"vader unavailable: {type(exc).__name__}")
        if "finbert" in requested:
            scorers.append(make_finbert(pipeline_factory))
        if "cryptobert" in requested:
            scorers.append(make_cryptobert(pipeline_factory))

        for scorer in scorers:
            started = time.monotonic()
            if isinstance(scorer, TransformersScorer):
                try:
                    scorer.load()
                except ImportError as exc:
                    log.warning("%s skipped: transformers/torch not installed (%s)", scorer.name, exc)
                    notes.append(f"{scorer.name}: not installed")
                    continue
                except Exception as exc:  # download failure, bad config, out of memory, ...
                    log.warning("%s skipped: could not load %s (%s: %s)", scorer.name, scorer.model_id, type(exc).__name__, exc)
                    notes.append(f"{scorer.name}: load failed ({type(exc).__name__})")
                    continue
            results: list[float | None] = []
            failed_chunks = 0
            limit = len(texts)
            if isinstance(scorer, TransformersScorer) and model_limit is not None:
                limit = min(limit, max(0, model_limit))
            deadline = started + model_budget_s if isinstance(scorer, TransformersScorer) and model_budget_s else None
            for start in range(0, limit, CHUNK):
                if deadline is not None and time.monotonic() > deadline:
                    log.warning("%s: time budget %.0fs used up after %d texts", scorer.name, model_budget_s, start)
                    notes.append(f"{scorer.name}: time budget reached after {start}/{len(texts)} texts")
                    break
                chunk = texts[start : min(start + CHUNK, limit)]
                try:
                    results.extend(scorer.score_batch(chunk))
                except Exception as exc:
                    failed_chunks += 1
                    log.warning("%s: batch %d failed (%s: %s)", scorer.name, start // CHUNK, type(exc).__name__, exc)
                    results.extend([None] * len(chunk))
                    if failed_chunks >= 3 and not any(r is not None for r in results):
                        log.warning("%s: giving up after %d failed batches", scorer.name, failed_chunks)
                        break
            results.extend([None] * (len(texts) - len(results)))
            produced = sum(1 for r in results if r is not None)
            if produced == 0:
                log.warning("%s produced no scores", scorer.name)
                notes.append(f"{scorer.name}: no scores")
                continue
            per_model[scorer.name] = results
            if limit < len(texts):
                notes.append(f"{scorer.name}: limited to {limit}/{len(texts)} texts")
            log.info("%s: scored %d/%d texts in %.1fs", scorer.name, produced, len(texts), time.monotonic() - started)

    models = [m for m in ALL_MODELS if m in per_model]
    scored: list[ScoredMessage] = []
    for m in messages:
        idx = unique[m.text]
        scores = {name: per_model[name][idx] for name in models if per_model[name][idx] is not None}
        comp = composite_score(scores)
        if comp is None:
            continue
        scored.append(ScoredMessage(message=m, scores=scores, composite=comp, label=label_for(comp)))
    return ScoringReport(models=models, notes=notes, scored=scored)
