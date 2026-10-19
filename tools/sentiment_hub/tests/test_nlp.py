"""Scorers and composite. transformers is always mocked: no model is ever downloaded."""

from __future__ import annotations

import sys
import types
from datetime import timedelta

import pytest

from sentiment_hub import nlp
from sentiment_hub.common import Message


def msg(text: str, now, i: int = 0) -> Message:
    return Message(source="rss", key=f"k{i}", text=text, ts=now - timedelta(hours=1), url=f"https://x/{i}", community="feed")


class FakePipe:
    """Mimics transformers' text-classification pipeline called with top_k=None."""

    def __init__(self, model_id: str, table: dict[str, dict[str, float]]):
        self.model_id = model_id
        self.table = table
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, inputs, **kwargs):
        self.calls.append((list(inputs), kwargs))
        out = []
        for text in inputs:
            probs = self.table.get(text, self.table["*"])
            out.append([{"label": k, "score": v} for k, v in probs.items()])
        return out


def fake_factory(registry: dict, fail: set[str] = frozenset()):
    tables = {
        "ProsusAI/finbert": {"*": {"positive": 0.7, "negative": 0.1, "neutral": 0.2}},
        "ElKulako/cryptobert": {"*": {"Bearish": 0.3, "Neutral": 0.2, "Bullish": 0.5}},
    }

    def factory(task, model=None, tokenizer=None, device=None, **kwargs):
        assert task == "text-classification"
        assert tokenizer == model and device == "cpu"
        if model in fail:
            raise OSError(f"We couldn't connect to 'https://huggingface.co' to load {model}")
        pipe = FakePipe(model, tables[model])
        registry[model] = pipe
        return pipe

    return factory


def test_composite_weights_and_renormalization():
    assert nlp.composite_score({"vader": 0.5, "finbert": 0.2, "cryptobert": -0.4}) == pytest.approx(0.45 * -0.4 + 0.30 * 0.5 + 0.25 * 0.2)
    assert nlp.composite_score({"vader": 0.5}) == pytest.approx(0.5)
    assert nlp.composite_score({"vader": 0.5, "finbert": -0.5}) == pytest.approx((0.30 * 0.5 - 0.25 * 0.5) / 0.55)
    assert nlp.composite_score({"vader": 0.2, "cryptobert": 0.6}) == pytest.approx((0.30 * 0.2 + 0.45 * 0.6) / 0.75)
    assert nlp.composite_score({}) is None
    assert nlp.composite_score({"vader": None}) is None
    assert sum(nlp.WEIGHTS.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("c,label", [(0.10, "bullish"), (0.5, "bullish"), (0.0999, "neutral"), (0.0, "neutral"),
                                     (-0.0999, "neutral"), (-0.10, "bearish"), (-1.0, "bearish")])
def test_labels(c, label):
    assert nlp.label_for(c) == label


def test_parse_models():
    assert nlp.parse_models("vader,finbert,cryptobert") == ["vader", "finbert", "cryptobert"]
    assert nlp.parse_models("cryptobert") == ["vader", "cryptobert"]  # vader always runs
    assert nlp.parse_models(" FinBERT , vader ") == ["vader", "finbert"]
    assert nlp.parse_models(None) == list(nlp.ALL_MODELS)
    with pytest.raises(ValueError):
        nlp.parse_models("vader,gpt")


def test_model_text_strips_urls():
    assert nlp.model_text("BTC  up https://x.y/z\n\nnow") == "BTC up now"
    assert len(nlp.model_text("a" * 5000)) == nlp.MAX_MODEL_CHARS


def test_all_three_models_with_mocked_transformers(now):
    registry: dict = {}
    messages = [msg("Bitcoin is pumping hard, great news", now, 0), msg("Bitcoin is pumping hard, great news", now, 1),
                msg("Total disaster, exchange hacked", now, 2)]
    report = nlp.score_messages(messages, ["vader", "finbert", "cryptobert"], pipeline_factory=fake_factory(registry))
    assert report.models == ["vader", "finbert", "cryptobert"]
    assert report.notes == []
    assert len(report.scored) == 3
    first = report.scored[0]
    assert first.scores["finbert"] == pytest.approx(0.6)  # p(positive) - p(negative)
    assert first.scores["cryptobert"] == pytest.approx(0.2)  # p(Bullish) - p(Bearish)
    assert first.scores["vader"] > 0
    expected = (0.45 * 0.2 + 0.30 * first.scores["vader"] + 0.25 * 0.6) / 1.0
    assert first.composite == pytest.approx(expected)
    assert first.label == nlp.label_for(expected)
    assert report.scored[2].scores["vader"] < 0
    # duplicates are scored once; batch size, truncation and all-label output are requested
    for model_id in ("ProsusAI/finbert", "ElKulako/cryptobert"):
        (inputs, kwargs), = registry[model_id].calls
        assert inputs == ["Bitcoin is pumping hard, great news", "Total disaster, exchange hacked"]
        assert kwargs == {"batch_size": 32, "truncation": True, "max_length": 128, "top_k": None}


def test_transformers_missing_falls_back_to_vader(now, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "transformers", None)  # import raises ImportError
    report = nlp.score_messages([msg("Great rally!", now)], ["vader", "finbert", "cryptobert"])
    assert report.models == ["vader"]
    assert report.notes == ["finbert: not installed", "cryptobert: not installed"]
    assert report.scored[0].scores.keys() == {"vader"}
    assert report.scored[0].composite == pytest.approx(report.scored[0].scores["vader"])
    assert "transformers/torch not installed" in caplog.text


def test_transformers_module_without_torch(now, monkeypatch):
    fake = types.ModuleType("transformers")

    def pipeline(*args, **kwargs):
        raise ImportError("TextClassificationPipeline requires the PyTorch library")

    fake.pipeline = pipeline
    monkeypatch.setitem(sys.modules, "transformers", fake)
    report = nlp.score_messages([msg("ok", now)], ["finbert"])
    assert report.models == ["vader"]
    assert report.notes == ["finbert: not installed"]


def test_one_model_download_fails(now):
    registry: dict = {}
    report = nlp.score_messages(
        [msg("SOL breaks out", now)], ["vader", "finbert", "cryptobert"],
        pipeline_factory=fake_factory(registry, fail={"ElKulako/cryptobert"}),
    )
    assert report.models == ["vader", "finbert"]
    assert report.notes == ["cryptobert: load failed (OSError)"]
    s = report.scored[0]
    assert s.composite == pytest.approx((0.30 * s.scores["vader"] + 0.25 * s.scores["finbert"]) / 0.55)


def test_inference_errors_drop_the_model(now):
    class Boom:
        def __call__(self, inputs, **kwargs):
            raise RuntimeError("CUDA? no, just broken")

    report = nlp.score_messages([msg("x", now)], ["vader", "finbert"], pipeline_factory=lambda *a, **k: Boom())
    assert report.models == ["vader"]
    assert report.notes == ["finbert: no scores"]


def test_label_fallback_names_and_single_dict_output():
    scorer = nlp.make_cryptobert(pipeline_factory=lambda *a, **k: None)
    assert scorer._to_score([{"label": "LABEL_2", "score": 0.6}, {"label": "LABEL_0", "score": 0.1}]) == pytest.approx(0.5)
    assert scorer._to_score({"label": "Bullish", "score": 0.9}) == pytest.approx(0.9)
    assert scorer._to_score([]) is None
    fin = nlp.make_finbert(pipeline_factory=lambda *a, **k: None)
    assert fin._to_score([{"label": "LABEL_0", "score": 0.2}, {"label": "LABEL_1", "score": 0.7}]) == pytest.approx(-0.5)


def test_no_messages_loads_nothing():
    def factory(*args, **kwargs):
        raise AssertionError("must not load models without texts")

    report = nlp.score_messages([], ["vader", "finbert", "cryptobert"], pipeline_factory=factory)
    assert report.models == [] and report.scored == []


def test_model_text_limit_keeps_vader_for_everything(now):
    registry: dict = {}
    messages = [msg(f"text {i}", now, i) for i in range(10)]
    report = nlp.score_messages(messages, ["vader", "finbert"], pipeline_factory=fake_factory(registry), model_limit=4)
    assert report.models == ["vader", "finbert"]
    assert [("finbert" in s.scores) for s in report.scored] == [True] * 4 + [False] * 6
    assert all("vader" in s.scores for s in report.scored)
    assert report.notes == ["finbert: limited to 4/10 texts"]
    ((inputs, _),) = registry["ProsusAI/finbert"].calls
    assert inputs == [f"text {i}" for i in range(4)]  # the first (most important) texts


def test_model_time_budget(now):
    report = nlp.score_messages([msg("x", now)], ["finbert"], pipeline_factory=fake_factory({}), model_budget_s=1e-9)
    assert report.models == ["vader"]
    assert report.notes[0].startswith("finbert: time budget reached after 0/1 texts")
