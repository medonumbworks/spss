"""Non-esoteric statistical controls, evaluated with exactly the same protocol.

They answer a different question: is there *any* pattern in the draw history that a
simple data-driven model could exploit? Both only ever see draws strictly before the
draw being predicted (``history = M[:t]``).

* ``hot_frequency``: the numbers drawn most often in the last ``window`` draws.
* ``logistic``: scikit-learn logistic regression on per-number features computed from
  the history (frequency in the last 10/30/100 draws, draws since last appearance,
  drawn last time). Fitted once on the training period; C is fixed a priori, not tuned.
  Time-ordered cross-validation (TimeSeriesSplit) inside the training period is
  reported for information only.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from methods.common import Pool, rank_pool

FEATURES = ["freq_last_10", "freq_last_30", "freq_last_100", "gap_since_last_capped_30", "drawn_last_time"]


def features_at(matrix: np.ndarray, t: int) -> np.ndarray:
    """Features for draw t from draws 0..t-1 only."""
    hist = matrix[:t]
    n_num = matrix.shape[1]
    if len(hist) == 0:
        return np.tile([0.5, 0.5, 0.5, 30.0, 0.0], (n_num, 1))
    cols = [hist[-w:].mean(axis=0) for w in (10, 30, 100)]
    gaps = np.full(n_num, 30.0)
    for j in range(n_num):
        idx = np.flatnonzero(hist[:, j])
        if len(idx):
            gaps[j] = min(30.0, len(hist) - 1 - idx[-1])
    cols += [gaps, hist[-1].astype(float)]
    return np.column_stack(cols)


class HotFrequencyControl:
    key = "control_hot_frequency"
    label = "Kontrola: najčešći brojevi u zadnjih 30 kola"

    def __init__(self, window: int = 30):
        self.window = window

    def ranking(self, matrix: np.ndarray, t: int, pool: Pool, key: str) -> list[int]:
        hist = matrix[max(0, t - self.window):t]
        scores = hist.mean(axis=0) if len(hist) else np.zeros(pool.size)
        return rank_pool(scores, pool, f"{self.key}|{key}")


class LogisticControl:
    key = "control_logistic"
    label = "Kontrola: logistička regresija (scikit-learn) na povijesnim značajkama"

    def __init__(self, warmup: int = 100, C: float = 1.0, cv_splits: int = 5):
        self.warmup, self.C, self.cv_splits = warmup, C, cv_splits
        self.model = None
        self.cv_report: dict = {}

    def _rows(self, matrix: np.ndarray, draws: range) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X, y, g = [], [], []
        for t in draws:
            X.append(features_at(matrix, t))
            y.append(matrix[t])
            g.append(np.full(matrix.shape[1], t))
        return np.vstack(X), np.concatenate(y), np.concatenate(g)

    def _new_model(self):
        return make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=1000))

    def fit(self, matrix: np.ndarray, train_end: int) -> "LogisticControl":
        """Fit on draws warmup..train_end-1; ``matrix`` rows from train_end on are never read."""
        train = matrix[:train_end]
        X, y, groups = self._rows(train, range(self.warmup, train_end))
        self.model = self._new_model().fit(X, y)
        draws = np.arange(self.warmup, train_end)
        folds = []
        for tr, te in TimeSeriesSplit(n_splits=self.cv_splits).split(draws):
            tr_mask, te_mask = np.isin(groups, draws[tr]), np.isin(groups, draws[te])
            m = self._new_model().fit(X[tr_mask], y[tr_mask])
            p = m.predict_proba(X[te_mask])[:, 1]
            folds.append({"train_draws": int(len(tr)), "test_draws": int(len(te)),
                          "log_loss": float(log_loss(y[te_mask], p, labels=[0, 1])),
                          "log_loss_p_0_5": float(np.log(2.0)),
                          "auc": float(roc_auc_score(y[te_mask], p))})
        coef = self.model[-1].coef_[0]
        self.cv_report = {"features": FEATURES, "C": self.C, "warmup_draws": self.warmup,
                          "training_rows": int(len(y)), "coefficients_standardized": dict(zip(FEATURES, coef.tolist())),
                          "time_series_cv": folds}
        return self

    def ranking(self, matrix: np.ndarray, t: int, pool: Pool, key: str) -> list[int]:
        p = self.model.predict_proba(features_at(matrix, t))[:, 1]
        return rank_pool(p, pool, f"{self.key}|{key}")
