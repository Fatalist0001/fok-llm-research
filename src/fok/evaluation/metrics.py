"""Metrics used throughout the evaluation and analysis stages.

All of these treat the probe output as a *score*. The core quantities:

* ``f1`` / ``auc`` / ``acc`` — basic discriminative quality (how separable the
  hidden states are w.r.t. the chosen target).
* ``delta`` — difference between probe score on positive vs negative examples.
* ``fok_correct_contrast`` — a contrast used in the multidimensionality /
  decoupling experiments: how well the probe score aligns with (later)
  correctness vs. with the input-condition target.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)


def binary_metrics(true: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    """Common classification metrics from binary ``true`` (0/1) and continuous
    ``score`` (higher => more of the positive class)."""
    true = np.asarray(true, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = (score > 0.5).astype(int)
    out: Dict[str, float] = {}
    out["acc"] = float(accuracy_score(true, pred))
    out["f1"] = float(f1_score(true, pred, zero_division=0))
    out["auc"] = float(roc_auc_score(true, score))
    # mean score gap between classes
    out["delta"] = float(0.5 * (score[true == 1].mean() - score[true == 0].mean()))
    return out


def regression_metrics(true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    from sklearn.metrics import r2_score

    return {
        "r2": float(r2_score(true, pred)),
        "mae": float(np.mean(np.abs(true - pred))),
    }


def signal_metrics(
    probe_score: np.ndarray,
    target: np.ndarray,
    correctness: np.ndarray,
) -> Dict[str, float]:
    """Contrast metrics for the multidimensionality experiments.

    ``probe_score`` is the probe's output on a held-out set. ``target`` is the
    input-condition the probe was trained on (e.g. ``info_relevant``).
    ``correctness`` is whether the generated answer was correct (0/1/None).

    Returns a dict describing (a) how well the probe separates its own target
    and (b) whether that separation is orthogonal to (later) correctness.
    """
    t = np.asarray(target, dtype=int)
    c = np.asarray(correctness, dtype=float)
    mask = ~np.isnan(c)
    out: Dict[str, float] = {}
    out.update(binary_metrics(t, probe_score))
    if mask.sum() > 1:
        out["correct_auc"] = float(roc_auc_score(c[mask].astype(int), probe_score[mask]))
        # correlation between probe score and correctness among the evaluable ones
        if np.std(probe_score[mask]) > 0 and np.std(c[mask]) > 0:
            out["rho_correct"] = float(np.corrcoef(probe_score[mask], c[mask])[0, 1])
        else:
            out["rho_correct"] = float("nan")
    return out
