"""Evaluation stage: train probes on the extracted features and compare.

This is the central diagnostic. For each input-condition target (see
:mod:`fok.evaluation.targets`) and for each selected hidden layer, an
``fok.signal`` probe is fit on the *train* split and scored on the hold-out
splits. The probe output is treated as a continuous ``fok_score`` which we
report per-layer.

Crucially, this stage also records the *confidence baselines* so that, in the
analysis stage, we can answer the project's key question: is the per-layer
signal *additive* / *distinct* from what a single scalar confidence value would
give, or is it just a re-encoding of it? Two layers having different probe
separability is itself informative (per-layer structure, spec §4/§12).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..config import ExperimentConfig
from ..probes.base import make_probe
from ..utils import save_json, write_csv
from .metrics import binary_metrics
from .targets import candidate_targets, target_values

logger = logging.getLogger("fok.evaluate")


def _split_masks(splits: List[str]):
    tr = np.array([s == "train" for s in splits])
    val = np.array([s == "val" for s in splits])
    te = np.array([s == "test" for s in splits])
    return tr, val, te


def _probe_scores(probe, X: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a continuous score (>=0.5 means predicted class 1)."""
    p = probe.predict_proba(X)
    return p[:, 1]


def _fit_eval(probe, X, y, tr, val, te):
    """Fit on train rows, return per-split metrics + scores."""
    probe.fit(X[tr], y[tr], X_val=X[val] if np.any(val) else None, y_val=y[val] if np.any(val) else None)
    scores = {s: _probe_scores(probe, X[m], y[m]) for s, m in
              (("train", tr), ("val", val), ("test", te))}
    m = {s: binary_metrics(y[m], scores[s]) for s, m in
         (("train", tr), ("val", val), ("test", te))}
    return {s: m[s] for s in m}, scores


def evaluate(
    cfg: ExperimentConfig,
    features_dir: Optional[Path] = None,
    targets: Optional[List[str]] = None,
) -> Dict:
    """Run the evaluation and persist per-layer probe results + confidence baselines.

    Returns a summary dict with paths to the written CSV/JSON artifacts.
    """
    if features_dir is None:
        features_dir = cfg.run_dir() / "features"
    features_dir = Path(features_dir)
    rows = _read_rows(features_dir)
    if not rows:
        raise SystemExit(f"no features found in {features_dir}; run 'extract' first")

    X, _layers = _load_hidden(features_dir, "A")
    if X is None:
        raise SystemExit("hidden_A.npy missing; run 'extract' with capture_A=True")
    splits = [r["split"] for r in rows]
    tr, val, te = _split_masks(splits)

    tgt_names = targets or candidate_targets(rows)
    tgt_names = [t for t in tgt_names if target_values(rows, t) is not None]
    if not tgt_names:
        logger.warning("no binary input-condition target found; probing 'correct' only")
        tgt_names = ["correct"]

    out_dir = cfg.run_dir() / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_rows = []
    layer_list = _layers
    for tgt in tgt_names:
        y = target_values(rows, tgt)
        if y is None:
            continue
        for li, layer in enumerate(layer_list):
            Xl = X[:, li, :]  # [N, hidden]
            probe = make_probe(cfg.probe, C=cfg.probe_C, tune_C=cfg.probe_tune_C)
            metrics, scores = _fit_eval(probe, Xl, y, tr, val, te)
            probe_rows.append({
                "target": tgt,
                "layer": int(layer),
                "kind": cfg.probe,
                # held-out numbers (primary)
                "val_acc": metrics["val"]["acc"],
                "val_f1": metrics["val"]["f1"],
                "val_auc": metrics["val"]["auc"],
                "test_acc": metrics["test"]["acc"],
                "test_f1": metrics["test"]["f1"],
                "test_auc": metrics["test"]["auc"],
                "delta": metrics["test"]["delta"],
                "coef_norm2": float(np.dot(probe.coef(), probe.coef())),
            })
        logger.info("probed target=%s across %d layers", tgt, len(layer_list))

    write_csv(probe_rows, out_dir / "probe_results.csv")

    # --- confidence baselines: the same targets scored from scalar confidence ---
    conf_rows = _confidence_baselines(rows, tgt_names, tr, val, te)
    write_csv(conf_rows, out_dir / "confidence_baselines.csv")

    save_json({"targets": tgt_names, "layers": [int(x) for x in layer_list]},
              out_dir / "eval_meta.json")
    logger.info("Evaluation complete -> %s", out_dir)
    return {
        "probe_csv": str(out_dir / "probe_results.csv"),
        "confidence_csv": str(out_dir / "confidence_baselines.csv"),
        "targets": tgt_names,
    }


def _confidence_baselines(rows, tgt_names, tr, val, te):
    """Fit a tiny logistic probe on each available *scalar confidence* feature.

    This is the explicit comparison requested in spec §8: how does the per-layer
    FOK signal compare to simply reading off token/sequence confidence? Each row
    records the val/test AUC of a logistic regression trained (on train) to
    recover the target from that one scalar confidence feature.
    """
    from sklearn.linear_model import LogisticRegression

    conf_feats = ["avg_logprob", "seq_logprob", "first_entropy", "mean_entropy", "top1_prob"]
    present = [c for c in conf_feats if c in rows[0]]
    out = []
    for tgt in tgt_names:
        y = target_values(rows, tgt)
        if y is None:
            continue
        for feat in present:
            vec = np.array([_to_float(r[feat]) for r in rows])
            mask = np.isfinite(vec)
            if mask.sum() < 8 or np.unique(y[mask]).size < 2:
                continue
            clf = LogisticRegression(max_iter=2000)
            clf.fit(vec[mask & tr].reshape(-1, 1), y[mask & tr])
            for split, m in (("val", (mask & val)), ("test", (mask & te))):
                out.append({
                    "target": tgt,
                    "feature": feat,
                    "split": split,
                    "auc": float(roc_from_vec(clf, vec[m], y[m])),
                    "acc": float(((clf.predict(vec[m].reshape(-1, 1)) == y[m]).mean())),
                })
    return out


def roc_from_vec(clf, x, y):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, clf.predict_proba(x.reshape(-1, 1))[:, 1]))


def _to_float(v):
    try:
        f = float(v)
        return f
    except (TypeError, ValueError):
        return float("nan")


def _read_rows(features_dir: Path) -> List[Dict]:
    import csv
    path = features_dir / "examples.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_hidden(features_dir: Path, tp: str):
    path = features_dir / f"hidden_{tp}.npy"
    if not path.exists():
        return None, []
    arr = np.load(path)
    lay = []
    try:
        with open(features_dir / "extraction_meta.json", encoding="utf-8") as f:
            import json
            lay = list(json.load(f).get("layers", []))
    except Exception:
        lay = list(range(arr.shape[1]))
    return arr, lay
