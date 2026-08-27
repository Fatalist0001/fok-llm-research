"""Analysis stage: post-hoc interpretation of the probe results.

This stage turns the raw per-layer numbers into the research claims the project
is really about. It implements the higher-level checks from the spec:

* :meth:`confidence_contrast` — does a single layer of hidden states beat the
  best scalar-confidence baseline at recovering the target? (spec §8)
* :meth:`per_layer_summary` — structure of the signal across layers (spec §4,§12).
* :meth:`multidim` — decoupling: is the probed signal independent of what a
  plain correctness label gives? And are two input-condition targets separately
  decodable? (spec §11)
* :meth:`timepoints` — does the signal move between A/B/C (before / while /
  after reading the answer)? (spec §9)
* :meth:`control` — permutation null so any "signal" is compared with chance
  (spec §13).

Every method writes a small CSV/JSON so the plot stage and the user can read
the interpretation without rerunning the probes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..config import ExperimentConfig
from ..probes.base import make_probe
from ..utils import save_json, write_csv
from .evaluation_split import load_rows, load_hidden, load_conf_and_probe

logger = logging.getLogger("fok.analyze")


def analyze(
    cfg: ExperimentConfig,
    features_dir: Optional[Path] = None,
    eval_dir: Optional[Path] = None,
    n_perm: int = 200,
) -> Dict[str, Path]:
    """Run the full analysis and return a mapping name -> written artifact path."""
    features_dir = Path(features_dir) if features_dir else cfg.run_dir() / "features"
    eval_dir = Path(eval_dir) if eval_dir else cfg.run_dir() / "eval"
    out_dir = cfg.run_dir() / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(features_dir)
    if not rows:
        raise SystemExit(f"no features in {features_dir}; run 'extract' first")

    probe_df, conf_df = load_conf_and_probe(eval_dir)

    artifacts: Dict[str, Path] = {}

    if probe_df is not None and not probe_df.empty:
        _per_layer_summary(probe_df, out_dir, artifacts)
    if probe_df is not None and conf_df is not None and not conf_df.empty:
        _confidence_contrast(probe_df, conf_df, out_dir, artifacts)

    _control(cfg, rows, features_dir, out_dir, artifacts, n_perm=n_perm)
    _timepoints(cfg, rows, features_dir, out_dir, artifacts)
    _multidim(cfg, rows, features_dir, eval_dir, out_dir, artifacts)

    logger.info("Analysis complete -> %s", out_dir)
    return artifacts


# --------------------------------------------------------------------------- #
def _per_layer_summary(probe_df, out_dir, artifacts):
    """Persist a per-layer, per-target AUC table (spec §4, §12)."""
    if "val_auc" not in probe_df.columns:
        return
    piv = probe_df.pivot_table(index="layer", columns="target", values="val_auc")
    path = out_dir / "per_layer_auc.csv"
    piv.to_csv(path)
    artifacts["per_layer_auc"] = path


def _confidence_contrast(probe_df, conf_df, out_dir, artifacts):
    """Spec §8: best single layer vs best scalar-confidence feature per target."""
    best_layer = (
        probe_df.loc[probe_df.groupby("target")["val_auc"].idxmax()]
        .set_index("target")["val_auc"].to_dict()
    )
    best_conf = (
        conf_df[conf_df["split"] == "val"]
        .groupby("target")["auc"].max().to_dict()
    )
    rows = []
    for tgt in best_layer:
        layers = float(best_layer[tgt])
        confs = float(best_conf.get(tgt, float("nan")))
        rows.append({
            "target": tgt,
            "best_layer_val_auc": layers,
            "best_confidence_val_auc": confs,
            "hidden_beats_confidence": bool(layers > confs),
            "advantage": float(layers - confs),
        })
    path = out_dir / "confidence_contrast.csv"
    write_csv(rows, path)
    artifacts["confidence_contrast"] = path


# --------------------------------------------------------------------------- #
def _control(cfg, rows, features_dir, out_dir, artifacts, n_perm=200):
    """Permutation null: is any per-layer AUC above chance? (spec §13)."""
    from fok.evaluation import candidate_targets, target_values

    X, _layers = load_hidden(features_dir, "A")
    if X is None:
        return
    tr, val, te, _ = _split_mask(rows)
    rng = np.random.default_rng(cfg.seed)
    res = []

    for tgt in candidate_targets(rows):
        y = target_values(rows, tgt)
        if y is None:
            continue
        real_auc, layer_used = _best_real_auc(X, y, tr, te, cfg)
        nulls = []
        for _ in range(n_perm):
            ys = rng.permutation(y)
            pd_probe = make_probe(cfg.probe, C=cfg.probe_C)
            pd_probe.fit(X[tr][:, layer_used, :], ys[tr])
            p = pd_probe.predict_proba(X[te][:, layer_used, :])[:, 1]
            nulls.append(float(roc_auc_score(ys[te], p)))
        nulls = np.array(nulls)
        res.append({
            "target": tgt,
            "layer": int(_layers[layer_used]) if _layers else -1,
            "real_test_auc": float(real_auc),
            "null_mean": float(nulls.mean()),
            "null_95": float(np.percentile(nulls, 95)),
            "p_value": float((int(np.sum(nulls >= real_auc)) + 1) / (n_perm + 1)),
            "signal_above_chance95": bool(real_auc > np.percentile(nulls, 95)),
        })
    path = out_dir / "control.csv"
    write_csv(res, path)
    artifacts["control"] = path


def _best_real_auc(X, y, tr, te, cfg):
    """AUC of best single layer (by test) on target y; returns (auc, layer_idx)."""
    best, bi = -1.0, 0
    for li in range(X.shape[1]):
        pr = make_probe(cfg.probe, C=cfg.probe_C)
        pr.fit(X[tr][:, li, :], y[tr])
        p = pr.predict_proba(X[te][:, li, :])[:, 1]
        if len(np.unique(y[te])) < 2:
            continue
        a = float(roc_auc_score(y[te], p))
        if a > best:
            best, bi = a, li
    return best, bi


# --------------------------------------------------------------------------- #
def _split_mask(rows):
    splits = np.array([r["split"] for r in rows])
    return (splits == "train"), (splits == "val"), (splits == "test"), splits


# --------------------------------------------------------------------------- #
def _timepoints(cfg, rows, features_dir, out_dir, artifacts):
    """Spec §9: compare A vs B vs C separability for each target."""
    from fok.evaluation import candidate_targets, target_values

    avail = [tp for tp in ("A", "B", "C") if (features_dir / f"hidden_{tp}.npy").exists()]
    if len(avail) < 2:
        return
    tr, val, te, _ = _split_mask(rows)
    res = []
    for tgt in candidate_targets(rows):
        y = target_values(rows, tgt)
        if y is None:
            continue
        row = {"target": tgt}
        for tp in avail:
            X, _layers = load_hidden(features_dir, tp)
            a, _ = _best_real_auc(X, y, tr, te, cfg)
            row[f"auc_{tp}"] = float(a)
        res.append(row)
    path = out_dir / "timepoints.csv"
    write_csv(res, path)
    artifacts["timepoints"] = path


# --------------------------------------------------------------------------- #
def _multidim(cfg, rows, features_dir, eval_dir, out_dir, artifacts):
    """Spec §11: is the FOK signal separable from correctness / from the other
    input-condition targets?"""
    from fok.evaluation import candidate_targets, target_values

    X, _layers = load_hidden(features_dir, "A")
    if X is None:
        return
    tr, val, te, _ = _split_mask(rows)
    res = []
    for tgt in candidate_targets(rows):
        y = target_values(rows, tgt)
        if y is None:
            continue
        a, li = _best_real_auc(X, y, tr, te, cfg)
        # correctness as a comparison axis (same-layer AUC on 'correct')
        cvec = _correct_col(rows)
        c_auc = _correct_auc(X, cvec, tr, te, li) if cvec is not None else float("nan")
        res.append({
            "target": tgt,
            "layer": int(_layers[li]) if _layers else -1,
            "target_test_auc": float(a),
            "correct_test_auc_same_layer": c_auc,
            "n_examples": int(len(rows)),
        })
    path = out_dir / "multidim.csv"
    write_csv(res, path)
    artifacts["multidim"] = path


def _correct_col(rows):
    vals = []
    missing = False
    for r in rows:
        c = r.get("correct", "")
        if c == "":
            missing = True
            vals.append(np.nan)
        else:
            vals.append(int(c))
    if missing:
        return None
    return np.asarray(vals, dtype=float)


def _correct_auc(X, cvec, tr, te, li):
    ok = ~np.isnan(cvec)
    if ok[tr].sum() < 4 or len(np.unique(cvec[tr][ok[tr]])) < 2:
        return float("nan")
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000)
    mtr = ok & tr
    clf.fit(X[mtr][:, li, :], cvec[mtr].astype(int))
    mte = ok & te
    if len(np.unique(cvec[mte])) < 2:
        return float("nan")
    return float(roc_auc_score(cvec[mte].astype(int), clf.predict_proba(X[mte][:, li, :])[:, 1]))


def _read_df(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)
