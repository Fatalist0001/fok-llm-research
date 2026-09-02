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
from sklearn.linear_model import LogisticRegression
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
    _pca_umap_clustering(cfg, rows, features_dir, out_dir, artifacts)

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
        real_auc, layer_used, _ = _select_layer_by_val(X, y, tr, val, te, cfg)
        nulls = []
        for _ in range(n_perm):
            ys = rng.permutation(y)
            # Skip iterations where the shuffled labels collapse to one class
            # in the test slice — otherwise roc_auc_score is undefined (NaN).
            if len(np.unique(ys[te])) < 2:
                continue
            pd_probe = make_probe(cfg.probe, C=cfg.probe_C)
            pd_probe.fit(X[tr][:, layer_used, :], ys[tr])
            p = pd_probe.predict_proba(X[te][:, layer_used, :])[:, 1]
            nulls.append(float(roc_auc_score(ys[te], p)))
        if not nulls:
            nulls = [float("nan")]
        nulls = np.array(nulls)
        res.append({
            "target": tgt,
            "layer": int(_layers[layer_used]) if _layers else -1,
            "real_test_auc": float(real_auc),
            "null_mean": float(np.nanmean(nulls)),
            "null_95": float(np.nanpercentile(nulls, 95)),
            "p_value": (
                float((int(np.sum(nulls >= real_auc)) + 1) / (len(nulls) + 1))
                if not np.isnan(real_auc) and not np.isnan(nulls).all()
                else float("nan")
            ),
            "signal_above_chance95": bool(real_auc > np.nanpercentile(nulls, 95)),
        })
    path = out_dir / "control.csv"
    write_csv(res, path)
    artifacts["control"] = path


def _select_layer_by_val(X, y, tr, val, te, cfg):
    """Select the best layer by *validation* AUC, then score it once on test.

    The test set is used exactly once, on the single layer chosen via validation,
    so the reported test AUC is not optimistic from layer-selection (no
    test-set leakage). Returns ``(test_auc, layer_idx, val_auc)``.
    """
    best_val, bi = -1.0, 0
    for li in range(X.shape[1]):
        pr = make_probe(cfg.probe, C=cfg.probe_C)
        pr.fit(X[tr][:, li, :], y[tr])
        if len(np.unique(y[val])) < 2:
            continue
        va = float(roc_auc_score(y[val], pr.predict_proba(X[val][:, li, :])[:, 1]))
        if va > best_val:
            best_val, bi = va, li
    # score the single selected layer once on the held-out test split
    pr = make_probe(cfg.probe, C=cfg.probe_C)
    pr.fit(X[tr][:, bi, :], y[tr])
    if len(np.unique(y[te])) < 2:
        test_auc = float("nan")
    else:
        test_auc = float(
            roc_auc_score(y[te], pr.predict_proba(X[te][:, bi, :])[:, 1])
        )
    return test_auc, bi, best_val


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
            a, _li, _ = _select_layer_by_val(X, y, tr, val, te, cfg)
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
        a, li, _ = _select_layer_by_val(X, y, tr, val, te, cfg)
        # correctness as a comparison axis: select its own layer by validation,
        # then score the chosen layer once on test (no reuse of the target's
        # test-selected layer).
        cvec = _correct_col(rows)
        c_auc, _c_li = _correct_auc(X, cvec, tr, val, te, cfg) if cvec is not None else (float("nan"), -1)
        res.append({
            "target": tgt,
            "layer": int(_layers[li]) if _layers else -1,
            "target_test_auc": float(a),
            "correct_test_auc": c_auc,
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


def _correct_auc(X, cvec, tr, val, te, cfg):
    """Best correctness AUC on *test*, with the layer selected by validation.

    Returns ``(test_auc, layer_idx)``. NaN when correctness is degenerate (too
    few examples or a single class) on either the validation or test slice.
    """
    ok = ~np.isnan(cvec)
    mtr = ok & tr
    mval = ok & val
    mte = ok & te
    if (mtr.sum() < 4
            or len(np.unique(cvec[mtr])) < 2
            or len(np.unique(cvec[mval])) < 2):
        return float("nan"), -1
    best_val, bi = -1.0, 0
    for li in range(X.shape[1]):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[mtr][:, li, :], cvec[mtr].astype(int))
        va = float(roc_auc_score(cvec[mval].astype(int), clf.predict_proba(X[mval][:, li, :])[:, 1]))
        if va > best_val:
            best_val, bi = va, li
    if len(np.unique(cvec[mte])) < 2:
        return float("nan"), bi
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X[mtr][:, bi, :], cvec[mtr].astype(int))
    return float(roc_auc_score(cvec[mte].astype(int), clf.predict_proba(X[mte][:, bi, :])[:, 1])), bi


def _read_df(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# PCA / UMAP / clustering  (spec §11)
# --------------------------------------------------------------------------- #

def _pca_umap_clustering(cfg, rows, features_dir, out_dir, artifacts):
    """PCA, UMAP projections and KMeans clustering of hidden states.

    For each candidate target, the best layer (selected by validation AUC) is
    projected to 2D via PCA and (optionally) UMAP, and clustered with
    KMeans(k=2).  Results are saved as CSVs that the plot stage reads.

    Spec §11: "Исследование многомерности" — PCA, UMAP, clustering.
    """
    from fok.evaluation import candidate_targets, target_values

    X, layers = load_hidden(features_dir, "A")
    if X is None:
        return
    tr, val, te, splits = _split_mask(rows)

    pca_rows = []
    umap_rows = []
    cluster_rows = []

    for tgt in candidate_targets(rows):
        y = target_values(rows, tgt)
        if y is None:
            continue

        # Select best layer by validation AUC (same logic as other analyses).
        best_auc, best_li, _ = _select_layer_by_val(X, y, tr, val, te, cfg)
        if np.isnan(best_auc):
            continue
        X_best = X[:, best_li, :]  # [n_examples, hidden_dim]
        layer_id = int(layers[best_li]) if layers else best_li

        # ── PCA ──────────────────────────────────────────────────────
        coords = _run_pca(X_best)
        for i, (cx, cy) in enumerate(coords):
            pca_rows.append({
                "x": float(cx), "y": float(cy),
                "target": tgt, "label": int(y[i]),
                "split": splits[i], "layer": layer_id,
            })

        # ── UMAP (optional) ──────────────────────────────────────────
        coords_u = _run_umap(X_best)
        if coords_u is not None:
            for i, (cx, cy) in enumerate(coords_u):
                umap_rows.append({
                    "x": float(cx), "y": float(cy),
                    "target": tgt, "label": int(y[i]),
                    "split": splits[i], "layer": layer_id,
                })

        # ── Clustering ───────────────────────────────────────────────
        ari, purity = _run_clustering(X_best, y)
        cluster_rows.append({
            "target": tgt,
            "layer": layer_id,
            "val_auc": float(best_auc),
            "ari": float(ari),
            "purity": float(purity),
            "n_examples": int(len(y)),
        })

    if pca_rows:
        path = out_dir / "multidim_pca.csv"
        write_csv(pca_rows, path)
        artifacts["multidim_pca"] = path

    if umap_rows:
        path = out_dir / "multidim_umap.csv"
        write_csv(umap_rows, path)
        artifacts["multidim_umap"] = path

    if cluster_rows:
        path = out_dir / "multidim_clusters.csv"
        write_csv(cluster_rows, path)
        artifacts["multidim_clusters"] = path


def _run_pca(X_2d: np.ndarray) -> np.ndarray:
    """Project [n, hidden_dim] to 2D via PCA. Returns [n, 2]."""
    from sklearn.decomposition import PCA

    n_components = min(2, X_2d.shape[0], X_2d.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    coords = pca.fit_transform(X_2d)
    # Pad to 2 columns if only 1 component was possible.
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros_like(coords)])
    return coords


def _run_umap(X_2d: np.ndarray):
    """Project [n, hidden_dim] to 2D via UMAP. Returns [n, 2] or None."""
    try:
        import umap
    except ImportError:
        logger.info("umap-learn not installed; skipping UMAP projection")
        return None
    reducer = umap.UMAP(n_components=2, random_state=0, n_neighbors=min(15, len(X_2d) - 1))
    return reducer.fit_transform(X_2d)


def _run_clustering(X_2d: np.ndarray, y: np.ndarray):
    """KMeans(k=2) on the hidden states; return (ARI, purity) vs ground truth."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    km = KMeans(n_clusters=2, n_init=10, random_state=0)
    pred = km.fit_predict(X_2d)
    ari = adjusted_rand_score(y, pred)
    purity = _purity(y, pred)
    return ari, purity


def _purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of examples in the majority class of their assigned cluster."""
    total = len(y_true)
    if total == 0:
        return 0.0
    correct = 0
    for k in np.unique(y_pred):
        mask = y_pred == k
        correct += int((y_true[mask] == np.bincount(y_true[mask].astype(int)).argmax()).sum())
    return correct / total
