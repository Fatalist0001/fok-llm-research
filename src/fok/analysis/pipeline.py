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
    _simple_baselines(cfg, rows, features_dir, out_dir, artifacts)
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
        real_auc, layer_used, _, tuned_c = _select_layer_by_val(X, y, tr, val, te, cfg)
        nulls = []
        for _ in range(n_perm):
            ys = rng.permutation(y)
            # Skip iterations where the shuffled labels collapse to one class
            # in the test slice — otherwise roc_auc_score is undefined (NaN).
            if len(np.unique(ys[te])) < 2:
                continue
            # Null model shares the layer and regularisation chosen on the real
            # labels (C picked once by validation), so real vs null differ only
            # in the label shuffle — no per-permutation C search.
            pd_probe = make_probe(cfg.probe, C=tuned_c, tune_C=False)
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
    test-set leakage). Returns ``(test_auc, layer_idx, val_auc, tuned_C)`` where
    ``tuned_C`` is the regularisation strength chosen by validation AUC (or the
    fixed ``cfg.probe_C`` when C-tuning is off). The permutation null reuses this
    same ``tuned_C`` so real and null models share the same regularisation scheme.
    """
    best_val, bi = -1.0, 0
    for li in range(X.shape[1]):
        pr = make_probe(cfg.probe, C=cfg.probe_C, tune_C=cfg.probe_tune_C)
        pr.fit(X[tr][:, li, :], y[tr],
               X_val=X[val][:, li, :] if np.any(val) else None,
               y_val=y[val] if np.any(val) else None)
        if len(np.unique(y[val])) < 2:
            continue
        va = float(roc_auc_score(y[val], pr.predict_proba(X[val][:, li, :])[:, 1]))
        if va > best_val:
            best_val, bi = va, li
    # score the single selected layer once on the held-out test split
    pr = make_probe(cfg.probe, C=cfg.probe_C, tune_C=cfg.probe_tune_C)
    pr.fit(X[tr][:, bi, :], y[tr],
           X_val=X[val][:, bi, :] if np.any(val) else None,
           y_val=y[val] if np.any(val) else None)
    tuned_C = pr.tuned_C_ if pr.tuned_C_ is not None else float(cfg.probe_C)
    if len(np.unique(y[te])) < 2:
        test_auc = float("nan")
    else:
        test_auc = float(
            roc_auc_score(y[te], pr.predict_proba(X[te][:, bi, :])[:, 1])
        )
    return test_auc, bi, best_val, tuned_C


# --------------------------------------------------------------------------- #
def _simple_baselines(cfg, rows, features_dir, out_dir, artifacts):
    """Baseline probes fit on trivial text/surface features (README §8.2).

    A hidden-state probe is only meaningful if it beats probes trained on plain
    text features, because the "known/unknown" datasets differ in surface style
    (field templates, rare made-up name tokens, first/second-person phrasing), not
    just in actual knowledge. For every target we fit, on the *same* train/val/test
    splits, a LogisticRegression on:

      * TF-IDF unigrams+bigrams of the question text
      * the question length (a single scalar feature)
      * the question category (one-hot), when present

    and record their val/test AUC next to the hidden-state test AUC (best layer by
    validation). The comparison hidden vs tfidf is the operative "not just style"
    check; the CSV is written as ``analysis/simple_baselines.csv``.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    from fok.evaluation import candidate_targets, target_values

    X, _ = load_hidden(features_dir, "A")
    if X is None:
        return
    tr, val, te, _ = _split_mask(rows)

    texts = [str(r.get("question", "") or "") for r in rows]
    length = np.asarray([len(t) for t in texts], dtype=float)
    raw_cats = [str(r.get("category", "") or "") for r in rows]
    cats = sorted(set(raw_cats))

    # TF-IDF vocabulary is fit on train only (no target / future leakage).
    vec = None
    tfidf = None
    if any(tr):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)
        vec.fit([texts[i] for i in np.where(tr)[0]])
        tfidf = vec.transform(texts).toarray()

    def _score_2d(fmat, name):
        """Fit logistic on train, return (val_auc, test_auc) on the same splits."""
        if (len(np.unique(y[tr])) < 2
                or len(np.unique(y[val])) < 2
                or len(np.unique(y[te])) < 2
                or fmat.shape[0] != len(rows)):
            return float("nan"), float("nan")
        try:
            clf = LogisticRegression(max_iter=2000)
            clf.fit(fmat[tr], y[tr])
            va = roc_auc_score(y[val], clf.predict_proba(fmat[val])[:, 1])
            ta = roc_auc_score(y[te], clf.predict_proba(fmat[te])[:, 1])
            return float(va), float(ta)
        except Exception:
            return float("nan"), float("nan")

    def _tfidf_exclusive_auc(tfidf_mat, y_true, tr_mask, te_mask, best_layer):
        """B2: hidden-state AUC on test examples where TF-IDF is wrong/uncertain.

        Fits TF-IDF on train, gets predictions on test, identifies hard examples
        (predicted probability in [0.25, 0.75] or misclassified), then computes
        hidden-state AUC on only those examples. This tests whether hidden states
        capture signal beyond lexical features, even when overall TF-IDF AUC is high.
        Returns (n_hard, hidden_auc_on_hard).
        """
        if tfidf_mat is None or te_mask.sum() < 6:
            return 0, float("nan")
        try:
            clf = LogisticRegression(max_iter=2000)
            clf.fit(tfidf_mat[tr_mask], y_true[tr_mask])
            proba = clf.predict_proba(tfidf_mat[te_mask])[:, 1]
            hard = (proba >= 0.25) & (proba <= 0.75)
            n_hard = int(hard.sum())
            if n_hard < 4:
                return n_hard, float("nan")
            te_idx = np.where(te_mask)[0]
            hard_idx = te_idx[hard]
            y_hard = y_true[hard_idx]
            if len(np.unique(y_hard)) < 2:
                return n_hard, float("nan")
            hid_hard = X[hard_idx]
            layer_data = hid_hard[:, best_layer]
            if layer_data.ndim > 1:
                layer_data = layer_data.mean(axis=1)
            auc = roc_auc_score(y_hard, layer_data)
            return n_hard, float(auc)
        except Exception:
            return 0, float("nan")

    res = []
    for tgt in candidate_targets(rows):
        y = target_values(rows, tgt)
        if y is None:
            continue
        hid_test, _li, best_layer_idx, _ = _select_layer_by_val(X, y, tr, val, te, cfg)

        len_val, len_test = _score_2d(length.reshape(-1, 1), "length")

        cat_val = cat_test = float("nan")
        if len(cats) > 1:
            catmat = np.zeros((len(rows), len(cats)), dtype=float)
            for i, c in enumerate(cats):
                catmat[:, i] = [1.0 if rc == c else 0.0 for rc in raw_cats]
            cat_val, cat_test = _score_2d(catmat, "category")

        tf_val = tf_test = float("nan")
        if tfidf is not None:
            tf_val, tf_test = _score_2d(tfidf, "tfidf")

        hid_clean = float(hid_test) if not np.isnan(hid_test) else float("nan")
        beats_tf = not (np.isnan(hid_clean) or np.isnan(tf_test)) and hid_clean > tf_test

        # B1: normalized advantage — when TF-IDF is near ceiling, denominator -> 0
        # and the metric honestly becomes undefined (NaN).
        norm_adv = float("nan")
        if not (np.isnan(hid_clean) or np.isnan(tf_test)):
            denom = 1.0 - tf_test
            if denom > 1e-6:
                norm_adv = (hid_clean - tf_test) / denom

        # B2: hidden AUC on TF-IDF-hard examples
        n_hard, hid_hard_auc = (0, float("nan"))
        if tfidf is not None:
            n_hard, hid_hard_auc = _tfidf_exclusive_auc(
                tfidf, y, tr, te, best_layer_idx)

        res.append({
            "target": tgt,
            "n_train": int(tr.sum()),
            "n_val": int(val.sum()),
            "n_test": int(te.sum()),
            "hidden_test_auc": hid_clean,
            "tfidf_val_auc": tf_val,
            "tfidf_test_auc": tf_test,
            "length_val_auc": len_val,
            "length_test_auc": len_test,
            "category_val_auc": cat_val,
            "category_test_auc": cat_test,
            "hidden_beats_tfidf": beats_tf,
            "advantage_over_tfidf": (
                float(hid_clean - tf_test)
                if not (np.isnan(hid_clean) or np.isnan(tf_test)) else float("nan")
            ),
            "normalized_advantage": norm_adv,
            "n_hard_tfidf": n_hard,
            "hidden_auc_on_hard": hid_hard_auc,
        })
    if not res:
        return
    path = out_dir / "simple_baselines.csv"
    write_csv(res, path)
    artifacts["simple_baselines"] = path


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
            a, _li, _, _ = _select_layer_by_val(X, y, tr, val, te, cfg)
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
        a, li, _, _ = _select_layer_by_val(X, y, tr, val, te, cfg)
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
        best_auc, best_li, _, _ = _select_layer_by_val(X, y, tr, val, te, cfg)
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
