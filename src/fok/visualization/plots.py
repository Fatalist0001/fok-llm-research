"""Visualization: turn the analysis CSVs into the plots described in the spec.

Every plot reads only the CSV/JSON artifacts produced by the analyse stage, so
plotting never touches the model. All figures are saved to the run's
``plots/`` directory as PNG (and some as PDF).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import ExperimentConfig
from ..utils import save_json
from ..analysis.evaluation_split import load_conf_and_probe, load_rows

logger = logging.getLogger("fok.plots")


def plot(
    cfg: ExperimentConfig,
    eval_dir: Optional[Path] = None,
    analysis_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    eval_dir = Path(eval_dir) if eval_dir else cfg.run_dir() / "eval"
    analysis_dir = Path(analysis_dir) if analysis_dir else cfg.run_dir() / "analysis"
    out_dir = Path(out_dir) if out_dir else cfg.run_dir() / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_df, conf_df = load_conf_and_probe(eval_dir)
    artifacts: Dict[str, Path] = {}

    if probe_df is not None and not probe_df.empty:
        artifacts["per_layer"] = _plot_per_layer(probe_df, out_dir)

    per_layer_auc = analysis_dir / "per_layer_auc.csv"
    if per_layer_auc.exists():
        artifacts["auc_by_layer"] = _plot_auc_by_layer(per_layer_auc, out_dir)

    contrast = analysis_dir / "confidence_contrast.csv"
    if contrast.exists():
        artifacts["confidence_contrast"] = _plot_confidence_contrast(contrast, out_dir)

    control = analysis_dir / "control.csv"
    if control.exists():
        artifacts["control"] = _plot_control(control, out_dir)

    timepoints = analysis_dir / "timepoints.csv"
    if timepoints.exists():
        artifacts["timepoints"] = _plot_timepoints(timepoints, out_dir)

    pca = analysis_dir / "multidim_pca.csv"
    if pca.exists():
        artifacts["multidim_pca"] = _plot_pca(pca, out_dir)

    umap = analysis_dir / "multidim_umap.csv"
    if umap.exists():
        artifacts["multidim_umap"] = _plot_umap(umap, out_dir)

    clusters = analysis_dir / "multidim_clusters.csv"
    if clusters.exists():
        artifacts["multidim_clusters"] = _plot_clusters(clusters, out_dir)

    save_json({k: str(v) for k, v in artifacts.items()}, out_dir / "plots_meta.json")
    logger.info("Plots written -> %s", out_dir)
    return artifacts


def _plot_per_layer(probe_df, out_dir):
    """Faceted per-layer val/test AUC for each target (spec §4, §12)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for tgt, grp in probe_df.groupby("target"):
        grp = grp.sort_values("layer")
        ax.plot(grp["layer"], grp["val_auc"], marker="o", label=f"{tgt} (val)")
        if "test_auc" in grp.columns:
            ax.plot(grp["layer"], grp["test_auc"], marker="s", ls="--", label=f"{tgt} (test)")
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.set_xlabel("layer")
    ax.set_ylabel("AUC")
    ax.set_title("Per-layer probe AUC (input-condition target)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "per_layer_auc.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _plot_auc_by_layer(per_layer_auc_csv, out_dir):
    df = pd.read_csv(per_layer_auc_csv, index_col=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in df.columns:
        ax.plot(df.index, df[col], marker="o", label=col)
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.set_xlabel("layer")
    ax.set_ylabel("val AUC")
    ax.set_title("Per-layer, per-target validation AUC")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "auc_by_layer.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _plot_confidence_contrast(contrast_csv, out_dir):
    df = pd.read_csv(contrast_csv)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w / 2, df["best_layer_val_auc"], w, label="best layer")
    ax.bar(x + w / 2, df["best_confidence_val_auc"], w, label="best confidence")
    ax.set_xticks(x)
    ax.set_xticklabels(df["target"])
    ax.set_ylabel("val AUC")
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.set_title("Hidden-state signal vs scalar confidence (spec §8)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = out_dir / "confidence_contrast.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _plot_control(control_csv, out_dir):
    df = pd.read_csv(control_csv)
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = df["target"]
    ax.bar(labels, df["real_test_auc"], label="real (layer chosen by val)")
    ax.bar(labels, df["null_95"], alpha=0.5, color="r", label="null 95th pct.")
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.set_ylabel("test AUC")
    ax.set_title("Swap-control: real vs permutation-chance AUC (spec §13)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = out_dir / "control.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _plot_timepoints(timepoints_csv, out_dir):
    df = pd.read_csv(timepoints_csv)
    tps = [c for c in df.columns if c.startswith("auc_")]
    if not tps:
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    for i, tp in enumerate(tps):
        ax.plot(x, df[tp], marker="o", label=tp[4:])
    ax.set_xticks(x)
    ax.set_xticklabels(df["target"])
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.set_ylabel("test AUC")
    ax.set_title("Signal across time points A/B/C (spec §9)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "timepoints.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# Multidimensionality plots (spec §11)
# --------------------------------------------------------------------------- #

def _plot_pca(csv_path, out_dir):
    """PCA 2D scatter plots colored by each target label (spec §11)."""
    df = pd.read_csv(csv_path)
    targets = df["target"].unique()
    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 5), squeeze=False)
    for ax, tgt in zip(axes[0], targets):
        sub = df[df["target"] == tgt]
        for label in sorted(sub["label"].unique()):
            mask = sub["label"] == label
            split_markers = {"train": "o", "val": "s", "test": "^"}
            for spl in sub[mask]["split"].unique():
                m = mask & (sub["split"] == spl)
                ax.scatter(sub.loc[m, "x"], sub.loc[m, "y"],
                           marker=split_markers.get(spl, "o"),
                           label=f"{tgt}={label} ({spl})", alpha=0.7, s=40)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"PCA — {tgt}")
        ax.legend(fontsize=6, loc="best")
        ax.grid(alpha=0.3)
    fig.suptitle("PCA projection of hidden states (best layer per target)", y=1.02)
    fig.tight_layout()
    p = out_dir / "multidim_pca.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def _plot_umap(csv_path, out_dir):
    """UMAP 2D scatter plots colored by each target label (spec §11)."""
    df = pd.read_csv(csv_path)
    targets = df["target"].unique()
    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 5), squeeze=False)
    for ax, tgt in zip(axes[0], targets):
        sub = df[df["target"] == tgt]
        for label in sorted(sub["label"].unique()):
            mask = sub["label"] == label
            split_markers = {"train": "o", "val": "s", "test": "^"}
            for spl in sub[mask]["split"].unique():
                m = mask & (sub["split"] == spl)
                ax.scatter(sub.loc[m, "x"], sub.loc[m, "y"],
                           marker=split_markers.get(spl, "o"),
                           label=f"{tgt}={label} ({spl})", alpha=0.7, s=40)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_title(f"UMAP — {tgt}")
        ax.legend(fontsize=6, loc="best")
        ax.grid(alpha=0.3)
    fig.suptitle("UMAP projection of hidden states (best layer per target)", y=1.02)
    fig.tight_layout()
    p = out_dir / "multidim_umap.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def _plot_clusters(csv_path, out_dir):
    """Bar chart of ARI and purity for KMeans clustering vs each target."""
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w / 2, df["ari"], w, label="Adjusted Rand Index")
    ax.bar(x + w / 2, df["purity"], w, label="Purity")
    ax.set_xticks(x)
    ax.set_xticklabels(df["target"])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.0, color="k", lw=0.8, ls=":")
    ax.set_title("KMeans(k=2) clustering alignment with targets (spec §11)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = out_dir / "multidim_clusters.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p
