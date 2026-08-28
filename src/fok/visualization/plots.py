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
