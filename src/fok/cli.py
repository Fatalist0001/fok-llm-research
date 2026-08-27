"""Command-line interface for the FOK research pipeline.

Commands (each driven by a shared YAML config):

    fok extract        run the model over a dataset and store hidden states + labels
    fok evaluate       train probes per layer and score them (features -> probe CSV)
    fok analyze        higher-level interpretation (control, timepoints, multidim)
    fok plot           produce the figures from the analysis CSVs
    fok run            convenience: extract -> evaluate -> analyze -> plot
    fok download-model download the default model into data/models (no config needed)

Example:

    fok run --config configs/fok.yaml --dataset synthetic_knowledge \
        --dataset-param n_per_class 150
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import click

from .config import ExperimentConfig, load_config
from .datasets import get_dataset
from .utils import set_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fok.cli")


@click.group()
@click.option("--config", "-c", "config_path", default=None, type=click.Path(), help="YAML config.")
@click.option("--seed", default=None, type=int, help="Override random seed.")
@click.option("--dataset", default=None, help="Override dataset name.")
@click.option("--dataset-param", "dataset_params", multiple=True, metavar="K=V",
              help="Override a dataset config key, repeatable.")
@click.option("--layers", default=None, help="Override layer selector (e.g. 'all').")
@click.pass_context
def cli(ctx, config_path, seed, dataset, dataset_params, layers):
    """FOK research pipeline."""
    ctx.ensure_object(dict)
    cfg = load_config(Path(config_path) if config_path else None)
    if seed is not None:
        cfg.seed = seed
    if dataset is not None:
        cfg.dataset = dataset
    for kv in dataset_params:
        key, _, val = kv.partition("=")
        if val.lower() in ("true", "false"):
            val = val.lower() == "true"
        elif val.lstrip("-").isdigit():
            val = int(val)
        else:
            try:
                val = float(val)
            except ValueError:
                pass
        cfg.dataset_config[key] = val
    if layers is not None:
        cfg.layers = layers
    set_seed(cfg.seed)
    ctx.obj["cfg"] = cfg
    ctx.obj["cfg_path"] = config_path


def _load_model(cfg):
    from .model.backend import HFBackend
    return HFBackend(cfg)


def _load_dataset(cfg):
    return get_dataset(cfg.dataset, cfg.dataset_config).build()


@cli.command()
@click.pass_context
def extract(ctx):
    """Run the model over the dataset and store hidden states + labels."""
    cfg: ExperimentConfig = ctx.obj["cfg"]
    backend = _load_model(cfg)
    ds = _load_dataset(cfg)
    from .extraction import collect
    out = collect(cfg, backend, ds)
    click.echo(f"Extraction done -> {out}")


@cli.command()
@click.option("--features-dir", default=None, type=click.Path(), help="Override features dir.")
@click.option("--targets", default=None, help="Comma-separated target columns to probe.")
@click.pass_context
def evaluate(ctx, features_dir, targets):
    """Train per-layer probes and score them (features -> probe CSV)."""
    cfg: ExperimentConfig = ctx.obj["cfg"]
    from .evaluation import evaluate as run_eval
    tgt = [t.strip() for t in targets.split(",")] if targets else None
    out = run_eval(cfg, features_dir=Path(features_dir) if features_dir else None, targets=tgt)
    click.echo(f"Evaluation done -> {out['probe_csv']}")


@cli.command()
@click.option("--n-perm", default=200, type=int, help="Permutations for the control.")
@click.pass_context
def analyze(ctx, n_perm):
    """Higher-level interpretation (control, timepoints, multidimensionality)."""
    cfg: ExperimentConfig = ctx.obj["cfg"]
    from .analysis import analyze as run_analysis
    arts = run_analysis(cfg, n_perm=n_perm)
    for name, p in arts.items():
        click.echo(f"  {name}: {p}")


@cli.command()
@click.pass_context
def plot(ctx):
    """Produce figures from the analysis CSVs."""
    cfg: ExperimentConfig = ctx.obj["cfg"]
    from .visualization import plot as run_plot
    arts = run_plot(cfg)
    for name, p in arts.items():
        click.echo(f"  {name}: {p}")


@cli.command()
@click.option("--extract-only", is_flag=True, help="Stop after extraction.")
@click.pass_context
def run(ctx, extract_only):
    """Convenience: extract -> evaluate -> analyze -> plot in one shot."""
    cfg: ExperimentConfig = ctx.obj["cfg"]
    backend = _load_model(cfg)
    ds = _load_dataset(cfg)
    from .extraction import collect
    feats = collect(cfg, backend, ds)
    if extract_only:
        click.echo(f"Extraction done -> {feats}")
        return
    from .evaluation import evaluate as run_eval
    run_eval(cfg, features_dir=feats)
    from .analysis import analyze as run_analysis
    arts = run_analysis(cfg)
    for name in arts:
        click.echo(f"  analysis/{name}")
    from .visualization import plot as run_plot
    run_plot(cfg)
    click.echo(f"All artifacts under {cfg.run_dir()}")


@cli.command()
@click.option("--repo", default="Qwen/Qwen3.5-2B", help="Hugging Face repo to download.")
@click.option("--out", default=None, type=click.Path(), help="Output directory.")
def download_model(repo, out):
    """Download a model into data/models (no config required)."""
    import subprocess
    import sys
    from .config import PROJECT_ROOT
    out_dir = Path(out) if out else PROJECT_ROOT / "data" / "models"
    script = PROJECT_ROOT / "scripts" / "download_model.py"
    cmd = [sys.executable, str(script), "--repo", repo, "--out", str(out_dir)]
    click.echo(" ".join(cmd))
    subprocess.run(cmd, check=True)


@cli.command()
@click.pass_context
def show_config(ctx):
    """Print the effective configuration."""
    import yaml
    click.echo(yaml.safe_dump(ctx.obj["cfg"].to_dict(), sort_keys=False))


if __name__ == "__main__":
    cli()
