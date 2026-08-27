"""Configuration handling for the FOK research project.

The whole pipeline is driven by a single YAML config file describing the model,
the dataset, the experiment and the probe settings. Every experiment saves the
(serialized) config alongside its results so the run is reproducible.

A single ``ExperimentConfig`` object is the standard currency passed between the
CLI commands (``extract``, ``train-probe``, ...). This keeps the pipeline stages
decoupled while ensuring they all agree on the same settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Project root = package dir /../.. (src/fok -> src -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def default_workspace() -> Path:
    """Where results/experiments/data live. Defaults to the project root."""
    return PROJECT_ROOT


@dataclass
class ExperimentConfig:
    """All settings for one reproducible experiment."""

    # --- general ---
    name: str = "fok_experiment"
    seed: int = 42

    # --- model ---
    model_path: str = str(PROJECT_ROOT / "data" / "models" / "Qwen3.5-2B")
    # Layers to extract. Use "all" or a comma list, e.g. "0,2,4,...".
    layers: str = "all"
    max_new_tokens: int = 96
    temperature: float = 0.0  # 0 => greedy (deterministic), matters for confidence
    dtype: str = "bfloat16"

    # --- dataset ---
    dataset: str = "fok_trivia"           # name of a built-in dataset
    dataset_config: Dict[str, Any] = field(default_factory=dict)
    train_ratio: float = 0.6
    val_ratio: float = 0.2               # remainder is test
    max_examples: Optional[int] = None   # cap for quick runs

    # --- representation ---
    # What we take from each layer's hidden states [batch, seq, hidden]:
    #   "last_token"  -> final token's vector
    #   "mean"        -> mean over question tokens
    #   "last_k_mean" -> mean over the last k tokens
    representation: str = "last_token"
    last_k: int = 4

    # --- probe ---
    # "logistic" (classification) or "ridge" (regression)
    probe: str = "logistic"
    probe_C: float = 1.0
    # For the MLP experiment we train the same probe on the same data with an MLP.
    mlp_hidden: List[int] = field(default_factory=lambda: [256])

    # --- experiment / target ---
    # Which target the probe is asked to predict. Keys are the target column
    # names written by the collector (see fok.extraction.collector).
    target: str = "knowledge_state"
    # For experiments that dichotomize a continuous target.
    target_threshold: Optional[float] = None

    # --- misc ---
    device: str = "cuda"
    results_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "results")
    experiment_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "experiments")
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")

    # ------------------------------------------------------------------ #
    @property
    def layer_indices(self) -> List[int]:
        """Resolve the ``layers`` string into a concrete list of indices.

        The caller resolves the concrete layer count and may override with a
        raw list; this property handles the ``str`` case (e.g. 'all', '0,5').
        """
        return _resolve_layers(self.layers)

    def run_id(self) -> str:
        """A short unique-ish identifier used for output folder names."""
        return f"{self.name}_rep-{self.representation}_tgt-{self.target}"

    def run_dir(self) -> Path:
        """Directory under results/ where this run's artifacts live."""
        d = self.results_dir / self.run_id()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("results_dir", "experiment_dir", "data_dir"):
            d[k] = str(d[k])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        c = cls()
        allowed = set(asdict(c).keys())
        for k, v in d.items():
            if k in allowed:
                setattr(c, k, v)
        # coercion back to Path for path fields
        for k in ("results_dir", "experiment_dir", "data_dir"):
            if getattr(c, k) is not None:
                setattr(c, k, Path(getattr(c, k)))
        return c

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as fh:
            d = yaml.safe_load(fh)
        return cls.from_dict(d)


def _resolve_layers(layers: str) -> List[int]:
    """Parse a layer selector into a sorted list of ints.

    'all' has no fixed meaning until the model layer count is known; it is
    represented here as the special token -1 which callers replace with
    ``list(range(n_layers))``.
    """
    layers = str(layers).strip().lower()
    if layers == "all":
        return [-1]
    out = []
    for part in layers.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def load_config(path: Optional[Path]) -> ExperimentConfig:
    """Load a config from a path, or return a default config if none given."""
    if path is None:
        cfg = ExperimentConfig()
        (cfg.results_dir.parent).mkdir(parents=True, exist_ok=True)
        return cfg
    return ExperimentConfig.load(Path(path))
