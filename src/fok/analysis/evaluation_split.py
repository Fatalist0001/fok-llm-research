"""Shared loaders for the analyse/plot stages.

These functions read exactly what the ``extract`` and ``evaluate`` stages wrote,
so analysis/visualization stay decoupled from the model and from the mining
details.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_rows(features_dir: Path) -> List[Dict]:
    """Load the examples.csv rows as a list of dicts."""
    p = features_dir / "examples.csv"
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_meta(features_dir: Path) -> Dict:
    p = features_dir / "extraction_meta.json"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_hidden(features_dir: Path, tp: str) -> Tuple[Optional[np.ndarray], List[int]]:
    """Load hidden_{tp}.npy plus the layer order from extraction_meta.json."""
    path = features_dir / f"hidden_{tp}.npy"
    if not path.exists():
        return None, []
    arr = np.load(path)
    meta = load_meta(features_dir)
    layers = list(meta.get("layers", [])) or list(range(arr.shape[1]))
    return arr, [int(l) for l in layers]


def load_features(features_dir: Path, hp: str = "A") -> List[Dict]:
    """Canonical entry: return the example rows (kept for API stability)."""
    return load_rows(features_dir)


def load_conf_and_probe(eval_dir: Path):
    import pandas as pd

    def _read(name):
        p = eval_dir / name
        if not p.exists():
            return None
        return pd.read_csv(p)

    return _read("probe_results.csv"), _read("confidence_baselines.csv")
