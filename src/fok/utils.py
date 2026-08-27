"""Small shared helpers: seeding, JSON/CSV serialization, device detection."""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def set_seed(seed: int) -> None:
    """Make a run reproducible: seed python, numpy and torch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_device(preferred: str = "cuda") -> str:
    """Return a usable torch device string ('cuda' if available else 'cpu')."""
    if preferred.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
    if preferred.startswith("mps"):
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
    return "cpu"


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_ndarray(arr, path: Path) -> None:
    """Save a numpy array to .npy (parent dir created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not serializable: {type(o)}")


def write_csv(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    """Write a list of dicts to a CSV file (keys taken from the first row)."""
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        open(path, "w", encoding="utf-8").close()
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def normalize(v: np.ndarray) -> np.ndarray:
    """Unit-normalize vectors along the last axis (useful for cosine-based reps)."""
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def stable_softmax(logits) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    logits = np.asarray(logits, dtype=np.float64)
    m = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(m)
    return e / e.sum(axis=-1, keepdims=True)


def make_parents_for(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
