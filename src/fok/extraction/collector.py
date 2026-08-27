"""Collector: run the model over a dataset and persist the research features.

For every example the collector stores:

  * ``examples.csv``   - one row per example with ids, questions, split,
    target labels (from dataset metadata), the generated answer, the
    correctness label, and all confidence baselines (token probs, avg/seq
    log-probs, entropy).
  * ``hidden_{A,B,C}.npy`` - the per-layer representation vectors at the
    requested time points, one row per example in the same order as the CSV.

We intentionally store activations only for the selected layers and the chosen
representation (not full per-token dumps), per the project rule about not
keeping activation files bigger than necessary.

The target a probe is trained on is chosen later (at probe time) from the
stored columns; the collector just records the raw evidence.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import tqdm

from ..config import ExperimentConfig
from ..datasets import Dataset
from ..model.backend import HFBackend, SnapshotSpec
from ..utils import save_json, write_csv
from .answers import check_answer

logger = logging.getLogger("fok.collector")


def collect(
    cfg: ExperimentConfig,
    backend: HFBackend,
    dataset: Dataset,
    out_dir: Optional[Path] = None,
    capture: Optional[SnapshotSpec] = None,
) -> Path:
    """Run extraction over the dataset and write artifacts to ``out_dir``.

    Returns the directory that was written (defaults to cfg.results_dir/<run>/features).
    """
    capture = capture or SnapshotSpec(
        capture_A=True,
        capture_B=cfg.dataset_config.get("capture_B", False),
        capture_C=cfg.dataset_config.get("capture_C", False),
        b_tokens=cfg.dataset_config.get("b_tokens", 8),
    )
    if out_dir is None:
        out_dir = cfg.run_dir() / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve concrete layer indices and representation metadata.
    layer_indices = backend.resolve_layers(cfg.layers)
    rep_name = cfg.representation
    extra_meta = {
        "model_path": str(cfg.model_path),
        "layers": layer_indices,
        "representation": rep_name,
        "n_layers_total": backend.n_layers + 1,
        "hidden_dim": backend.hidden_dim,
        "max_new_tokens": cfg.max_new_tokens,
        "seed": cfg.seed,
    }
    save_json(extra_meta, out_dir / "extraction_meta.json")

    # Collect all examples (train+val+test together) in a stable order.
    all_examples = dataset.examples

    rows = []
    hstates = {"A": [], "B": [], "C": []}
    to_capture = [p for p in ("A", "B", "C") if capture.__dict__.get(f"capture_{p}")]

    for ex in tqdm.tqdm(all_examples, desc="extracting", unit="ex"):
        result = backend.generate(ex.question, spec=capture)

        correct = check_answer(result.answer, ex.correct_answer)
        meta = dict(ex.metadata or {})

        row = {
            "id": ex.id,
            "split": ex.split,
            "question": ex.question,
            "category": ex.category,
            "duration": "" if ex.difficulty is None else ex.difficulty,
            "correct_answer": ex.correct_answer or "",
            "generated": result.answer,
            "correct": "" if correct is None else int(correct),
            "avg_logprob": result.avg_logprob,
            "seq_logprob": result.seq_logprob,
            "first_entropy": result.first_entropy,
            "mean_entropy": result.mean_entropy,
            "top1_prob": result.token_probs[0] if result.token_probs else "",
        }
        # copy through any dataset-defined target/auxiliary fields
        for k, v in meta.items():
            row[k] = v
        rows.append(row)

        for p in to_capture:
            snap = result.snapshots.get(p)
            if snap is None:
                hstates[p].append(np.full((len(layer_indices), backend.hidden_dim), np.nan, dtype=np.float32))
            else:
                # keep only the requested layers
                hstates[p].append(snap[layer_indices])
        # ---- save progressively so a crash doesn't lose everything ----
        _flush(out_dir, rows, layer_indices, hstates, to_capture, flush_every=25)

    _flush(out_dir, rows, layer_indices, hstates, to_capture, final=True)

    cfg.save(out_dir / "config.yaml")
    logger.info("Extraction complete: %d examples -> %s", len(rows), out_dir)
    return out_dir


def _flush(
    out_dir: Path,
    rows,
    layer_indices,
    hstates,
    to_capture,
    flush_every: int = 100,
    final: bool = False,
) -> None:
    """Write the accumulated rows and hidden-state arrays to disk.

    Called periodically (every ``flush_every`` examples) and at the end
    (``final=True``) so partial results survive interruption/errors.
    """
    n = len(rows)
    if n == 0:
        return
    if not final and n % flush_every != 0:
        return
    write_csv(rows, out_dir / "examples.csv")
    for p in to_capture:
        arr = np.stack(hstates[p], axis=0) if hstates[p] else np.zeros((0, len(layer_indices), 0))
        np.save(out_dir / f"hidden_{p}.npy", arr.astype(np.float32))
    if final:
        save_json({"layer_indices": layer_indices}, out_dir / "layer_index.json")
