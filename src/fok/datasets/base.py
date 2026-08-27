"""Unified dataset interface for the FOK project.

Every dataset yields :class:`Example` objects with at least ``id``, ``question``,
``correct_answer`` and ``category``, plus optional ``difficulty`` and free-form
``metadata``. Each example also carries an explicit ``split`` (train/val/test),
because one of the project's rules is that nearly-identical questions must never
appear in both train and test.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Example:
    """One labeled item in a research dataset."""

    id: str
    question: str
    correct_answer: Optional[str] = None
    category: str = "qa"
    difficulty: Optional[float] = None      # 0 easy .. 1 hard (optional)
    split: str = "train"                    # train | val | test
    metadata: Dict[str, Any] = field(default_factory=dict)


def stable_id(text: str, salt: str = "") -> str:
    """Deterministic short id for an example (used for reproducible splits)."""
    raw = f"{salt}::{text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def hsplit(question_text: str) -> str:
    """Deterministic train/val/test partition key derived from the question text."""
    h = int(hashlib.sha1(question_text.encode("utf-8")).hexdigest()[:8], 16)
    return h % 1000


class Dataset:
    """Base class for datasets.

    Subclasses implement :meth:`_build` which returns a flat list of
    :class:`Example` (splits already assigned). The base :meth:`iter_split`
    filters by split and :meth:`to_rows` flattens metadata into a dict for
    saving to CSV.
    """

    name: str = "base"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.examples: List[Example] = []
        self._built = False

    def build(self) -> "Dataset":
        if not self._built:
            self.examples = self._build()
            self._built = True
        return self

    # subclasses override
    def _build(self) -> List[Example]:  # pragma: no cover - abstract
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    def iter_split(self, split: str):
        for e in self.examples:
            if e.split == split:
                yield e

    def rows(self, split: Optional[str] = None) -> List[Dict[str, Any]]:
        out = []
        for e in self.examples:
            if split and e.split != split:
                continue
            r = {
                "id": e.id,
                "question": e.question,
                "correct_answer": e.correct_answer or "",
                "category": e.category,
                "difficulty": e.difficulty if e.difficulty is not None else "",
                "split": e.split,
            }
            r.update(e.metadata or {})
            out.append(r)
        return out

    def counts(self) -> Dict[str, int]:
        c = {"train": 0, "val": 0, "test": 0}
        for e in self.examples:
            c[e.split] = c.get(e.split, 0) + 1
        return c


def assign_splits(
    examples: List[Example],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    split_key=None,
) -> List[Example]:
    """Deterministically (by question text) assign train/val/test splits.

    Using a hash of the question text (rather than position) guarantees that
    identical questions map to the same split, which is exactly what we need to
    prevent a (near-)identical question from appearing in both train and test.

    ``split_key`` is an optional callable ``key(example) -> str`` used instead of
    the question text. Datasets whose examples are variants of one underlying
    item (e.g. ``info_variant``) pass a key derived from that item's id so all
    variants land in the same split.
    """
    for e in examples:
        key = e.question if split_key is None else split_key(e)
        h = hsplit(str(key))
        if h / 1000 < train_ratio:
            e.split = "train"
        elif h / 1000 < train_ratio + val_ratio:
            e.split = "val"
        else:
            e.split = "test"
    return examples


def balanced_subset(dataset: Dataset, n: int, split: str, seed: int = 0) -> List[Example]:
    """Return a balanced subsample of a split (for quick end-to-end runs)."""
    rng = np.random.default_rng(seed)
    examples = list(dataset.iter_split(split))
    if len(examples) <= n:
        return examples
    return list(rng.choice(examples, size=n, replace=False))
