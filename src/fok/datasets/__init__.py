"""Dataset loading for the FOK pipeline.

Use :func:`get_dataset` to obtain a built-in dataset by name:

    from fok.datasets import get_dataset, available
    ds = get_dataset("synthetic_knowledge", {"n_per_class": 200}).build()

The dataset interface (Example/Dataset/split helpers) lives in :mod:`fok.datasets.base`.
"""

from .base import Dataset, Example, assign_splits, balanced_subset, stable_id
from .builtin import available, get_dataset

__all__ = [
    "Dataset",
    "Example",
    "assign_splits",
    "balanced_subset",
    "stable_id",
    "available",
    "get_dataset",
]
