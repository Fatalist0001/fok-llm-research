"""Answerability dataset: cleanly answerable vs. unanswerable questions.

A second independent axis of the knowledge state. ``answerable=1`` questions have
a definitive factual answer; ``answerable=0`` are phrased so that no definite
answer exists (personal, subjective, unobservable, or impossible-for-the-model).

Because ``answerable`` is a property of the question (not of the produced text),
the probe target stays independent of correctness.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..base import Dataset, Example, assign_splits

_ANSWERABLE = [
    ("What is the boiling point of water at sea level in Celsius?", "100"),
    ("Who was the first person to walk on the Moon?", "Neil Armstrong"),
    ("What is the SI unit of force?", "Newton"),
    ("Which country has the largest population?", "India"),
    ("What is the process by which plants make their own food?", "Photosynthesis"),
    ("How many sides does a hexagon have?", "Six"),
    ("Which ocean separates Africa and South America?", "Atlantic"),
    ("What year did World War II end?", "1945"),
]

_UNANSWERABLE = [
    "What is the exact number of stars visible from your kitchen window tonight?",
    "What color is the dragon you dreamed about last Tuesday?",
    "How many hours of sleep did the author of this question get in 1999?",
    "What is the precise weight of the apple in my backpack right now?",
    "What is the name of your favorite song when you were a child?",
    "How many grains of sand are on the specific beach I visited yesterday?",
    "What did my grandfather say to me the last time we met?",
    "What is the flavor of the cake that does not exist anywhere?",
]


class AnswerabilityDataset(Dataset):
    name = "answerability"

    def _build(self) -> List[Example]:
        out = []
        i = 0
        for q, a in _ANSWERABLE:
            out.append(Example(
                id=f"ans-y-{i:03d}", question=q, correct_answer=a,
                category="answerable", difficulty=0.2, metadata={"answerable": 1},
            ))
            i += 1
        for q in _UNANSWERABLE:
            out.append(Example(
                id=f"ans-n-{i:03d}", question=q, correct_answer=None,
                category="unanswerable", difficulty=0.8, metadata={"answerable": 0},
            ))
            i += 1
        return assign_splits(out)


def build_dataset(config: Dict[str, Any] | None = None) -> Dataset:
    return AnswerabilityDataset(config)
