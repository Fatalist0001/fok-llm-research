"""Curated trivia-style dataset (audit C2 rewrite).

Th purpose is the simplest "knowledge state" contrast:
``knowable=1`` -> common facts the model verifiably knows;
``knowable=0`` -> facts about *invented* entities that no model can have.

C2 fix: the old dataset contrasted short, ordinary known questions against long
questions full of invented rare-token names (Zoltir, Zvarkovo, ...), so TF-IDF
and length separated the classes with AUC 1.0 (audit A2). This version is a
set of **hand-matched pairs**: each knowable question has an unknowable twin
that uses the *same sentence frame and comparable length*, differing only in a
normal-looking invented entity. The known and unknown lists are therefore
stylistically indistinguishable, which the ``simple_baselines`` check verifies.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..base import Dataset, Example, assign_splits

# Each entry: (known_question, known_answer, unknown_question, unknown_answer).
_PAIRS: List[tuple] = [
    # --- person / creator frames (matched structure + length) ---
    ("Who painted the Sistine Chapel ceiling?", "Michelangelo",
     "Who painted the Tarnley Cathedral ceiling?", "Alaric Venn"),
    ("Who wrote the novel 'Pride and Prejudice'?", "Jane Austen",
     "Who wrote the novel 'The Ashen Coast'?", "Dora Fennel"),
    ("Who composed the symphony 'Eroica'?", "Beethoven",
     "Who composed the 'Westermarck' symphony?", "Ilse North"),
    ("Who invented the telephone?", "Alexander Graham Bell",
     "Who invented the field telegraph?", "Owen Barrow"),
    ("Who devised the theory of relativity?", "Einstein",
     "Who devised the theory of resonance?", "Vera Sork"),
    ("Who painted the 'Pearl Earring'?", "Johannes Vermeer",
     "Who painted the 'Green Shawl' portrait?", "Pierre Vantel"),
    ("Who wrote the poem 'The Raven'?", "Edgar Allan Poe",
     "Who wrote the poem 'The Hollow Lark'?", "Sonia Merr"),
    # --- geography frames ---
    ("Which is the largest desert in the world?", "Antarctica",
     "Which is the largest desert on Vexland?", "the Duneveld"),
    ("What is the tallest mountain on Earth?", "Mount Everest",
     "What is the tallest peak in Pellink?", "Mount Dorvan"),
    ("Which river flows through Paris?", "the Seine",
     "Which river flows through Kelmworth?", "the Brennd"),
    ("What is the capital of Canada?", "Ottawa",
     "What is the capital of Orvaine?", "Tulis"),
    ("Which ocean is the largest?", "the Pacific",
     "Which ocean borders Osmark?", "the Merrow Sea"),
    ("How many continents are there on Earth?", "Seven",
     "How many provinces are in Maribou?", "Nine"),
    # --- science / numbers frames ---
    ("What is the chemical symbol for gold?", "Au",
     "What is the symbol for the element quorium?", "Qu"),
    ("How many sides does a hexagon have?", "Six",
     "How many angles does a septagon have?", "Seven"),
    ("What is the boiling point of water in Celsius?", "100",
     "What is the boiling point of novine in Celsius?", "87"),
    ("What gas do plants mainly absorb?", "Carbon dioxide",
     "What gas do the flora of Duval absorb?", "Nitrous oxide"),
    # --- everyday-object / institution frames ---
    ("What is the currency of Japan?", "Yen",
     "What is the currency of Selwick?", "Quell"),
    ("Which metal is liquid at room temperature?", "Mercury",
     "Which metal stays liquid below freezing?", "Halmine"),
    ("What is the fastest land animal?", "Cheetah",
     "What is the fastest bird of Tarmeath?", "the Calver"),
]


def _pairs() -> List[Dict[str, Any]]:
    out = []
    for (kq, ka, uq, _ua) in _PAIRS:
        out.append({"question": kq, "correct_answer": ka,
                    "category": "known", "knowable": 1})
        # Unknown/invented have no ground-truth answer -> correct stays NaN.
        out.append({"question": uq, "correct_answer": None,
                    "category": "unknown", "knowable": 0})
    return out


class TriviaDataset(Dataset):
    name = "fok_trivia"

    def _build(self) -> List[Example]:
        out = []
        for i, p in enumerate(_pairs()):
            out.append(
                Example(
                    id=f"trivia-{i:04d}",
                    question=p["question"],
                    correct_answer=p["correct_answer"],
                    category=p["category"],
                    difficulty=0.0 if p["knowable"] else 1.0,
                    metadata={"knowable": int(p["knowable"])},
                )
            )
        return assign_splits(out)


def build_dataset(config: Dict[str, Any] | None = None) -> Dataset:
    return TriviaDataset(config)
