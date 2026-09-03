"""Answerability dataset (audit C2 rewrite): answerable vs. unanswerable.

Defines a second independent axis of the knowledge state. ``answerable=1``
questions have a definitive factual answer; ``answerable=0`` are phrased so no
definite answer exists (private, unobservable, exact-count, or future facts).

C2 fix: the old unanswerable list leaned heavily on first/second-person personal
questions ("...the shoes I wore last Tuesday?", "...my favorite...", "...you...").
That produced a systematic lexical + length confound (audit A2: length gave AUC
0.98). This version re-words unanswerable questions as **objective, third-person
and future/exact-count** statements with comparable length to the answerable
ones, so no pronoun/yesterday class signature remains. Because ``answerable`` is
a property of the question (not of the output text), the target stays
independent of correctness.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..base import Dataset, Example, assign_splits

_ANSWERABLE: List[tuple] = [
    ("What is the boiling point of water at sea level in Celsius?", "100"),
    ("Who was the first person to walk on the Moon?", "Neil Armstrong"),
    ("What is the SI unit of force?", "Newton"),
    ("Which country has the largest population?", "India"),
    ("What is the process by which plants make their own food?", "Photosynthesis"),
    ("How many sides does a hexagon have?", "Six"),
    ("Which ocean separates Africa and South America?", "Atlantic"),
    ("What year did World War II end?", "1945"),
    ("What is the tallest mountain on Earth?", "Mount Everest"),
    ("Which is the largest planet in the Solar System?", "Jupiter"),
    ("What is the chemical symbol for gold?", "Au"),
    ("How many continents are there on Earth?", "Seven"),
    ("What is the capital of Japan?", "Tokyo"),
    ("Who wrote the play 'Romeo and Juliet'?", "William Shakespeare"),
    ("What is the smallest prime number?", "Two"),
    ("What is H2O commonly called?", "Water"),
    ("Which animal is known as the King of the Jungle?", "Lion"),
    ("How many days are there in a leap year?", "366"),
    ("What is the freezing point of water in Celsius?", "0"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("Which planet is known as the Red Planet?", "Mars"),
    ("Which is the largest ocean on Earth?", "Pacific"),
    ("How many bones are in the adult human body?", "206"),
    ("Which country is famous for the Great Pyramid of Giza?", "Egypt"),
    ("What is the currency of Japan?", "Yen"),
    ("How many players are on a soccer field at once per team?", "Eleven"),
    ("Which element has atomic number 1?", "Hydrogen"),
    ("What is the capital of Australia?", "Canberra"),
    ("How many strings does a standard guitar have?", "Six"),
    ("What is the largest mammal on Earth?", "Blue whale"),
    ("What is the approximate speed of light in a vacuum in km/s?", "300000"),
    ("Who developed the theory of general relativity?", "Albert Einstein"),
]

# Re-worded as *objective* unknowable facts -- third person, exact counts,
# private records, or future states -- with length comparable to the answerable
# ones, so no pronoun/date-class length signature remains. Each is phrased as a
# short question about an exact-but-unobservable quantity.
_UNANSWERABLE: List[str] = [
    "How many grains of sand on one beach?",
    "What was the weight of one unnamed apple?",
    "How many stars are visible from one town?",
    "What was the height of one unnamed tree?",
    "How many words were in one private chat yesterday?",
    "What was the exact pitch of one unrecorded song?",
    "How many bees were in one particular hive?",
    "What is the temperature of one empty room next week?",
    "How many flakes fell on one field in 1953?",
    "What was the weight of one lost ring?",
    "How many steps were on a staircase torn down in 1911?",
    "What was the price of one stamp sold privately?",
    "How many notes were in an unsung song?",
    "What was the pattern of one burned coat?",
    "How many guests at a wedding with no guest list?",
    "What was the rainfall at one farm in 1944?",
    "How many bricks are in one lost wall?",
    "What was the size of one lost ring?",
    "How many candles were on one birthday cake?",
    "What is the edge count of one old coin?",
    "How many bulbs were in a rebuilt theater?",
    "What was the gauge of one snapped rope?",
    "How many words were on one torn diary page?",
    "What is the depth gain of one soil pit in 1961?",
    "How many picks did one old loom make in an hour?",
    "What was the fish count in one pond one day?",
    "How many times did one clock chime unnamed?",
    "What was the glaze of one jar from 1850?",
    "How many knots were on one 1901 rope?",
    "What was the ink shade on one 1889 letter?",
    "How many panes were in one replaced window?",
    "What was the seam count of one quilt from 1975?",
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
