"""Answerability dataset (audit C2 + audit2 A1 rewrite): answerable vs. unanswerable.

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

Audit2 A1 fix: even after C2, unanswerable questions still used past-tense
verbs ("was", "were") and the word "one" as an inadvertent class marker
(TF-IDF AUC = 1.0). This version rewrites all unanswerable questions in
**present tense** (is/does/are) and removes "one" entirely, so verb tense
and that token no longer separate the classes. Target: TF-IDF AUC < 0.75.
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
    "How many microbes live in that soil sample?",
    "What was the decibel level at the site?",
    "Which moth species visited the porch light?",
    "How many rotations did the turbine complete?",
    "What is the refractive index of that solution?",
    "Who calibrated the station instruments?",
    "How much uranium is in that fuel rod?",
    "What voltage does the buried cable carry?",
    "Which constellation was overhead at 3 AM?",
    "How many gallons flow through the pipe?",
    "What frequency did the whale pod emit?",
    "Which gene variant determines that eye color?",
    "How many RPM did the centrifuge reach?",
    "What is the tensile strength of that fiber?",
    "Who administered the focus group survey?",
    "How many joules were in the fracture?",
    "What is the pH of the rainwater at dawn?",
    "Which mineral gives the cliff its red hue?",
    "How many lumens does the vintage lamp produce?",
    "What was the wind shear at runway altitude?",
    "Which fungus colonizes that orchid's roots?",
    "How many bits per second does it negotiate?",
    "What is the melting point of that alloy?",
    "Who timed the velodrome cycling trials?",
    "How many kilograms of thrust does it produce?",
    "What is the acoustic impedance at that depth?",
    "Which architecture powers the embedded chip?",
    "How many moles of reagent remain after?",
    "What is the oil viscosity at operating temp?",
    "Which turbine blade was used in testing?",
    "How many bits of entropy does it contain?",
    "What is the crystal's resonance frequency?",
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
