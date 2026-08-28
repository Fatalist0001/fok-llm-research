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

_UNANSWERABLE: List[str] = [
    "What is the exact number of stars visible from your kitchen window tonight?",
    "What color is the dragon you dreamed about last Tuesday?",
    "How many hours of sleep did the author of this question get in 1999?",
    "What is the precise weight of the apple in my backpack right now?",
    "What is the name of your favorite song when you were a child?",
    "How many grains of sand are on the specific beach I visited yesterday?",
    "What did my grandfather say to me the last time we met?",
    "What is the flavor of the cake that does not exist anywhere?",
    "What color were the shoes I wore last Tuesday?",
    "How many times did I blink while answering this question?",
    "What is my favorite childhood memory?",
    "What exactly did I eat for breakfast one year ago today?",
    "What is the name of the neighbor's imaginary friend?",
    "How many letters are in the book I have not finished reading yet?",
    "What will the exact temperature be in my bedroom at 3:15 am tomorrow?",
    "What song was the first reader of this question listening to?",
    "What is the street address of the house I visited in a dream last night?",
    "What color is the coat worn by the person behind me right now?",
    "What was the last thing the last person to use this keyboard said?",
    "How many pancakes did I eat on my third birthday?",
    "What is the plot of the movie I will watch next Friday?",
    "What was my favorite ice cream flavor in 2010?",
    "What did my grandmother whisper to me when I was two days old?",
    "How many words did I speak yesterday?",
    "What is the name of the song playing in my head right now?",
    "What time will I wake up next Tuesday?",
    "What is the middle name of my mail carrier's oldest sibling?",
    "What color was the first car I saw while traveling in 2008?",
    "How many years old is the oldest leaf on the neighbor's tree?",
    "What is the secret password I intended to set but never did?",
    "How much money is currently in the wallet I left at home?",
    "What is the title of the biography no one has written about me?",
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
