"""Curated trivia-style dataset: a hand-written set of questions that the model
verifiably knows vs. verifiably does not know.

This is the simplest instantiation of the "knowledge state" (FOK-like) contrast:

    * ``knowable=1``  -> questions about common facts in the model's training data.
    * ``knowable=0``  -> questions about *invented* entities / private facts that
                         no model can have in its training data.

The split (train/val/test) is assigned deterministically from the question text
so identical questions never straddle train and test.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..base import Dataset, Example, assign_splits


def _pairs() -> List[Dict[str, Any]]:
    """(question, correct_answer, knowable) curated entries.

    ``knowable`` is 1 if the model objectively has this knowledge by construction.
    This target is deliberately *not* the same as whether the model answered
    correctly in a given run - a model may hallucinate confidently (knowable=0
    but an answer was still produced) or miss an answer it should know.
    """
    known = [
        # (question, expected answer)
        ("What is the capital of France?", "Paris"),
        ("What is the largest planet in our solar system?", "Jupiter"),
        ("Who wrote the play Romeo and Juliet?", "Shakespeare"),
        ("What is the chemical symbol for gold?", "Au"),
        ("How many continents are there on Earth?", "Seven"),
        ("What is the capital of Japan?", "Tokyo"),
        ("Which metal is liquid at room temperature?", "Mercury"),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
        ("What is the largest ocean on Earth?", "Pacific"),
        ("How many bones are in the adult human body?", "206"),
        ("What is the freezing point of water in Celsius?", "0"),
        ("Who developed the theory of general relativity?", "Einstein"),
        ("What country hosted the 2016 Summer Olympics?", "Brazil"),
        ("What is the tallest mountain on Earth?", "Everest"),
        ("Which element has the atomic number 8?", "Oxygen"),
        ("What is the smallest prime number?", "Two"),
        ("Who is known as the father of computers?", "Charles Babbage"),
        ("What is the currency of the United Kingdom?", "Pound"),
        ("Which planet has the most prominent rings?", "Saturn"),
        ("How many hours are in a day?", "24"),
    ]
    unknown = [
        # Questions constructed so that no language model can have the answer.
        # These target the interesting case where the model may still confidently
        # hallucinate (high FOK, high confidence, but objectively incorrect).
        ("According to the unpublished 1987 Zoltir manuscript, what was the main claim?", None),
        ("What is the middle name of the inventor of the Zvarkovo device in Ghent?", None),
        ("In the 2019 internal review of company Quiptex, which product line was discontinued?", None),
        ("What color were the walls of room 712 in the now-demolished Halver Hotel?", None),
        ("What was the exact birth weight of the fictional character Meera Thorne?", None),
        ("Which street was the Zorbin laboratory on before it moved?", None),
        ("The third draft of the Kren heretic saga assigns what name to the twin moons?", None),
        ("What password did the inventor of the Teollet clock use for his safe deposit box?", None),
        ("In the 1974 Korveth census, how many residents reported being left-handed?", None),
        ("What was the code name of the silent film projector project at Olmer & Co. in 1912?", None),
        ("What is the name of the mythical bird that guards the Ghalvor mountain pass?", None),
        ("Which novel mentions the town of Ingeld-on-Wrex in its appendix?", None),
        ("What was the original color of the Luckstone of Thorburn before it was painted?", None),
        ("How many steps are on the hidden staircase of the Vantell castle ruins?", None),
        ("What is the traditional festival dish of the (now isolated) Brixham Isles?", None),
        ("What was the name of the ninth member of the founding council of Norheim?", None),
        ("When exactly did the inventor of the Pisquet galvanometer file its patent?", None),
        ("What instrument did the composer Vello Rastik play in his youth?", None),
        ("What is the registry number of the last tram that ran in Broling in 1958?", None),
        ("In the parable of the Amethyst Thread, what color is the thread?", None),
    ]
    pairs = [("known", q, a, 1) for (q, a) in known] + [
        ("unknown", q, a, 0) for (q, a) in unknown
    ]
    out = []
    for i, (cat, q, a, knowable) in enumerate(pairs):
        out.append(
            {
                "question": q,
                "correct_answer": a,
                "category": cat,
                "knowable": knowable,
            }
        )
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
