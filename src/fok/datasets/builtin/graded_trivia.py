"""Difficulty-graded trivia dataset: real facts at three difficulty tiers plus
invented (unknowable) questions.

Purpose (contrast vs. ``fok_trivia`` / ``synthetic_knowledge``): those datasets
contrast very easy, well-known facts with fabricated entities, so the model is
confidently right on one side and confidently hallucinates on the other -- both
confidence measures sit at roughly the ceiling and are hard to separate from a
hidden-state signal. Here we deliberately add *hard-but-real* facts (niche
geography, obscure science, etc.) where Qwen's generation confidence is
moderate and not pinned at the ceiling, so that the ordinary-confidence
baseline is informative instead of saturated.

The target ``knowable`` still means "this fact really exists in the model's
training data by construction": 1 for real facts (any tier), 0 for invented
entities. The point of the experiment is whether a hidden-state probe can
separate knowable/un-knowable strongly while confidence stays below its
ceiling, i.e. whether the internal signal carries information beyond ordinary
confidence.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

from ..base import Dataset, Example, assign_splits

# Each entry: (question, correct_answer).
_EASY: List[tuple] = [
    ("What is the capital of France?", "Paris"),
    ("Which planet is known as the Red Planet?", "Mars"),
    ("What is the chemical formula for water?", "H2O"),
    ("Who wrote 'Romeo and Juliet'?", "William Shakespeare"),
    ("What is the largest ocean on Earth?", "Pacific"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("Which metal is liquid at room temperature?", "Mercury"),
    ("What is the currency of Japan?", "Yen"),
    ("How many continents are there?", "Seven"),
    ("What is the tallest mountain on Earth?", "Mount Everest"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("What is the SI unit of force?", "Newton"),
    ("Which country has the largest population?", "India"),
    ("What is the freezing point of water in Celsius?", "0"),
    ("Who developed the theory of general relativity?", "Albert Einstein"),
    ("What is the largest mammal on Earth?", "Blue whale"),
    ("How many sides does a hexagon have?", "Six"),
    ("What gas do plants absorb during photosynthesis?", "Carbon dioxide"),
    ("What is the capital of Japan?", "Tokyo"),
    ("Which planet has the most prominent rings?", "Saturn"),
    ("What is the chemical symbol for gold?", "Au"),
    ("Who is known as the father of computers?", "Charles Babbage"),
    ("What is the currency of the United Kingdom?", "Pound"),
    ("What year did World War II end?", "1945"),
]

_MEDIUM: List[tuple] = [
    ("Which composer wrote the opera 'The Barber of Seville'?", "Rossini"),
    ("What is the capital of New Zealand?", "Wellington"),
    ("In which sport is a shuttlecock used?", "Badminton"),
    ("What is the main ingredient of hummus?", "Chickpeas"),
    ("Who invented the telephone?", "Alexander Graham Bell"),
    ("What is the chemical element with atomic number 6?", "Carbon"),
    ("Which country has the longest coastline?", "Canada"),
    ("What is the capital of Turkey?", "Ankara"),
    ("Who was the first woman to win a Nobel Prize?", "Marie Curie"),
    ("What is the fastest land animal?", "Cheetah"),
    ("What enzyme breaks down starch in saliva?", "Amylase"),
    ("Which is the largest hot desert in the world?", "Sahara"),
    ("Who authored 'Pride and Prejudice'?", "Jane Austen"),
    ("What is the capital of Brazil?", "Brasilia"),
    ("Which ocean separates Africa and South America?", "Atlantic"),
    ("Who painted 'The Starry Night'?", "Vincent van Gogh"),
    ("What is the capital of South Korea?", "Seoul"),
    ("Which element has atomic number 8?", "Oxygen"),
    ("What is the capital of Australia?", "Canberra"),
    ("Which element has atomic number 1?", "Hydrogen"),
    ("What is the largest planet in the Solar System?", "Jupiter"),
    ("What is H2O commonly called?", "Water"),
    ("Which animal is known as the King of the Jungle?", "Lion"),
    ("How many days are in a leap year?", "366"),
]

_HARD: List[tuple] = [
    ("What is the capital of Eswatini?", "Mbabane"),
    ("Which element has the atomic number 49?", "Indium"),
    ("What is the longest river that flows entirely within Germany?", "Weser"),
    ("What is the capital of the island nation of Vanuatu?", "Port Vila"),
    ("Which chemical element has the symbol W?", "Tungsten"),
    ("What is the oldest continuously inhabited city in Europe?", "Plovdiv"),
    ("Who was the first woman to win the Fields Medal?", "Maryam Mirzakhani"),
    ("What is the capital of the Federated States of Micronesia?", "Palikir"),
    ("Which element has the highest melting point?", "Tungsten"),
    ("What is the only mammal that can truly fly?", "Bat"),
    ("What is the capital of Bhutan?", "Thimphu"),
    ("What is the deepest point of the Pacific Ocean?", "Mariana Trench"),
    ("Who discovered the element radium?", "Marie Curie"),
    ("What is the capital of the Central African Republic?", "Bangui"),
    ("Which is the smallest country in the world by area?", "Vatican City"),
    ("What is the SI unit of magnetic flux?", "Weber"),
    ("What is the capital of Lesotho?", "Maseru"),
    ("Which element has the atomic number 88?", "Radium"),
    ("What is the capital of the island of Borneo's Indonesian part?", "Samarinda"),
    ("Who composed the symphonic poem 'Tone Poems' cycle, Op. 137, catalogued as 'Má vlast'?", "Smetana"),
    ("What is the capital of the Central Asian country Kyrgyzstan?", "Bishkek"),
    ("Which metal is used to make the red pigment vermilion?", "Mercury"),
    ("What is the capital of the Pacific territory of Tokelau?", "Nukunonu"),
    ("Which element has the symbol I?", "Iodine"),
]

# --- Invented-entity questions (unknowable) --------------------------------
# Templated so that no model can have the answer, but phrased like the hard
# known items so the *wording* difficulty is comparable.
_INVENTED_PEOPLE = ["Mirza Krenov", "Calla Voss", "Thane Orme", "Yelena Quirke", "Osbert Fenn", "Drea Vasquez"]
_INVENTED_PLACES = ["Norvald", "Zaravia", "Illaria", "Torth", "Maribou", "Dravenia", "Vespath", "Sarl", "Bracken Fold", "Ostrell", "Vandermeer"]
_INVENTED_ORGS = ["Zoltir Industries", "Quiptex Ltd", "the Olmer Consortium", "the Norheim Council", "the Pennestone Institute"]
_INVENTED_THINGS = ["Quorium", "thulanium", "silveon", "hypochloran", "galleon", "the Luckstone of Thorburn", "the Amethyst Thread", "the Vantell ruin"]
_INVENTED_SCI_UNITS = ["luminance flux", "magnetograde", "quantic charge"]

_INVENTED_TEMPLATES = [
    "What is the capital of the {place}?",
    "Which element, called '{thing}', was discovered in {year}?",
    "What is the longest river of the {place}?",
    "Who composed the unpublished opera 'The {thing} of {place}' in {year}?",
    "Who was the first {profession} to win the imaginary Pennestone {prize}?",
    "Which metal, '{thing}', has the highest density ever measured?",
    "What is the only mammal endemic to the {place}?",
    "What is the capital of the {place}?",
    "What is the deepest point of the fictitious {thing} Trench?",
    "Who first isolated the invented element '{thing}' in {year}?",
    "What is the SI unit of the made-up quantity '{unit}'?",
    "Which museum houses the lost sculpture of '{thing}'?",
    "What is the oldest settlement ever found in the {place}?",
    "Who authored the apocryphal treatise 'On {thing}' in {year}?",
]


def _make_invented(rng: random.Random) -> str:
    tmpl = rng.choice(_INVENTED_TEMPLATES)
    place = rng.choice(_INVENTED_PLACES)
    thing = rng.choice(_INVENTED_THINGS)
    year = rng.randrange(1700, 1999)
    return tmpl.format(
        place=place,
        thing=thing,
        year=year,
        profession=rng.choice(["philosopher", "chemist", "cartographer", "composer"]),
        prize=rng.choice(["Medal", "Prize", "Award"]),
        unit=rng.choice(_INVENTED_SCI_UNITS),
    )


class GradedTriviaDataset(Dataset):
    name = "graded_trivia"

    def _build(self) -> List[Example]:
        cfg = self.config or {}
        n_invented = int(cfg.get("n_invented", 48))
        rng = random.Random(int(cfg.get("seed", 7)))

        out = []
        i = 0
        for tier, facts in (("easy", _EASY), ("medium", _MEDIUM), ("hard", _HARD)):
            d = {"easy": 0.2, "medium": 0.5, "hard": 0.8}[tier]
            for q, ans in facts:
                out.append(Example(
                    id=f"gtr-k-{i:04d}", question=q, correct_answer=ans,
                    category=f"tier_{tier}", difficulty=d,
                    metadata={"knowable": 1, "tier": tier},
                ))
                i += 1
        for _ in range(n_invented):
            q = _make_invented(rng)
            out.append(Example(
                id=f"gtr-u-{i:04d}", question=q, correct_answer=None,
                category="invented", difficulty=0.9,
                metadata={"knowable": 0, "tier": "invented"},
            ))
            i += 1
        return assign_splits(out)


def build_dataset(config: Dict[str, Any] | None = None) -> Dataset:
    return GradedTriviaDataset(config)
