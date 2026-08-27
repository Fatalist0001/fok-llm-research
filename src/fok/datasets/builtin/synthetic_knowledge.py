"""Procedural "synthetic knowledge" dataset.

Goal: scale up the know/not-know contrast with guaranteed ground truth, using
curated fact banks for the *knowable* questions (so the model definitely has
them) and templated *invented* questions for the *unknowable* ones.

The ``knowable`` target indicates, by construction, whether the model can have
the knowledge. It is independent of the model's actual answer, which lets us
later analyse the four cases: knowable x correct/incorrect, unknowable x
correct/incorrect (e.g. a confident hallucination on an unknowable question).
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

from ..base import Dataset, Example, assign_splits

# --- Fact banks for knowable questions (guaranteed in training data) ---------
CAPITALS = {
    "Portugal": "Lisbon", "Chile": "Santiago", "Egypt": "Cairo",
    "Canada": "Ottawa", "Kenya": "Nairobi", "Norway": "Oslo",
    "Argentina": "Buenos Aires", "New Zealand": "Wellington",
}
ELEMENTS = {
    "hydrogen": "H", "carbon": "C", "sodium": "Na", "iron": "Fe",
    "silver": "Ag", "potassium": "K", "calcium": "Ca", "nitrogen": "N",
}
MONTHS = {
    "the first month": "January", "the month after March": "April",
    "the last month": "December", "the month with 28 days": "February",
    "the sixth month": "June", "the month before July": "June",
}
PLANETS = {
    "the Red Planet": "Mars", "the planet closest to the Sun": "Mercury",
    "the planet with the Great Red Spot": "Jupiter", "the coldest planet": "Uranus",
    "the planet known as the Evening Star": "Venus",
}
SCI = {
    "the speed of sound in air at 20C in m/s": "343",
    "Avogadro's number in scientific notation": "6.022 x 10^23",
    "the number of neutrons in a carbon-12 atom": "6",
    "the unit of electrical resistance": "ohm",
}

# --- Invented-entity banks for unknowable questions ---------------------------
INVENTED_PEOPLE = ["Mirza Krenov", "Calla Voss", "Thane Orme", "Yelena Quirke"]
INVENTED_PLACES = ["the town of Bracken Fold", "the village of Ostrell", "the city of Vandermeer"]
INVENTED_ORGS = ["Zoltir Industries", "Quiptex Ltd", "the Olmer Consortium", "the Norheim Council"]
INVENTED_THINGS = ["the Vantell ruin", "the Luckstone of Thorburn", "the Amethyst Thread", "the Ghalvor bird"]

_UNKNOWN_TEMPLATES = [
    "According to the 1987 internal report of {org}, what was the code name of the {thing} project?",
    "What is the middle name of {person}, who lived in {place}?",
    "In which year did {org} first publish its {thing} manual?",
    "What was the registry number of the first {thing} built in {place}?",
    "How many chapters are in the private diary kept by {person} at {place}?",
    "What color was the front gate of the {thing} workshop run by {org}?",
]


def _make_known(rng: random.Random, bank_key: str) -> (str, str):
    bank = {
        "capitals": CAPITALS, "elements": ELEMENTS, "months": MONTHS,
        "planets": PLANETS, "sci": SCI,
    }[bank_key]
    d, ans = rng.choice(list(bank.items()))
    if bank_key == "capitals":
        q = f"{ans} is the capital of which country?"
    elif bank_key == "elements":
        q = f"What is the chemical symbol for {d}?"
    elif bank_key == "months":
        q = f"Which month is {d}?"
    elif bank_key == "planets":
        q = f"Which planet is {d}?"
    else:
        q = f"What is {d}?"
    return q, ans


def _make_unknown(rng: random.Random) -> (str, None):
    tmpl = rng.choice(_UNKNOWN_TEMPLATES)
    q = tmpl.format(
        org=rng.choice(INVENTED_ORGS),
        thing=rng.choice(INVENTED_THINGS),
        person=rng.choice(INVENTED_PEOPLE),
        place=rng.choice(INVENTED_PLACES),
    )
    return q, None


class SyntheticKnowledgeDataset(Dataset):
    name = "synthetic_knowledge"

    def _build(self) -> List[Example]:
        cfg = self.config or {}
        n_per_class = int(cfg.get("n_per_class", 300))
        seed = int(cfg.get("seed", 7))
        rng = random.Random(seed)

        out = []
        i = 0
        for _ in range(n_per_class):
            bank_key = ["capitals", "elements", "months", "planets"][rng.randrange(4)]
            q, ans = _make_known(rng, bank_key)
            out.append(Example(
                id=f"synth-k-{i:05d}", question=q, correct_answer=ans,
                category="knowable", difficulty=0.0, metadata={"knowable": 1},
            ))
            i += 1
        for _ in range(n_per_class):
            q, ans = _make_unknown(rng)
            out.append(Example(
                id=f"synth-u-{i:05d}", question=q, correct_answer=ans,
                category="unknowable", difficulty=1.0, metadata={"knowable": 0},
            ))
            i += 1
        return assign_splits(out)


def build_dataset(config: Dict[str, Any] | None = None) -> Dataset:
    return SyntheticKnowledgeDataset(config)
