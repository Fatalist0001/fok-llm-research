"""Synthetic "knowable vs unknowable" dataset (audit C2 rewrite).

Purpose (unchanged): scale up the know/not-know contrast with guaranteed ground
truth. ``knowable=1`` questions are about real facts the model is known to have
in training data; ``knowable=0`` asks about invented-but-plausible entities that
no model can know.

C2 fix (the point of this rewrite): the old dataset used *different* template
sets for known (5 short templates) and unknown (6 long, "According to the 1987
..." templates), so the two classes differed by template and sentence length;
TF-IDF and length reached AUC 1.0 with no hidden-state signal (audit A2).

This version draws BOTH classes from the SAME small template set, where the only
difference is the entity filling the slot: a real country for knowable, an
invented-but-*normal-looking* country for unknowable. Invented country names use
ordinary phonotactics and comparable length, and both classes pair a country with
the same five attribute questions, so template/lexis/length should no longer
separate the classes. The cross-validated TF-IDF/length baseline
(``simple_baselines``) is the honest check.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from ..base import Dataset, Example, assign_splits

# --- Attribute choices shared by BOTH classes. Order matters for reproducibility
# --- of the sampled distribution.
_ATTRS = ["capital", "longest river", "highest mountain", "currency", "main export"]

# --- Real countries: nation -> (capital, longest river, highest mountain, currency, main export)
_REAL_PLACES: Dict[str, Tuple[str, str, str, str, str]] = {
    "Portugal": ("Lisbon", "the Tagus", "Mount Pico", "the euro", "wine"),
    "Chile": ("Santiago", "the Loa", "Ojos del Salado", "the peso", "copper"),
    "Egypt": ("Cairo", "the Nile", "Mount Catherine", "the pound", "cotton"),
    "Norway": ("Oslo", "the Glomma", "Galdhopiggen", "the krone", "salmon"),
    "Kenya": ("Nairobi", "the Tana", "Mount Kenya", "the shilling", "tea"),
    "Argentina": ("Buenos Aires", "the Parana", "Aconcagua", "the peso", "beef"),
    "Canada": ("Ottawa", "the Mackenzie", "Mount Logan", "the dollar", "maple syrup"),
    "Japan": ("Tokyo", "the Shinano", "Mount Fuji", "the yen", "electronics"),
    "India": ("New Delhi", "the Ganges", "Kangchenjunga", "the rupee", "textiles"),
    "Brazil": ("Brasilia", "the Amazon", "Pico da Neblina", "the real", "coffee"),
    "Australia": ("Canberra", "the Murray", "Mount Kosciuszko", "the dollar", "iron ore"),
    "Nigeria": ("Abuja", "the Niger", "Chappal Waddi", "the naira", "oil"),
    "Turkey": ("Ankara", "the Kizilirmak", "Mount Ararat", "the lira", "textiles"),
    "New Zealand": ("Wellington", "the Waikato", "Aoraki", "the dollar", "dairy"),
}

# --- Invented but normal-looking countries, with the same attribute slots.
_INV_PLACES: Dict[str, Tuple[str, str, str, str, str]] = {
    "Barrow": ("Elsworth", "the Tearng", "Mount Falver", "the mark", "wheat"),
    "Marlton": ("Heston", "the Brennd", "Mount Askel", "the dur", "tin"),
    "Duval": ("Merrow", "the Vallen", "Mount Ostry", "the calm", "hemp"),
    "Orvaine": ("Tulis", "the Calder", "Mount Prenn", "the nove", "salt"),
    "Selwick": ("Armonde", "the Thell", "Mount Gadden", "the quell", "wool"),
    "Tarmeath": ("Vexley", "the Ombre", "Mount Dirren", "the rote", "flax"),
    "Bramley": ("Coreham", "the Ellsw", "Mount Valant", "the fenn", "barley"),
    "Osmark": ("Fered", "the Ullan", "Mount Grenn", "the vane", "ore"),
    "Yewdon": ("Talworth", "the Smoor", "Mount Pellon", "the line", "linen"),
    "Haverly": ("Goston", "the Narvan", "Mount Culey", "the prand", "pottery"),
    "Pellmoor": ("Iver", "the Quelln", "Mount Dorvan", "the pall", "timber"),
    "Rookvale": ("Denworth", "the Falmar", "Mount Sherren", "the kern", "fish"),
}


def _question(place: str, attr: str, values: Tuple[str, ...]) -> (str, str):
    """Return (question, expected_answer) for a given attribute of a place."""
    table = {
        "capital": 0, "longest river": 1, "highest mountain": 2,
        "currency": 3, "main export": 4,
    }
    ans = values[table[attr]]
    if attr == "capital":
        return f"What is the capital of {place}?", ans
    if attr == "longest river":
        return f"What is the longest river in {place}?", ans
    if attr == "highest mountain":
        return f"What is the highest mountain in {place}?", ans
    if attr == "currency":
        return f"What is the currency of {place}?", ans
    return f"What is the main export of {place}?", ans


def _balanced_questions(
    n_per_class: int, rng: random.Random,
) -> (List[Tuple[str, str]], List[Tuple[str, str]]):
    """Generate `n_per_class` (question, answer) for each class, using the SAME
    template distribution so template/lexis/length cannot separate classes."""
    known: List[Tuple[str, str]] = []
    unknown: List[Tuple[str, str]] = []
    while len(known) < n_per_class:
        place = rng.choice(list(_REAL_PLACES))
        attr = rng.choice(_ATTRS)
        q, a = _question(place, attr, _REAL_PLACES[place])
        if a:
            known.append((q, a))
    while len(unknown) < n_per_class:
        place = rng.choice(list(_INV_PLACES))
        attr = rng.choice(_ATTRS)
        q, a = _question(place, attr, _INV_PLACES[place])
        unknown.append((q, a))
    return known, unknown


class SyntheticKnowledgeDataset(Dataset):
    name = "synthetic_knowledge"

    def _build(self) -> List[Example]:
        cfg = self.config or {}
        n_per_class = int(cfg.get("n_per_class", 200))
        seed = int(cfg.get("seed", 7))
        rng = random.Random(seed)

        known, unknown = _balanced_questions(n_per_class, rng)

        out = []
        i = 0
        for q, a in known:
            out.append(Example(
                id=f"synth-k-{i:05d}", question=q, correct_answer=a,
                category="knowable", difficulty=0.0, metadata={"knowable": 1},
            ))
            i += 1
        # Unknown/invented questions have no ground-truth answer (as before), so
        # ``correct`` stays NaN on those rows (a known audit caveat).
        for q, _a in unknown:
            out.append(Example(
                id=f"synth-u-{i:05d}", question=q, correct_answer=None,
                category="unknowable", difficulty=1.0, metadata={"knowable": 0},
            ))
            i += 1
        return assign_splits(out)


def build_dataset(config: Dict[str, Any] | None = None) -> Dataset:
    return SyntheticKnowledgeDataset(config)
