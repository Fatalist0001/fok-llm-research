"""Informative-variant dataset: the same question under four conditions.

The core FOK-mechanism experiment. A base question (about an obscure-but-real
fact the model probably does NOT know) is presented in four ways:

    * base       - the bare question            (no added info)
    * relevant   - question + a clue that gives the answer  -> becomes knowable
    * irrelevant - question + a neutral fact that does not help
    * misleading - question + a plausible-but-wrong clue

The hypothesis (a "mechanistic" FOK test) is that the model's internal state
right after reading the *relevant* variant should differ from the others in a
way that reflects "now I can answer this". We probe this with the binary
target ``info_relevant`` = 1 for the relevant variant and 0 for the rest.

Crucially, the ``info_relevant`` target is NOT: is the answer correct? It is:
was the relevant information present in the input? This keeps FOK distinct from
correctness, as required. The misleading-variant target is deliberately not
grouped with relevant, so the probe cannot just map "has text" -> 1.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..base import Dataset, Example, assign_splits

# base question -> (correct answer, relevant clue, misleading clue, irrelevant clue)
_FACTS: List[Dict[str, str]] = [
    {
        "q": "Which river is the longest in South America?",
        "ans": "The Amazon",
        "rel": "The Amazon River flows across much of northern South America.",
        "mis": "The Parana River flows across much of northern South America.",
        "irr": "South America is mostly in the Southern Hemisphere.",
    },
    {
        "q": "In which country is the ancient city of Petra located?",
        "ans": "Jordan",
        "rel": "Petra is the rock-carved ancient capital located in Jordan.",
        "mis": "Petra is the rock-carved ancient capital located in Egypt.",
        "irr": "Petra was carved from rose-colored sandstone.",
    },
    {
        "q": "What is the hard outer layer of a mammal's tooth called?",
        "ans": "Enamel",
        "rel": "Tooth enamel is the hard mineralized outer layer of teeth.",
        "mis": "Dentin is the hard mineralized outer layer of teeth.",
        "irr": "Adult humans normally have 32 teeth.",
    },
    {
        "q": "Which composer wrote the opera 'The Barber of Seville'?",
        "ans": "Rossini",
        "rel": "Gioachino Rossini composed 'The Barber of Seville'.",
        "mis": "Giuseppe Verdi composed 'The Barber of Seville'.",
        "irr": "'The Barber of Seville' premiered in Rome.",
    },
    {
        "q": "What gas do plants primarily take in during photosynthesis?",
        "ans": "Carbon dioxide",
        "rel": "Plants absorb carbon dioxide during photosynthesis.",
        "mis": "Plants absorb oxygen during photosynthesis.",
        "irr": "Photosynthesis produces glucose and oxygen.",
    },
    {
        "q": "What is the main ingredient in traditional hummus?",
        "ans": "Chickpeas",
        "rel": "Hummus is made primarily from mashed chickpeas.",
        "mis": "Hummus is made primarily from mashed lentils.",
        "irr": "Hummus is often served with olive oil.",
    },
    {
        "q": "Which metal is the best conductor of electricity?",
        "ans": "Silver",
        "rel": "Silver is the best electrical conductor among metals.",
        "mis": "Copper is the best electrical conductor among metals.",
        "irr": "Conductors allow electric current to flow.",
    },
    {
        "q": "In which sport would you use a shuttlecock?",
        "ans": "Badminton",
        "rel": "Badminton is played with a shuttlecock.",
        "mis": "Tennis is played with a shuttlecock.",
        "irr": "The shuttlecock has feathers or plastic skirt.",
    },
]


class InfoVariantDataset(Dataset):
    name = "info_variant"

    def _build(self) -> List[Example]:
        examples = []
        base_id = 0
        for f in _FACTS:
            variants = [
                ("base", f["q"], False),
                ("relevant", f"{f['q']}\nHint: {f['rel']}", True),
                ("irrelevant", f"{f['q']}\nNote: {f['irr']}", False),
                ("misleading", f"{f['q']}\nHint: {f['mis']}", False),
            ]
            for variant, question, is_rel in variants:
                examples.append(
                    Example(
                        id=f"info-{base_id:03d}-{variant}",
                        question=question,
                        correct_answer=f["ans"],
                        category=f"info.{variant}",
                        difficulty=0.5,
                        metadata={
                            "variant": variant,
                            "base_id": f"{base_id:03d}",
                            "info_relevant": int(is_rel),
                        },
                    )
                )
            base_id += 1
        return assign_splits(examples, split_key=lambda e: e.metadata["base_id"])


def build_dataset(config: Dict[str, Any] | None = None) -> Dataset:
    return InfoVariantDataset(config)
