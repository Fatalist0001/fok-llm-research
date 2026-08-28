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
    {
        "q": "What is the capital of New Zealand?",
        "ans": "Wellington",
        "rel": "Wellington is the capital city of New Zealand.",
        "mis": "Auckland is the capital city of New Zealand.",
        "irr": "New Zealand is an island country in the Pacific.",
    },
    {
        "q": "Which metal is liquid at standard room temperature?",
        "ans": "Mercury",
        "rel": "Mercury is the metal that is liquid at room temperature.",
        "mis": "Lead is the metal that is liquid at room temperature.",
        "irr": "Metals usually conduct electricity well.",
    },
    {
        "q": "What is the capital city of Canada?",
        "ans": "Ottawa",
        "rel": "Ottawa is the capital city of Canada.",
        "mis": "Toronto is the capital city of Canada.",
        "irr": "Canada has ten provinces.",
    },
    {
        "q": "Which organ pumps blood around the body?",
        "ans": "The heart",
        "rel": "The heart is the organ that pumps blood.",
        "mis": "The liver is the organ that pumps blood.",
        "irr": "Blood carries oxygen to tissues.",
    },
    {
        "q": "Who is the author of 'The Great Gatsby'?",
        "ans": "F. Scott Fitzgerald",
        "rel": "F. Scott Fitzgerald wrote 'The Great Gatsby'.",
        "mis": "Ernest Hemingway wrote 'The Great Gatsby'.",
        "irr": "The novel is set in the 1920s.",
    },
    {
        "q": "Which is the largest hot desert in the world?",
        "ans": "The Sahara",
        "rel": "The Sahara is the largest hot desert.",
        "mis": "The Arabian Desert is the largest hot desert.",
        "irr": "Deserts receive very little rainfall.",
    },
    {
        "q": "What is the capital of Spain?",
        "ans": "Madrid",
        "rel": "Madrid is the capital of Spain.",
        "mis": "Barcelona is the capital of Spain.",
        "irr": "Spain is in southwestern Europe.",
    },
    {
        "q": "Which gas makes up most of Earth's atmosphere?",
        "ans": "Nitrogen",
        "rel": "Nitrogen makes up about 78 percent of the atmosphere.",
        "mis": "Oxygen makes up the majority of the atmosphere.",
        "irr": "The atmosphere has several distinct layers.",
    },
    {
        "q": "Who invented the telephone?",
        "ans": "Alexander Graham Bell",
        "rel": "Alexander Graham Bell invented the telephone.",
        "mis": "Thomas Edison invented the telephone.",
        "irr": "The telephone transmits sound over distance.",
    },
    {
        "q": "What is the capital of Italy?",
        "ans": "Rome",
        "rel": "Rome is the capital of Italy.",
        "mis": "Milan is the capital of Italy.",
        "irr": "Italy is shaped like a boot.",
    },
    {
        "q": "Which planet is closest to the Sun?",
        "ans": "Mercury",
        "rel": "Mercury is the planet closest to the Sun.",
        "mis": "Venus is the planet closest to the Sun.",
        "irr": "Planets in the Solar System orbit the Sun.",
    },
    {
        "q": "What is the largest organ of the human body?",
        "ans": "The skin",
        "rel": "The skin is the largest organ of the human body.",
        "mis": "The liver is the largest organ of the human body.",
        "irr": "Organs perform specific functions in the body.",
    },
    {
        "q": "Who composed the 'Moonlight Sonata'?",
        "ans": "Beethoven",
        "rel": "Ludwig van Beethoven composed the Moonlight Sonata.",
        "mis": "Wolfgang Amadeus Mozart composed the Moonlight Sonata.",
        "irr": "It is a famous piano work.",
    },
    {
        "q": "What is the chemical formula for table salt?",
        "ans": "NaCl",
        "rel": "Table salt is sodium chloride, whose formula is NaCl.",
        "mis": "Table salt is potassium chloride, whose formula is KCl.",
        "irr": "Table salt is commonly used in cooking.",
    },
    {
        "q": "Which country has the longest coastline in the world?",
        "ans": "Canada",
        "rel": "Canada has the longest coastline in the world.",
        "mis": "Russia has the longest coastline in the world.",
        "irr": "Coastlines border oceans or seas.",
    },
    {
        "q": "What is the capital of Turkey?",
        "ans": "Ankara",
        "rel": "Ankara is the capital of Turkey.",
        "mis": "Istanbul is the capital of Turkey.",
        "irr": "Turkey straddles Europe and Asia.",
    },
    {
        "q": "Who was the first woman to win a Nobel Prize?",
        "ans": "Marie Curie",
        "rel": "Marie Curie was the first woman to win a Nobel Prize.",
        "mis": "Rosalind Franklin was the first woman to win a Nobel Prize.",
        "irr": "Nobel Prizes honor major scientific achievements.",
    },
    {
        "q": "Which is the fastest land animal?",
        "ans": "The cheetah",
        "rel": "The cheetah is the fastest land animal.",
        "mis": "The leopard is the fastest land animal.",
        "irr": "Many animals are adapted for speed.",
    },
    {
        "q": "What is the capital of Brazil?",
        "ans": "Brasília",
        "rel": "Brasília is the capital of Brazil.",
        "mis": "Rio de Janeiro is the capital of Brazil.",
        "irr": "Brazil is a large country in South America.",
    },
    {
        "q": "Which element has the atomic number 6?",
        "ans": "Carbon",
        "rel": "Carbon is the element with atomic number 6.",
        "mis": "Nitrogen is the element with atomic number 6.",
        "irr": "Elements are arranged in the periodic table.",
    },
    {
        "q": "Who painted 'The Starry Night'?",
        "ans": "Vincent van Gogh",
        "rel": "Vincent van Gogh painted 'The Starry Night'.",
        "mis": "Claude Monet painted 'The Starry Night'.",
        "irr": "The painting shows a swirling night sky.",
    },
    {
        "q": "What is the capital of South Korea?",
        "ans": "Seoul",
        "rel": "Seoul is the capital of South Korea.",
        "mis": "Busan is the capital of South Korea.",
        "irr": "South Korea is on a peninsula in East Asia.",
    },
    {
        "q": "Which enzyme breaks down starch in saliva?",
        "ans": "Amylase",
        "rel": "Amylase is the enzyme that breaks down starch in saliva.",
        "mis": "Pepsin is the enzyme that breaks down starch in saliva.",
        "irr": "Saliva begins the digestion of food.",
    },
    {
        "q": "Who authored 'Pride and Prejudice'?",
        "ans": "Jane Austen",
        "rel": "Jane Austen authored 'Pride and Prejudice'.",
        "mis": "Charlotte Bronte authored 'Pride and Prejudice'.",
        "irr": "It is a classic English novel.",
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
