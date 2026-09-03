"""Answer-correctness checking.

Correctness is one of the three quantities the project deliberately keeps
separate from FOK and confidence. It is computed from what the model actually
wrote vs. a canonical expected answer. It is used only as a *secondary* check
on whether the FOK-related signal tracks later behavior - never to define FOK.

C1 (audit): the old checker was a crude substring match that could both miss
common phrasings and falsely accept a sub-word (e.g. "Paris" inside "Parisian").
The new version normalises harder (strip leading articles), matches on
word boundaries, and folds simple spelling/morphology variants so that only a
genuine whole-answer agreement counts as correct.
"""

from __future__ import annotations

import re
from typing import Optional

# Leading articles / titles that carry no answer content.
_STRIP_PREFIX = re.compile(r"^(?:the|a|an|one)\s+")
# Keep only letters/digits for the "core" comparison.
_TOKENIFY = re.compile(r"[^a-z0-9]+")

# Replacement table (lowercase, space-separated keys) for common spelling
# variations so that correct answers differing only in abbreviation/spelling are
# not marked wrong. Keys are matched as substrings.
_ALIASES = {
    "u s a": "usa",
    "u s": "us",
    "u k": "uk",
    "g b": "uk",
    "2nd": "second",
    "3rd": "third",
    "mts": "meters",
}


def _norm(s: str) -> str:
    """Normalise to a lowercase space-separated core, dropping leading articles."""
    s = (s or "").lower().strip()
    s = _STRIP_PREFIX.sub("", s)
    s = _TOKENIFY.sub(" ", s)
    for k, v in _ALIASES.items():
        s = s.replace(k, " " + v + " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _contains_word(haystack: str, needle: str) -> bool:
    """True if ``needle`` occurs as a space-delimited chunk in ``haystack``."""
    tokens = haystack.split()
    needle_tokens = needle.split()
    if not needle_tokens:
        return False
    k = len(needle_tokens)
    for i in range(len(tokens) - k + 1):
        if tokens[i : i + k] == needle_tokens:
            return True
    return False


def check_answer(generated: Optional[str], expected: Optional[str]) -> Optional[bool]:
    """Return True/False if the generated answer matches the expected one, or
    None when there is no expected answer (unanswerable / invented questions).

    The generated text may embed the answer inside a sentence (e.g. "The
    capital is Paris."); we therefore look for the expected answer (normalised)
    as a *word-boundary* sequence anywhere in the generated output. Word
    boundaries avoid false matches like "Paris" inside "Parisian". A few simple
    alias/spelling variants are folded in.
    """
    if not expected:
        return None
    gen = _norm(generated)
    exp = _norm(expected)
    if not gen or not exp:
        return None
    # Accept the (normalised) expected answer appearing as a word-boundary
    # sequence anywhere in the generated text, or an exact equality (covers the
    # reverse case where the whole generated text is exactly the answer).
    if _contains_word(gen, exp) or gen == exp:
        return True
    return False
