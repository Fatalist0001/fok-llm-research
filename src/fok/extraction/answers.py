"""Lightweight answer-correctness checking.

Correctness is one of the three quantities the project deliberately keeps
separate from FOK and confidence. It is computed from what the model actually
wrote vs. a canonical expected answer. It is used only as a *secondary* check
on whether the FOK-related signal tracks later behavior - never to define FOK.
"""

from __future__ import annotations

import re
from typing import Optional


def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def check_answer(generated: Optional[str], expected: Optional[str]) -> Optional[bool]:
    """Return True/False if the generated answer matches the expected one, or
    None when there is no expected answer (unanswerable / invented questions).

    Matching is forgiving: we accept the expected answer appearing as a
    (normalized) substring of the generated text, because free-form generations
    often embed the answer inside a full sentence (e.g. "The capital is Paris.").
    """
    if not expected:
        return None
    gen = _norm(generated)
    exp = _norm(expected)
    if not gen or not exp:
        return None
    if exp in gen or gen in exp:
        return True
    # account for common holiday roundings / alternative phrasings is out of
    # scope; simple exact-or-substring match is the baseline.
    return False
