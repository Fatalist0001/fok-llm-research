"""Target selection and value extraction for the probe experiments."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


# Columns that are always present but never treated as a probe target.
_RESERVED = {
    "id", "split", "question", "category", "duration",
    "correct_answer", "generated", "correct",
    "avg_logprob", "seq_logprob", "first_entropy", "mean_entropy", "top1_prob",
}


def candidate_targets(rows: List[Dict]) -> List[str]:
    """Return metadata columns that look like binary input-condition labels.

    These are the ``knowable`` / ``info_relevant`` / ``answerable`` style
    targets that datasets attach in ``Example.metadata``. A column qualifies if
    it is present in every row, contains exactly the integers {0,1}, and is not
    a reserved/probe/confidence column.
    """
    if not rows:
        return []
    keys = set(rows[0].keys())
    cands = []
    for k in sorted(keys):
        if k in _RESERVED:
            continue
        vals = []
        ok = True
        for r in rows:
            v = r.get(k)
            if isinstance(v, str):
                v = v.strip()
                if v == "":
                    v = None
            if v is None or v in ("",):
                ok = False
                break
            try:
                f = float(v)
            except (TypeError, ValueError):
                ok = False
                break
            if f not in (0.0, 1.0):
                ok = False
                break
            vals.append(int(f))
        if ok and len(set(vals)) == 2:
            cands.append(k)
    return cands


def target_values(rows: List[Dict], target: str) -> np.ndarray:
    """Extract the 0/1 label vector for a target column, or None if missing."""
    if target not in candidate_targets(rows):
        return None
    out = []
    for r in rows:
        v = r.get(target)
        out.append(int(float(str(v).strip())))
    return np.asarray(out, dtype=int)
