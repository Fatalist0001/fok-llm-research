"""Built-in English datasets for the FOK project.

Each builder returns a list of :class:`Example` with splits already assigned.
The module registry maps dataset names to builders so ``get_dataset`` can be
used uniformly by the rest of the pipeline.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from ..base import Dataset


def _registry() -> Dict[str, Callable[[Dict[str, Any]], Dataset]]:
    from . import answerability as _ans
    from . import info_variant as _info
    from . import synthetic_knowledge as _synth
    from . import trivia as _trivia

    return {
        "fok_trivia": _trivia.build_dataset,
        "synthetic_knowledge": _synth.build_dataset,
        "info_variant": _info.build_dataset,
        "answerability": _ans.build_dataset,
    }


def available() -> list:
    return sorted(_registry().keys())


def get_dataset(name: str, config: Dict[str, Any] | None = None) -> Dataset:
    reg = _registry()
    if name not in reg:
        raise KeyError(
            f"Unknown dataset '{name}'. Available: {available()}"
        )
    return reg[name](config).build()
