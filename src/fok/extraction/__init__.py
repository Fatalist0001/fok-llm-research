"""Extraction stage: turn a dataset + model into stored research features."""

from .answers import check_answer
from .collector import collect

__all__ = ["collect", "check_answer"]
