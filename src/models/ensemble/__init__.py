"""Stacking ensemble: Ridge regression meta-learner for combining model outputs."""

from __future__ import annotations

from healthrisk_ai.models.ensemble.model import (
    StackingEnsemble,
    build_stacking_ensemble,
)

__all__ = [
    "StackingEnsemble",
    "build_stacking_ensemble",
]
