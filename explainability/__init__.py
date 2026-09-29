"""Explainability module: SHAP attributions and model card generation."""

from __future__ import annotations

from healthrisk_ai.explainability.model import (
    ExplainabilityModule,
    ModelCardGenerator,
    SHAPExplainer,
)

__all__ = [
    "ExplainabilityModule",
    "ModelCardGenerator",
    "SHAPExplainer",
]
