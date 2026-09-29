"""Tabular baseline models: XGBoost and LightGBM for clinical prediction."""

from __future__ import annotations

from healthrisk_ai.models.tabular.model import (
    TabularModel,
    build_tabular_model,
)

__all__ = [
    "TabularModel",
    "build_tabular_model",
]
