"""Clinical NLP engine: Bio_ClinicalBERT fine-tuning for clinical complexity embeddings.

This package provides the NLP component of the HealthRisk AI ensemble:
- ClinicalComplexityEncoder: Bio_ClinicalBERT with projection to 64-dim embeddings
- ClinicalNLPModel: High-level train/predict/save/load interface
- ClinicalNoteDataset: PyTorch Dataset for clinical notes
"""

from __future__ import annotations

from healthrisk_ai.models.clinical_nlp.model import (
    ClinicalComplexityEncoder,
    ClinicalNLPModel,
    ClinicalNoteDataset,
    build_clinical_nlp_model,
)

__all__ = [
    "ClinicalComplexityEncoder",
    "ClinicalNLPModel",
    "ClinicalNoteDataset",
    "build_clinical_nlp_model",
]
