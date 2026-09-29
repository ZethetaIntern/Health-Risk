# HealthRisk AI - acquisition submodules lazy import shim
#
# This shim lets the rest of the package import acquisition modules without
# pulling in heavy I/O at package import time. Real adapters are imported
# on demand inside functions.

from __future__ import annotations

from healthrisk_ai.acquisition import base, clinicaltrials, mimic_iv, openfda

__all__ = ["base", "mimic_iv", "openfda", "clinicaltrials"]
