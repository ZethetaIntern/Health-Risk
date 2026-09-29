# HealthRisk AI - acquisition subpackage

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
if 'healthrisk_ai' not in sys.modules:
    parent = here.parent
    pkg_spec = importlib.util.spec_from_file_location(
        'healthrisk_ai', parent / 'healthrisk_ai' / '__init__.py'
    )
    pkg = importlib.util.module_from_spec(pkg_spec)
    sys.modules['healthrisk_ai'] = pkg
    pkg_spec.loader.exec_module(pkg)

_base_spec = importlib.util.spec_from_file_location(
    'healthrisk_ai.acquisition.base', here / 'base.py'
)
_base = importlib.util.module_from_spec(_base_spec)
_base_spec.loader.exec_module(_base)
sys.modules.setdefault('healthrisk_ai.acquisition.base', _base)

_ct_spec = importlib.util.spec_from_file_location(
    'healthrisk_ai.acquisition.clinicaltrials', here / 'clinicaltrials.py'
)
_ct = importlib.util.module_from_spec(_ct_spec)
_ct_spec.loader.exec_module(_ct)
sys.modules.setdefault('healthrisk_ai.acquisition.clinicaltrials', _ct)

_mimic_spec = importlib.util.spec_from_file_location(
    'healthrisk_ai.acquisition.mimic_iv', here / 'mimic_iv.py'
)
_mimic = importlib.util.module_from_spec(_mimic_spec)
_mimic_spec.loader.exec_module(_mimic)
sys.modules.setdefault('healthrisk_ai.acquisition.mimic_iv', _mimic)

_open_spec = importlib.util.spec_from_file_location(
    'healthrisk_ai.acquisition.openfda', here / 'openfda.py'
)
_open = importlib.util.module_from_spec(_open_spec)
_open_spec.loader.exec_module(_open)
sys.modules.setdefault('healthrisk_ai.acquisition.openfda', _open)

_acq_name = 'healthrisk_ai.acquisition'
_acq = sys.modules.get(_acq_name)
if _acq is None:
    _acq_spec = importlib.util.spec_from_file_location(
        _acq_name, here / '__init__.py', submodule_search_locations=[str(here)]
    )
    _acq = importlib.util.module_from_spec(_acq_spec)
    sys.modules[_acq_name] = _acq
_acq.__path__ = [_acq.__path__[0] if hasattr(_acq, '__path__') and _acq.__path__ else str(here)]
__all__ = ["base", "clinicaltrials", "mimic_iv", "openfda"]
