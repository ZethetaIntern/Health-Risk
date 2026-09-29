# HealthRisk AI / HealthRisk Lab — app package
#
# This package is the thin public surface for configuration, entrypoints, and
# orchestration. Heavy logic lives in the domain subpackages.

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

logger = logging.getLogger("healthrisk_ai")

__all__ = ["load_config", "project_root"]


def project_root() -> Path:
    """Return the repository root as a Path.

    The root is defined as the directory two levels up from this module file.
    """
    return Path(__file__).resolve().parents[2]


def _project_env_prefix() -> str:
    """Return an env var prefix used by the local config resolver.

    The proxy ``${VAR:-default}`` syntax in config.yaml is resolved against
    environment variables first, then falls back to the literal default text.
    """
    return "HR_"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load and resolve the runtime config.

    Resolution order:
    1. Explicit `path` if provided.
    2. `configs/config.yaml` relative to the project root.

    Environment variables referenced with the ``${VAR:-default}`` syntax in
    YAML string values are resolved against the process environment.
    """
    path = project_root() / "configs" / "config.yaml" if path is None else Path(path)

    if not path.exists():        raise FileNotFoundError(f"Config file not found: {path}")

    load_dotenv()

    import os as _os

    with path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    return _resolve_env_references(raw, base_env=_os.environ.copy())




def _resolve_env_references(value: Any, *, base_env: dict[str, str] | None = None) -> Any:
    """Recursively resolve ``${VAR:-default}`` placeholders in string values."""
    if isinstance(value, str):
        return _resolve_env_template(value, base_env=base_env)
    if isinstance(value, dict):
        return {key: _resolve_env_references(val, base_env=base_env) for key, val in value.items()}
    if isinstance(value, list):
        return [_resolve_env_references(item, base_env=base_env) for item in value]
    return value


def _resolve_env_template(text: str, *, base_env: dict[str, str] | None = None) -> str:
    """Resolve a single ``${VAR:-default}`` or ``${VAR}`` template string."""
    if "${" not in text:
        return text

    envs: dict[str, str]
    if base_env is None:
        import os

        envs = os.environ.copy()
    else:
        envs = dict(base_env)

    result: list[str] = []
    remainder = text
    while remainder:
        start = remainder.find("${")
        if start == -1:
            result.append(remainder)
            break

        result.append(remainder[:start])
        end = remainder.find("}", start)
        if end == -1:
            result.append(remainder[start:])
            break

        inner = remainder[start + 2 : end]
        var, _, default = inner.partition(":-")
        # _env_lookup_project_prefixed adds the HR_ prefix itself when needed.
        env_key = var
        value = Path(__file__).resolve().parents[2].joinpath(".env")  # noqa: F841 kept for readability
        resolved = _env_lookup_project_prefixed(env_key, envs=envs)
        if resolved is None:
            resolved = default or ""
        if resolved is None:
            resolved = default
        result.append(resolved)
        remainder = remainder[end + 1 :]
    return "".join(result)


def _env_lookup(name: str) -> str | None:
    """Lookup an env var exactly; return None if missing.

    This helper is kept deliberately thin so CI and tests can control the
    environment without depending on dotenv internals.
    """
    import os

    return os.environ.get(name)


def _env_lookup_project_prefixed(name: str, *, envs: dict[str, str] | None = None) -> str | None:
    """Lookup a project-prefixed env var (HR_...) first, then unprefixed.

    This mirrors the behavior expected by existing config tests.
    """
    envs = envs if envs is not None else __import__("os").environ
    prefixed = f"{_project_env_prefix()}{name}"
    if prefixed in envs:
        return envs[prefixed]
    return envs.get(name)


def instrument_logging(level: int = logging.INFO) -> None:
    """Configure package logging to emit to stderr with a stable format."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(name)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)


def _register_acquisition_package() -> None:
    """Register the acquisition package so submodule imports resolve correctly.

    The acquisition package is implemented as a flat module bundle; this helper
    ensures Python's import machinery can find submodules like mimic_iv during
    normal imports and during pytest collection.
    """
    if "healthrisk_ai.acquisition" in sys.modules:
        return

    here = Path(__file__).resolve().parents[1] / "acquisition"
    acq_spec = importlib.util.spec_from_file_location(
        "healthrisk_ai.acquisition",
        here / "__init__.py",
        submodule_search_locations=[str(here)],
    )
    if acq_spec is None:
        return

    acq_mod = importlib.util.module_from_spec(acq_spec)
    sys.modules["healthrisk_ai.acquisition"] = acq_mod
    try:
        acq_spec.loader.exec_module(acq_mod)
    except Exception:
        # If the acquisition package fails to initialize, leave the stub so
        # imports downstream can still raise a clearer error when they need it.
        raise


def _register_processing_package() -> None:
    """Register the processing package so submodule imports resolve correctly."""
    if "healthrisk_ai.processing" in sys.modules:
        return

    here = Path(__file__).resolve().parents[1] / "processing"
    proc_spec = importlib.util.spec_from_file_location(
        "healthrisk_ai.processing",
        here / "__init__.py",
        submodule_search_locations=[str(here)],
    )
    if proc_spec is None:
        return

    proc_mod = importlib.util.module_from_spec(proc_spec)
    sys.modules["healthrisk_ai.processing"] = proc_mod
    try:
        proc_spec.loader.exec_module(proc_mod)
    except Exception:
        raise


def _register_models_package() -> None:
    """Register the models package for imports."""
    if "healthrisk_ai.models" in sys.modules:
        return

    here = Path(__file__).resolve().parents[1] / "models"
    models_spec = importlib.util.spec_from_file_location(
        "healthrisk_ai.models",
        here / "__init__.py" if (here / "__init__.py").exists() else None,
        submodule_search_locations=[str(here)],
    )
    if models_spec is None:
        return

    models_mod = importlib.util.module_from_spec(models_spec)
    sys.modules["healthrisk_ai.models"] = models_mod
    try:
        models_spec.loader.exec_module(models_mod)
    except Exception:
        pass


def _register_other_packages() -> None:
    """Register other domain packages."""
    packages = ["insurance", "credit_risk", "pharma", "explainability", "simulation"]
    for pkg_name in packages:
        pkg_key = f"healthrisk_ai.{pkg_name}"
        if pkg_key in sys.modules:
            continue

        here = Path(__file__).resolve().parents[1] / pkg_name
        pkg_spec = importlib.util.spec_from_file_location(
            pkg_key,
            here / "__init__.py" if (here / "__init__.py").exists() else None,
            submodule_search_locations=[str(here)],
        )
        if pkg_spec is None:
            continue

        pkg_mod = importlib.util.module_from_spec(pkg_spec)
        sys.modules[pkg_key] = pkg_mod
        try:
            pkg_spec.loader.exec_module(pkg_mod)
        except Exception:
            pass


def __init_submodules() -> None:
    _register_acquisition_package()
    _register_processing_package()
    _register_models_package()
    _register_other_packages()


__init_submodules()

