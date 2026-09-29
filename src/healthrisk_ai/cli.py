# HealthRisk AI / HealthRisk Lab — CLI entrypoint
#
# This is the ``hrl-serve`` console script declared in pyproject.toml. It is a
# thin dispatcher; real serving logic lives in the domain modules and (later)
# the Streamlit Lab app.

from __future__ import annotations

import argparse
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from healthrisk_ai import instrument_logging, project_root

logger = logging.getLogger("healthrisk_ai.cli")

__all__ = ["console_entry"]


def _parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="hrl-serve",
        description=(
            "HealthRisk AI / HealthRisk Lab entrypoint. Without arguments, "
            "this prints the project layout and config summary."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a config.yaml to load (defaults to project root configs/config.yaml).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Console logging level.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved configuration and exit.",
    )
    return parser


def console_entry(argv: Sequence[str] | None = None) -> int:
    """Main entrypoint used by the console_script declaration."""
    parser = _parser()
    args = parser.parse_args(argv)

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    instrument_logging(log_level)

    try:
        from healthrisk_ai.config import cfg
    except Exception as exc:  # pragma: no cover - startup diagnostic
        logger.error("Failed to load configuration: %s", exc)
        return 2

    if args.print_config:
        config = cfg()
        for key, value in config.items():
            logger.info("%s = %r", key, value)
        return 0

    logger.info("HealthRisk AI / HealthRisk Lab v0.1.0")
    logger.info("Project root: %s", project_root())
    logger.info("Config loaded from: %s", args.config or "configs/config.yaml")

    print("Run `hrl-serve --help` for options.")
    print("For the interactive simulation, start the Streamlit Lab app from Docker or the CLI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(console_entry())
