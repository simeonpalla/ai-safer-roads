"""
logger.py — Centralised logging configuration for the SSS pipeline.

Usage in every module:
    from logger import get_logger
    log = get_logger(__name__)

main.py calls configure() once at startup based on --verbose / --quiet flags.
"""

import logging
import sys

_FMT_PLAIN   = "%(message)s"
_FMT_VERBOSE = "%(levelname)-8s %(name)s: %(message)s"

_configured = False


def configure(verbose: bool = False, quiet: bool = False) -> None:
    global _configured
    if _configured:
        return

    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    fmt   = _FMT_VERBOSE if verbose else _FMT_PLAIN

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger. Call configure() before first use."""
    return logging.getLogger(name)
