"""Scoped console/file logging for road-link pretraining runs."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def run_logging(log_file: Path | None = None) -> Iterator[logging.Logger]:
    """Create a scoped logger for console and optional file output.

    Args:
        log_file: Optional file path for persistent logs. When omitted,
            logging writes to the console without creating any directories.

    Yields:
        Configured logger that sends messages below ERROR to stdout, ERROR
        and above to stderr, and all messages to ``log_file`` when supplied.

    Handlers are detached and closed on exit to avoid duplicate messages
    between subsequent CLI invocations.
    """
    logger = logging.getLogger("urban_syn.link_pretrain")
    old_level, old_propagate = logger.level, logger.propagate
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    standard = logging.StreamHandler(sys.stdout)
    standard.addFilter(lambda record: record.levelno < logging.ERROR)
    errors = logging.StreamHandler(sys.stderr)
    errors.setLevel(logging.ERROR)
    handlers: list[logging.Handler] = [standard, errors]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        yield logger
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(old_level)
        logger.propagate = old_propagate
