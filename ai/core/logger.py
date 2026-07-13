"""
Logging utilities for KartikAI.
"""

from __future__ import annotations

import logging
from pathlib import Path


def get_logger(
    name: str = "KartikAI",
    log_file: str | None = "logs/kartikai.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create and return a configured logger.

    Logs are printed to the terminal and optionally saved to a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            log_path,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger