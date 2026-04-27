from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console

from rich.console import Console as RichConsole
from rich.logging import RichHandler

_console: RichConsole | None = None


def get_console() -> RichConsole:
    global _console
    if _console is None:
        _console = RichConsole(force_terminal=True)
    return _console


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("gdf")
    if not logger.handlers:
        handler = RichHandler(
            console=get_console(),
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        handler.setLevel(level)
        fmt = logging.Formatter("%(message)s", datefmt="[%X]")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


log = setup_logging()
