from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path


def setup_logging(process: str, logs_dir: Path | None, level: str = "INFO", console: bool = True) -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            logs_dir / f"{process}.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)
    if console:
        try:
            from rich.logging import RichHandler

            ch: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
        except ImportError:  # pragma: no cover
            ch = logging.StreamHandler(sys.stderr)
            ch.setFormatter(fmt)
        root.addHandler(ch)
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
