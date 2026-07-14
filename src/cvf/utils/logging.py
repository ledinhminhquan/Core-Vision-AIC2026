"""One-call logging setup shared by scripts, notebooks and the app."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if root.handlers:  # already configured (e.g. inside Streamlit) — just set level
        root.setLevel(level.upper())
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S"))
    root.addHandler(handler)
    root.setLevel(level.upper())
    # Quieten noisy third parties
    for name in ("urllib3", "PIL", "httpx", "filelock"):
        logging.getLogger(name).setLevel(logging.WARNING)
