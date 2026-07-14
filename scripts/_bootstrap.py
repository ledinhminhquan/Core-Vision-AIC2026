"""Shared script bootstrap: make `cvf` importable + set up logging.

Also forces UTF-8 on stdout/stderr (round-3 fix M-R3-2): script docstrings and
progress prints contain characters like "→" that do not exist in cp1252, so on
Windows any piped/redirected run (``--help | more``, ``… --zip > run.log``,
subprocess capture, CI logs) would die with UnicodeEncodeError — in the worst
case AFTER writing submission CSVs but BEFORE the --zip packaging block. The
reconfigure runs at import time, before any argparse/print in the scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if (getattr(_stream, "encoding", "") or "").lower() not in ("utf-8", "utf8"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 — exotic stream without reconfigure
            pass

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cvf.config import Settings, load_settings  # noqa: E402
from cvf.utils.logging import setup_logging  # noqa: E402


def init(settings_path: str | None = None) -> Settings:
    settings = load_settings(settings_path)
    setup_logging(settings.logging.level)
    return settings
