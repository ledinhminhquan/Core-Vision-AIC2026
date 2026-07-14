"""Crash-safe file I/O.

Everything that writes to Google Drive (or any flaky filesystem) goes through the
atomic_* helpers: write to a sibling ``.tmp`` file first, then ``os.replace`` —
so a Colab disconnect can never leave a half-written artifact behind. The tmp
name is unique per call (pid + random suffix) so concurrent writers to the same
path can never rename each other's half-written bytes; it stays in the SAME
directory so the rename remains same-filesystem (atomic).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Iterator
from uuid import uuid4


def _tmp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.{os.getpid()}.{uuid4().hex[:8]}.tmp")


def _replace_atomic(tmp: Path, path: Path) -> None:
    """``os.replace`` with a short retry: Windows transiently locks the
    destination while a CONCURRENT replace of the same path is in flight
    (PermissionError) — the last writer still wins, atomically."""
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.01 * (attempt + 1))


def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        tmp.write_bytes(data)
        _replace_atomic(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_text(path: str | os.PathLike, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | os.PathLike, obj: Any, indent: int | None = 2) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=indent))


def read_json(path: str | os.PathLike, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: str | os.PathLike) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict]) -> int:
    """Write rows as JSONL atomically. Returns row count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    n = 0
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
        _replace_atomic(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return n
