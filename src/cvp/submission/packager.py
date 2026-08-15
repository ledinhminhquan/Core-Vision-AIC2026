"""Codabench submission packaging: validate CSVs → zip → manifest.

The organiser contract (see ``cvp.submission.writer``) is enforced *again*
here, on the finished files — a wrong row silently costs a submission slot
(2026: max 20 total, 5/day), so we re-check everything right before upload:

    * task inferred from the filename (``trake`` → ``avs`` → ``qa`` → ``kis``)
    * headerless UTF-8, 1..MAX_SUBMISSION_ROWS rows
    * ``video_id`` matches the organiser regex, frame indexes are
      non-negative integers
    * TRAKE frames strictly increasing; QA answers ≤100 chars (an EMPTY QA
      answer is only a warning — it costs score, not format validity)

``package_codabench`` zips every CSV under ``<package_name>/`` (the layout
Codabench expects) and writes a ``MANIFEST.json`` (sha256 + row counts) next
to the zip so a submission can be audited after the fact. Packaging is
REFUSED while any ``error``-severity issue exists.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cvp.constants import MAX_QA_ANSWER_CHARS, MAX_SUBMISSION_ROWS, VIDEO_ID_RE
from cvp.utils.io import atomic_write_json

log = logging.getLogger(__name__)

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"

# Filename → task. Order matters: "trake" wins over the "ke"/"qa" substrings,
# "avs" and "qa" before the catch-all "kis" default.
_TASK_PRIORITY = ("trake", "avs", "qa", "kis")

_HEADER_TOKENS = {"video_id", "frame_idx", "answer", "video", "frame", "pts_time", "fps"}
_UINT_RE = re.compile(r"^\d+$")


@dataclass(frozen=True)
class ValidationIssue:
    """One problem found in a submission CSV.

    ``line`` is the 1-based line number inside the file (0 = file-level).
    ``severity`` is ``"error"`` (blocks packaging) or ``"warning"`` (advisory).
    """

    file: str
    line: int
    severity: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.severity}] {self.file}:{self.line}: {self.message}"


def has_errors(issues: list[ValidationIssue]) -> bool:
    """True when any issue would block packaging/submission."""
    return any(i.severity == SEVERITY_ERROR for i in issues)


def infer_task(filename: str) -> str:
    """Task encoded in an organiser query/submission filename (kis default)."""
    stem = filename.lower()
    for task in _TASK_PRIORITY:
        if task in stem:
            return task
    return "kis"


# ── per-file validation ──────────────────────────────────────────────────────


def _err(file: str, line: int, msg: str) -> ValidationIssue:
    return ValidationIssue(file=file, line=line, severity=SEVERITY_ERROR, message=msg)


def _warn(file: str, line: int, msg: str) -> ValidationIssue:
    return ValidationIssue(file=file, line=line, severity=SEVERITY_WARNING, message=msg)


def _validate_row(name: str, lineno: int, row: list[str], task: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    vid = row[0].strip()
    if not VIDEO_ID_RE.match(vid):
        issues.append(_err(name, lineno, f"bad video id {vid!r} (expected e.g. L21_V001)"))

    def check_uint(field: str, label: str) -> int | None:
        if not _UINT_RE.match(field.strip()):
            issues.append(_err(name, lineno, f"{label} is not a non-negative integer: {field!r}"))
            return None
        return int(field)

    if task in ("kis", "avs"):
        if len(row) != 2:
            issues.append(_err(name, lineno, f"expected 2 columns video_id,frame_idx — got {len(row)}"))
        else:
            check_uint(row[1], "frame_idx")
    elif task == "qa":
        if len(row) != 3:
            issues.append(_err(
                name, lineno,
                f"expected 3 columns video_id,frame_idx,answer — got {len(row)} "
                "(answers containing commas must be double-quoted)",
            ))
        else:
            check_uint(row[1], "frame_idx")
            answer = row[2].strip()
            if not answer:
                # A scoring loss, not a format violation — a valid submission
                # may carry an empty answer, so this must not block packaging.
                issues.append(_warn(name, lineno, "QA answer is empty — the row can never score"))
            elif len(answer) > MAX_QA_ANSWER_CHARS:
                issues.append(_err(
                    name, lineno,
                    f"QA answer longer than {MAX_QA_ANSWER_CHARS} chars ({len(answer)})",
                ))
    elif task == "trake":
        if len(row) < 2:
            issues.append(_err(name, lineno, "expected video_id plus at least 1 frame column"))
        else:
            frames = [check_uint(f, f"frame #{k + 1}") for k, f in enumerate(row[1:])]
            if all(f is not None for f in frames):
                ok = [f for f in frames if f is not None]
                if any(b <= a for a, b in zip(ok, ok[1:])):
                    issues.append(_err(name, lineno, f"TRAKE frames not strictly increasing: {ok}"))
    return issues


def validate_file(path: Path, task: str | None = None, strict: bool = True) -> list[ValidationIssue]:
    """Validate one submission CSV. ``task=None`` → inferred from the filename."""
    path = Path(path)
    name = path.name
    task = task or infer_task(name)
    issues: list[ValidationIssue] = []

    try:
        data = path.read_bytes()
    except OSError as e:
        return [_err(name, 0, f"unreadable: {e}")]
    if not data.strip():
        return [_err(name, 0, "empty submission file")]
    if data.startswith(b"\xef\xbb\xbf"):
        if strict:
            issues.append(_warn(name, 1, "UTF-8 BOM present — writers should emit plain UTF-8"))
        data = data[3:]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        issues.append(_err(name, 0, f"not valid UTF-8: {e}"))
        return issues

    rows = list(csv.reader(io.StringIO(text)))
    header_row = False
    if rows and rows[0]:
        fields = [f.strip().lower() for f in rows[0]]
        # Header iff the id column is a known column name, or EVERY field is one
        # (a legit QA answer like "frame" in column 3 must not trip this).
        if fields[0] in ("video_id", "video", "videoid") or set(fields) <= _HEADER_TOKENS:
            header_row = True
            issues.append(_err(name, 1, "header row detected — submission CSVs must be headerless"))

    n_data = len(rows) - (1 if header_row else 0)
    if n_data <= 0:
        issues.append(_err(name, 0, "no data rows"))
        return issues
    if n_data > MAX_SUBMISSION_ROWS:
        issues.append(_err(name, 0, f"{n_data} rows exceed the {MAX_SUBMISSION_ROWS}-row cap"))
    elif strict and n_data < MAX_SUBMISSION_ROWS and task != "trake":
        issues.append(_warn(
            name, 0, f"only {n_data}/{MAX_SUBMISSION_ROWS} rows — extra ranked rows can only help R@k"
        ))

    seen: set[tuple[str, ...]] = set()
    for lineno, row in enumerate(rows, 1):
        if header_row and lineno == 1:
            continue
        if not row or all(not f.strip() for f in row):
            issues.append(_err(name, lineno, "blank line inside the file"))
            continue
        issues.extend(_validate_row(name, lineno, row, task))
        key = tuple(f.strip() for f in row)
        if key in seen and strict:
            issues.append(_warn(name, lineno, "duplicate row — wastes a ranked slot"))
        seen.add(key)
    return issues


def validate_submission_dir(directory: str | os.PathLike, strict: bool = True) -> list[ValidationIssue]:
    """Validate every ``*.csv`` in a submission folder.

    Returns ALL issues found; ``error``-severity ones block ``package_codabench``.
    With ``strict=False`` advisory warnings are suppressed.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return [_err("", 0, f"not a directory: {directory}")]
    csvs = sorted(directory.glob("*.csv"))
    if not csvs:
        return [_err("", 0, f"no CSV files found in {directory}")]
    issues: list[ValidationIssue] = []
    for f in csvs:
        issues.extend(validate_file(f, strict=strict))
    if not strict:
        issues = [i for i in issues if i.severity == SEVERITY_ERROR]
    return issues


# ── packaging ────────────────────────────────────────────────────────────────


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _row_count(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    return sum(1 for ln in text.splitlines() if ln.strip())


def package_codabench(
    directory: str | os.PathLike,
    out_zip: str | os.PathLike,
    package_name: str = "submission",
    files: list[Path] | None = None,
) -> list[ValidationIssue]:
    """Zip a validated submission folder Codabench-style.

    Layout inside the zip: ``<package_name>/<file>.csv``. By default every CSV
    in ``directory`` is packaged; pass ``files`` (the CSVs THIS run wrote) so
    only those are validated and zipped — stale files from earlier runs in the
    same folder must never ride along. A ``MANIFEST.json`` (file list, sha256,
    row counts, created_utc, zip sha256) is written next to the zip for
    auditability.

    Returns the validation issues. When any ``error``-severity issue exists
    the zip is NOT created (submission slots are precious) — fix and re-run.
    """
    directory = Path(directory)
    out_zip = Path(out_zip)
    if files is not None:
        csvs = sorted(Path(f) for f in files)
        if not csvs:
            return [_err("", 0, "no CSV files to package")]
        issues: list[ValidationIssue] = []
        for f in csvs:
            issues.extend(validate_file(f, strict=True))
    else:
        issues = validate_submission_dir(directory, strict=True)
        csvs = sorted(directory.glob("*.csv"))
    if has_errors(issues):
        return issues

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_zip.with_name(out_zip.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in csvs:
            zf.write(f, arcname=f"{package_name}/{f.name}")

    # ORDER (round-9): manifest + ledger + history are written from the tmp
    # bytes BEFORE the zip lands at its final path. A crash mid-sequence then
    # leaves audit records that are merely NEWER than the visible zip (healed
    # by the next successful package) — never a fresh zip whose manifest,
    # ledger and history all describe the PREVIOUS package.
    manifest = {
        "package_name": package_name,
        "zip": out_zip.name,
        "zip_sha256": _sha256(tmp),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": [
            {
                "name": f.name,
                "task": infer_task(f.name),
                "rows": _row_count(f),
                "sha256": _sha256(f),
            }
            for f in csvs
        ],
    }
    atomic_write_json(out_zip.parent / "MANIFEST.json", manifest)

    # Audit trail (round-6): each repackage OVERWRITES submission.zip +
    # MANIFEST.json in place, and nothing tracked the 5/day-20-total ration.
    # Keep an append-only ledger + a timestamped zip copy so every package
    # ever built stays diffable (scripts/41) and countable.
    try:
        history = out_zip.parent / "history"
        history.mkdir(exist_ok=True)
        stamp = manifest["created_utc"].replace(":", "").replace("-", "")
        archived = history / f"{stamp}-{manifest['zip_sha256'][:8]}.zip"
        if not archived.exists():
            shutil.copy2(tmp, archived)
        ledger = out_zip.parent / "submissions_log.jsonl"
        with open(ledger, "a", encoding="utf-8") as lf:
            lf.write(json.dumps({
                "created_utc": manifest["created_utc"],
                "zip_sha256": manifest["zip_sha256"],
                "archived": archived.name,
                "files": [f["name"] for f in manifest["files"]],
            }, ensure_ascii=False) + "\n")
        n_packages = sum(1 for _ in open(ledger, encoding="utf-8"))
        log.info("Package archived → %s (ledger now lists %d builds; the prelim "
                 "ration is 20 UPLOADS total / ≤5 per day — track uploads, not builds)",
                 archived.name, n_packages)
    except OSError as e:
        log.warning("Could not archive the package for the audit trail: %s", e)
    os.replace(tmp, out_zip)
    return issues
