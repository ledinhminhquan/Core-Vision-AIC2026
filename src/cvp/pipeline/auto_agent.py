"""Automatic-track runner (new for 2026): query pack → validated submission, no human.

Per query file the task is inferred from the filename (``trake`` → ``avs`` →
``qa`` → ``kis``) and routed to the matching engine call:

    KIS   → ``engine.search_text``            → ``write_kis``
    QA    → search + grouped auto-VQA answers → ``write_qa``
    TRAKE → ``engine.search_trake``           → ``write_trake``
    AVS   → ``engine.search_avs``             → ``write_kis`` (same row shape)

Afterwards the CSVs written by THIS run are re-validated (``packager.validate_file``;
stale files from earlier runs never block or ride along) and, when clean,
zipped Codabench-style with a sha256 manifest. With ``submit=True`` the top-1
row of each CSV is pushed through the DRES client — but ONLY when the run is
error-free, ``settings.submission.auto_submit`` is true and
``settings.submission.dres_base_url`` is set (wrong submissions are penalised
at the finals, so the gate defaults to off).

Heavy imports (``SearchEngine`` → numpy/faiss stack) happen lazily inside
``run_auto`` so this module imports clean on machines without torch; tests
inject a stub engine via ``engine_factory``.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from cvp.config import Settings
from cvp.submission.dres_client import DresClient, SubmitResult
from cvp.submission.packager import (
    ValidationIssue,
    has_errors,
    infer_task,
    package_codabench,
    validate_file,
)

log = logging.getLogger(__name__)


@dataclass
class AutoRunReport:
    """What one automatic-track run produced."""

    written: list[Path] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)
    zip_path: Path | None = None
    submitted: list[tuple[str, SubmitResult]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.written) and not has_errors(self.issues)


def _first_row(csv_path: Path) -> list[str]:
    """First data row of a submission CSV (quote-aware), [] when unreadable."""
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.reader(f):
                if row and any(x.strip() for x in row):
                    return row
    except OSError as e:
        log.warning("Cannot read %s for submission: %s", csv_path.name, e)
    return []


def _submit_top1(written: list[Path], client: Any,
                 top1_times: dict[str, list[float]] | None = None) -> list[tuple[str, SubmitResult]]:
    """Push the top-1 row of each CSV. One attempt each — NEVER retried.

    ``top1_times`` maps CSV stem → the top-1 candidate's pts times in seconds
    (one per TRAKE event, a single element for KIS/QA/AVS); DRES v2 answers
    need millisecond timestamps, so they are preferred over bare frame indexes.
    """
    top1_times = top1_times or {}
    out: list[tuple[str, SubmitResult]] = []
    for path in written:
        row = _first_row(path)
        if not row:
            out.append((path.stem, SubmitResult(False, 0, "no rows to submit")))
            continue
        task = infer_task(path.name)
        times = top1_times.get(path.stem)
        try:
            if task == "trake":
                frames = [int(x) for x in row[1:]]
                times_ms = (
                    [round(t * 1000) for t in times]
                    if times and len(times) == len(frames)
                    else None
                )
                res = client.submit_trake(row[0], frames, times_ms=times_ms)
            elif task == "qa":
                time_ms = round(times[0] * 1000) if times else None
                res = client.submit_qa(row[0], int(row[1]), row[2] if len(row) > 2 else "",
                                       time_ms=time_ms)
            else:  # kis and avs share the row shape
                time_ms = round(times[0] * 1000) if times else None
                res = client.submit_kis(row[0], frame_idx=int(row[1]), time_ms=time_ms)
        except Exception as e:  # noqa: BLE001 — a bad row must not stop the run
            res = SubmitResult(False, 0, f"submit failed: {e}")
        out.append((path.stem, res))
        log.info("DRES %s: %s (%s) %s", path.stem, "OK" if res.ok else "REJECTED", res.status, res.message)
    return out


def run_auto(
    query_dir: str | Path,
    out_dir: str | Path,
    settings: Settings,
    submit: bool = False,
    *,
    engine_factory: Callable[[Settings], Any] | None = None,
    vqa: Any | None = None,
    client_factory: Callable[[Settings], Any] | None = None,
) -> AutoRunReport:
    """Run the full automatic track over a query pack.

    Args:
        query_dir: folder of organiser ``*.txt`` query files.
        out_dir: where submission CSVs (+ zip + MANIFEST.json) are written.
        settings: full app settings (vqa caps, submission gates, ...).
        submit: also push top-1 per query to DRES — gated by
            ``settings.submission.auto_submit`` AND ``dres_base_url``.
        engine_factory: optional ``Settings -> engine`` override (tests inject
            a stub; default builds the real ``SearchEngine`` lazily).
        vqa: optional pre-built VQA assistant (default: ``VqaAssistant`` when
            ``settings.vqa.provider != "none"``).
        client_factory: optional ``Settings -> DresClient`` override for tests.

    Returns:
        AutoRunReport with written CSVs, validation issues, zip path (when
        packaging succeeded) and per-query submit results.
    """
    # Lazy heavy imports: keep `import cvp.pipeline.auto_agent` torch-free.
    from cvp.pipeline.run_queries import run_query_file

    query_dir = Path(query_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if engine_factory is not None:
        engine = engine_factory(settings)
    else:
        from cvp.search.engine import SearchEngine  # heavy: numpy/faiss/pandas

        engine = SearchEngine(settings)

    if vqa is None and settings.vqa.provider != "none":
        try:
            from cvp.search.vqa import VqaAssistant

            vqa = VqaAssistant(settings)
        except Exception as e:  # noqa: BLE001 — QA falls back to empty answers
            log.warning("VQA assistant unavailable: %s", e)
            vqa = None

    report = AutoRunReport()
    top1_times: dict[str, list[float]] = {}
    for qf in sorted(query_dir.glob("*.txt")):
        try:
            p = run_query_file(engine, qf, out_dir, vqa, top1_times=top1_times)
        except Exception as e:  # noqa: BLE001 — one bad query must not stop the pack
            log.error("Query %s failed: %s", qf.name, e)
            continue
        if p:
            report.written.append(p)
            log.info("%s → %s", qf.name, p.name)

    # Validate (and later package) ONLY the CSVs THIS run wrote — stale files
    # from earlier runs in the same folder must never block or ride along.
    report.issues = [i for p in report.written for i in validate_file(p, strict=True)]
    if not report.written:
        log.warning("Packaging skipped — no submission CSVs were written.")
    elif has_errors(report.issues):
        log.warning("Packaging skipped — validation errors present.")
    else:
        zip_path = out_dir / f"{settings.submission.package_name}.zip"
        pkg_issues = package_codabench(out_dir, zip_path,
                                       package_name=settings.submission.package_name,
                                       files=report.written)
        if not has_errors(pkg_issues):
            report.zip_path = zip_path

    if submit:
        sub = settings.submission
        if has_errors(report.issues):
            log.error("submit requested but validation errors are present — NOT submitting "
                      "(wrong submissions are penalised at the finals).")
        elif not sub.dres_base_url:
            log.warning("submit requested but submission.dres_base_url is empty — skipping.")
        elif not sub.auto_submit:
            log.warning("submit requested but submission.auto_submit is false — safety gate, skipping.")
        else:
            client = (
                client_factory(settings)
                if client_factory is not None
                else DresClient(sub.dres_base_url, timeout=sub.dres_timeout_s)
            )
            login = client.login()  # env DRES_USER / DRES_PASSWORD
            if not login.ok:
                log.error("DRES login failed (%s): %s — NOT submitting.", login.status, login.message)
                return report
            report.submitted = _submit_top1(report.written, client, top1_times)
    return report
