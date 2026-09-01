"""Automatic-track runner (new for 2026): query pack → validated submission, no human.

Per query file the task is inferred from the filename (``trake`` → ``avs`` →
``qa`` → ``kis``) and routed to the matching engine call:

    KIS   → ``engine.search_text``            → ``write_kis``
    QA    → search + grouped auto-VQA answers → ``write_qa``
    TRAKE → ``engine.search_trake``           → ``write_trake``
    AVS   → ``engine.search_avs``             → ``write_kis`` (same row shape)

Afterwards the CSVs written by THIS run are re-validated (``packager.validate_file``;
stale files from earlier runs never block or ride along — EXCEPT with
``resume=True``, which deliberately keeps same-stem CSVs that pass the
three-shield gate in :func:`_keep_resumed_csv`) and, when clean,
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
    # Queries that produced NO submission CSV (engine exception, empty file):
    # {stem: reason}. The automatic track forfeits these silently otherwise —
    # every caller must be able to see the shortfall (review R3-C26).
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Everything WRITTEN validated clean (a short pack can still be ok)."""
        return bool(self.written) and not has_errors(self.issues)

    @property
    def complete(self) -> bool:
        """ok AND no query file was dropped — the finals-ready bar."""
        return self.ok and not self.failed


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
        # ok = server ACCEPTED the call; the verdict says whether the answer was
        # judged CORRECT/WRONG (review C21 — an accepted-but-WRONG answer must
        # never read as a win in the finals log).
        log.info("DRES %s: %s%s (%s) %s", path.stem,
                 "ACCEPTED" if res.ok else "REJECTED",
                 f" verdict={res.verdict}" if res.verdict else "",
                 res.status, res.message)
    return out


def _keep_resumed_csv(prev: Path, qf: Path) -> bool:
    """Round-77 resume gate — giữ CSV cũ CHỈ khi cả ba lá chắn đều qua.

    (1) tồn tại và không rỗng (OSError kiểu DriveFS = coi như không có, chạy
    lại thay vì sập cả pack); (2) KHÔNG cũ hơn file đề ``qf`` — đề pack MỚI
    trùng tên stem với pack cũ là bẫy nộp nhầm đáp án vòng trước (audit r77);
    (3) validate sạch — CSV hỏng mà giữ lại sẽ phủ quyết zip của TẤT CẢ các
    câu, nên tự chữa bằng cách chạy lại câu đó.
    """
    try:
        if not (prev.is_file() and prev.stat().st_size > 0):
            return False
        if prev.stat().st_mtime < qf.stat().st_mtime:
            log.warning("resume: %s CŨ HƠN file đề — coi như đề mới, chạy lại "
                        "(chống nộp nhầm đáp án pack trước).", prev.name)
            return False
    except OSError as e:
        log.warning("resume: không đọc được %s (%s) — chạy lại câu này.",
                    prev.name, e)
        return False
    if has_errors(validate_file(prev, strict=True)):
        log.warning("resume: %s có lỗi validate — bỏ bản cũ, chạy lại câu này "
                    "(giữ lại sẽ chặn zip của cả pack).", prev.name)
        return False
    return True


def run_auto(
    query_dir: str | Path,
    out_dir: str | Path,
    settings: Settings,
    submit: bool = False,
    *,
    engine_factory: Callable[[Settings], Any] | None = None,
    vqa: Any | None = None,
    client_factory: Callable[[Settings], Any] | None = None,
    resume: bool = False,
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
        resume: keep existing VALID query CSVs in ``out_dir`` (same stem, not
            older than the query file, validate-clean) and only run the rest —
            round-77, for crashed-VM recovery mid-pack. Kept files DO enter
            validation + packaging.

    Returns:
        AutoRunReport with written CSVs, validation issues, zip path (when
        packaging succeeded) and per-query submit results.
    """
    # Lazy heavy imports: keep `import cvp.pipeline.auto_agent` torch-free.
    from cvp.pipeline.run_queries import run_query_file

    query_dir = Path(query_dir)
    out_dir = Path(out_dir)

    # The automatic track runs with NO human watching — a typo'd dir exiting 0
    # with "Answered 0/0" forfeits the whole pack silently (round-7; mirrors
    # run_query_folder's round-6 guard). Checked BEFORE the heavy engine build.
    if not query_dir.is_dir():
        raise FileNotFoundError(f"query dir not found: {query_dir}")
    qfiles = sorted(query_dir.glob("*.txt"))
    if not qfiles:
        log.error("No *.txt query files directly in %s — wrong folder, or did the "
                  "pack unzip into a SUBFOLDER?", query_dir)

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
    # Round-77 (bài học đêm 28/08): resume sau khi VM chết + thống đốc thời
    # gian chống bão 504. Cả hai mặc định TẮT — hành vi cũ y nguyên.
    import time as _time
    _t0 = _time.time()
    _deadline = float(getattr(settings.submission, "pack_deadline_min", 0.0) or 0.0)
    _sprint = False
    _resumed = 0
    # Nước rút chỉ áp cho CÁC CÂU CÒN LẠI CỦA PACK NÀY — settings là object
    # dùng chung của kernel, không trả lại nguyên trạng thì lượt pack sau
    # trong cùng phiên sẽ âm thầm kẹt ở chế độ nước rút.
    _pre_sprint = (settings.vqa.self_consistency,
                   settings.vqa.answer_neighbor_frames,
                   settings.search.vlm_rerank)
    try:
        for qf in qfiles:
            if resume and _keep_resumed_csv(out_dir / f"{qf.stem}.csv", qf):
                report.written.append(out_dir / f"{qf.stem}.csv")
                _resumed += 1
                continue
            if _deadline and not _sprint and (_time.time() - _t0) / 60.0 > _deadline:
                _sprint = True
                settings.vqa.self_consistency = 1
                settings.vqa.answer_neighbor_frames = 0
                settings.search.vlm_rerank = False
                log.warning(
                    "⏰ QUÁ %.0f phút — BẬT CHẾ ĐỘ NƯỚC RÚT cho các câu còn lại: "
                    "QA votes 1, tắt neighbor strips, tắt VLM rerank (giữ nguyên "
                    "retrieval 2 lane + cross-rerank local). Nộp sớm vẫn hơn "
                    "chạy đẹp mà trễ giờ.", _deadline)
            try:
                p = run_query_file(engine, qf, out_dir, vqa, top1_times=top1_times)
            except Exception as e:  # noqa: BLE001 — one bad query must not stop the pack
                log.error("Query %s failed: %s", qf.name, e)
                report.failed[qf.stem] = f"error: {e}"
                continue
            if p:
                report.written.append(p)
                log.info("%s → %s", qf.name, p.name)
            else:
                report.failed[qf.stem] = "no submission produced (empty/unparseable query?)"
    finally:
        # Audit r77: restore phải nằm trong finally — Ctrl+C/Stop giữa nước
        # rút mà không trả settings là mọi query sau trong phiên âm thầm chạy
        # cấu hình bị hạ cấp.
        if _sprint:
            (settings.vqa.self_consistency, settings.vqa.answer_neighbor_frames,
             settings.search.vlm_rerank) = _pre_sprint
            log.warning("nước rút KẾT THÚC — settings đã trả về nguyên trạng "
                        "cho các lượt chạy sau trong phiên.")
    if _resumed:
        log.warning("resume: giữ nguyên %d CSV hợp lệ từ lượt trước (không "
                    "chạy lại) — chúng vẫn được validate + đóng gói.", _resumed)
        if submit:
            log.error("resume + DRES submit: %d câu giữ lại KHÔNG có timestamp "
                      "top-1 (top1_times chỉ ghi khi chạy thật) — DRES sẽ nhận "
                      "answer thiếu vùng thời gian cho các câu đó.", _resumed)
    if report.failed:
        log.error("%d/%d query files produced NO submission: %s — these queries "
                  "score 0 unless fixed.", len(report.failed),
                  len(report.failed) + len(report.written), sorted(report.failed))

    # Validate (and later package) the CSVs this run wrote OR kept (resume) — stale files
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
                else DresClient(sub.dres_base_url, timeout=sub.dres_timeout_s,
                                evaluation_id=sub.dres_evaluation_id)
            )
            login = client.login()  # env DRES_USER / DRES_PASSWORD
            if not login.ok:
                log.error("DRES login failed (%s): %s — NOT submitting.", login.status, login.message)
                return report
            report.submitted = _submit_top1(report.written, client, top1_times)
    return report
