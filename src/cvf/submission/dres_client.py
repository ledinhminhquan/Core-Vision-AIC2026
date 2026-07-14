"""Minimal DRES-style submission client for the on-site finals.

The finals run a DRES-like evaluation server (5-minute queries, time decay,
WRONG-SUBMISSION PENALTY). This client keeps the HTTP surface tiny and fully
configurable, because the real 2026 endpoints are only announced at the
finals; defaults follow the public DRES v2 API:

    POST {base_url}{login_path}                     body: {"username", "password"}
         → JSON containing a session id ("sessionId" / "session" / "token")
    POST {base_url}{submit_path}?session={session}  body: {"answerSets": [...]}

``submit_path`` is a template; ``{evaluation_id}`` is substituted when set
(e.g. ``/api/v2/submit/{evaluation_id}``) and the trailing slash is dropped
when it is empty. Answer payloads follow the DRES v2 ``ApiClientSubmission``
shape: the ``ApiClientAnswer`` schema has ONLY ``{text, mediaItemName,
mediaItemCollectionName, start, end}`` (``additionalProperties: false``) —
there is NO frame field, so temporal answers must carry ``start``/``end`` in
milliseconds. Frame indexes are converted via ``fps``
(``start = end = round(frame_idx / fps * 1000)``); without ``time_ms`` or
``fps`` the answer is sent with only ``mediaItemName`` (plus ``text`` for QA)
and a warning is logged — swap the templates/fields at the finals if the
organisers deviate.

Design rules:
    * pure stdlib ``urllib`` — no requests dependency on the competition box;
    * every call is exception-wrapped and returns a ``SubmitResult``;
    * a rejected submission is NEVER auto-retried (wrong submissions carry a
      score penalty — a human or the auto-agent gate decides what to do next).
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Sequence

log = logging.getLogger(__name__)

DEFAULT_LOGIN_PATH = "/api/v2/login"
DEFAULT_SUBMIT_PATH = "/api/v2/submit/{evaluation_id}"


@dataclass(frozen=True)
class SubmitResult:
    """Outcome of one HTTP call. ``status`` is the HTTP code (0 = transport error)."""

    ok: bool
    status: int
    message: str


class DresClient:
    """One authenticated session against a DRES-style endpoint."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 6.0,
        *,
        login_path: str = DEFAULT_LOGIN_PATH,
        submit_path: str = DEFAULT_SUBMIT_PATH,
        evaluation_id: str = "",
        session_id: str = "",
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.timeout = float(timeout)
        self.login_path = login_path
        self.submit_path = submit_path
        self.evaluation_id = evaluation_id
        self.session_id = session_id

    # ── transport ────────────────────────────────────────────────────────

    def _url(self, path_template: str, with_session: bool = False) -> str:
        path = path_template.format(evaluation_id=self.evaluation_id)
        if path.endswith("/"):  # empty evaluation_id in the template
            path = path.rstrip("/")
        url = self.base_url + path
        if with_session and self.session_id:
            sep = "&" if "?" in url else "?"
            url += f"{sep}session={urllib.parse.quote(self.session_id)}"
        return url

    def _post_json(self, url: str, payload: dict) -> tuple[int, dict[str, Any], str]:
        """POST payload as JSON → (status, parsed_body, raw_text). Never raises."""
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                status = int(getattr(resp, "status", None) or resp.getcode() or 0)
        except urllib.error.HTTPError as e:  # server answered with 4xx/5xx
            try:
                raw = e.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                raw = ""
            status = int(e.code)
        except Exception as e:  # noqa: BLE001 — DNS/timeout/refused/...
            return 0, {}, str(e)
        try:
            body = json.loads(raw) if raw.strip() else {}
            if not isinstance(body, dict):
                body = {"value": body}
        except json.JSONDecodeError:
            body = {}
        return status, body, raw

    @staticmethod
    def _result(status: int, body: dict, raw: str) -> SubmitResult:
        message = str(body.get("description") or body.get("message") or raw).strip()
        ok = 200 <= status < 300 and body.get("status") is not False
        return SubmitResult(ok=ok, status=status, message=message[:500])

    # ── public API ───────────────────────────────────────────────────────

    def login(self, user: str | None = None, password: str | None = None) -> SubmitResult:
        """Authenticate; falls back to env ``DRES_USER`` / ``DRES_PASSWORD``."""
        user = user or os.environ.get("DRES_USER", "")
        password = password or os.environ.get("DRES_PASSWORD", "")
        if not user or not password:
            return SubmitResult(False, 0, "missing credentials (args or DRES_USER/DRES_PASSWORD)")
        status, body, raw = self._post_json(
            self._url(self.login_path), {"username": user, "password": password}
        )
        result = self._result(status, body, raw)
        if result.ok:
            session = body.get("sessionId") or body.get("session") or body.get("token") or ""
            if session:
                self.session_id = str(session)
            else:
                log.warning("DRES login OK but no session id in response: %s", raw[:200])
        return result

    def _submit(self, answers: list[dict]) -> SubmitResult:
        payload = {"answerSets": [{"answers": answers}]}
        url = self._url(self.submit_path, with_session=True)
        status, body, raw = self._post_json(url, payload)
        result = self._result(status, body, raw)
        if not result.ok:
            # NEVER auto-retry: wrong submissions are penalised at the finals.
            log.warning("DRES submission rejected (%s): %s", result.status, result.message)
        return result

    @staticmethod
    def _answer(video_id: str, time_ms: int | None = None, text: str | None = None) -> dict:
        """One DRES v2 ``ApiClientAnswer`` — the schema has no frame field."""
        a: dict[str, Any] = {"mediaItemName": str(video_id)}
        if text is not None:
            a["text"] = str(text)
        if time_ms is not None:
            a["start"] = int(time_ms)
            a["end"] = int(time_ms)
        return a

    @staticmethod
    def _time_ms(video_id: str, frame_idx: int | None, time_ms: int | None,
                 fps: float | None) -> int | None:
        """Millisecond position: ``time_ms`` as-is, else ``frame_idx``/``fps``."""
        if time_ms is not None:
            return int(round(time_ms))
        if frame_idx is not None and fps:
            return round(int(frame_idx) / float(fps) * 1000)
        log.warning(
            "%s: no time_ms and no fps to convert frame %s — sending the answer "
            "without start/end (DRES v2 ApiClientAnswer has no frame field).",
            video_id, frame_idx,
        )
        return None

    def submit_kis(self, video_id: str, frame_idx: int | None = None,
                   time_ms: int | None = None, fps: float | None = None) -> SubmitResult:
        """KIS: one video + one moment.

        ``time_ms`` is preferred; ``frame_idx`` needs ``fps`` to convert
        (``start = end = round(frame_idx / fps * 1000)``). With neither, a
        warning is logged and only ``mediaItemName`` is sent.
        """
        return self._submit(
            [self._answer(video_id, time_ms=self._time_ms(video_id, frame_idx, time_ms, fps))]
        )

    def submit_qa(self, video_id: str, frame_idx: int | None, answer: str,
                  time_ms: int | None = None, fps: float | None = None) -> SubmitResult:
        """QA: video + moment + short textual answer.

        ``time_ms`` is preferred; ``frame_idx`` needs ``fps`` to convert. With
        neither, only ``mediaItemName`` + ``text`` are sent (with a warning).
        """
        return self._submit(
            [self._answer(video_id, time_ms=self._time_ms(video_id, frame_idx, time_ms, fps),
                          text=answer)]
        )

    def submit_trake(self, video_id: str, frames: Sequence[int] = (),
                     times_ms: Sequence[int] | None = None,
                     fps: float | None = None) -> SubmitResult:
        """TRAKE: one answer PER EVENT, in event order, in a single answer set.

        Per-event millisecond timestamps (``times_ms``) are preferred;
        otherwise each frame in ``frames`` is converted with ``fps``
        (``start = end = round(frame / fps * 1000)``). With neither, a warning
        is logged and every answer carries only ``mediaItemName`` — the DRES
        v2 ``ApiClientAnswer`` schema has no frame field.
        """
        if times_ms is not None:
            answers = [self._answer(video_id, time_ms=int(round(t))) for t in times_ms]
        elif fps:
            answers = [
                self._answer(video_id, time_ms=round(int(f) / float(fps) * 1000))
                for f in frames
            ]
        else:
            if frames:
                log.warning(
                    "%s: no times_ms and no fps for TRAKE frames %s — sending answers "
                    "without start/end (DRES v2 ApiClientAnswer has no frame field).",
                    video_id, list(frames),
                )
            answers = [self._answer(video_id) for _ in frames]
        if not answers:
            return SubmitResult(False, 0, "TRAKE submission needs at least one event")
        return self._submit(answers)
