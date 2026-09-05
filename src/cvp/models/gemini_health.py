"""Round-96: cầu dao bão Gemini (circuit breaker) — process-wide health tracker.

Sơ tuyển 3 (04/09/2026): Google trả 503 "high demand" / 504 DEADLINE_EXCEEDED
trên 80–90 % cuộc gọi suốt 8 giờ. Mỗi cuộc gọi hỏng vẫn đi hết chuỗi dự phòng
5 model, mỗi model chờ trọn wall-clock 45–90 s trước khi "trying next" — một
câu KIS mất 4–7 phút cho ba phiếu VLM rerank trả về RỖNG, một câu QA mất tới
30 phút. 22/36 câu không kịp nộp.

Cầu dao: đếm kết quả gần nhất THEO MODEL. Một model rớt liên tiếp ``min_consecutive``
lần và ≥ ``fail_ratio`` của ``window`` lần gần nhất → cầu dao MỞ: mọi call site bỏ
qua model đó NGAY (không chờ timeout) trong ``cooldown_s`` giây. Hết cooldown →
NỬA MỞ: đúng một cuộc gọi thăm dò được đi qua; thành công → ĐÓNG lại (xóa lịch
sử), hỏng → mở tiếp. Khi MỌI model trong chuỗi đều mở, call site được báo ngay
(``all_open``) để bỏ qua bước tùy chọn (VLM rerank) hoặc rơi thẳng về model cục bộ.

Khi API khỏe, cầu dao không đổi một byte hành vi: không có chuỗi rớt thì không bao
giờ mở. Tắt mặc định (``settings.breaker.enabled``); nb03/nb09 bật qua env.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

TRANSIENT_MARKERS = ("503", "504", "429", "UNAVAILABLE", "DEADLINE", "RESOURCE_EXHAUSTED",
                     "wall clock", "timed out", "timeout", "Timeout", "overloaded")


def classify(exc: BaseException | str) -> str:
    """Rough error kind for the ledger: 503 | 504 | 429 | timeout | 4xx | other."""
    msg = str(exc)
    for code in ("503", "504", "429"):
        if code in msg:
            return code
    low = msg.lower()
    if "wall clock" in low or "timed out" in low or "timeout" in low or "deadline" in low:
        return "timeout"
    if "400" in msg or "404" in msg or "403" in msg or "INVALID_ARGUMENT" in msg:
        return "4xx"
    return "other"


def is_transient(exc: BaseException | str) -> bool:
    msg = str(exc)
    return any(m in msg for m in TRANSIENT_MARKERS)


@dataclass
class _ModelState:
    outcomes: deque = field(default_factory=lambda: deque(maxlen=8))   # (ok: bool, t)
    consecutive_fail: int = 0
    opened_at: float | None = None      # monotonic time the circuit opened (None = closed)
    probing: bool = False               # half-open: one probe in flight / allowed
    skipped: int = 0                    # calls avoided while open (telemetry)


class GeminiHealth:
    """Per-model circuit breaker. Thread-safe; one instance per process (``HEALTH``)."""

    def __init__(self, *, window: int = 8, fail_ratio: float = 0.75,
                 min_consecutive: int = 3, cooldown_s: float = 120.0, clock=time.monotonic):
        self.enabled = False
        self.window = int(window)
        self.fail_ratio = float(fail_ratio)
        self.min_consecutive = int(min_consecutive)
        self.cooldown_s = float(cooldown_s)
        self._clock = clock
        self._lock = threading.RLock()
        self._models: dict[str, _ModelState] = {}

    # ── configuration ─────────────────────────────────────────────────────
    def configure(self, settings) -> "GeminiHealth":
        """Adopt ``settings.breaker`` (idempotent; called by every client factory)."""
        cfg = getattr(settings, "breaker", None)
        if cfg is None:
            return self
        with self._lock:
            self.enabled = bool(getattr(cfg, "enabled", False))
            self.window = int(getattr(cfg, "window", self.window))
            self.fail_ratio = float(getattr(cfg, "fail_ratio", self.fail_ratio))
            self.min_consecutive = int(getattr(cfg, "min_consecutive", self.min_consecutive))
            self.cooldown_s = float(getattr(cfg, "cooldown_s", self.cooldown_s))
        return self

    def reset(self) -> None:
        with self._lock:
            self._models.clear()

    def _st(self, model: str) -> _ModelState:
        st = self._models.get(model)
        if st is None or st.outcomes.maxlen != self.window:
            st = _ModelState(outcomes=deque(maxlen=self.window))
            self._models[model] = st
        return st

    # ── ledger ────────────────────────────────────────────────────────────
    def record(self, model: str, ok: bool, kind: str = "") -> None:
        """Record one attempt. Only TRANSIENT failures (5xx/429/timeout) count
        against the model; a 400/404 is a caller bug or a dead id, not a storm."""
        if not model:
            return
        with self._lock:
            st = self._st(model)
            now = self._clock()
            if ok:
                st.outcomes.append((True, now))
                st.consecutive_fail = 0
                if st.opened_at is not None or st.probing:
                    log.warning("cầu dao bão: %r trả lời lại — ĐÓNG cầu dao", model)
                st.opened_at, st.probing = None, False
                return
            if kind == "4xx":
                return
            st.outcomes.append((False, now))
            st.consecutive_fail += 1
            if st.probing:                       # the half-open probe failed → open again
                st.probing = False
                st.opened_at = now
                log.warning("cầu dao bão: thăm dò %r vẫn hỏng (%s) — mở thêm %.0f s",
                            model, kind or "?", self.cooldown_s)
                return
            if st.opened_at is None and self._should_open(st):
                st.opened_at = now
                log.warning("cầu dao bão: %r rớt %d lần liên tiếp (%d/%d gần nhất) — MỞ cầu "
                            "dao %.0f s, các cuộc gọi tới bỏ qua model này không chờ timeout",
                            model, st.consecutive_fail, self._n_fail(st), len(st.outcomes),
                            self.cooldown_s)

    @staticmethod
    def _n_fail(st: _ModelState) -> int:
        return sum(1 for ok, _ in st.outcomes if not ok)

    def _should_open(self, st: _ModelState) -> bool:
        """Two triggers: a RUN of ``min_consecutive`` transient failures (the
        storm just started — every extra attempt costs a 45–90 s wall clock), OR
        a full window with ≥ ``fail_ratio`` failures (flapping: F T F F T F F F)."""
        if st.consecutive_fail >= self.min_consecutive:
            return True
        n = len(st.outcomes)
        if n < self.window:
            return False
        return self._n_fail(st) >= math.ceil(self.fail_ratio * self.window)

    # ── decisions ─────────────────────────────────────────────────────────
    def state(self, model: str) -> str:
        with self._lock:
            st = self._models.get(model)
            if st is None or st.opened_at is None:
                return "closed"
            if self._clock() - st.opened_at >= self.cooldown_s:
                return "half-open"
            return "open"

    def should_skip(self, model: str) -> bool:
        """True = do NOT call this model now. Half-open lets exactly one probe
        through (the first caller after the cooldown); others keep skipping
        until the probe reports."""
        if not self.enabled:
            return False
        with self._lock:
            st = self._models.get(model)
            if st is None or st.opened_at is None:
                return False
            if self._clock() - st.opened_at < self.cooldown_s:
                st.skipped += 1
                return True
            if st.probing:
                st.skipped += 1
                return True
            st.probing = True                    # this caller is the probe
            return False

    def all_open(self, models) -> bool:
        """True when EVERY model in the chain would be skipped right now."""
        if not self.enabled:
            return False
        models = [m for m in models if m]
        if not models:
            return False
        with self._lock:
            for m in models:
                st = self._models.get(m)
                if st is None or st.opened_at is None:
                    return False
                if self._clock() - st.opened_at >= self.cooldown_s and not st.probing:
                    return False                 # a probe is available → not all closed off
            return True

    def snapshot(self) -> dict:
        with self._lock:
            return {m: {"state": self.state(m), "consecutive_fail": st.consecutive_fail,
                        "recent_fail": self._n_fail(st), "recent": len(st.outcomes),
                        "skipped": st.skipped}
                    for m, st in self._models.items()}


HEALTH = GeminiHealth()


class AllModelsOpen(RuntimeError):
    """Raised by a call site when the whole Gemini chain is behind an open breaker."""
