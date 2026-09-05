"""Round-90: a POOL of Gemini API keys, rotated when one runs out of quota.

Bench session 03/09 (nb09, 6 h): ``gemini-3.1-pro-preview`` answered
``429 RESOURCE_EXHAUSTED — You exceeded your current quota`` 253 times after
the first two arms; every QA answer for the rest of the session was the
fallback ('không rõ'), QA = 0 on every arm. One billed key is one daily
bucket, and a battle night burns the very same bucket.

The Colab secrets ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) plus optional
``GEMINI_API_KEY_2`` … ``GEMINI_API_KEY_5`` form a pool. A quota-exhausted
429 moves EVERY caller in the process to the next key (the exhausted key is
never tried again) and the failing call is retried once per remaining key.
Rate-limit 429s carry the same wording, so they rotate too — another key is
another bucket. With a single key nothing changes: the error propagates and
the model chain degrades exactly as before.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable

log = logging.getLogger(__name__)

KEY_ENV_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY_2",
                 "GEMINI_API_KEY_3", "GEMINI_API_KEY_4", "GEMINI_API_KEY_5")
_LOCK = threading.Lock()
_INDEX = 0          # process-wide: once a key is exhausted nobody goes back to it

# Round-96: a SECOND PIPE for the same Gemini models — Vertex "express mode"
# (verified 05/09/2026: genai.Client(vertexai=True, api_key=<express key>) →
# aiplatform.googleapis.com global endpoint, same ids, same prices, separate
# serving path from the AI Studio 503 storms; partially correlated hedge).
# Chain entries are spelled "vertex:<model id>" and only exist when the Colab
# secret GEMINI_VERTEX_KEY is set. "hf:<model>" entries route to the Hugging
# Face Inference Providers router (cvp.models.hf_router) — an independent vendor.
VERTEX_ENV = "GEMINI_VERTEX_KEY"
VERTEX_PREFIX = "vertex:"


def vertex_key() -> str:
    return (os.environ.get(VERTEX_ENV) or "").strip()


def vertex_available() -> bool:
    return bool(vertex_key())


def expand_chain(primary: str, fallbacks) -> list[str]:
    """``[primary, vertex:primary?, *fallbacks]`` with lanes whose secret is
    missing dropped and duplicates removed (order kept).

    The vertex twin of the PRIMARY sits right behind it: same model, other
    pipe — the cheapest hedge when AI Studio alone is stormy. ``vertex:`` /
    ``hf:`` ids anywhere in ``fallbacks`` are kept only when configured.
    """
    from cvp.models import hf_router

    chain: list[str] = [primary]
    if vertex_available() and primary and not primary.startswith(VERTEX_PREFIX) \
            and not hf_router.is_hf_id(primary):
        chain.append(VERTEX_PREFIX + primary)
    chain.extend(str(m) for m in (fallbacks or []))
    out: list[str] = []
    for m in chain:
        if not m or m in out:
            continue
        if m.startswith(VERTEX_PREFIX) and not vertex_available():
            continue
        if hf_router.is_hf_id(m) and not hf_router.available():
            continue
        out.append(m)
    return out


def api_keys() -> list[str]:
    """Distinct non-empty keys in pool order."""
    keys: list[str] = []
    for name in KEY_ENV_NAMES:
        v = (os.environ.get(name) or "").strip()
        if v and v not in keys:
            keys.append(v)
    return keys


def is_quota_exhausted(e: BaseException) -> bool:
    s = str(e)
    return "429" in s or "RESOURCE_EXHAUSTED" in s


def current_key() -> str | None:
    keys = api_keys()
    if not keys:
        return None
    return keys[min(_INDEX, len(keys) - 1)]


def key_position() -> tuple[int, int]:
    """(1-based index of the active key, pool size) — for logs and payloads."""
    keys = api_keys()
    return (min(_INDEX, max(len(keys) - 1, 0)) + 1, len(keys))


def rotate(reason: str = "") -> bool:
    """Advance the whole process to the next key; False when none is left."""
    global _INDEX
    with _LOCK:
        keys = api_keys()
        if _INDEX + 1 >= len(keys):
            return False
        _INDEX += 1
        log.warning("Gemini key #%d exhausted (%s) — switching to key #%d of %d",
                    _INDEX, reason.replace("\n", " ")[:90], _INDEX + 1, len(keys))
        return True


def reset_for_tests() -> None:
    global _INDEX
    _INDEX = 0


def _build_genai(http_options: dict | None, **client_kwargs):
    """``genai.Client`` with graceful degradation of ``http_options``: an SDK
    that rejects ``retry_options`` (round-96 field names chưa xác minh trên mọi
    phiên bản) gets the same options without it; one that rejects
    ``http_options`` altogether gets none. Never let a knob kill the client."""
    from google import genai

    attempts: list[dict | None] = [http_options]
    if http_options and "retry_options" in http_options:
        attempts.append({k: v for k, v in http_options.items() if k != "retry_options"})
    attempts.append(None)
    last: Exception | None = None
    for opts in attempts:
        try:
            if opts:
                return genai.Client(http_options=opts, **client_kwargs)
            return genai.Client(**client_kwargs)
        except Exception as e:  # noqa: BLE001 — degrade the options, then re-raise
            last = e
            log.warning("genai.Client rejected http_options=%s (%s) — retrying with fewer options",
                        sorted((opts or {}).keys()), str(e)[:120])
    raise last if last else RuntimeError("genai.Client could not be built")


def _default_factory(key: str, http_options: dict | None):
    return _build_genai(http_options, api_key=key)


def _default_vertex_factory(key: str, http_options: dict | None):
    return _build_genai(http_options, vertexai=True, api_key=key)


class _Models:
    def __init__(self, owner: "RotatingGeminiClient"):
        self._owner = owner

    def generate_content(self, **kwargs: Any):
        return self._owner._call("generate_content", **kwargs)


class RotatingGeminiClient:
    """Drop-in for ``genai.Client`` as the repo uses it
    (``client.models.generate_content(model=…, contents=…, config=…)``).

    Builds the inner client lazily for the pool's ACTIVE key; when a call
    fails with a quota-exhausted 429 and another key is left, the pool
    rotates (process-wide) and the same call is retried on the new key.
    """

    def __init__(self, http_options: dict | None = None,
                 client_factory: Callable[[str, dict | None], Any] | None = None,
                 vertex_factory: Callable[[str, dict | None], Any] | None = None):
        self._http = http_options
        self._factory = client_factory or _default_factory
        self._vertex_factory = vertex_factory or _default_vertex_factory
        self._client: Any = None
        self._built_for: str | None = None
        self._vertex_client: Any = None
        self.models = _Models(self)

    def _timeout_s(self) -> float:
        try:
            return float((self._http or {}).get("timeout", 45000)) / 1000.0
        except (TypeError, ValueError):
            return 45.0

    def _vertex(self):
        key = vertex_key()
        if not key:
            raise RuntimeError(f"vertex: lane needs the {VERTEX_ENV} secret (chưa cấu hình)")
        if self._vertex_client is None:
            self._vertex_client = self._vertex_factory(key, self._http)
        return self._vertex_client

    def _inner(self):
        key = current_key()
        if key is None:
            raise RuntimeError("GEMINI_API_KEY not set")
        if self._client is None or self._built_for != key:
            self._client = self._factory(key, self._http)
            self._built_for = key
        return self._client

    def _call(self, method: str, **kwargs: Any):
        # Round-96: prefixed ids are OTHER PIPES, never the AI Studio key pool.
        model = str(kwargs.get("model") or "")
        if model.startswith("hf:"):
            from cvp.models import hf_router

            return hf_router.generate(model, kwargs.get("contents"), timeout_s=self._timeout_s())
        if model.startswith(VERTEX_PREFIX):
            kwargs = dict(kwargs, model=model[len(VERTEX_PREFIX):])
            return getattr(self._vertex().models, method)(**kwargs)
        while True:
            key_used = current_key()
            try:
                return getattr(self._inner().models, method)(**kwargs)
            except Exception as e:  # noqa: BLE001 — classify, rotate or re-raise
                if not is_quota_exhausted(e):
                    raise
                # Parallel callers (vqa.parallel_calls) may hit the same
                # exhausted key at once: if someone already rotated since this
                # call started, retry on the NEW key instead of rotating again
                # (which would skip a key, or give up while a fresh one exists).
                if current_key() != key_used or rotate(str(e)):
                    continue                     # same call, next key
                raise


def build_client(timeout_s: float | None = None) -> RotatingGeminiClient:
    """The one Gemini client factory for every call site (query enhancement,
    VQA, VLM rerank). ``timeout_s`` becomes the HTTP deadline (google-genai
    takes milliseconds)."""
    if not api_keys():
        raise RuntimeError("GEMINI_API_KEY not set")
    http: dict | None = {"timeout": int(float(timeout_s) * 1000)} if timeout_s else None
    # Round-96: the google-genai SDK silently retries 429/5xx up to FOUR times with
    # 1-2-4-8 s backoff (Gemini troubleshooting guide, 04/09/2026) — under a storm
    # one failed call burns >15 s before the breaker even sees it. With the breaker
    # on, cap the SDK at two attempts; the chain + breaker do the rest. Field names
    # follow google-genai HttpRetryOptions (chưa xác minh trên mọi phiên bản SDK →
    # the RotatingGeminiClient factory falls back to no retry options on TypeError).
    from cvp.models.gemini_health import HEALTH

    if HEALTH.enabled:
        http = dict(http or {})
        http["retry_options"] = {"attempts": 2, "initial_delay": 1.0, "max_delay": 4.0}
    return RotatingGeminiClient(http_options=http)
