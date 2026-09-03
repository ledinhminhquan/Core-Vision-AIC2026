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


def _default_factory(key: str, http_options: dict | None):
    from google import genai

    try:
        if http_options:
            return genai.Client(api_key=key, http_options=http_options)
        return genai.Client(api_key=key)
    except TypeError:  # older google-genai without http_options
        return genai.Client(api_key=key)


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
                 client_factory: Callable[[str, dict | None], Any] | None = None):
        self._http = http_options
        self._factory = client_factory or _default_factory
        self._client: Any = None
        self._built_for: str | None = None
        self.models = _Models(self)

    def _inner(self):
        key = current_key()
        if key is None:
            raise RuntimeError("GEMINI_API_KEY not set")
        if self._client is None or self._built_for != key:
            self._client = self._factory(key, self._http)
            self._built_for = key
        return self._client

    def _call(self, method: str, **kwargs: Any):
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
    http = {"timeout": int(float(timeout_s) * 1000)} if timeout_s else None
    return RotatingGeminiClient(http_options=http)
