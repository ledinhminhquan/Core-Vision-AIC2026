"""Vietnamese query understanding: translate → visually re-describe → expand.

One Gemini call per query returns translation + CLIP-style enhancement + N
paraphrase expansions as JSON (single round-trip — venue networks are slow).
Failures degrade gracefully: gemini → free Google Translate → passthrough, so
the search bar always works even fully offline (SigLIP-2 reads Vietnamese
directly; translation just tends to score higher for English-centric towers).

Results are disk-cached per (provider, query) — during a competition round the
same query is often re-run with tweaks.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field

from cvp.config import Settings
from cvp.utils.io import atomic_write_json, read_json

log = logging.getLogger(__name__)

_ASCII_RE = re.compile(r"^[\x00-\x7F]+$")


def _call_with_timeout(fn, timeout_s: float):
    """Run ``fn()`` with a hard wall-clock timeout.

    Raises ``concurrent.futures.TimeoutError`` on expiry so the caller's
    fallback chain engages (venue networks stall for minutes otherwise).

    Uses a DAEMON thread, not a ThreadPoolExecutor (round-7): pool workers are
    non-daemon and concurrent.futures joins every worker at interpreter exit —
    one hung HTTP read would keep a finished batch script alive forever.
    A daemon thread is simply abandoned; the process exits cleanly.
    """
    import threading
    from concurrent.futures import TimeoutError as FuturesTimeoutError

    outcome: list = []

    def _runner():
        try:
            outcome.append((True, fn()))
        except BaseException as e:  # noqa: BLE001 — re-raised in the caller below
            outcome.append((False, e))

    t = threading.Thread(target=_runner, daemon=True, name="cvp-gemini-call")
    t.start()
    t.join(timeout_s)
    if not outcome:
        raise FuturesTimeoutError(f"call exceeded {timeout_s:.1f}s wall clock")
    ok, value = outcome[0]
    if ok:
        return value
    raise value

_GEMINI_PROMPT = """You help a video-retrieval team search Vietnamese TV news with a CLIP-style model.
Given a Vietnamese query, return STRICT JSON (no markdown) with keys:
  "translation": faithful English translation.
  "enhanced": one English sentence describing exactly what the TARGET VIDEO FRAME shows,
              concrete and visual (colors, objects, actions, scene type, on-screen text),
              dropping meta-words like "tìm cảnh" / "đoạn video".
  "expansions": {n} alternative English phrasings emphasizing different visual details.
Query: {query}
JSON:"""


@dataclass
class ProcessedQuery:
    original: str
    translation: str | None = None
    enhanced: str | None = None
    expansions: list[str] = field(default_factory=list)
    provider_used: str = "none"

    def has_english(self) -> bool:
        """True when at least one English-usable variant exists (translation,
        enhancement or expansions). An English-only tower fed only ``original``
        Vietnamese produces near-random rankings — callers with more than one
        lane should skip such a lane instead of fusing noise."""
        return bool(self.enhanced or self.translation or self.expansions)

    def texts_for_search(self, multilingual_model: bool) -> list[str]:
        """Query variants worth encoding, deduped, original first when useful."""
        texts: list[str] = []
        if multilingual_model:
            texts.append(self.original)
        if self.enhanced:
            texts.append(self.enhanced)
        if self.translation and self.translation != self.enhanced:
            texts.append(self.translation)
        texts.extend(self.expansions)
        if not texts:
            texts = [self.original]
        seen: set[str] = set()
        out = []
        for t in texts:
            t = t.strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
        return out


class QueryProcessor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cfg = settings.query
        self.cache_path = settings.paths.art("cache", "query_cache.json")
        self._cache: dict[str, dict] | None = None
        self._gemini_client = None

    # ── cache ────────────────────────────────────────────────────────────

    def _cache_key(self, query: str) -> str:
        """Key over EVERY knob that changes the processed output — switching
        ``gemini_model`` or ``enhance_english`` mid-competition must produce
        fresh enhancements, not replay entries from the previous setting."""
        raw = "|".join([
            self.cfg.provider, self.cfg.gemini_model, str(self.cfg.enhance),
            str(self.cfg.enhance_english), str(self.cfg.expansions), query.strip(),
        ])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def _load_cache(self) -> dict[str, dict]:
        if self._cache is None:
            self._cache = read_json(self.cache_path, default={}) or {}
        return self._cache

    def _save_cache(self) -> None:
        if self._cache is not None:
            try:
                atomic_write_json(self.cache_path, self._cache, indent=None)
            except OSError as e:
                log.warning("Query cache write failed: %s", e)

    # ── providers ────────────────────────────────────────────────────────

    def _gemini(self):
        if self._gemini_client is None:
            from google import genai

            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY not set")
            try:
                # google-genai HttpOptions.timeout is in MILLISECONDS.
                self._gemini_client = genai.Client(
                    api_key=api_key,
                    http_options={"timeout": int(self.cfg.timeout_s * 1000)},
                )
            except TypeError:  # older google-genai without http_options
                self._gemini_client = genai.Client(api_key=api_key)
        return self._gemini_client

    def _process_gemini(self, query: str) -> ProcessedQuery:
        client = self._gemini()
        prompt = _GEMINI_PROMPT.format(n=self.cfg.expansions, query=query)
        # Hard wall-clock cap on top of the client-side HTTP timeout: on expiry
        # this raises and the gemini → google → passthrough chain takes over.
        # Model ids churn (previews get retired mid-season) — try the configured
        # model first, then each fallback, so a stale id degrades to the next
        # Gemini model instead of dropping all the way to Google Translate.
        models = [self.cfg.gemini_model] + [
            m for m in self.cfg.gemini_model_fallbacks if m != self.cfg.gemini_model
        ]
        resp = None
        last_err: Exception | None = None
        for model_id in models:
            try:
                resp = _call_with_timeout(
                    lambda m=model_id: client.models.generate_content(model=m, contents=prompt),
                    self.cfg.timeout_s,
                )
                break
            except Exception as e:  # noqa: BLE001 — fall through to next model id
                last_err = e
                log.warning("Gemini model %r failed (%s) — trying next fallback", model_id, e)
        if resp is None:
            raise last_err if last_err else RuntimeError("no Gemini model succeeded")
        text = (resp.text or "").strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        data = json.loads(text)
        # Defend against LLM shape drift: a string 'expansions' would be
        # iterated CHARACTER by character without this guard.
        raw_exps = data.get("expansions")
        if isinstance(raw_exps, str):
            raw_exps = [raw_exps]
        exps = [str(x).strip() for x in (raw_exps or []) if str(x).strip()][: self.cfg.expansions]

        def _str_or_none(v) -> str | None:
            return v.strip() if isinstance(v, str) and v.strip() else None

        return ProcessedQuery(
            original=query,
            translation=_str_or_none(data.get("translation")),
            enhanced=_str_or_none(data.get("enhanced")),
            expansions=exps,
            provider_used="gemini",
        )

    def _process_google(self, query: str) -> ProcessedQuery:
        from deep_translator import GoogleTranslator

        translated = _call_with_timeout(
            lambda: GoogleTranslator(source="vi", target="en").translate(query),
            self.cfg.timeout_s,
        )
        return ProcessedQuery(original=query, translation=translated, provider_used="google")

    # ── public ───────────────────────────────────────────────────────────

    def process(self, query: str) -> ProcessedQuery:
        query = query.strip()
        if not query:
            return ProcessedQuery(original=query)
        # Already English (pure ASCII)? No translation needed — but the visual
        # re-description + expansions still help, so only shortcut when
        # enhancement is off/unavailable.
        if _ASCII_RE.match(query) and not (
            self.cfg.enhance and self.cfg.enhance_english and self.cfg.provider == "gemini"
        ):
            return ProcessedQuery(original=query, translation=query, provider_used="none")

        if self.cfg.cache:
            cached = self._load_cache().get(self._cache_key(query))
            if cached:
                return ProcessedQuery(**cached)

        result = ProcessedQuery(original=query)
        provider = self.cfg.provider
        if provider == "gemini":
            try:
                result = self._process_gemini(query)
            except Exception as e:  # noqa: BLE001 — any API failure must degrade, not crash
                log.warning("Gemini query processing failed (%s) — falling back to google", e)
                provider = "google"
        if provider == "google" and result.provider_used == "none" and _ASCII_RE.match(query):
            # English input needs no translation; don't round-trip it through vi→en.
            return ProcessedQuery(original=query, translation=query, provider_used="none")
        if provider == "google" and result.provider_used in ("none",):
            try:
                result = self._process_google(query)
            except Exception as e:  # noqa: BLE001
                log.warning("Google translate failed (%s) — using raw query", e)

        # Cache only FULL-QUALITY results: a transient Gemini failure must not
        # pin its Google-translate fallback in the cache forever.
        if self.cfg.cache and result.provider_used == self.cfg.provider:
            self._load_cache()[self._cache_key(query)] = {
                "original": result.original,
                "translation": result.translation,
                "enhanced": result.enhanced,
                "expansions": result.expansions,
                "provider_used": result.provider_used,
            }
            self._save_cache()
        return result
