"""KIS-C / conversational assistant: hint accumulation + clarifying questions.

The 2026 direction (per the organisers' training session) is agent-style
retrieval: progressive-KIS reveals extra hints over time, and KIS-C may add a
dialogue where the system asks "critical questions". This module keeps a
running dialogue state and, per turn, produces:

* one consolidated visual search query merging ALL hints so far
  (progressive hints are cumulative — never search only the latest one), and
* up to 3 clarifying questions ranked by how much they would shrink the
  candidate space (colour? indoor/outdoor? on-screen text? camera angle?).

Requires Gemini; without a key it degrades to simple hint concatenation.

Also home of :class:`EpisodicLog` — the append-only interaction log the
organisers' buổi-3 lecture explicitly recommends for conversational retrieval
agents ("ghi lại mọi sự kiện: query đã chạy, filter đã dùng, ảnh đã click…" so
follow-up turns can reference earlier events). One JSONL per session under
``artifacts/agent_logs/``; doubles as the paper's failure-case transcript.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field

from cvp.config import Settings

log = logging.getLogger(__name__)


class EpisodicLog:
    """Append-only event stream for one KIS-C / agent session.

    Events are ``{"t": epoch_s, "kind": ..., **payload}`` JSON lines. Appends
    are best-effort: a full disk or locked file must NEVER break a live
    competition turn (failures are logged and swallowed).
    """

    def __init__(self, settings: Settings, session: str | None = None):
        stamp = session or time.strftime("%Y%m%d-%H%M%S")
        self.path = settings.paths.art("agent_logs") / f"session-{stamp}.jsonl"

    def append(self, kind: str, **payload) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            record = {"t": round(time.time(), 3), "kind": str(kind), **payload}
            with open(self.path, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001 — logging must never block a turn
            log.warning("Episodic log append failed (%s) — continuing", e)

    def events(self) -> list[dict]:
        """All events so far (skips corrupt lines rather than raising)."""
        if not self.path.is_file():
            return []
        out: list[dict] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def recent_summary(self, n: int = 8) -> str:
        """Compact one-line-per-event digest of the last ``n`` events —
        injectable into agent prompts so the assistant can resolve references
        like "tìm lại video giống kết quả lúc nãy"."""
        lines = []
        for ev in self.events()[-n:]:
            extras = {k: v for k, v in ev.items() if k not in ("t", "kind")}
            lines.append(f"[{ev.get('kind', '?')}] " + json.dumps(extras, ensure_ascii=False))
        return "\n".join(lines)

_AGENT_PROMPT = """You assist a team searching Vietnamese TV-news video by text query (CLIP-style retrieval).
The target scene is described by progressive hints; later hints add detail to earlier ones.

Hints so far (in order):
{hints}

{results_note}

Return STRICT JSON (no markdown) with keys:
  "query": ONE English sentence describing the target FRAME visually, merging every hint
           (concrete: objects, colors, actions, scene type, on-screen text).
  "query_vi": the same in Vietnamese.
  "questions": up to 3 short Vietnamese clarifying questions whose answers would most
               narrow the search (ask about discriminative visuals only).
JSON:"""


@dataclass
class DialogueState:
    hints: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # answers / extra observations


@dataclass
class AgentTurn:
    query: str
    query_vi: str
    questions: list[str] = field(default_factory=list)


class ConversationalAssistant:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None

    def _gemini(self):
        if self._client is None:
            from google import genai

            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY not set")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def step(self, state: DialogueState, results_summary: str | None = None) -> AgentTurn:
        """Produce the consolidated query + clarifying questions for this turn."""
        all_hints = state.hints + state.notes
        fallback = AgentTurn(query=" . ".join(all_hints), query_vi=" . ".join(all_hints))
        if not all_hints:
            return fallback
        try:
            prompt = _AGENT_PROMPT.format(
                hints="\n".join(f"{i + 1}. {h}" for i, h in enumerate(all_hints)),
                results_note=(
                    f"Current top results look like: {results_summary}" if results_summary else ""
                ),
            )
            from cvp.search.vqa import gemini_model_chain, generate_with_fallback

            raw = generate_with_fallback(
                self._gemini(),
                gemini_model_chain(self.settings, self.settings.query.gemini_model),
                prompt,
            )
            text = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
            data = json.loads(text)

            def _clean_str(v, default: str) -> str:
                return v.strip() if isinstance(v, str) and v.strip() else default

            raw_q = data.get("questions")
            if isinstance(raw_q, str):  # a string would be char-iterated below
                raw_q = [raw_q]
            questions = [str(q).strip() for q in (raw_q or []) if str(q).strip()][:3]
            return AgentTurn(
                query=_clean_str(data.get("query"), fallback.query),
                query_vi=_clean_str(data.get("query_vi"), fallback.query_vi),
                questions=questions,
            )
        except Exception as e:  # noqa: BLE001 — assistant is advisory, never blocking
            log.warning("Conversational assistant failed (%s) — using hint concatenation", e)
            return fallback
