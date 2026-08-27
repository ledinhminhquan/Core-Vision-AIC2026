"""Round-73: merge of the Cursor đợt-2 handoff + fixes for its audit findings.

A 38-agent adversarial review (9 reviewers, 2 refuters per finding) confirmed
the four ALWAYS-ON perf rewrites are value-identical and every knob-off path is
bit-identical, and kept 5 findings, fixed here:

1. (major) strip_decoration ran preamble-strip before edge-quote-strip and
   never retried — '"Đáp án: 5"' keyed to 'đáp án 5' while 'Đáp án: 5' keyed
   to '5' (split vote classes, canonical_key non-idempotent). Now the two
   passes alternate to a fixpoint.
2. (major) '-' sat in the edge-strip set and the inner-punct fold, so
   canonical_key merged OPPOSITE answers: 'tầng -1' ≡ 'tầng 1', '-5' ≡ '5'.
   A leading dash directly before a digit is now preserved as a sign.
3. (minor) 67_qa_diag let GT answers that normalize to "" ("?", "...") into
   gt_norm, so an empty-normalizing row answer was verdicted OK while
   r_score_qa scores it 0. Empty-normalized answers are dropped on both sides;
   all-empty GT → GT_UNUSABLE, matching official._entry_answers.
4. (minor) 66_trake_diag's markdown crashed (TypeError on None:.0f) whenever a
   map CSV yielded <2 parsable frames. Renders "—" now.
5. (minor) 66 grid_ceiling reported 0.0/"no legal row" when keyframes < events
   — but a SHORT row is writer-legal and scores partial credit; the ceiling is
   the DP over the first p_count events (denominator still K).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

from cvp.eval.official import normalize_answer  # noqa: E402
from cvp.search.answer_norm import canonical_key, majority_vote, strip_decoration  # noqa: E402


def _load_script(stem: str):
    path = REPO / "scripts" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"_r73_{stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── 1 · quoted/bracketed preamble peels to a fixpoint ────────────────────────

def test_r73_quoted_preamble_folds_with_bare_answer():
    assert canonical_key('"Đáp án: 5"') == canonical_key("5") == "5"
    assert canonical_key("(Trả lời: màu đỏ)") == canonical_key("màu đỏ")
    assert canonical_key('"Đáp án: 300 kg"', 2) == canonical_key("300kg", 2)
    assert strip_decoration('"Đáp án: 5"') == "5"


def test_r73_canonical_key_idempotent_on_decorated_forms():
    for s in ['"Đáp án: 5"', "(Trả lời: màu đỏ)", '"Trả lời: Đáp án là X"',
              "Đáp án: 27", '"tầng -1"', "-5 độ"]:
        for tier in (1, 2):
            k = canonical_key(s, tier)
            assert canonical_key(k, tier) == k, (s, tier)


# ── 2 · numeric sign survives; opposite answers never merge ──────────────────

def test_r73_minus_sign_is_preserved():
    assert canonical_key("tầng -1", 2) != canonical_key("tầng 1", 2)
    assert canonical_key("-5", 2) != canonical_key("5", 2)
    assert canonical_key("-5 độ", 2) != canonical_key("5 độ", 2)
    # the sign survives edge-stripping at tier 1 too, like the official scorer
    assert canonical_key("-5", 1) == "-5" == normalize_answer("-5")


def test_r73_dash_still_folds_when_not_a_sign():
    # hyphenated unit spellings and word-joins still become spaces at tier 2
    assert canonical_key("ba trăm ki-lô-gam", 2) == canonical_key("300 kg", 2)
    assert canonical_key("B-52", 2) == canonical_key("b 52", 2)
    # decoration-only dashes still strip; "-" alone survives via the fallback
    assert strip_decoration("- màu đỏ") == "màu đỏ"
    assert strip_decoration("-") == "-"


def test_r73_signed_votes_do_not_pool():
    on = majority_vote(["-5", "5", "5"], canonicalize=True)
    assert on.answer == "5" and on.votes_for == 2 and on.total == 3


# ── 3 · 67: empty-normalizing GT answers can no longer fake an OK ────────────

def test_r73_67_empty_normalized_gt_answer_is_not_a_match():
    m = _load_script("67_qa_diag")
    entry = {"video_id": "L01_V001", "answers": ["màu xanh", "?"],
             "ranges": [[100, 200]]}
    rows = [["L01_V001", "150", ""]]
    got = m.classify_qa("q", rows, entry)
    assert got["verdict"] != "OK"
    entry_all_empty = {"video_id": "L01_V001", "answers": ["?", "..."],
                       "ranges": [[100, 200]]}
    got2 = m.classify_qa("q", rows, entry_all_empty)
    assert got2["verdict"] == "GT_UNUSABLE"


# ── 4+5 · 66: short-grid ceiling + report never crashes on a thin map ────────

def test_r73_66_short_row_ceiling_and_none_gap_render():
    m = _load_script("66_trake_diag")
    got = m.grid_ceiling([100], [(95, 105), (195, 205)])
    assert got["ceiling"] == 0.5 and got["hits"] == 1 and not got["feasible"]
    diag = {"stem": "q", "video": "V", "k_events": 2, "n_keyframes": 1,
            "median_gap": None, "h1_ceiling": 0.5, "h1_feasible": False,
            "h1_per_event": [True, False]}
    md = m.render_markdown([diag], {"h1": "x", "h2": "y", "h6": "z"}, "gt", None)
    assert "—" in md and "0.500" in md
    assert "KHÔNG dựng nổi" not in md
