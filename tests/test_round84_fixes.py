"""Round-84: KIS is not done — two upgrades aimed at the real-pack KIS gap.

Public 9.4 ≈ 0.5-0.63/câu on the sotuyen2 KIS subset vs ~0.87 on the drill.
Two concrete suspects, both attackable in code:
1. search.kis_multi_event — 6/19 sotuyen2 KIS queries describe a SEQUENCE of
   scenes; the single-text KIS path discards that structure. When on, scene
   sentences become DANTE events (the TRAKE engine) and the top chains' frames
   are RRF-fused with the single-text ranking. Chain frames are materialised
   through the catalog and VERIFIED (video, frame) — never guessed.
2. nb04 BENCH_PACK "ABX" — the max-effort configuration (votes 5/5, topk 800,
   expansions 4, parallel QA, variant rows, exact transcription, multi-event
   KIS) is now affordable (sharded night) but was never benched. Bench-gated.
"""
from __future__ import annotations

import json
from pathlib import Path

from cvp.config import Settings
from cvp.pipeline.run_queries import kis_events

REPO = Path(__file__).resolve().parents[1]


def test_r84_kis_events_detects_sequences_only():
    two_with_cue = ["Đoạn clip bắt đầu với cảnh một người chụp ảnh bức tranh tê giác. "
                    "Đoạn clip kết thúc với cảnh một người chụp ảnh ba chú khỉ trên cầu"]
    assert len(kis_events(two_with_cue)) == 2
    four_lines = ["Một đầu bếp chế biến món ăn trong chảo với dồi trường và rau xanh.",
                  "Đầu bếp cho bông hẹ vào chảo rồi dùng dụng cụ đảo các nguyên liệu.",
                  "Các đoạn bông hẹ dài màu xanh được trộn cùng dồi trường trong chảo.",
                  "Máy quay chuyển sang cận cảnh chảo khi đầu bếp tiếp tục xào."]
    assert len(kis_events(four_lines)) == 4
    single = ["Sân khấu với dòng chữ nổi 3D có nội dung SẮC CỔ đặt ở mép sân khấu."]
    assert kis_events(single) == []
    two_no_cue = ["Một người áo đỏ đội nón trắng đang rưới nước vào mặt. "
                  "Khung hình có hai người đi xe đạp đang đuổi theo nhau trên đường."]
    assert kis_events(two_no_cue) == []


class _Ref:
    def __init__(self, vid, frame):
        self.video_id, self.frame_idx, self.pts_time = vid, frame, frame / 25.0
        self.path = f"{vid}_{frame}.jpg"


class _Res:
    def __init__(self, vid, frame, score):
        self.ref, self.score, self.signals = _Ref(vid, frame), score, {}
    video_id = property(lambda s: s.ref.video_id)
    frame_idx = property(lambda s: s.ref.frame_idx)


class _Chain:
    def __init__(self, vid, ns, frames, score):
        self.video_id, self.ns, self.frame_idxs, self.score = vid, ns, frames, score
        self.pts_times = [f / 25.0 for f in frames]


class _Catalog:
    # manifest: video L01_V002 keyframes n = 1..100 (BTC numbering, 1-based)
    # with global ids 100..199 and frame_idx = 10*n
    def load(self):
        import pandas as pd
        return pd.DataFrame({"video_id": ["L01_V002"] * 100,
                             "n": list(range(1, 101)),
                             "global_id": list(range(100, 200))})

    def ref(self, gid):
        n = gid - 100 + 1
        return _Ref("L01_V002", 10 * n)


class _Engine:
    def __init__(self, on):
        self.settings = Settings()
        self.settings.search.kis_multi_event = on
        self.catalog = _Catalog()
        self.trake_calls = 0

    def search_trake(self, events, **kw):
        self.trake_calls += 1
        return [_Chain("L01_V002", [3, 7], [30, 70], 0.9)]


def test_r84_multi_event_off_is_identity_and_on_fuses_chain_frames():
    from cvp.pipeline.run_queries import _maybe_kis_multi_event
    lines = ["Đoạn clip bắt đầu với cảnh một người chụp ảnh bức tranh tê giác trên tường. "
             "Đoạn clip kết thúc với cảnh người đó chụp ảnh ba chú khỉ trên một cây cầu"]
    base = [_Res("L01_V009", 500, 0.9), _Res("L01_V002", 30, 0.8), _Res("L01_V005", 7, 0.7)]
    off = _Engine(False)
    assert _maybe_kis_multi_event(off, lines, base) is base and off.trake_calls == 0
    on = _Engine(True)
    out = _maybe_kis_multi_event(on, lines, base)
    assert on.trake_calls == 1
    keys = [(r.video_id, r.frame_idx) for r in out]
    # the chain frame present in both rankings wins RRF → rank 1
    assert keys[0] == ("L01_V002", 30)
    # the chain-only frame was materialised through the catalog and added
    assert ("L01_V002", 70) in keys
    assert len(out) >= len(base)


def test_r84_multi_event_is_loud_when_no_frame_materialises(caplog):
    import logging
    from cvp.pipeline.run_queries import _maybe_kis_multi_event
    lines = ["Đoạn clip bắt đầu với cảnh một người chụp ảnh bức tranh tê giác trên tường. "
             "Đoạn clip kết thúc với cảnh người đó chụp ảnh ba chú khỉ trên một cây cầu"]
    base = [_Res("L01_V009", 500, 0.9)]
    eng = _Engine(True)
    eng.catalog = None                       # no way to materialise chain frames
    with caplog.at_level(logging.WARNING, logger="cvp.pipeline.run_queries"):
        out = _maybe_kis_multi_event(eng, lines, base)
    assert out is base
    assert "KNOB VÔ HIỆU" in caplog.text     # never a silent no-op again


def test_r84_defaults_and_bench_pack_x():
    assert Settings().search.kis_multi_event is False
    nb = json.loads((REPO / "notebooks" / "04_lab_artifacts.ipynb")
                    .read_text(encoding="utf-8"))
    src = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert 'assert BENCH_PACK in ("off", "A", "AB", "ABK", "ABX")' in src
    for k in ("CVP_SEARCH__VLM_RERANK_VOTES", "CVP_VQA__SELF_CONSISTENCY",
              "CVP_SEARCH__TOPK", "CVP_QUERY__EXPANSIONS", "CVP_SEARCH__KIS_MULTI_EVENT"):
        assert k in src, k
    # pack X's votes must not be clobbered by the battle-mirror block
    assert 'setdefault("CVP_SEARCH__VLM_RERANK_VOTES", "3")' in src
