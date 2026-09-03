"""Round-88: the ONE-SHOT campaign — every remaining idea, one Colab session.

1. search.query_adaptive_weights (heuristic, zero API): a query that quotes
   on-screen text / speech scales the OCR / ASR fusion weight per query.
2. notebooks/09_campaign.ipynb: nine arms run back-to-back in one session —
   ABK baseline (in-session noise gauge), +W, +V5, +RRF, DIVERSE full-stack
   (first full-stack measurement of the attempt-2 lineup), TUNE -> candidate
   weights (the battle tuning file is never written), +TUNED, MERGE2/MERGE3
   offline. Each arm is saved to Drive lab/campaign/ and skipped on re-run;
   an API-storm counter is recorded per arm; the verdict table is computed.
3. Record correction: the 48x 429 in bench ABKD happened with QA
   parallel_calls=1 (pack D never sets it) — Google-side quota/load, not our
   concurrency. QA_PARALLEL=2 in nb03 stays as insurance for 4 VMs.
"""
from __future__ import annotations

import json
from pathlib import Path

from cvp.config import Settings
from cvp.search.query_cues import adaptive_weight_scale, query_signal_cues

REPO = Path(__file__).resolve().parents[1]


def test_r88_cues_vietnamese():
    assert query_signal_cues("Dòng chữ trên biển hiệu ghi gì?") == {"ocr"}
    assert query_signal_cues("Người đàn ông phát biểu về việc gì") == {"asr"}
    assert query_signal_cues("Bài hát vang lên khi dòng chữ hiện trên màn hình") == {"asr", "ocr"}
    assert query_signal_cues("Một người đàn ông mặc áo đỏ đi xe đạp") == set()
    # substrings of other words must not fire ("hát" in "nhất", "nói" vs "nơi")
    assert query_signal_cues("khoảnh khắc đẹp nhất ở nơi này") == set()
    # a visual scene of people talking is not quoted speech
    assert query_signal_cues("người đàn ông đang nói chuyện với bạn") == set()
    assert query_signal_cues("") == set()


def test_r88_cues_checked_on_real_aic_queries():
    # OCR cues fire on genuine on-screen-text queries (sơ tuyển 1/2, đề nháp)
    assert query_signal_cues("Con số hiển thị cuối cùng trên cân là bao nhiêu?") == {"ocr"}
    assert query_signal_cues("con số được ghi trên biển báo bên trái của cây cầu") == {"ocr"}
    assert query_signal_cues("Sân khấu với dòng chữ nổi 3D có nội dung: “SẮC CỔ”") == {"ocr"}
    # question idioms and segment-type words must NOT be read as quoted speech
    assert query_signal_cues("Hãy cho biết loài cây (II) đạt tốc độ sinh trưởng nào") == set()
    assert query_signal_cues("Mẩu tin giới thiệu về đàn hổ tại một địa phương") == set()
    assert query_signal_cues("Cảnh phim lần lượt giới thiệu các nguyên liệu của món ăn") == set()
    assert query_signal_cues("Hỏi quán trọ được nhắc đến trong đoạn phim nằm trên đường nào?") == {"asr"}


def test_r88_cues_reproduced_false_positives_are_gone():
    # audit r88 reproduced each of these firing on purely VISUAL descriptions
    assert query_signal_cues("Cảnh quay được ghi lại từ trên cao cho thấy một số xe ô tô đi qua cầu") == set()
    assert query_signal_cues("chiếc áo in trên ngực một bông hoa") == set()
    assert query_signal_cues("màn hình hiển thị hình ảnh một con mèo") == set()
    assert query_signal_cues("a man reads a newspaper on a bench") == set()
    assert query_signal_cues("trước nhà hát lớn, hai hàng xe chạy song song") == set()
    assert query_signal_cues("người dẫn đầu đoàn đua mặc áo vàng") == set()
    # ...while the object-bearing forms still fire
    assert query_signal_cues("có thông tin về giá dầu được hiển thị trong khung hình") == {"ocr"}
    assert query_signal_cues("người dẫn chương trình nói rằng buổi lễ bắt đầu") == {"asr"}
    # audit r88b: bare "tiêu đề"/"băng rôn" describe visual objects; object-bearing forms fire
    assert query_signal_cues("phía trên có thanh tiêu đề xanh dương chứa họa tiết địa cầu") == set()
    assert query_signal_cues("trên băng rôn còn có hình ảnh 02 em bé vùng khó khăn") == set()
    assert query_signal_cues("Hỏi tiêu đề của công thức nấu ăn (tên món ăn) này là gì?") == {"ocr"}
    assert query_signal_cues("băng rôn ghi dòng chữ chào mừng") == {"ocr"}
    assert query_signal_cues("con số trong bảng số liệu là bao nhiêu") == {"ocr"}
    assert query_signal_cues("đoạn đối thoại giữa hai người") == {"asr"}


def test_r88_cues_english_prepared_path():
    assert query_signal_cues("a banner reads 'welcome' above the stage") == {"ocr"}
    assert query_signal_cues("the reporter says the bridge reopened") == {"asr"}


def test_r88_scale_off_is_identity_and_defaults():
    s = Settings()
    assert s.search.query_adaptive_weights is False
    assert s.search.adaptive_ocr_boost == 2.0 and s.search.adaptive_asr_boost == 2.0
    assert adaptive_weight_scale("Một người đi bộ") == {}
    assert adaptive_weight_scale("dòng chữ ghi gì", ocr_boost=1.5) == {"ocr": 1.5}
    assert adaptive_weight_scale("dòng chữ ghi gì", ocr_boost=1.0) == {}
    assert adaptive_weight_scale("") == {}


def test_r88_engine_wiring():
    src = (REPO / "src" / "cvp" / "search" / "engine.py").read_text(encoding="utf-8")
    assert "adaptive_weight_scale(" in src
    assert "weights.append(weight * _scale.get(name, 1.0))" in src
    assert "if self.settings.search.query_adaptive_weights:" in src
    # audit r88: the cue is taken from description + question, threaded through
    # search_text / search_prepared / the low-confidence retry
    assert "cue_text or query_text_for_bm25" in src
    assert src.count("cue_text: str | None = None") == 3
    rq = (REPO / "src" / "cvp" / "pipeline" / "run_queries.py").read_text(encoding="utf-8")
    assert "cue_text=_cue" in rq and "**_cue_kwargs(search, cue_text)" in rq


def test_r88_cue_text_reaches_qa_questions_and_spares_stub_engines():
    from cvp.pipeline.run_queries import _cue_kwargs, parse_query_lines
    lines = ["Đoạn phim ghi lại cảnh mạnh thường quân hỗ trợ một quán trọ dành cho "
             "người cao tuổi, sau đó chuyển sang cảnh một nhóm người nước ngoài.",
             "Hỏi quán trọ được nhắc đến trong đoạn phim nằm trên đường nào?"]
    retrieval_text, question = parse_query_lines("qa", lines)
    assert question and "nhắc đến" in question
    assert query_signal_cues(retrieval_text) == set()        # cue lives in the question
    cue = " ".join(t for t in (retrieval_text, question) if t)
    assert query_signal_cues(cue) == {"asr"}

    def old_engine(text, skip_rerank=False):
        return []

    def new_engine(text, skip_rerank=False, cue_text=None):
        return []

    assert _cue_kwargs(old_engine, cue) == {}
    assert _cue_kwargs(new_engine, cue) == {"cue_text": cue}
    assert _cue_kwargs(new_engine, "") == {}


def test_r88_campaign_notebook():
    nb = json.loads((REPO / "notebooks" / "09_campaign.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    camp = next(s for s in cells if "_ARM_ORDER" in s)
    assert 'ARMS = "all"' in camp
    for arm in ("TUNE", "ABK", "ABK+TUNED", "ABK+W", "ABK+RRF", "DIVERSE", "ABK+V5",
                "MERGE2", "MERGE3", "MERGE_SIB", "MERGE2_NOHEDGE"):
        assert f'"{arm}"' in camp, arm
    assert "best_weights-candidate.json" in camp            # battle weights never written
    assert '"tuning" / "best_weights.json"' in camp         # ...only READ, once, up front
    assert "_tune_out" not in camp
    assert 'os.environ["CVP_VQA__PARALLEL_CALLS"] = "2"' in camp   # = nb03 round-87
    assert "class _StormCounter" in camp and "campaign_summary.json" in camp
    assert "bỏ qua" in camp                                  # resume per arm
    assert "qa_keep_all_answers=hedge" in camp
    assert "engine.member_names == _want" in camp            # no lane-short bench
    # audit r88 hardening
    assert "GEMINI_API_KEY" in camp                          # preflight, not 9 h later
    assert "_cr._get_reranker(settings) is not None" in camp  # eager reranker build
    assert "torch.cuda.mem_get_info()" in camp               # VRAM gate
    assert "if rep.failed:" in camp and "if _storm.fatal:" in camp   # degraded is never saved
    assert "_write_json(_camp" in camp and '_run.__tmp-{SESSION}"' in camp   # CSV first, marker last
    assert '_run-prev"' in camp                              # old run dir rotated, never deleted
    # audit r88b: Gemini VLM storms are SOFT (counted), only the cross-encoder is FATAL
    assert '"Cross-encoder rerank failed",' in camp        # tuple content checked in test_r88b
    assert '"VLM rerank failed"' in camp and "_vlm_off > len(_qfiles) // 4" in camp
    assert "battle_insample" in camp and '< _cvm.get("default", 0)' in camp
    assert "sessions_of_parts" in camp and "parse_query_lines" in camp
    assert 'SESSION = globals().get("SESSION") or uuid' in camp and "≠phiên" in camp   # provenance
    assert "mean_heldout" in camp                            # TUNE cross-validation
    assert "bench_full-prev.json" in camp                    # second ABK draw → noise
    assert "_bar = max(2 * _noise, 2 * _step)" in camp       # win rule above noise
    assert "cue_hits" in camp
    # nb04 default untouched
    nb4 = json.loads((REPO / "notebooks" / "04_lab_artifacts.ipynb").read_text(encoding="utf-8"))
    src4 = "".join("".join(c["source"]) for c in nb4["cells"] if c["cell_type"] == "code")
    assert 'BENCH_PACK = "ABK"' in src4


def test_r88b_fatal_phrases_never_match_gemini_storm_warnings():
    import re
    nb = json.loads((REPO / "notebooks" / "09_campaign.ipynb").read_text(encoding="utf-8"))
    camp = next("".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code" and "_ARM_ORDER" in "".join(c["source"]))
    fatal = re.findall(r'"([^"]+)"', re.search(r"FATAL = \((.*?)\)\n", camp, re.S).group(1))
    assert "Cross-encoder rerank failed" in fatal and len(fatal) >= 8
    vlm = (REPO / "src" / "cvp" / "search" / "vlm_rerank.py").read_text(encoding="utf-8")
    for msg in ("VLM rerank failed (%s) — keeping original order",
                "VLM rerank returned no usable scores — keeping original order"):
        assert msg in vlm                                    # the real Gemini-side strings
        assert not any(f.lower() in msg.lower() for f in fatal), msg
    cr = (REPO / "src" / "cvp" / "search" / "cross_rerank.py").read_text(encoding="utf-8")
    assert "Cross-encoder rerank failed" in cr                # the real structural string
