"""Round-78: the Qwen3-VL-Embedding-8B third-lane campaign notebook (nb08).

The 29/08 frontier research (5 verified investigators) found ONE clear
adopt-candidate: Qwen/Qwen3-VL-Embedding-8B — open SOTA (MMEB-V2 77.8, video
retrieval 58.7), Vietnamese explicit in its 33 languages, Apache-2.0, same
family as the battle reranker. The repo already carried a correct wrapper
(qwen_embed, EOS-pooling + MRL) pointing at the 2B; nb08 runs the 8B end to
end: embed 177K keyframes (resumable) -> FAISS -> Drive sync -> retrieval-only
A/B of battle(f+m) vs two 3-lane mixes vs qwen solo -> lab/lane_qwen_ab.json.
Bench-gated: nothing enters nb03 unless 3-lane WINS.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _nb08_src() -> str:
    nb = json.loads((REPO / "notebooks" / "08_qwen_lane.ipynb")
                    .read_text(encoding="utf-8"))
    return "".join("".join(c["source"]) for c in nb["cells"])


def test_r78_nb08_targets_the_8b_with_mrl_1536():
    src = _nb08_src()
    assert 'QWEN_ID  = "Qwen/Qwen3-VL-Embedding-8B"' in src
    assert "QWEN_DIM = 1536" in src
    assert 'CVP_EMBEDDING__QWEN_EMBED_ID' in src
    assert '"--model", "qwen_embed"' in src


def test_r78_ab_covers_battle_and_two_3lane_mixes():
    src = _nb08_src()
    assert '"[0.6, 0.4]"' in src                       # battle reference
    assert '"[0.45, 0.3, 0.25]"' in src
    assert '"[0.4, 0.25, 0.35]"' in src
    assert '"finetuned", "metaclip2", "qwen_embed"' in src
    assert "lane_qwen_ab.json" in src


def test_r78_ab_hygiene():
    src = _nb08_src()
    # pack knobs must never leak into the lane A/B (round-74 lesson)
    assert "CVP_SEARCH__NEIGHBOR_CONSISTENCY_BOOST" in src
    assert "os.environ.pop(_k, None)" in src
    # finetuned lane must use the battle checkpoint
    assert "vi_siglip2_best" in src
    # Drive sync uses the declared-name pattern only
    assert '"*qwen_embed*"' in src


def test_r78_qwen_lane_is_registered_and_ensemble_takes_three():
    from cvp.config import Settings
    from cvp.models.registry import index_key_for
    assert index_key_for("qwen_embed") == "qwen_embed"
    s = Settings()
    s.embedding.ensemble_members = ["finetuned", "metaclip2", "qwen_embed"]
    s.embedding.ensemble_weights = [0.45, 0.3, 0.25]   # validator accepts 3


def test_r78_builder_has_single_keepalive_definition():
    build = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")
    assert build.count("LAB_KEEPALIVE = r'''") == 1    # duplicate removed


# ── audit r78 hardening pins ────────────────────────────────────────────────

WRAPPER = (REPO / "src" / "cvp" / "models" / "qwen_embed.py").read_text(encoding="utf-8")


def test_r78_wrapper_matches_official_recipe():
    # official Qwen3VLEmbedder: add_generation_prompt=True + system-turn
    # instruction on EVERY input (docs get the official default)
    assert "add_generation_prompt=True" in WRAPPER
    assert "add_generation_prompt=False" not in WRAPPER
    assert 'DOC_INSTRUCTION = "Represent the user\'s input."' in WRAPPER


def test_r78_wrapper_survives_transformers5_dtype_rename():
    assert "dtype=self.dtype," in WRAPPER          # 5.x kwarg first
    assert "except TypeError:" in WRAPPER          # 4.x fallback


def test_r78_nb08_batch_cap_and_background_sync():
    src = _nb08_src()
    assert "QWEN_BATCH = 16" in src                # audit: batch 64 = OOM on 40GB
    assert "CVP_EMBEDDING__BATCH_SIZE" in src
    assert "_sync_stop.wait(900)" in src           # 15-min background Drive sync
    assert "chốt cuối" in src                      # final sync in finally
    # embed runs even without GT; only the A/B is gated on it
    assert "BỎ QUA A/B" in src
