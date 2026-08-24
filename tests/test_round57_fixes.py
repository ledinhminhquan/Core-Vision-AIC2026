"""Round-57: fixes for the 10 findings of the multi-agent adversarial audit.

Confirmed findings (24/08, 27-agent workflow, 2 refuters each) and their fixes:
1. captioner loaded Vintern straight from the Drive-FUSE HF cache with no
   torn-cache fallback (ASR/SigLIP had one) → resilient_from_pretrained now
   wraps model+tokenizer, so a persistent bad cache file re-downloads locally
   instead of crash-looping a 17-33h session.
2. dead-mount blindness: the syncer swallowed every error silently and the
   final sync / finalize ran without _ensure_drive → the syncer now revives
   the mount and prints failures; the final sync retries with _ensure_drive
   and, on failure, says "KHÔNG xóa runtime".
3. non-atomic finalize lock → session token + 90s confirm re-read, lock
   heartbeat (_touch_lock) between long steps, staleness threshold 2h→6h,
   and crash-release only removes the session's OWN lock.
4. _vids came from a live Drive glob (short/empty listing would prune valid
   staging files and finalize short stores) → local materialized dir first +
   a hard floor assert.
5. torn jsons passed the name-only prune and got locked into the battle store
   by the existence-only ASR resume → _prune_staging now json.loads every
   staged file.
6. self-referential aux staging gate (_need from the same FUSE listing) →
   _need is now floored at len(_vids).
7. seed/staging copies no longer rmtree or overwrite local atomic-written
   results — they only fill what is missing, so a failed final sync plus a
   cell rerun cannot lose local work; the seed also retries with
   _ensure_drive, as does the catalog stage.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUILDER = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")
CAPTIONER = (REPO / "src" / "cvp" / "auxindex" / "captioner.py").read_text(encoding="utf-8")

_SWEEPS = {"05": ("NB5_SWEEP = r", "NB6_TITLE"), "06": ("NB6_SWEEP = r", "def main")}


def _frag(fam: str) -> str:
    start, end = _SWEEPS[fam]
    return BUILDER.split(start)[1].split(end)[0]


def _code_sources(nb_name: str):
    nb = json.loads((REPO / "notebooks" / nb_name).read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_r57_captioner_survives_torn_drive_cache():
    assert "resilient_from_pretrained" in CAPTIONER
    # the tokenizer must ride the same fallback path as the model
    assert "return model, load_tokenizer(src)" in CAPTIONER


def test_r57_vids_come_from_local_materialized_dir_with_floor():
    for fam in _SWEEPS:
        frag = _frag(fam)
        assert '_mk_local = Path("/content/data/map-keyframes")' in frag
        assert "len(_vids) > 800" in frag
        # the local dir must be preferred BEFORE the Drive fallback
        assert frag.index("_mk_local if _mk_local.is_dir()") < frag.index("_my = _vids")


def test_r57_prune_validates_json_content():
    for fam in _SWEEPS:
        frag = _frag(fam)
        assert "_json.load(_fh)" in frag
        assert "file lạ/rách" in frag


def test_r57_syncer_revives_mount_and_reports_failures():
    for fam in _SWEEPS:
        frag = _frag(fam)
        syncer = frag.split("def _syncer():")[1].split("threading.Thread")[0]
        assert "_ensure_drive()" in syncer
        assert "LỖI" in syncer                       # failures are printed
        assert "except Exception:  # noqa: BLE001 — syncer" not in syncer


def test_r57_final_sync_retries_and_never_tells_user_to_delete_runtime_first():
    for fam in _SWEEPS:
        frag = _frag(fam)
        assert "sync chốt lỗi" in frag
        assert "KHÔNG xóa runtime" in frag


def test_r57_seed_fills_missing_only_and_keeps_local_results():
    for fam in _SWEEPS:
        frag = _frag(fam)
        assert "rmtree(_job_local)" not in frag       # local results survive reruns
        assert "Chỉ bù những file THIẾU" in frag
        assert "seed staging lỗi" in frag             # retried with _ensure_drive


def test_r57_finalize_lock_token_confirm_heartbeat():
    for fam in _SWEEPS:
        frag = _frag(fam)
        assert "_uuid.uuid4().hex" in frag
        assert "_tm.sleep(90)" in frag and "phiên này NHƯỜNG" in frag
        assert frag.count("_touch_lock()") >= 4       # def + acquire + heartbeats
        # crash-release must be token-guarded
        assert 'if _lock.read_text(encoding="utf-8").strip() == _token:' in frag


def test_r57_aux_need_anchored_on_video_universe():
    for fam in _SWEEPS:
        frag = _frag(fam)
        assert 'max(len(list(_src.glob("*.json"))), len(_vids))' in frag


def test_r57_generated_shards_carry_it_all():
    for fam, job in (("05", "asr_large"), ("06", "caption_dense")):
        for i, letter in enumerate("abc"):
            src = "\n".join(_code_sources(f"{fam}{letter}_{job}_shard{i + 1}.ipynb"))
            assert "_mk_local" in src
            assert "_uuid.uuid4().hex" in src
            assert "_json.load(_fh)" in src
