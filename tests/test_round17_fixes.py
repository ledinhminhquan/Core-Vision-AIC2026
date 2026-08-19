"""Round-17 — live-run-7 root cause: mp4s extracted onto Drive during the
FUSE-data-loss era are TRUNCATED ("moov atom not found" from a seekable read).
The source Videos_*.zip files are intact — ASR now pulls the single affected
video straight from its zip (local temp copy) and retries, so a corrupt or
missing Drive extraction can no longer kill the speech channel.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from cvp.auxindex import asr
from cvp.utils.io import read_json

GOOD = b"good-mp4-bytes"


class _ContentBackend:
    """Raises on files holding b'broken'; transcribes anything else."""

    def transcribe(self, path):
        data = Path(path).read_bytes()
        if data == b"broken":
            raise RuntimeError("moov atom not found")
        return [{"start": 0.0, "end": 1.0, "text": "xin chào"}]


def _setup(corpus, drive_bytes: bytes | None, with_zip: bool):
    from cvp.data.catalog import KeyframeCatalog

    catalog = KeyframeCatalog(corpus)
    catalog.build()
    vids = catalog.videos()
    data_root = Path(str(corpus.paths.data_root))
    if drive_bytes is not None:
        vdir = data_root / corpus.paths.videos_dir
        vdir.mkdir(parents=True, exist_ok=True)
        for v in vids:
            (vdir / f"{v}.mp4").write_bytes(drive_bytes)
    if with_zip:
        with zipfile.ZipFile(data_root / "Videos_L21_a.zip", "w") as zh:
            for v in vids:
                zh.writestr(f"video/{v}.mp4", GOOD)
    return catalog, vids


def test_truncated_drive_mp4_recovered_from_source_zip(corpus, monkeypatch):
    catalog, vids = _setup(corpus, drive_bytes=b"broken", with_zip=True)
    monkeypatch.setattr(asr, "_build_backend", lambda s: _ContentBackend())
    done = asr.asr_all_videos(corpus, catalog)               # must NOT raise
    assert done == len(vids)
    for v in vids:
        segs = read_json(corpus.paths.art("asr") / f"{v}.json")["segments"]
        assert segs and segs[0]["text"] == "xin chào"


def test_missing_drive_mp4_recovered_from_source_zip(corpus, monkeypatch):
    catalog, vids = _setup(corpus, drive_bytes=None, with_zip=True)
    monkeypatch.setattr(asr, "_build_backend", lambda s: _ContentBackend())
    done = asr.asr_all_videos(corpus, catalog)               # no files_found raise
    assert done == len(vids)


def test_broken_files_with_no_zip_still_abort_systemically(corpus, monkeypatch):
    catalog, vids = _setup(corpus, drive_bytes=b"broken", with_zip=False)
    monkeypatch.setattr(asr, "_build_backend", lambda s: _ContentBackend())
    if len(vids) >= 5:
        with pytest.raises(RuntimeError, match="5 consecutive"):
            asr.asr_all_videos(corpus, catalog)
    else:
        assert asr.asr_all_videos(corpus, catalog) == 0


def test_silent_video_from_zip_marked_no_audio(corpus, monkeypatch):
    catalog, vids = _setup(corpus, drive_bytes=b"broken", with_zip=True)

    class _SilentOnGood(_ContentBackend):
        def transcribe(self, path):
            super().transcribe(path)                          # raise on broken
            return None                                       # zip copy: silent

    monkeypatch.setattr(asr, "_build_backend", lambda s: _SilentOnGood())
    done = asr.asr_all_videos(corpus, catalog)
    assert done == len(vids)
    out = read_json(corpus.paths.art("asr") / f"{vids[0]}.json")
    assert out == {"segments": [], "no_audio": True}


def test_zip_map_matches_only_video_zips(tmp_path):
    (tmp_path / "videos").mkdir()
    with zipfile.ZipFile(tmp_path / "Videos_L21_a.zip", "w") as zh:
        zh.writestr("video/L21_V001.mp4", GOOD)
    with zipfile.ZipFile(tmp_path / "Keyframes_L21.zip", "w") as zh:
        zh.writestr("keyframes/L21_V001/001.jpg", b"jpg")     # must NOT be indexed
    zmap = asr._video_zip_map(tmp_path / "videos")
    assert set(zmap) == {"L21_V001"}
    assert zmap["L21_V001"][0].name == "Videos_L21_a.zip"


def test_extract_video_from_zip_roundtrip_and_cleanup(tmp_path):
    zp = tmp_path / "Videos_L21_a.zip"
    with zipfile.ZipFile(zp, "w") as zh:
        zh.writestr("video/L21_V001.mp4", GOOD)
    tmp = asr._extract_video_from_zip(zp, "video/L21_V001.mp4")
    try:
        assert Path(tmp).read_bytes() == GOOD and tmp.endswith(".mp4")
    finally:
        Path(tmp).unlink()


# ── R22 · torn shared-cache load → clean local re-download ───────────────────
def test_transformers_backend_falls_back_to_local_download(monkeypatch, tmp_path):
    """Live run 12 (01c): the Drive-hosted HF cache served a truncated model
    file — pipeline() rejected PhoWhisper with every class. The backend must
    re-download to LOCAL disk and retry from that path."""
    import types

    calls = {"pipeline": [], "snapshot": []}

    def fake_pipeline(model, **kw):
        calls["pipeline"].append(model)
        if model == "vinai/PhoWhisper-medium":
            raise ValueError("Could not load model with any of the following classes")
        return types.SimpleNamespace(model=model)

    def fake_snapshot(model_id, cache_dir=None):
        calls["snapshot"].append((model_id, cache_dir))
        return str(tmp_path / "local_snapshot")

    import huggingface_hub

    monkeypatch.setattr(asr, "_make_pipeline", fake_pipeline)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot,
                        raising=False)
    settings = types.SimpleNamespace(asr=types.SimpleNamespace(
        model="vinai/PhoWhisper-medium", chunk_length_s=30, batch_size=8))
    backend = asr._TransformersBackend(settings)
    assert calls["pipeline"][0] == "vinai/PhoWhisper-medium"      # shared cache first
    assert calls["snapshot"][0][0] == "vinai/PhoWhisper-medium"   # then clean download
    assert "hf_asr_cache" in calls["snapshot"][0][1]              # to LOCAL disk
    assert backend.pipe.model == str(tmp_path / "local_snapshot") # loaded from local
