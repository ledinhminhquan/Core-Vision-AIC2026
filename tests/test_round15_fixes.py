"""Round-15 — live-run-6 lessons + adversarial-verify hardening.

  R15-1  ASR: transformers' ffmpeg_read pipes file BYTES through ffmpeg stdin,
         which cannot decode moov-at-end mp4s (every Batch-1 video failed).
         _extract_wav decodes from the seekable path itself (with timeouts);
         no-audio videos → valid silent transcript; audio-present-but-empty
         output is NOT written (would lock the corruption in behind resume)
         and 5 in a row aborts; zero resolvable video files aborts.
  R15-2  Captions: remote-code InternVL never calls post_init(), so the
         transformers-5 finalize step's `all_tied_weights_keys` was missing —
         shared read-only shim in hf_compat, applied at ALL Vintern load sites
         (captioner, VQA local fallback, VLM reranker) + tokenizer fallback;
         sticky mid-sweep failures abort after 3 all-fail videos.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from cvp.auxindex import asr

REPO = Path(__file__).resolve().parents[1]


# ── R15-1 · _extract_wav classification ──────────────────────────────────────
def _fake_run(script):
    """script: list of (returncode, stdout, side_effect(cmd))"""
    calls = []

    def run(cmd, capture_output=True, text=True, timeout=None):
        rc, out, effect = script[len(calls)]
        calls.append(cmd)
        if effect:
            effect(cmd)
        return subprocess.CompletedProcess(cmd, rc, stdout=out, stderr="boom")

    return run, calls


def test_extract_wav_success(monkeypatch):
    def write_wav(cmd):
        Path(cmd[-1]).write_bytes(b"RIFF" + b"\0" * 100)   # > 44-byte header

    run, calls = _fake_run([(0, "", write_wav)])
    monkeypatch.setattr("subprocess.run", run)
    wav = asr._extract_wav("clip.mp4")
    assert wav is not None and Path(wav).exists()
    Path(wav).unlink()
    assert calls[0][0] == "ffmpeg" and "-nostdin" in calls[0]
    assert calls[0][calls[0].index("-i") + 1] == "clip.mp4"  # seekable PATH, not stdin


def test_extract_wav_silent_video_returns_none(monkeypatch):
    run, calls = _fake_run([(1, "", None), (0, "", None)])   # extract fails, probe: no audio
    monkeypatch.setattr("subprocess.run", run)
    assert asr._extract_wav("silent.mp4") is None
    assert calls[1][0] == "ffprobe"


def test_extract_wav_real_breakage_raises(monkeypatch):
    run, _ = _fake_run([(1, "", None), (0, "audio", None)])  # probe: audio EXISTS → real failure
    monkeypatch.setattr("subprocess.run", run)
    with pytest.raises(RuntimeError, match="ffmpeg audio extraction failed"):
        asr._extract_wav("broken.mp4")


def test_extract_wav_missing_ffmpeg_binary(monkeypatch):
    def run(cmd, capture_output=True, text=True, timeout=None):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr("subprocess.run", run)
    with pytest.raises(RuntimeError, match="ffmpeg binary not found"):
        asr._extract_wav("clip.mp4")


def test_extract_wav_timeout_becomes_per_video_failure(monkeypatch):
    def run(cmd, capture_output=True, text=True, timeout=None):
        assert timeout is not None                           # Drive-FUSE stall guard
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr("subprocess.run", run)
    with pytest.raises(RuntimeError, match="timed out"):
        asr._extract_wav("stalled.mp4")


def test_transformers_backend_silent_video_returns_none(monkeypatch):
    backend = asr._TransformersBackend.__new__(asr._TransformersBackend)
    backend.pipe = lambda *_a, **_k: pytest.fail("pipeline must not run on silent video")
    monkeypatch.setattr(asr, "_extract_wav", lambda p: None)
    assert backend.transcribe("silent.mp4") is None


def test_transformers_backend_cleans_up_wav(monkeypatch, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 100)
    backend = asr._TransformersBackend.__new__(asr._TransformersBackend)
    backend.pipe = lambda p: {"chunks": [{"timestamp": (0.0, 1.5), "text": " xin chào "}]}
    monkeypatch.setattr(asr, "_extract_wav", lambda p: str(wav))
    segs = backend.transcribe("clip.mp4")
    assert segs == [{"start": 0.0, "end": 1.5, "text": "xin chào"}]
    assert not wav.exists()                                  # temp wav removed


def test_transformers_backend_salvages_text_without_timestamps(monkeypatch, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 100)
    backend = asr._TransformersBackend.__new__(asr._TransformersBackend)
    backend.pipe = lambda p: {"text": " bản tin thời sự ", "chunks": [
        {"timestamp": (None, None), "text": "bản tin thời sự"}]}
    monkeypatch.setattr(asr, "_extract_wav", lambda p: str(wav))
    segs = backend.transcribe("clip.mp4")                    # PhoWhisper degenerate ts
    assert len(segs) == 1 and segs[0]["text"] == "bản tin thời sự"


def _corpus_with_videos(corpus):
    from cvp.data.catalog import KeyframeCatalog

    catalog = KeyframeCatalog(corpus)
    catalog.build()
    vids = catalog.videos()
    vdir = Path(str(corpus.paths.data_root)) / corpus.paths.videos_dir
    vdir.mkdir(parents=True, exist_ok=True)
    for v in vids:
        (vdir / f"{v}.mp4").write_bytes(b"x")
    return catalog, vids


def test_silent_videos_do_not_trip_systemic_guard(corpus, monkeypatch):
    catalog, vids = _corpus_with_videos(corpus)

    class _Silent:
        def transcribe(self, path):
            return None                                      # no audio track at all

    monkeypatch.setattr(asr, "_build_backend", lambda s: _Silent())
    done = asr.asr_all_videos(corpus, catalog)               # must NOT raise
    assert done == len(vids)
    from cvp.utils.io import read_json
    out = read_json(corpus.paths.art("asr") / f"{vids[0]}.json")
    assert out == {"segments": [], "no_audio": True}


def test_empty_output_with_audio_never_written_and_aborts(corpus, monkeypatch):
    catalog, vids = _corpus_with_videos(corpus)

    class _EmptyDecoder:
        def transcribe(self, path):
            return []                                        # audio exists, decoder mute

    monkeypatch.setattr(asr, "_build_backend", lambda s: _EmptyDecoder())
    if len(vids) >= 5:
        with pytest.raises(RuntimeError, match="0 segments for 5 consecutive"):
            asr.asr_all_videos(corpus, catalog)
    else:
        asr.asr_all_videos(corpus, catalog)
    # the poison must never reach disk — resume would trust it forever
    assert not list((corpus.paths.art("asr")).glob("*.json"))


def test_no_video_files_at_all_raises(corpus, monkeypatch):
    from cvp.data.catalog import KeyframeCatalog

    catalog = KeyframeCatalog(corpus)
    catalog.build()
    monkeypatch.setattr(asr, "_build_backend",
                        lambda s: pytest.fail("backend must not load with no files"))
    with pytest.raises(RuntimeError, match="KHÔNG tìm thấy file video"):
        asr.asr_all_videos(corpus, catalog)                  # videos dir absent


# ── R15-2 · shared transformers-5 remote-code shim ───────────────────────────
def test_shim_gives_readonly_class_level_default():
    pytest.importorskip("transformers")
    from transformers.modeling_utils import PreTrainedModel

    from cvp.models.hf_compat import ensure_remote_code_compat

    ensure_remote_code_compat()
    keys = PreTrainedModel.all_tied_weights_keys              # exists after shim
    assert hasattr(keys, "keys")                              # dict-like contract
    if not isinstance(keys, dict):                            # our default → immutable
        with pytest.raises(TypeError):
            keys["x"] = 1                                     # loud, not silent poison


def test_shim_applied_at_all_three_vintern_load_sites():
    for rel in ("src/cvp/auxindex/captioner.py", "src/cvp/search/vqa.py",
                "src/cvp/search/vlm_rerank.py"):
        src = (REPO / rel).read_text(encoding="utf-8")
        assert "ensure_remote_code_compat()" in src, rel
        body = src.split("ensure_remote_code_compat()")[1]
        assert "from_pretrained" in body, rel                 # shim BEFORE load
        assert "load_tokenizer(" in src, rel                  # v5 slow-tokenizer fallback


def test_captions_sticky_midsweep_failure_aborts(corpus, monkeypatch):
    import pandas as pd

    from cvp.auxindex import captioner

    vids = [f"L99_V00{i}" for i in range(1, 6)]               # 5 synthetic videos

    class _Cat:
        def load(self):
            return pd.DataFrame({
                "video_id": [v for v in vids for _ in range(2)],
                "n": [1, 2] * len(vids),
                "path": [f"{v}/00{j}.jpg" for v in vids for j in (1, 2)],
            })

        def resolve_path(self, p):
            return p

    state = {"seen": set()}

    class _StickyAfterFirst:
        def caption(self, path):
            vid = Path(path).parent.name
            state["seen"].add(vid)
            if len(state["seen"]) == 1:
                return "một cảnh tin tức"                     # video 1 fine
            raise RuntimeError("poisoned CUDA context")       # sticky from video 2

    monkeypatch.setattr(captioner, "VinternCaptioner", lambda s: _StickyAfterFirst())
    with pytest.raises(RuntimeError, match="3 consecutive videos"):
        captioner.caption_all_keyframes(corpus, _Cat())
