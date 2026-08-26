"""Vietnamese keyframe captioning with Vintern-1B (offline, resumable).

Two consumers:
* BM25 ``caption`` field at search time (captions catch things OCR/objects
  miss — actions, scene types, weather, clothing colours).
* the LiT/LoRA training set for the Vietnamese text tower (caption ↔ keyframe
  pairs are exactly the supervision SigLIP-2 needs).

Output: artifacts/captions/{video_id}.json — {"n_to_caption": {"1": "..."}}
Resumable per video; safe to re-run after any crash/disconnect.
"""

from __future__ import annotations

import logging

from cvp.config import Settings
from cvp.data.catalog import KeyframeCatalog
from cvp.utils.io import atomic_write_json, read_json

log = logging.getLogger(__name__)

_CAPTION_PROMPT = (
    "<image>\nMô tả ngắn gọn khung hình này trong 1-2 câu tiếng Việt: "
    "cảnh gì, ai/cái gì xuất hiện, hành động, màu sắc nổi bật, chữ trên màn hình (nếu có)."
)


class VinternCaptioner:
    def __init__(self, settings: Settings):
        import torch
        from transformers import AutoModel

        from cvp.models.hf_compat import (
            ensure_remote_code_compat,
            load_tokenizer,
            resilient_from_pretrained,
        )

        self.settings = settings
        model_id = settings.caption.model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        log.info("Loading captioner %s (%s)", model_id, self.device)
        ensure_remote_code_compat()

        # Round-57 (audit): the shared HF cache on Drive FUSE can serve TORN
        # files (live run 12 killed the ASR load that way, and the torn bytes
        # PERSIST across reruns). ASR and SigLIP already fall back to a local
        # re-download; the captioner was the last unguarded heavy loader —
        # one bad cache file would crash-loop an entire 17-33h 06 session.
        def _load(src: str):
            model = AutoModel.from_pretrained(
                src, torch_dtype=self.dtype, trust_remote_code=True
            ).to(self.device).eval()
            return model, load_tokenizer(src)

        self.model, self.tokenizer = resilient_from_pretrained(_load, model_id)

    def caption(self, image_path: str) -> str:
        from cvp.auxindex.vintern_preprocess import load_image_tiles

        pixel_values = load_image_tiles(image_path, max_num=self.settings.caption.max_tiles)
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        gen_cfg = dict(max_new_tokens=self.settings.caption.max_new_tokens, do_sample=False, num_beams=2)
        out = self.model.chat(self.tokenizer, pixel_values, _CAPTION_PROMPT, gen_cfg)
        return str(out).strip()


def caption_all_keyframes(settings: Settings, catalog: KeyframeCatalog,
                          videos: list[str] | None = None,
                          stride: int = 1, overwrite: bool = False) -> int:
    """Caption all (or every ``stride``-th) keyframe per video; skip finished videos."""
    out_dir = settings.paths.art("captions")
    df = catalog.load()
    todo_videos = videos or [str(v) for v in df["video_id"].unique()]
    captioner = None
    done = 0
    consecutive_video_failures = 0
    for vid in todo_videos:
        out_path = out_dir / f"{vid}.json"
        grp = df[df["video_id"] == vid].sort_values("n")
        wanted = grp.iloc[::stride]
        if out_path.exists() and not overwrite:
            existing = read_json(out_path, default={}) or {}
            # done == every wanted frame was PROCESSED (empty captions are valid)
            done_count = existing.get("processed_count", len(existing.get("n_to_caption", {})) or -1)
            if done_count >= len(wanted):
                continue
        if captioner is None:
            captioner = VinternCaptioner(settings)
        n_to_caption: dict[str, str] = {}
        processed = 0
        last_err: Exception | None = None
        for _, row in wanted.iterrows():
            try:
                cap = captioner.caption(catalog.resolve_path(str(row["path"])))
            except Exception as e:  # noqa: BLE001 — one bad frame must not kill hours of work
                last_err = e
                log.warning("Caption failed on %s n=%s: %s", vid, row["n"], e)
                continue
            processed += 1
            if cap:
                n_to_caption[str(int(row["n"]))] = cap
        if vid == todo_videos[0] and len(wanted) > 0 and processed == 0 and last_err is not None:
            # Round-13: all-fail on the FIRST video = systemic breakage (e.g.
            # a transformers-5 semantics change inside the remote code) — fail
            # loud now, not after hours of empty artifacts.
            # Round-61 (audit): anchored on the first video of the todo LIST —
            # a single video with corrupt local keyframes gets re-attempted
            # first on every rerun (the rest resume-skip) and must not be
            # misdiagnosed as systemic, wedging the shard forever.
            raise RuntimeError(
                f"Captioning failed on every frame of the first video ({vid}) — "
                "systemic failure, aborting the sweep"
            ) from last_err
        if len(wanted) > 0 and processed == 0 and last_err is not None:
            # verify-R15: a STICKY mid-sweep failure (poisoned CUDA context,
            # dead keyframe path) fails every frame of every later video — the
            # first-video guard alone would let it burn the whole session.
            consecutive_video_failures += 1
            if consecutive_video_failures >= 3:
                raise RuntimeError(
                    "Captioning failed on every frame of 3 consecutive videos "
                    f"(latest: {vid}) — systemic failure, aborting the sweep"
                ) from last_err
        else:
            consecutive_video_failures = 0
        atomic_write_json(out_path, {"n_to_caption": n_to_caption, "processed_count": processed})
        done += 1
        log.info("Captions %s: %d frames (%d/%d videos)", vid, len(n_to_caption), done, len(todo_videos))
    return done
