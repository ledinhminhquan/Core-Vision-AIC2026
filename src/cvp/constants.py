"""Competition-fixed constants for the HCMC AI Challenge (AIC).

These encode the organiser-defined data and submission contracts. Do not change
them unless the organisers change the rules — everything else is configurable
in ``configs/settings.yaml``.
"""

from __future__ import annotations

import re

# ── Video / keyframe identity ────────────────────────────────────────────────
# Video ids look like L21_V001 (batch letter + 2-digit group, V + 3-digit video).
VIDEO_ID_RE = re.compile(r"^[A-Z]\d{2}_V\d{3}$")

# Keyframe files are 1-indexed zero-padded ordinals inside keyframes/{video_id}/.
# Some organiser packs use 3 digits, self-extracted packs may need more.
KEYFRAME_NAME_RE = re.compile(r"^(\d{3,5})\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE)


def keyframe_name(n: int, width: int = 3) -> str:
    """Canonical keyframe filename for ordinal ``n`` (1-indexed)."""
    return f"{n:0{width}d}.jpg"


# ── map-keyframes CSV contract ───────────────────────────────────────────────
# Columns of map-keyframes/{video_id}.csv. `frame_idx` is THE value submissions
# require; `n` is the keyframe ordinal (row n-1 of the per-video feature .npy).
MAP_KEYFRAMES_COLUMNS = ("n", "pts_time", "fps", "frame_idx")

# ── Tasks ────────────────────────────────────────────────────────────────────
TASK_KIS = "kis"
TASK_QA = "qa"
TASK_TRAKE = "trake"
ALL_TASKS = (TASK_KIS, TASK_QA, TASK_TRAKE)

# DRES submission limits: at most 100 ranked lines per query, best first.
MAX_SUBMISSION_ROWS = 100

# QA answers are graded as short text; the organiser cap is 100 characters.
MAX_QA_ANSWER_CHARS = 100

# ── Known embedding dims (informational; real dim is always probed at load) ──
KNOWN_DIMS = {
    "provided_clip32": 512,   # organiser clip-features-32 (CLIP ViT-B/32)
    "siglip2": 1152,          # google/siglip2-so400m
    "openclip_h14": 1024,     # laion CLIP-ViT-H-14
    "openclip_l14": 768,
    "mclip": 640,
}
