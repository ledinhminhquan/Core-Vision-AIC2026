"""Round-96: mốc giờ đóng cổng → thống đốc thời gian, không cần tính tay.

Sơ tuyển 3: ``PACK_DEADLINE_MIN`` để 0 vì chờ hỏi giờ đóng cổng; bão Gemini kéo
mỗi câu KIS lên 4–7 phút và 22/36 câu không kịp nộp. Giờ nb03 nhận
``CLOSE_TIME = "22:30"`` (giờ Việt Nam) và tự suy ra số phút cho thống đốc.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

VN_TZ = timezone(timedelta(hours=7))
_HHMM = re.compile(r"^\s*(\d{1,2})\s*[:hH]\s*(\d{2})\s*$")


def parse_hhmm(text: str) -> tuple[int, int]:
    m = _HHMM.match(text or "")
    if not m:
        raise ValueError(f"CLOSE_TIME phải dạng HH:MM (giờ VN), nhận {text!r}")
    h, mi = int(m.group(1)), int(m.group(2))
    if not (0 <= h <= 23 and 0 <= mi <= 59):
        raise ValueError(f"CLOSE_TIME ngoài khoảng: {text!r}")
    return h, mi


def minutes_until(close_hhmm: str, now: datetime | None = None) -> float:
    """Minutes from ``now`` (VN time) to today's ``close_hhmm``; 0 when passed.

    A close time earlier than now by more than 12 h is read as TOMORROW (a
    session opened before midnight for a 01:00 close); a close time in the
    last 12 h is 'already closed' → 0.
    """
    h, mi = parse_hhmm(close_hhmm)
    now = (now or datetime.now(VN_TZ)).astimezone(VN_TZ)
    close = now.replace(hour=h, minute=mi, second=0, microsecond=0)
    delta = (close - now).total_seconds() / 60.0
    if delta < -12 * 60:
        delta += 24 * 60
    return max(0.0, delta)


def governor_minutes(close_hhmm: str, *, fraction: float = 0.45, floor: float = 10.0,
                     reserve_min: float = 25.0, now: datetime | None = None) -> float:
    """PACK_DEADLINE_MIN derived from the close time.

    ``reserve_min`` is kept for gather + zip + upload + one rescue pass; of the
    rest, ``fraction`` runs at full quality and the remainder is the sprint
    tail. Never below ``floor`` (a value of 0 would DISABLE the governor).
    """
    left = minutes_until(close_hhmm, now)
    usable = max(0.0, left - reserve_min)
    return round(max(floor, usable * fraction), 1)
