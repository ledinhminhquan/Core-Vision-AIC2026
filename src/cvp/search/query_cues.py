"""Round-88: query-conditioned fusion weights (heuristic, zero API).

A query that quotes on-screen text ("dòng chữ", "biển hiệu", "tiêu đề") is
decided by the OCR field; one that quotes speech ("phát biểu", "nói rằng",
"bài hát") by ASR. The fusion weights are tuned GLOBALLY over the drill pack,
so for exactly those queries the decisive signal is under-weighted. When
``search.query_adaptive_weights`` is on, cue patterns scale the OCR / ASR
fusion weight per query (default ×2.0); a neutral query is bit-identical to
off. Cues are matched with diacritics on word boundaries — folding would
merge "nói" (speak) with "nơi" (place), and a bare substring would fire
"hát" (sing) inside "nhất" (most). Bare "nói" is deliberately NOT a cue:
"đang nói chuyện" describes a visual scene, not quoted speech.
"""
from __future__ import annotations

import re
import unicodedata

_OCR_CUES = (
    # Vietnamese — on-screen text is being quoted / asked about
    "dòng chữ", "chữ viết", "hàng chữ", "có chữ", "ghi chữ", "chữ trên", "chữ in",
    "biển hiệu", "biển báo", "biển số", "tấm biển", "bảng hiệu", "bảng tên",
    "bảng chữ", "tấm bảng", "bảng điện tử", "bảng thông tin", "bảng số liệu",
    "tiêu đề là", "tiêu đề của", "tiêu đề ghi", "tiêu đề có", "dòng tiêu đề",
    "phụ đề", "logo", "banner", "băng rôn ghi", "băng rôn có chữ",
    "băng rôn với dòng chữ", "chữ trên băng rôn", "nội dung băng rôn", "câu thơ",
    "khẩu hiệu", "slogan", "nhãn hiệu", "tên cửa hàng", "tên quán",
    "tên công ty", "tên đường", "được ghi trên", "được ghi là", "được ghi trong",
    "được ghi ở", "ghi là", "viết là", "viết trên", "ký tự", "dòng text", "poster",
    "áp phích", "màn hình ghi", "được hiển thị", "số hiển thị", "hiển thị số",
    "hiển thị dòng", "hiển thị chữ", "hiển thị thông tin", "hiển thị giá",
    "hiển thị tên",
    # English — the prepared / enhanced-text path
    "the text", "text on", "written", "sign says", "sign reads", "subtitle",
    "subtitles", "caption", "headline", "signboard",
    # NOT cues (audit r88, reproduced false positives): bare "được ghi" ("được
    # ghi lại từ trên cao" = filmed), "số xe" ("một số xe" = some cars), "in trên"
    # ("in trên áo" = printed), bare "hiển thị" ("màn hình hiển thị hình ảnh"),
    # bare "reads" ("a man reads a newspaper"); audit r88b: bare "tiêu đề"
    # ("thanh tiêu đề xanh dương chứa họa tiết" = a title BAR) and bare "băng
    # rôn" ("băng rôn ... có hình ảnh" = a banner's pictures) are visual.
)
_ASR_CUES = (
    # Vietnamese — speech is being quoted / asked about
    "nói rằng", "nói về", "nói là", "nói câu", "nói đến", "nói gì", "phát biểu",
    "lời thoại", "câu thoại", "đoạn thoại", "đối thoại", "bài hát", "hát bài", "đang hát",
    "hát rằng", "lời hát", "tiếng hát", "kể về", "kể rằng", "tuyên bố", "chia sẻ",
    "phỏng vấn", "bình luận", "nhắc đến", "đề cập", "người dẫn chương trình",
    "người dẫn chuyện", "thuyết minh", "phát ngôn", "kêu gọi", "giọng nói",
    "lời nói", "nhấn mạnh", "khẳng định", "diễn giả", "lời bài hát",
    # English
    "says", "said", "saying", "speaks", "speech", "talks about", "narrator",
    "narrates", "interview", "interviewed", "mentions", "announces", "sings",
    "lyrics",
    # NOT cues (checked on 79 real AIC queries, round-88): "giới thiệu" names
    # the segment type ("mẩu tin giới thiệu về đàn hổ"), "cho biết"/"trả lời"
    # are question idioms ("hãy cho biết loài cây…"); audit r88 reproduced
    # bare "hát" inside "nhà hát" (theatre), English "song" = Vietnamese "song
    # song" (parallel), "người dẫn" inside "người dẫn đầu" (the leader).
)


def _compile(cues: tuple[str, ...]) -> re.Pattern[str]:
    alts = "|".join(re.escape(c) for c in sorted(cues, key=len, reverse=True))
    return re.compile(r"(?<!\w)(?:%s)(?!\w)" % alts)


_OCR_RE = _compile(_OCR_CUES)
_ASR_RE = _compile(_ASR_CUES)


def query_signal_cues(text: str) -> set[str]:
    """Which text signals the query explicitly leans on: subset of {ocr, asr}."""
    if not text:
        return set()
    t = unicodedata.normalize("NFC", text).lower()
    out: set[str] = set()
    if _OCR_RE.search(t):
        out.add("ocr")
    if _ASR_RE.search(t):
        out.add("asr")
    return out


def adaptive_weight_scale(text: str, *, ocr_boost: float = 2.0,
                          asr_boost: float = 2.0) -> dict[str, float]:
    """Per-query multipliers for the fusion weights ({} = leave every weight)."""
    cues = query_signal_cues(text)
    scale: dict[str, float] = {}
    if "ocr" in cues and ocr_boost != 1.0:
        scale["ocr"] = float(ocr_boost)
    if "asr" in cues and asr_boost != 1.0:
        scale["asr"] = float(asr_boost)
    return scale
