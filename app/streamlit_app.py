"""Core Vision Perfect V1 — competition UI.

Run:  streamlit run app/streamlit_app.py
Env:  CVP_PATHS__DATA_ROOT=... CVP_EMBEDDING__MODEL=siglip2|ensemble|finetuned
      GEMINI_API_KEY=... (optional — query enhancement, VQA, KIS-C assistant)

Tabs = tasks: KIS · QA · TRAKE · AVS · KIS-C (chat). Every tab shares the same
result grid with per-tile actions (similar / context / basket) and a
per-task submission basket exporting the exact Codabench/DRES CSV format.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from cvp.config import load_settings  # noqa: E402
from cvp.utils.logging import setup_logging  # noqa: E402

st.set_page_config(page_title="Core Vision Perfect V1", layout="wide", page_icon="🎯")


# ── cached singletons ────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading search engine (models + index)...")
def get_engine():
    from cvp.search.engine import SearchEngine

    settings = load_settings()
    setup_logging(settings.logging.level)
    return SearchEngine(settings)


@st.cache_resource
def get_media_store():
    from cvp.data.metadata import MediaInfoStore

    return MediaInfoStore(load_settings())


@st.cache_resource
def get_vqa():
    from cvp.search.vqa import VqaAssistant

    return VqaAssistant(load_settings())


@st.cache_resource
def get_assistant():
    from cvp.models.agent import ConversationalAssistant

    return ConversationalAssistant(load_settings())


def _init_state() -> None:
    defaults = {
        "results": [],           # list[SearchResult] for KIS/QA/AVS
        "results_task": None,    # which tab produced `results` (export guard, M-R3-3)
        "trake_results": [],
        "basket_kis": [],        # (video_id, frame_idx, global_id)
        "basket_qa": [],         # (video_id, frame_idx, answer, global_id)
        "basket_trake": [],      # (video_id, [frames])
        "basket_avs": [],
        "marks_pos": set(),      # feedback global_ids
        "marks_neg": set(),
        "last_query": "",
        "kisc_hints": [],
        "kisc_questions": [],
        "kisc_started_at": None,   # progressive-KIS 5-minute clock anchor
        "vqa_suggestions": {},
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


# ── shared widgets ───────────────────────────────────────────────────────────


def group_results_by_video(results) -> list:
    """VISIONE'23-style ordering: videos by their best rank, frames of each
    video CONSECUTIVE in temporal order. One glance judges a whole video —
    a flat grid wastes screen on interleaved near-identical frames (organiser
    tập-huấn buổi 2 teaches exactly this display pattern)."""
    order: list[str] = []
    by_video: dict[str, list] = {}
    for r in results:
        vid = r.video_id
        if vid not in by_video:
            by_video[vid] = []
            order.append(vid)
        by_video[vid].append(r)
    out = []
    for vid in order:
        out.extend(sorted(by_video[vid], key=lambda r: r.ref.n))
    return out


def render_result_grid(results, cols: int, task: str, engine, qa_answer: str = "") -> None:
    if not results:
        return
    if st.session_state.get("group_by_video"):
        results = group_results_by_video(results)
    for row_start in range(0, len(results), cols):
        columns = st.columns(cols)
        for col, r in zip(columns, results[row_start : row_start + cols]):
            with col:
                try:
                    st.image(r.ref.path, use_container_width=True)
                except Exception:
                    st.caption("(image unavailable)")
                mm, ss = divmod(int(r.ref.pts_time), 60)
                st.caption(
                    f"**{r.video_id}** · n={r.ref.n} · frame **{r.frame_idx}** · "
                    f"{mm:02d}:{ss:02d} · s={r.score:.3f}"
                )
                # 💡 chỉ có nghĩa cho tab QA — các tab khác không có câu hỏi
                # (round-3 fix L-R3-1: hết gợi ý cũ lạc chỗ trên KIS/AVS/KIS-C).
                if task == "qa" and st.session_state.vqa_suggestions.get(r.global_id):
                    st.caption(f"💡 {st.session_state.vqa_suggestions[r.global_id]}")
                url = get_media_store().watch_url(r.video_id, r.ref.pts_time)
                if url:
                    st.caption(f"[▶ YouTube @{mm:02d}:{ss:02d}]({url})")
                b1, b2, b3, b4, b5 = st.columns(5)
                gid = r.global_id
                if b1.button("🧺", key=f"add-{task}-{gid}", help="Add to basket"):
                    _add_to_basket(task, r, qa_answer)
                if b2.button("🔍", key=f"sim-{task}-{gid}", help="Find similar frames"):
                    _set_results(engine.nearest(gid, k=60), task)
                    st.rerun()
                if b3.button("🎞", key=f"ctx-{task}-{gid}", help="Show video context"):
                    _show_context(engine, gid)
                if b4.button("✓", key=f"pos-{task}-{gid}", help="Mark relevant (feedback)"):
                    st.session_state.marks_pos.add(gid)
                if b5.button("✗", key=f"neg-{task}-{gid}", help="Mark irrelevant (feedback)"):
                    st.session_state.marks_neg.add(gid)


def render_concept_chips(results, engine, top_n: int = 30) -> None:
    """Exploitation concepts (organiser buổi-2 recipe): the most frequent
    detected objects among the top results — one glance tells the operator
    which countable/visible nouns to ADD to the query to discriminate."""
    if not results:
        return
    store = getattr(getattr(engine, "object_booster", None), "store", None)
    if store is None:
        return
    counts: dict[str, int] = {}
    try:
        for r in results[:top_n]:
            for d in store.get(r.video_id, r.ref.n):
                if d.score >= 0.4:
                    counts[d.entity] = counts.get(d.entity, 0) + 1
    except Exception:  # noqa: BLE001 — chips are advisory, never blocking
        return
    if counts:
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:10]
        st.caption("🧩 Concepts trong top kết quả (thêm vào mô tả để phân biệt / loại trừ): "
                   + " · ".join(f"{e}×{c}" for e, c in top))


@st.cache_resource
def get_episodic_log():
    from cvp.models.agent import EpisodicLog

    return EpisodicLog(load_settings())


@st.fragment(run_every="1s")
def _kisc_clock() -> None:
    """Progressive-KIS 5-minute countdown, ticking in REAL TIME.

    ``st.fragment(run_every=…)`` reruns only this block every second, so the
    clock moves without any user interaction (a plain Streamlit block only
    updates on reruns). Starts at the first hint, resets with the dialogue.
    """
    if not (st.session_state.get("kisc_hints") and st.session_state.get("kisc_started_at")):
        return
    elapsed = time.monotonic() - st.session_state.kisc_started_at
    remaining = max(0.0, 300.0 - elapsed)
    (st.error if remaining < 60 else st.warning if remaining < 150 else st.info)(
        f"⏱ {int(remaining // 60)}:{int(remaining % 60):02d} còn lại trong 5 phút"
    )


@st.dialog("Video context", width="large")
def _show_context(engine, gid: int) -> None:
    refs = engine.temporal_neighbors(gid, window=10)
    cols = st.columns(7)
    for i, ref in enumerate(refs):
        with cols[i % 7]:
            try:
                st.image(ref.path, use_container_width=True)
            except Exception:
                st.caption("(x)")
            marker = "▶ " if ref.global_id == gid else ""
            st.caption(f"{marker}n={ref.n} f={ref.frame_idx}")


def _set_results(results, task: str) -> None:
    """Replace the shared ranking + record WHICH tab produced it.

    The provenance tag is the export guard (round-3 fix M-R3-3): without it,
    clicking another tab's Export writes a foreign ranking as that task's
    submission. Stale VQA suggestions die with the old ranking (L-R3-1).
    """
    st.session_state.results = results
    st.session_state.results_task = task
    st.session_state.vqa_suggestions = {}


def _add_to_basket(task: str, r, qa_answer: str = "") -> None:
    if task == "qa":
        st.session_state.basket_qa.append((r.video_id, r.frame_idx, qa_answer, r.global_id))
    elif task == "avs":
        st.session_state.basket_avs.append((r.video_id, r.frame_idx, r.global_id))
    else:
        st.session_state.basket_kis.append((r.video_id, r.frame_idx, r.global_id))
    st.toast(f"Added {r.video_id} / {r.frame_idx}")


def _export_path(out_dir: Path, task: str) -> Path:
    """Date-qualified, collision-proof export filename (round-3 fix L-R3-3).

    The old ``HHMMSS`` stamp silently overwrote same-second re-exports and
    same-time-of-day exports across prelim DAYS — losing the audit trail of
    what was actually submitted.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{task}-{stamp}.csv"
    n = 2
    while path.exists():
        path = out_dir / f"{task}-{stamp}-{n}.csv"
        n += 1
    return path


def _current_results_for(task: str) -> list:
    """The shared ranking, ONLY if it was produced by this task's tab.

    Round-3 fix M-R3-3: KIS/QA/AVS/KIS-C share one results buffer; exporting
    another tab's ranking as this task's CSV is a silent wrong-task submission
    (5 prelim submissions/day make that expensive). KIS-C rankings export as
    KIS — same row format, same task family.
    """
    produced_by = st.session_state.get("results_task")
    compatible = {"kis": ("kis", "kisc"), "qa": ("qa",), "avs": ("avs",)}.get(task, (task,))
    if produced_by in compatible:
        return st.session_state.results
    if st.session_state.results:
        st.warning(
            f"Kết quả đang hiển thị thuộc tab **{produced_by or '?'}** — không tự thêm vào "
            f"CSV {task.upper()}. Chỉ export basket; hãy search lại trong tab này nếu muốn."
        )
    return []


def _export(task: str, engine) -> None:
    from cvp.submission.writer import write_kis, write_qa, write_trake

    settings = engine.settings
    out_dir = settings.paths.art("submissions")
    path = None
    if task == "kis":
        results = _current_results_for("kis")
        if st.session_state.basket_kis or results:
            ranked = [(v, f) for v, f, _ in st.session_state.basket_kis]
            ranked += [(r.video_id, r.frame_idx) for r in results]
            path = write_kis(_export_path(out_dir, "kis"), ranked)
    elif task == "qa":
        results = _current_results_for("qa")
        if st.session_state.basket_qa or results:
            answer = st.session_state.get("qa_answer_input", "")
            ranked = [(v, f, a or answer) for v, f, a, _ in st.session_state.basket_qa]
            ranked += [(r.video_id, r.frame_idx, answer) for r in results]
            path = write_qa(_export_path(out_dir, "qa"), ranked)
    elif task == "trake":
        if st.session_state.basket_trake or st.session_state.trake_results:
            ranked = list(st.session_state.basket_trake)
            ranked += [(c.video_id, c.frame_idxs) for c in st.session_state.trake_results]
            path = write_trake(_export_path(out_dir, "trake"), ranked)
    elif task == "avs":
        results = _current_results_for("avs")
        if st.session_state.basket_avs or results:
            ranked = [(v, f) for v, f, _ in st.session_state.basket_avs]
            ranked += [(r.video_id, r.frame_idx) for r in results]
            path = write_kis(_export_path(out_dir, "avs"), ranked)
    if path:
        st.success(f"Exported → {path}")
        st.code(Path(path).read_text(encoding="utf-8")[:1500])
    else:
        st.warning("Nothing to export — search or add tiles to the basket first.")


# ── main app ─────────────────────────────────────────────────────────────────


def main() -> None:
    _init_state()
    engine = get_engine()

    with st.sidebar:
        st.title("🎯 Core Vision Perfect V1")
        st.caption(
            f"{len(engine.catalog)} keyframes · members: "
            + ", ".join(m.key for m, _ in engine.members)
        )
        cols = st.slider("Grid columns", 3, 8, 5)
        display_k = st.slider("Results shown", 20, 300, engine.settings.search.display_k, step=20)
        st.checkbox("📺 Group by video (VISIONE-style)", key="group_by_video",
                    help="Videos ordered by best rank; each video's frames "
                         "consecutive in temporal order — judge a whole video at a glance.")

    tab_kis, tab_qa, tab_trake, tab_avs, tab_chat = st.tabs(
        ["🔎 KIS", "❓ QA", "⛓ TRAKE", "🌊 AVS", "💬 KIS-C"]
    )

    # ── KIS ──────────────────────────────────────────────────────────────
    with tab_kis:
        with st.form("kis_form"):
            q = st.text_input("Mô tả cảnh cần tìm (tiếng Việt hoặc English)", key="kis_query")
            c1, c2, c3 = st.columns([1, 1, 4])
            do_search = c1.form_submit_button("🔎 Search", use_container_width=True)
            do_feedback = c2.form_submit_button("♻ Refine (feedback)", use_container_width=True)
        if do_search and q.strip():
            t0 = time.time()
            _set_results(engine.search_text(q, display_k=display_k), "kis")
            st.session_state.last_query = q
            st.session_state.marks_pos, st.session_state.marks_neg = set(), set()
            st.caption(f"{len(st.session_state.results)} results in {time.time() - t0:.2f}s")
        if do_feedback and st.session_state.last_query:
            _set_results(engine.search_with_feedback(
                st.session_state.last_query,
                list(st.session_state.marks_pos),
                list(st.session_state.marks_neg),
                display_k=display_k,
            ), "kis")
            st.caption(
                f"Refined with {len(st.session_state.marks_pos)}✓ / {len(st.session_state.marks_neg)}✗"
            )
        with st.expander("🖼 KIS-V — tìm bằng ẢNH (clip chỉ được XEM: vẽ/sinh ảnh lại rồi tải lên)"):
            up = st.file_uploader("Ảnh truy vấn", type=["png", "jpg", "jpeg", "webp"],
                                  key="kisv_upload")
            if up is not None and st.button("🔎 Search bằng ảnh", key="kisv_go"):
                from PIL import Image as _Img

                t0 = time.time()
                img = _Img.open(up).convert("RGB")
                _set_results(engine.search_image(img, display_k=display_k), "kis")
                st.caption(f"{len(st.session_state.results)} results in "
                           f"{time.time() - t0:.2f}s (image query, ensemble-fused)")
        if st.button("📤 Export KIS CSV"):
            _export("kis", engine)
        render_result_grid(st.session_state.results, cols, "kis", engine)
        render_concept_chips(st.session_state.results, engine)

    # ── QA ───────────────────────────────────────────────────────────────
    with tab_qa:
        with st.form("qa_form"):
            qd = st.text_input("Mô tả cảnh (để tìm khung hình)", key="qa_desc")
            qq = st.text_input("Câu hỏi", key="qa_question")
            c1, c2 = st.columns([1, 1])
            do_search = c1.form_submit_button("🔎 Search")
            do_vqa = c2.form_submit_button("💡 Suggest answers (VQA)")
        answer = st.text_input("Đáp án sẽ ghi vào CSV (≤100 ký tự)", key="qa_answer_input")
        if do_search and qd.strip():
            _set_results(engine.search_text(qd, display_k=display_k), "qa")
        if do_vqa and st.session_state.results and qq.strip():
            try:
                suggestions = get_vqa().suggest(
                    qq, [(r.global_id, r.ref.path) for r in st.session_state.results[:5]]
                )
                st.session_state.vqa_suggestions = {s.global_id: s.answer for s in suggestions}
                if suggestions:
                    st.info("Suggestions (verify before submitting!): "
                            + " | ".join(s.answer for s in suggestions))
            except Exception as e:  # noqa: BLE001
                st.warning(f"VQA unavailable: {e}")
        if st.button("📤 Export QA CSV"):
            _export("qa", engine)
        render_result_grid(st.session_state.results, cols, "qa", engine, qa_answer=answer)

    # ── TRAKE ────────────────────────────────────────────────────────────
    with tab_trake:
        with st.form("trake_form"):
            events_text = st.text_area(
                "Các sự kiện theo THỨ TỰ, mỗi dòng một sự kiện",
                height=120, key="trake_events",
                placeholder="vận động viên chuẩn bị chạy đà\nvận động viên giậm nhảy\nvận động viên tiếp đất",
            )
            do_search = st.form_submit_button("⛓ Search sequences")
        if do_search:
            events = [ln.strip() for ln in events_text.splitlines() if ln.strip()]
            if len(events) >= 2:
                t0 = time.time()
                st.session_state.trake_results = engine.search_trake(events)
                st.caption(f"{len(st.session_state.trake_results)} sequences in {time.time() - t0:.1f}s")
            else:
                st.warning("Nhập ít nhất 2 sự kiện.")
        if st.button("📤 Export TRAKE CSV"):
            _export("trake", engine)
        for i, cand in enumerate(st.session_state.trake_results[:25]):
            st.markdown(
                f"**#{i + 1} · {cand.video_id}** · score {cand.score:.3f} · frames {cand.frame_idxs}"
            )
            refs = engine.catalog.video_rows(cand.video_id)
            columns = st.columns(len(cand.ns))
            for col, n, fi, pt in zip(columns, cand.ns, cand.frame_idxs, cand.pts_times):
                row = refs[refs["n"] == n]
                with col:
                    if not row.empty:
                        try:
                            st.image(engine.catalog.resolve_path(str(row.iloc[0]["path"])),
                                     use_container_width=True)
                        except Exception:
                            st.caption("(x)")
                    st.caption(f"n={n} · f={fi} · {int(pt) // 60:02d}:{int(pt) % 60:02d}")
            if st.button("🧺 Add sequence", key=f"trake-add-{i}"):
                st.session_state.basket_trake.append((cand.video_id, cand.frame_idxs))
                st.toast("Sequence added")

    # ── AVS ──────────────────────────────────────────────────────────────
    with tab_avs:
        with st.form("avs_form"):
            q = st.text_input("Mô tả chung — tìm CÀNG NHIỀU cảnh khớp càng tốt", key="avs_query")
            c1, c2 = st.columns(2)
            cap = c1.number_input("Max kết quả / video", 1, 10, 3)
            gap = c2.number_input("Khoảng cách tối thiểu giữa 2 cảnh cùng video (s)", 1.0, 120.0, 10.0)
            do_search = st.form_submit_button("🌊 Search (diversified)")
        if do_search and q.strip():
            _set_results(
                engine.search_avs(q, per_video_cap=int(cap), min_gap_s=float(gap)), "avs"
            )
            st.caption(
                f"{len(st.session_state.results)} diversified results across "
                f"{len({r.video_id for r in st.session_state.results})} videos"
            )
        if st.button("📤 Export AVS CSV"):
            _export("avs", engine)
        render_result_grid(st.session_state.results, cols, "avs", engine)

    # ── KIS-C (conversational) ───────────────────────────────────────────
    with tab_chat:
        st.caption(
            "Progressive/Conversational KIS: thêm từng gợi ý khi ban tổ chức tiết lộ; "
            "trợ lý gộp MỌI gợi ý thành một truy vấn và đề xuất câu hỏi làm rõ."
        )
        # 5-minute finals clock — real-time tick via st.fragment(run_every).
        _kisc_clock()
        for i, h in enumerate(st.session_state.kisc_hints):
            st.markdown(f"**Hint {i + 1}.** {h}")
        with st.form("kisc_form", clear_on_submit=True):
            new_hint = st.text_input("Gợi ý mới / câu trả lời làm rõ")
            c1, c2 = st.columns(2)
            do_add = c1.form_submit_button("➕ Add hint & search")
            do_reset = c2.form_submit_button("🗑 Reset dialogue")
        if do_reset:
            st.session_state.kisc_hints, st.session_state.kisc_questions = [], []
            _set_results([], "kisc")
            st.session_state.kisc_started_at = None
        if do_add and new_hint.strip():
            import time as _time

            from cvp.models.agent import DialogueState

            if not st.session_state.get("kisc_started_at"):
                st.session_state.kisc_started_at = _time.monotonic()
            st.session_state.kisc_hints.append(new_hint.strip())
            # Result-aware clarification: a summary of the current top hits lets
            # the assistant ask questions that discriminate between them.
            summary = None
            if st.session_state.results:
                summary = "; ".join(
                    f"{r.video_id}@{r.ref.pts_time:.0f}s" for r in st.session_state.results[:8]
                )
            turn = get_assistant().step(
                DialogueState(hints=list(st.session_state.kisc_hints)),
                results_summary=summary,
            )
            st.session_state.kisc_questions = turn.questions
            st.markdown(f"**Truy vấn gộp:** {turn.query}")
            _set_results(engine.search_text(turn.query, display_k=display_k), "kisc")
            st.session_state.last_query = turn.query
            # Marks made on a previous KIS query must not refine THIS query
            # (round-3 fix L-R3-2 — last_query changed, so the marks expire).
            st.session_state.marks_pos, st.session_state.marks_neg = set(), set()
            # Episodic memory (organiser buổi-3 recipe): every hint + merged
            # query + top hits go to the append-only session log, so later
            # turns can resolve "tìm lại video giống kết quả lúc nãy".
            get_episodic_log().append(
                "kisc_turn", hint=new_hint.strip(), merged_query=turn.query,
                top=[f"{r.video_id}@{int(r.ref.pts_time)}s"
                     for r in st.session_state.results[:5]],
            )
        if st.session_state.kisc_questions:
            st.info("❓ Câu hỏi làm rõ nên đặt: " + " · ".join(st.session_state.kisc_questions))
        if st.button("📤 Export KIS CSV", key="kisc_export"):
            _export("kis", engine)
        render_result_grid(st.session_state.results, cols, "kisc", engine)

    # Basket counters render AFTER the tab handlers ran, so they show the
    # post-click state in the same rerun (round-3 fix C-R3-2). Streamlit lets
    # the sidebar be appended to at any point of the script.
    with st.sidebar:
        st.divider()
        st.subheader("🧺 Baskets")
        st.caption(
            f"KIS {len(st.session_state.basket_kis)} · QA {len(st.session_state.basket_qa)} · "
            f"TRAKE {len(st.session_state.basket_trake)} · AVS {len(st.session_state.basket_avs)}"
        )
        if st.button("Clear all baskets"):
            for key in ("basket_kis", "basket_qa", "basket_trake", "basket_avs"):
                st.session_state[key] = []
            st.rerun()
        if st.button("Clear feedback marks"):
            st.session_state.marks_pos, st.session_state.marks_neg = set(), set()
            st.rerun()


main()
