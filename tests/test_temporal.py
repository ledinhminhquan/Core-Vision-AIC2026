import numpy as np
import pytest

from cvp.config import Settings
from cvp.search.temporal import dante_best_sequences, dp_best_sequences, trake_search


def _settings(**temporal) -> Settings:
    return Settings.model_validate({"temporal": temporal})


def _basis(i: int, dim: int = 16) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


# ── beam DP (legacy solver) ──────────────────────────────────────────────────


def test_dp_finds_planted_ordered_sequence():
    n_frames, n_events = 40, 3
    sim = np.full((n_frames, n_events), 0.05, dtype=np.float32)
    sim[5, 0] = 0.9
    sim[18, 1] = 0.9
    sim[30, 2] = 0.9
    pts = np.arange(n_frames, dtype=np.float64) * 2.0  # 2s per keyframe
    s = _settings(max_gap_s=60.0, sim_floor=0.0, beam_size=4, gap_penalty_per_s=0.0)
    seqs = dp_best_sequences(sim, pts, s)
    assert seqs, "no sequence found"
    rows, score, per_event = seqs[0]
    assert rows == [5, 18, 30]
    assert all(a < b for a, b in zip(rows, rows[1:]))


def test_dp_respects_max_gap():
    sim = np.full((40, 2), 0.05, dtype=np.float32)
    sim[0, 0] = 0.9
    sim[39, 1] = 0.9  # 78s away — outside max_gap 20s
    pts = np.arange(40, dtype=np.float64) * 2.0
    s = _settings(max_gap_s=20.0, sim_floor=0.0, beam_size=4, gap_penalty_per_s=0.0)
    seqs = dp_best_sequences(sim, pts, s)
    for rows, _, _ in seqs:
        assert pts[rows[1]] - pts[rows[0]] <= 20.0


def test_gap_penalty_prefers_tighter_chain():
    sim = np.full((30, 2), 0.0, dtype=np.float32)
    sim[0, 0] = 0.9
    sim[2, 1] = 0.5   # close, slightly weaker
    sim[25, 1] = 0.52  # far, slightly stronger
    pts = np.arange(30, dtype=np.float64) * 2.0
    s = _settings(max_gap_s=100.0, sim_floor=0.0, beam_size=4, gap_penalty_per_s=0.01)
    seqs = dp_best_sequences(sim, pts, s)
    assert seqs[0][0] == [0, 2]  # penalty (46s * 0.01) outweighs the 0.02 sim edge


# ── DANTE DP (SOICT'25 exact O(N·T) solver) ─────────────────────────────────


def test_dante_finds_planted_ordered_sequence():
    n_frames, n_events = 40, 3
    sim = np.full((n_frames, n_events), 0.05, dtype=np.float32)
    sim[5, 0] = 0.9
    sim[18, 1] = 0.9
    sim[30, 2] = 0.9
    pts = np.arange(n_frames, dtype=np.float64) * 2.0
    s = _settings(max_gap_s=60.0, sim_floor=0.0, beam_size=4, gap_penalty_per_s=0.0)
    seqs = dante_best_sequences(sim, pts, s)
    assert seqs, "no sequence found"
    rows, score, per_event = seqs[0]
    assert list(rows) == [5, 18, 30]
    assert score == pytest.approx(0.9, abs=1e-6)
    assert per_event == pytest.approx([0.9, 0.9, 0.9], abs=1e-6)


def test_dante_matches_beam_winner_on_easy_synthetic():
    sim = np.full((50, 3), 0.02, dtype=np.float32)
    sim[4, 0] = 0.8
    sim[20, 1] = 0.7
    sim[41, 2] = 0.9
    pts = np.arange(50, dtype=np.float64) * 1.5
    s = _settings(max_gap_s=60.0, sim_floor=0.0, beam_size=6, gap_penalty_per_s=0.001)
    beam = dp_best_sequences(sim, pts, s)
    dante = dante_best_sequences(sim, pts, s)
    assert beam and dante
    assert list(dante[0][0]) == list(beam[0][0]) == [4, 20, 41]
    assert dante[0][1] == pytest.approx(beam[0][1], abs=1e-9)  # same scoring formula


def test_dante_respects_max_gap():
    sim = np.full((40, 2), 0.05, dtype=np.float32)
    sim[0, 0] = 0.9
    sim[39, 1] = 0.9  # 78s away — outside max_gap 20s
    pts = np.arange(40, dtype=np.float64) * 2.0
    s = _settings(max_gap_s=20.0, sim_floor=0.0, beam_size=4, gap_penalty_per_s=0.0)
    seqs = dante_best_sequences(sim, pts, s)
    assert seqs
    for rows, _, _ in seqs:
        assert pts[rows[1]] - pts[rows[0]] <= 20.0
        assert rows[0] < rows[1]


def test_dante_respects_min_gap():
    sim = np.full((30, 2), 0.0, dtype=np.float32)
    sim[0, 0] = 0.9
    sim[1, 1] = 0.95  # 2s away — inside min_gap, must be excluded
    sim[10, 1] = 0.6  # 20s away — valid
    pts = np.arange(30, dtype=np.float64) * 2.0
    s = _settings(min_gap_s=10.0, max_gap_s=100.0, sim_floor=0.0, beam_size=4,
                  gap_penalty_per_s=0.0)
    seqs = dante_best_sequences(sim, pts, s)
    assert seqs
    assert list(seqs[0][0]) == [0, 10]
    for rows, _, _ in seqs:
        assert pts[rows[1]] - pts[rows[0]] >= 10.0


def test_dante_gap_penalty_prefers_tighter_chain():
    sim = np.full((30, 2), 0.0, dtype=np.float32)
    sim[0, 0] = 0.9
    sim[2, 1] = 0.5   # close, slightly weaker
    sim[25, 1] = 0.52  # far, slightly stronger
    pts = np.arange(30, dtype=np.float64) * 2.0
    s = _settings(max_gap_s=100.0, sim_floor=0.0, beam_size=4, gap_penalty_per_s=0.01)
    seqs = dante_best_sequences(sim, pts, s)
    assert list(seqs[0][0]) == [0, 2]  # penalty (46s * 0.01) outweighs the 0.02 sim edge


def test_dante_sim_floor_masks_weak_frames():
    sim = np.full((30, 2), 0.0, dtype=np.float32)
    sim[0, 0] = 0.9
    sim[5, 1] = 0.45   # tighter — would win under the gap penalty if unmasked
    sim[20, 1] = 0.55
    pts = np.arange(30, dtype=np.float64) * 2.0
    common = dict(max_gap_s=100.0, beam_size=4, gap_penalty_per_s=0.02)
    unmasked = dante_best_sequences(sim, pts, _settings(sim_floor=0.0, **common))
    assert list(unmasked[0][0]) == [0, 5]  # 0.45−0.2 beats 0.55−0.8
    masked = dante_best_sequences(sim, pts, _settings(sim_floor=0.5, **common))
    assert list(masked[0][0]) == [0, 20]  # frame 5 is below the floor
    # Relaxation: a floor above every similarity must still yield sequences.
    relaxed = dante_best_sequences(sim, pts, _settings(sim_floor=0.99, **common))
    assert relaxed


def test_dante_diverse_last_frames():
    sim = np.full((20, 2), 0.05, dtype=np.float32)
    sim[0, 0] = 0.9
    sim[5, 1] = 0.8
    sim[9, 1] = 0.7
    pts = np.arange(20, dtype=np.float64) * 2.0
    s = _settings(max_gap_s=100.0, sim_floor=0.0, beam_size=4, gap_penalty_per_s=0.0)
    seqs = dante_best_sequences(sim, pts, s)
    lasts = [rows[-1] for rows, _, _ in seqs]
    assert len(lasts) == len(set(lasts))
    assert list(seqs[0][0]) == [0, 5]


# ── trake_search: algo selection + ensemble scoring ─────────────────────────


def test_trake_algo_selected_via_env_override(corpus_with_index, monkeypatch):
    from cvp.config import load_settings
    from cvp.data.catalog import KeyframeCatalog
    from cvp.index.store import IndexStore
    import cvp.search.temporal as temporal_mod

    monkeypatch.setenv("CVP_TEMPORAL__ALGO", "beam")
    env_settings = load_settings()
    assert env_settings.temporal.algo == "beam"

    # trake_search(algo=None) must dispatch on settings.temporal.algo.
    corpus_with_index.temporal.algo = env_settings.temporal.algo
    calls: list[str] = []
    real_beam, real_dante = temporal_mod.dp_best_sequences, temporal_mod.dante_best_sequences
    monkeypatch.setattr(temporal_mod, "dp_best_sequences",
                        lambda *a, **k: calls.append("beam") or real_beam(*a, **k))
    monkeypatch.setattr(temporal_mod, "dante_best_sequences",
                        lambda *a, **k: calls.append("dante") or real_dante(*a, **k))

    catalog = KeyframeCatalog(corpus_with_index)
    store = IndexStore(corpus_with_index, "fake")
    vecs = np.load(store.embedding_path("L21_V001"))
    event_vecs = np.stack([vecs[1], vecs[4]]).astype(np.float32)
    out = trake_search(event_vecs, store.search, store.embedding_path,
                       catalog, corpus_with_index)
    assert out and "beam" in calls and "dante" not in calls

    calls.clear()  # explicit keyword beats the settings value
    out = trake_search(event_vecs, store.search, store.embedding_path,
                       catalog, corpus_with_index, algo="dante")
    assert out and "dante" in calls and "beam" not in calls


def test_trake_ensemble_finds_sequence_only_visible_in_combination(corpus_with_index):
    """Member A sees only event 1, member B only event 2 in L21_V001; each member
    alone prefers its own decoy video — the weighted combination must find the
    planted L21_V001 chain (n=1 → n=4)."""
    from cvp.data.catalog import KeyframeCatalog
    from cvp.index.store import IndexStore

    catalog = KeyframeCatalog(corpus_with_index)
    q_a = np.stack([_basis(0), _basis(1)])  # member A's event encodings
    q_b = np.stack([_basis(2), _basis(3)])  # member B's event encodings
    mix = lambda i, j: (0.6 * _basis(i) + 0.8 * _basis(j)).astype(np.float32)  # noqa: E731

    member_vecs = {
        "mema": {  # sees event 1 at V001 n=1; decoy chain (0.6, 0.6) in V002
            "L21_V001": np.stack([_basis(0), _basis(8), _basis(9), _basis(10), _basis(9), _basis(8)]),
            "L21_V002": np.stack([_basis(11), mix(0, 6), _basis(11), mix(1, 7), _basis(11)]),
            "K01_V001": np.stack([_basis(12)] * 4),
        },
        "memb": {  # sees event 2 at V001 n=4; decoy chain (0.6, 0.6) in K01
            "L21_V001": np.stack([_basis(8), _basis(9), _basis(10), _basis(3), _basis(9), _basis(8)]),
            "L21_V002": np.stack([_basis(11)] * 5),
            "K01_V001": np.stack([_basis(14), mix(2, 12), _basis(15), mix(3, 13)]),
        },
    }
    stores: dict[str, IndexStore] = {}
    for name, by_vid in member_vecs.items():
        store = IndexStore(corpus_with_index, name)
        for vid, vecs in by_vid.items():
            store.embedding_path(vid).parent.mkdir(parents=True, exist_ok=True)
            np.save(store.embedding_path(vid), vecs.astype(np.float32))
        store.build(catalog)
        stores[name] = store

    # Each member alone locks onto its decoy video.
    solo_a = trake_search(q_a, stores["mema"].search, stores["mema"].embedding_path,
                          catalog, corpus_with_index)
    assert solo_a and solo_a[0].video_id == "L21_V002"
    solo_b = trake_search(q_b, stores["memb"].search, stores["memb"].embedding_path,
                          catalog, corpus_with_index)
    assert solo_b and solo_b[0].video_id == "K01_V001"

    # The 50/50 combination surfaces the cross-member chain in L21_V001.
    out = trake_search(
        q_a, stores["mema"].search, stores["mema"].embedding_path,
        catalog, corpus_with_index,
        event_vecs_by_member={"mema": q_a, "memb": q_b},
        member_weights={"mema": 0.5, "memb": 0.5},
        stores_by_member=stores,
    )
    assert out
    best = out[0]
    assert best.video_id == "L21_V001"
    assert best.ns == [1, 4]
    assert best.frame_idxs == [100, 400]
    assert best.frame_idxs[0] < best.frame_idxs[1]


def test_trake_ensemble_single_member_equals_classic_path(corpus_with_index):
    from cvp.data.catalog import KeyframeCatalog
    from cvp.index.store import IndexStore

    catalog = KeyframeCatalog(corpus_with_index)
    store = IndexStore(corpus_with_index, "fake")
    vecs = np.load(store.embedding_path("L21_V001"))
    event_vecs = np.stack([vecs[1], vecs[4]]).astype(np.float32)
    classic = trake_search(event_vecs, store.search, store.embedding_path,
                           catalog, corpus_with_index)
    via_ensemble = trake_search(
        event_vecs, store.search, store.embedding_path, catalog, corpus_with_index,
        event_vecs_by_member={"fake": event_vecs},
        stores_by_member={"fake": store},
    )
    assert classic and via_ensemble
    assert via_ensemble[0].video_id == classic[0].video_id == "L21_V001"
    assert via_ensemble[0].ns == classic[0].ns == [2, 5]
    assert via_ensemble[0].score == pytest.approx(classic[0].score, abs=1e-6)
