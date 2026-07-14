import pytest

from cvp.submission.writer import sanitize_answer, write_kis, write_qa, write_trake


def test_kis_dedup_and_cap(tmp_path):
    ranked = [("L21_V001", 100), ("L21_V001", 100)] + [("L21_V002", i) for i in range(200)]
    p = write_kis(tmp_path / "kis.csv", ranked)
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 100
    assert lines[0] == "L21_V001,100"
    assert lines[1] == "L21_V002,0"


def test_kis_rejects_bad_video_id(tmp_path):
    # an ENTIRELY-invalid input still raises (nothing valid to write)
    with pytest.raises(ValueError):
        write_kis(tmp_path / "x.csv", [("badid", 1)])


def test_writers_skip_invalid_rows_with_warning(tmp_path, caplog):
    # one bad candidate must not abort the whole 100-row export (FIX: unified
    # skip-with-warning across write_kis / write_qa / write_trake)
    with caplog.at_level("WARNING", logger="cvp.submission.writer"):
        p = write_kis(tmp_path / "kis.csv", [("badid", 1), ("L21_V001", -5), ("L21_V002", 7)])
    assert p.read_text(encoding="utf-8").strip() == "L21_V002,7"
    assert sum("KIS row skipped" in r.message for r in caplog.records) == 2

    caplog.clear()
    with caplog.at_level("WARNING", logger="cvp.submission.writer"):
        p2 = write_qa(tmp_path / "qa.csv", [("badid", 1, "hai"), ("L21_V001", 5, "ba")])
    assert p2.read_text(encoding="utf-8").strip() == "L21_V001,5,ba"
    assert any("QA row skipped" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level("WARNING", logger="cvp.submission.writer"):
        p3 = write_trake(tmp_path / "t.csv", [("badid", [1, 2]), ("K01_V001", [-1, 2]),
                                              ("K01_V002", [5, 6])])
    assert p3.read_text(encoding="utf-8").strip() == "K01_V002,5,6"
    assert sum("TRAKE row skipped" in r.message for r in caplog.records) == 2


def test_writers_raise_when_every_candidate_invalid(tmp_path):
    with pytest.raises(ValueError, match="invalid"):
        write_qa(tmp_path / "qa.csv", [("badid", 1, "hai"), ("L21_V001", -5, "ba")])
    with pytest.raises(ValueError, match="invalid"):
        write_trake(tmp_path / "t.csv", [("K01_V001", [10, 10]), ("badid", [1, 2])])
    # an empty input is not "entirely invalid" — it still writes an empty file
    p = write_kis(tmp_path / "empty.csv", [])
    assert p.read_text(encoding="utf-8") == ""


def test_qa_answer_sanitization(tmp_path):
    p = write_qa(tmp_path / "qa.csv", [("L21_V001", 5, 'xã "Cam, Hải" Đông\nOK')])
    line = p.read_text(encoding="utf-8").strip()
    assert line.startswith("L21_V001,5,")
    assert "\n" not in line
    assert '"xã ""Cam, Hải"" Đông OK"' in line


def test_sanitize_answer_plain():
    assert sanitize_answer("  42  ") == "42"


def test_sanitize_answer_nfc_normalization():
    # VQA output may arrive decomposed (NFD); the organisers grade composed text.
    import unicodedata

    decomposed = unicodedata.normalize("NFD", "màu xanh")
    assert decomposed != "màu xanh"  # really is a different byte sequence
    assert sanitize_answer(decomposed) == "màu xanh"
    assert unicodedata.is_normalized("NFC", sanitize_answer(decomposed))


def test_sanitize_answer_truncates_overlong_to_organiser_cap(caplog):
    # Fix L3 (review 2026-07-08): a hand-typed UI answer >100 chars must be
    # truncated (with a warning), never written into a rejectable CSV.
    from cvp.constants import MAX_QA_ANSWER_CHARS

    long_answer = "một " * 40  # 160 chars
    with caplog.at_level("WARNING", logger="cvp.submission.writer"):
        out = sanitize_answer(long_answer)
    assert len(out) <= MAX_QA_ANSWER_CHARS
    assert any("truncated" in r.message for r in caplog.records)
    # exactly at the cap → untouched, no warning
    caplog.clear()
    exact = "x" * MAX_QA_ANSWER_CHARS
    assert sanitize_answer(exact) == exact
    assert not caplog.records


def test_sanitize_answer_truncation_happens_before_csv_quoting():
    # The 100-char cap applies to the ANSWER text; CSV quoting is added after,
    # so a truncated comma-carrying answer still round-trips through csv.
    import csv
    import io

    from cvp.constants import MAX_QA_ANSWER_CHARS

    long_with_comma = ("a," * 80)  # 160 chars, full of commas
    out = sanitize_answer(long_with_comma)
    parsed = next(csv.reader(io.StringIO(out)))
    assert len(parsed) == 1
    assert len(parsed[0]) <= MAX_QA_ANSWER_CHARS


def test_trake_strictly_increasing(tmp_path):
    p = write_trake(tmp_path / "t.csv", [("K01_V001", [10, 20, 30])])
    assert p.read_text(encoding="utf-8").strip() == "K01_V001,10,20,30"
    # invalid sequences are skipped, not fatal — the rest of the export survives
    p2 = write_trake(
        tmp_path / "t2.csv",
        [("K01_V001", [10, 10, 30]), ("K01_V002", [5, 6, 7])],
    )
    assert p2.read_text(encoding="utf-8").strip() == "K01_V002,5,6,7"
