from cvp.utils.text import fold_diacritics, normalize_text, tokenize_vi


def test_fold_diacritics():
    assert fold_diacritics("Hà Nội") == "Ha Noi"
    assert fold_diacritics("đường Đinh Tiên Hoàng") == "duong Dinh Tien Hoang"


def test_normalize_strips_punct_and_case():
    assert normalize_text("Xin CHÀO,   bạn!") == "xin chào bạn"


def test_tokenize_folds_by_default():
    assert tokenize_vi("Hà Nội, đường phố!") == ["ha", "noi", "duong", "pho"]


def test_tokenize_handles_empty():
    assert tokenize_vi("") == []
    assert tokenize_vi("   ") == []
