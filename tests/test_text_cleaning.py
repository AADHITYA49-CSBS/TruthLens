from app.text_cleaning import clean_text


def test_clean_text_removes_noise() -> None:
    cleaned = clean_text("Hello!!! Visit <b>this</b> site, now.")
    assert "hello" in cleaned
    assert "site" in cleaned
    assert "<b>" not in cleaned
    assert "!" not in cleaned
