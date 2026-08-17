import re
from functools import lru_cache

import pandas as pd


@lru_cache(maxsize=1)
def _stopwords() -> set[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return set(ENGLISH_STOP_WORDS)


def clean_text(value: str) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = text.lower()
    words = [word for word in text.split() if word not in _stopwords()]
    return re.sub(r"\s+", " ", " ".join(words)).strip()


def normalize_text_column(frame: pd.DataFrame, column: str = "text") -> pd.DataFrame:
    result = frame.copy()
    result[column] = result[column].fillna("").map(clean_text)
    return result
