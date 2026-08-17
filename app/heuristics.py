from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


SENSATIONAL_TERMS = {
    "shocking",
    "breaking",
    "miracle",
    "secret",
    "hoax",
    "exposed",
    "busted",
    "you won\'t believe",
    "everyone is talking about",
    "urgent",
    "warning",
    "censored",
}

TRUSTED_NEWS_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "nytimes.com",
    "theguardian.com",
    "washingtonpost.com",
    "cnn.com",
    "npr.org",
    "economist.com",
}


@dataclass
class HeuristicPrediction:
    label: int
    label_name: str
    confidence: float
    model_type: str = "heuristic-modern"


def _domain_from_url(source_url: str | None) -> str:
    if not source_url:
        return ""
    hostname = urlparse(source_url).hostname or ""
    return hostname.lower().removeprefix("www.")


def score_article(text: str, source_url: str | None = None) -> HeuristicPrediction:
    normalized = (text or "").strip()
    if not normalized:
        return HeuristicPrediction(label=0, label_name="fake", confidence=0.55)

    lower_text = normalized.lower()
    word_count = max(len(normalized.split()), 1)
    uppercase_ratio = sum(1 for character in normalized if character.isupper()) / max(len(normalized), 1)
    exclamation_count = normalized.count("!")
    question_count = normalized.count("?")
    sensational_hits = sum(1 for term in SENSATIONAL_TERMS if term in lower_text)

    score = 0.0
    score += min(sensational_hits * 0.18, 0.54)
    score += min(uppercase_ratio * 2.0, 0.20)
    score += min((exclamation_count + question_count) * 0.04, 0.20)
    if word_count < 120:
        score += 0.08

    domain = _domain_from_url(source_url)
    if domain in TRUSTED_NEWS_DOMAINS:
        score -= 0.20
    elif domain:
        score += 0.04

    score = max(0.0, min(score, 1.0))
    fake_confidence = round(0.55 + score * 0.4, 3)
    real_confidence = round(1.0 - fake_confidence, 3)

    if score >= 0.42:
        return HeuristicPrediction(label=0, label_name="fake", confidence=fake_confidence)

    return HeuristicPrediction(label=1, label_name="real", confidence=max(real_confidence, 0.55))
