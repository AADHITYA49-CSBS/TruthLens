from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from app.heuristics import HeuristicPrediction, score_article
from app.text_cleaning import clean_text


@dataclass
class SklearnPrediction:
    label: int
    label_name: str
    confidence: float
    model_type: str = "sklearn-logistic-regression"


class NewsPredictor:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.pipeline = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. Run training first to create it."
            )
        self.pipeline = joblib.load(self.model_path)

    def predict(self, text: str, source_url: str | None = None) -> SklearnPrediction | HeuristicPrediction:
        if self.pipeline is None:
            try:
                self.load()
            except FileNotFoundError:
                return score_article(text, source_url=source_url)

        cleaned_text = clean_text(text)
        probabilities = self.pipeline.predict_proba([cleaned_text])[0]
        label = int(np.argmax(probabilities))
        confidence = float(probabilities[label])
        label_name = "real" if label == 1 else "fake"
        return SklearnPrediction(label=label, label_name=label_name, confidence=confidence)
