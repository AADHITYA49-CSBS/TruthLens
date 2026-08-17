import os
from pathlib import Path

from pydantic import BaseModel


class AppSettings(BaseModel):
    model_path: Path = Path(os.getenv("TRUTHLENS_MODEL_PATH", "model_artifacts/model.joblib"))
    vectorizer_path: Path = Path(os.getenv("TRUTHLENS_VECTORIZER_PATH", "model_artifacts/vectorizer.joblib"))
    transformer_model_name: str = os.getenv("TRUTHLENS_TRANSFORMER_MODEL", "distilbert-base-uncased")
    service_name: str = "TruthLens"


settings = AppSettings()
