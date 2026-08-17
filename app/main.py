from fastapi import FastAPI, HTTPException

from app.content_extraction import fetch_article_text
from app.predictor import NewsPredictor
from app.schemas import PredictionRequest, PredictionResponse
from app.settings import settings


app = FastAPI(title=settings.service_name, version="0.1.0")
predictor = NewsPredictor(settings.model_path)


def _predict_with_model(text: str):
    try:
        return predictor.predict(text)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    if payload.text is not None:
        article_text = payload.text.strip()
        source_type = "text"
    else:
        try:
            article_text = fetch_article_text(str(payload.source_url))
        except Exception as error:
            raise HTTPException(status_code=400, detail=f"could not fetch article text: {error}") from error
        source_type = "url"

    if not article_text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    result = predictor.predict(article_text, source_url=str(payload.source_url) if payload.source_url else None)
    return PredictionResponse(
        label=result.label,
        label_name=result.label_name,
        confidence=result.confidence,
        model_type=result.model_type,
        source_type=source_type,
        engine="ml" if result.model_type.startswith("sklearn") else "heuristic",
    )


@app.get("/model-status")
def model_status() -> dict[str, str | bool]:
    return {
        "model_path": str(settings.model_path),
        "model_exists": settings.model_path.exists(),
        "transformer_model": settings.transformer_model_name,
    }
