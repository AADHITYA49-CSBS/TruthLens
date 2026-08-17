def test_health_check() -> None:
    from app.main import health

    assert health() == {"status": "ok"}


def test_predict_route_logic(monkeypatch) -> None:
    from app import main as main_module
    from app.schemas import PredictionRequest

    class DummyPredictor:
        def predict(self, text: str, source_url: str | None = None):
            return type(
                "Result",
                (),
                {"label": 1, "label_name": "real", "confidence": 0.91, "model_type": "dummy"},
            )()

    monkeypatch.setattr(main_module, "predictor", DummyPredictor())
    response = main_module.predict(PredictionRequest(text="Example text"))

    assert response.label == 1
    assert response.label_name == "real"
    assert response.confidence == 0.91
    assert response.model_type == "dummy"


def test_predictor_falls_back_without_model(monkeypatch, tmp_path) -> None:
    from app.predictor import NewsPredictor

    predictor = NewsPredictor(tmp_path / "missing.joblib")
    result = predictor.predict("Breaking secret shocking announcement!!!", source_url="https://example.com/story")

    assert result.label in {0, 1}
    assert result.model_type == "heuristic-modern"


def test_prediction_request_requires_input() -> None:
    from app.schemas import PredictionRequest

    try:
        PredictionRequest()
        assert False, "Expected validation error"
    except Exception as exc:
        assert "Provide either text or source_url" in str(exc)
