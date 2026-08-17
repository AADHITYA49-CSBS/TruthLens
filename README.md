


# TruthLens

TruthLens is being migrated from a Colab-era notebook into a small production-style NLP API.

## Current direction

The repository now includes a FastAPI service skeleton, a reusable text-cleaning layer, an offline sklearn training script, pytest tests, Docker support, GitHub Actions CI, and Render deployment metadata.

## Planned stack

- Python 3.12
- FastAPI
- Uvicorn
- Pydantic
- Pandas
- scikit-learn
- Hugging Face Transformers
- BeautifulSoup
- httpx
- pytest
- Docker
- GitHub Actions
- Render

## Project layout

- `app/` FastAPI application code
- `training/` offline model training
- `tests/` API and preprocessing tests
- `model_artifacts/` saved model files created locally or in CI/CD

## Notes

- The legacy Colab notebook file is retained for reference.
- The current API expects article text and returns a binary fake/real prediction with confidence.
- Before deploying, train the baseline model and place the resulting artifact in `model_artifacts/model.joblib`.
- If no trained artifact exists, the API automatically uses a modern heuristic fallback so the service still works immediately.

## Run locally

Train the baseline model:

```bash
python training/train_model.py --fake-csv data/Fake.csv --real-csv data/True.csv --output model_artifacts/model.joblib
```

Start the API:

```bash
uvicorn app.main:app --reload
```

If you want to point the API at a different model file, set `TRUTHLENS_MODEL_PATH` before starting the server.

## Modern data note

The old 2016-2017 CSVs are no longer required for the application to run.
For a supervised retrain later, use a current public dataset source of your choice and feed it into `training/train_model.py`.








