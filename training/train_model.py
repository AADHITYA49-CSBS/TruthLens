import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.text_cleaning import normalize_text_column


def load_dataset(fake_path: Path, real_path: Path) -> pd.DataFrame:
    fake_frame = pd.read_csv(fake_path)
    real_frame = pd.read_csv(real_path)
    fake_frame["label"] = 0
    real_frame["label"] = 1
    return pd.concat([fake_frame, real_frame], ignore_index=True)


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def train_and_save(fake_path: Path, real_path: Path, output_path: Path) -> dict[str, float]:
    dataset = load_dataset(fake_path, real_path)
    dataset = dataset.drop(columns=[col for col in ["title", "subject", "date"] if col in dataset.columns])
    dataset = normalize_text_column(dataset, "text")

    train_frame, test_frame = train_test_split(
        dataset,
        test_size=0.2,
        random_state=42,
        stratify=dataset["label"],
    )

    pipeline = build_pipeline()
    pipeline.fit(train_frame["text"], train_frame["label"])
    predictions = pipeline.predict(test_frame["text"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)

    accuracy = float(accuracy_score(test_frame["label"], predictions))
    report = classification_report(test_frame["label"], predictions, output_dict=True)

    return {
        "accuracy": accuracy,
        "fake_f1": float(report["0"]["f1-score"]),
        "real_f1": float(report["1"]["f1-score"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the TruthLens baseline model.")
    parser.add_argument("--fake-csv", type=Path, default=Path("data/Fake.csv"))
    parser.add_argument("--real-csv", type=Path, default=Path("data/True.csv"))
    parser.add_argument("--output", type=Path, default=Path("model_artifacts/model.joblib"))
    args = parser.parse_args()

    fake_csv = args.fake_csv
    real_csv = args.real_csv
    model_output = args.output
    metrics = train_and_save(fake_csv, real_csv, model_output)
    print(metrics)
