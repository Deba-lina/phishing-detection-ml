"""Train and evaluate a phishing-website classifier.

Example:
    python code.py --data phishing.csv --save-model artifacts/phishing_model.joblib
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger("phishing_detection")


def parse_arguments() -> argparse.Namespace:
    """Read command-line options for a reproducible experiment."""
    parser = argparse.ArgumentParser(description="Train a phishing website detection model.")
    parser.add_argument("--data", type=Path, default=Path("phishing.csv"), help="Path to the CSV dataset.")
    parser.add_argument("--target", default="Result", help="Name of the target column (default: Result).")
    parser.add_argument("--test-size", type=float, default=0.20, help="Test-set fraction (default: 0.20).")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--save-model", type=Path, default=None, help="Optional path for the trained .joblib model.")
    parser.add_argument("--metrics-out", type=Path, default=Path("artifacts/metrics.json"), help="Path for the metrics JSON file.")
    return parser.parse_args()


def load_dataset(data_path: Path, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load, validate, and prepare features and labels from a CSV file."""
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    data = pd.read_csv(data_path)
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found. Available columns: {list(data.columns)}")
    if data.empty:
        raise ValueError("Dataset is empty.")

    # Row numbers are identifiers rather than website attributes and would leak data.
    identifiers = [column for column in data.columns if column.lower() in {"index", "id", "unnamed: 0"}]
    features = data.drop(columns=[target_column, *identifiers], errors="ignore").apply(pd.to_numeric, errors="coerce")
    if features.empty:
        raise ValueError("No usable feature columns remain after preprocessing.")
    if features.isna().any().any():
        missing_columns = features.columns[features.isna().any()].tolist()
        raise ValueError(f"Features contain missing or non-numeric values: {missing_columns}")

    labels = pd.to_numeric(data[target_column], errors="raise")
    unique_labels = set(labels.unique())
    # In the UCI/Kaggle phishing dataset, -1 denotes phishing and 1 legitimate.
    # Normalize to the conventional internal encoding: 0 = legitimate, 1 = phishing.
    if unique_labels == {-1, 1}:
        labels = labels.map({-1: 1, 1: 0})
    elif unique_labels != {0, 1}:
        raise ValueError("Target must use either {-1, 1} or {0, 1} labels.")
    if labels.nunique() != 2:
        raise ValueError("Target must contain both legitimate and phishing examples.")
    return features, labels.astype(int)


def build_model() -> Pipeline:
    """Create the baseline model pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]
    )


def evaluate_model(model: Pipeline, features: pd.DataFrame, labels: pd.Series) -> dict[str, Any]:
    """Calculate classification metrics on a held-out test set."""
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "precision": round(float(precision_score(labels, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(labels, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(labels, probabilities)), 4),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "classification_report": classification_report(
            labels, predictions, target_names=["legitimate", "phishing"], output_dict=True, zero_division=0
        ),
    }


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    """Persist experiment output in a portable format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    """Run one deterministic training and evaluation experiment."""
    arguments = parse_arguments()
    if not 0 < arguments.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")
    features, labels = load_dataset(arguments.data, arguments.target)
    LOGGER.info("Loaded %s records with %s features.", len(features), features.shape[1])
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=arguments.test_size, random_state=arguments.random_state, stratify=labels
    )
    model = build_model()
    model.fit(x_train, y_train)
    metrics = evaluate_model(model, x_test, y_test)
    metrics.update({"training_samples": len(x_train), "test_samples": len(x_test), "feature_count": features.shape[1], "random_state": arguments.random_state})
    write_json(metrics, arguments.metrics_out)
    print(json.dumps({key: value for key, value in metrics.items() if key != "classification_report"}, indent=2))
    print(f"\nMetrics saved to: {arguments.metrics_out}")
    if arguments.save_model:
        arguments.save_model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, arguments.save_model)
        print(f"Model saved to: {arguments.save_model}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        main()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        LOGGER.error("%s", error)
        raise SystemExit(1) from error
