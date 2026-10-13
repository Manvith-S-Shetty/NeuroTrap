"""Baseline Logistic Regression training and evaluation for NeuroTap.

This module uses the existing participant-aware split. It does not tune
hyperparameters, perform feature selection, export a model, or modify raw data.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from ml.training.dataset import (  # type: ignore
        MLDataset,
        ParticipantSplit,
        build_feature_table,
        build_ml_dataset,
        split_by_participant,
        validate_participant_split,
    )
else:
    from ml.training.dataset import (
        MLDataset,
        ParticipantSplit,
        build_feature_table,
        build_ml_dataset,
        split_by_participant,
        validate_participant_split,
    )


CLASS_LABELS = ["CNT", "PD"]


@dataclass(frozen=True)
class BaselineResult:
    scaler: StandardScaler
    model: LogisticRegression
    split: ParticipantSplit
    y_pred: np.ndarray
    metrics: dict[str, object]


def validate_finite_features(X: pd.DataFrame, name: str) -> None:
    """Reject missing or infinite feature values before scaling/training."""
    if X.isna().any().any():
        columns = sorted(X.columns[X.isna().any()].tolist())
        raise ValueError(f"{name} contains NaN feature values in columns: {columns}")

    numeric = X.to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        bad_columns = sorted(X.columns[~np.isfinite(numeric).all(axis=0)].tolist())
        raise ValueError(f"{name} contains infinite feature values in columns: {bad_columns}")


def validate_split_for_baseline(split: ParticipantSplit) -> None:
    """Validate participant isolation, numeric X values, and supported labels."""
    validate_participant_split(split)
    validate_finite_features(split.X_train, "X_train")
    validate_finite_features(split.X_test, "X_test")

    unsupported_train = sorted(set(split.y_train) - set(CLASS_LABELS))
    unsupported_test = sorted(set(split.y_test) - set(CLASS_LABELS))
    if unsupported_train or unsupported_test:
        raise ValueError(
            "Unsupported class labels found: "
            f"train={unsupported_train}, test={unsupported_test}"
        )


def train_baseline_from_split(
    split: ParticipantSplit,
    random_state: int = 42,
    max_iter: int = 1000,
) -> BaselineResult:
    """Fit scaler on X_train only, train Logistic Regression, and evaluate X_test."""
    validate_split_for_baseline(split)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(split.X_train)
    X_test_scaled = scaler.transform(split.X_test)

    model = LogisticRegression(random_state=random_state, max_iter=max_iter)
    model.fit(X_train_scaled, split.y_train)
    y_pred = model.predict(X_test_scaled)
    metrics = evaluate_predictions(split.y_test, y_pred)
    return BaselineResult(scaler=scaler, model=model, split=split, y_pred=y_pred, metrics=metrics)


def evaluate_predictions(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, object]:
    """Calculate baseline metrics with PD as the positive class."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, labels=CLASS_LABELS, pos_label="PD", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, labels=CLASS_LABELS, pos_label="PD", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, labels=CLASS_LABELS, pos_label="PD", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASS_LABELS),
        "class_labels": CLASS_LABELS,
    }


def run_baseline(
    ml_dataset: MLDataset | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> BaselineResult:
    """Build/reuse the ML dataset, use the existing participant split, and train baseline."""
    dataset = build_ml_dataset() if ml_dataset is None else ml_dataset
    split = split_by_participant(
        dataset,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return train_baseline_from_split(split, random_state=random_state)


def _participant_overlap(split: ParticipantSplit) -> int:
    train = set(split.train_metadata["participant_id"].astype(str))
    test = set(split.test_metadata["participant_id"].astype(str))
    return len(train & test)


def main() -> int:
    feature_table = build_feature_table()
    ml_dataset = build_ml_dataset(feature_table)
    result = run_baseline(ml_dataset=ml_dataset, test_size=0.2, random_state=42, stratify=True)
    split = result.split

    print("# NeuroTap Baseline Model")
    print("\n## Dataset")
    print(f"Total samples: {len(ml_dataset.X)}")
    print(f"Total features: {ml_dataset.X.shape[1]}")
    print(f"Train samples: {len(split.X_train)}")
    print(f"Test samples: {len(split.X_test)}")
    print(f"Train participants: {split.train_metadata['participant_id'].nunique()}")
    print(f"Test participants: {split.test_metadata['participant_id'].nunique()}")
    print(f"Participant overlap: {_participant_overlap(split)}")
    print("\n## Model")
    print("Model: LogisticRegression")
    print("Scaler: StandardScaler")
    print("\n## Test Class Counts")
    print(split.y_test.value_counts().reindex(CLASS_LABELS, fill_value=0).to_string())
    print("\n## Results")
    print(f"Accuracy: {result.metrics['accuracy']:.4f}")
    print(f"Precision: {result.metrics['precision']:.4f}")
    print(f"Recall: {result.metrics['recall']:.4f}")
    print(f"F1: {result.metrics['f1']:.4f}")
    print("Confusion matrix labels: CNT, PD")
    print(result.metrics["confusion_matrix"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
