"""Random Forest training and evaluation for NeuroTap.

This module uses the existing participant-aware split. It does not tune
hyperparameters, perform feature selection, export a model, scale features, or
modify raw data.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from ml.training.baseline import CLASS_LABELS, evaluate_predictions, validate_finite_features  # type: ignore
    from ml.training.dataset import (  # type: ignore
        FEATURE_COLUMNS,
        MLDataset,
        ParticipantSplit,
        build_feature_table,
        build_ml_dataset,
        split_by_participant,
        validate_participant_split,
    )
else:
    from ml.training.baseline import CLASS_LABELS, evaluate_predictions, validate_finite_features
    from ml.training.dataset import (
        FEATURE_COLUMNS,
        MLDataset,
        ParticipantSplit,
        build_feature_table,
        build_ml_dataset,
        split_by_participant,
        validate_participant_split,
    )


DEFAULT_RANDOM_STATE = 42
DEFAULT_N_ESTIMATORS = 200
DEFAULT_MAX_DEPTH = None
DEFAULT_MIN_SAMPLES_SPLIT = 2
DEFAULT_MIN_SAMPLES_LEAF = 1


@dataclass(frozen=True)
class RandomForestResult:
    model: RandomForestClassifier
    split: ParticipantSplit
    y_pred: np.ndarray
    metrics: dict[str, object]


def validate_expected_features(X: pd.DataFrame, name: str) -> None:
    """Validate that X contains exactly the expected NeuroTap feature columns."""
    missing = sorted(set(FEATURE_COLUMNS) - set(X.columns))
    if missing:
        raise ValueError(f"{name} is missing expected feature columns: {missing}")

    unexpected = sorted(set(X.columns) - set(FEATURE_COLUMNS))
    if unexpected:
        raise ValueError(f"{name} contains unexpected feature columns: {unexpected}")

    if list(X.columns) != FEATURE_COLUMNS:
        raise ValueError(f"{name} feature columns are not in the expected order.")


def validate_split_for_random_forest(split: ParticipantSplit) -> None:
    """Validate feature, label, and participant-leakage requirements."""
    validate_participant_split(split)
    validate_expected_features(split.X_train, "X_train")
    validate_expected_features(split.X_test, "X_test")
    validate_finite_features(split.X_train, "X_train")
    validate_finite_features(split.X_test, "X_test")

    unsupported_train = sorted(set(split.y_train) - set(CLASS_LABELS))
    unsupported_test = sorted(set(split.y_test) - set(CLASS_LABELS))
    if unsupported_train or unsupported_test:
        raise ValueError(
            "Unsupported class labels found: "
            f"train={unsupported_train}, test={unsupported_test}"
        )


def train_random_forest_from_split(
    split: ParticipantSplit,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_depth: int | None = DEFAULT_MAX_DEPTH,
    min_samples_split: int = DEFAULT_MIN_SAMPLES_SPLIT,
    min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
) -> RandomForestResult:
    """Train RandomForestClassifier directly on validated numerical features."""
    validate_split_for_random_forest(split)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(split.X_train, split.y_train)
    y_pred = model.predict(split.X_test)
    metrics = evaluate_predictions(split.y_test, y_pred)
    return RandomForestResult(model=model, split=split, y_pred=y_pred, metrics=metrics)


def run_random_forest(
    ml_dataset: MLDataset | None = None,
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratify: bool = True,
) -> RandomForestResult:
    """Build/reuse the ML dataset, use the existing participant split, and train Random Forest."""
    dataset = build_ml_dataset() if ml_dataset is None else ml_dataset
    split = split_by_participant(
        dataset,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return train_random_forest_from_split(split, random_state=random_state)


def _participant_overlap(split: ParticipantSplit) -> int:
    train = set(split.train_metadata["participant_id"].astype(str))
    test = set(split.test_metadata["participant_id"].astype(str))
    return len(train & test)


def main() -> int:
    feature_table = build_feature_table()
    ml_dataset = build_ml_dataset(feature_table)
    result = run_random_forest(ml_dataset=ml_dataset, test_size=0.2, random_state=DEFAULT_RANDOM_STATE, stratify=True)
    split = result.split

    print("# NeuroTap Random Forest Experiment")
    print("\n## Dataset")
    print(f"Total samples: {len(ml_dataset.X)}")
    print(f"Total features: {ml_dataset.X.shape[1]}")
    print(f"Train samples: {len(split.X_train)}")
    print(f"Test samples: {len(split.X_test)}")
    print(f"Train participants: {split.train_metadata['participant_id'].nunique()}")
    print(f"Test participants: {split.test_metadata['participant_id'].nunique()}")
    print(f"Participant overlap: {_participant_overlap(split)}")
    print("\n## Model")
    print("Model: RandomForestClassifier")
    print(f"n_estimators: {DEFAULT_N_ESTIMATORS}")
    print(f"max_depth: {DEFAULT_MAX_DEPTH}")
    print(f"min_samples_split: {DEFAULT_MIN_SAMPLES_SPLIT}")
    print(f"min_samples_leaf: {DEFAULT_MIN_SAMPLES_LEAF}")
    print(f"random_state: {DEFAULT_RANDOM_STATE}")
    print("\n## Test Class Counts")
    print(split.y_test.value_counts().reindex(CLASS_LABELS, fill_value=0).to_string())
    print("\n## Results")
    print(f"Accuracy: {result.metrics['accuracy']:.4f}")
    print(f"Precision: {result.metrics['precision']:.4f}")
    print(f"Recall: {result.metrics['recall']:.4f}")
    print(f"F1: {result.metrics['f1']:.4f}")
    print("Confusion matrix labels: CNT, PD")
    print(result.metrics["confusion_matrix"])
    print("\n## Comparison")
    print("Metric | Logistic Regression | Random Forest")
    print("--- | ---: | ---:")
    print(f"Accuracy | 0.6522 | {result.metrics['accuracy']:.4f}")
    print(f"Precision | 0.6111 | {result.metrics['precision']:.4f}")
    print(f"Recall | 0.9167 | {result.metrics['recall']:.4f}")
    print(f"F1 | 0.7333 | {result.metrics['f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
