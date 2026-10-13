"""Participant-aware cross-validation for NeuroTap ML experiments.

Cross-validation operates only on the existing training/development set. The
final held-out test participants are reserved for final evaluation and are not
used in any CV fold.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


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
    from ml.training.random_forest import (  # type: ignore
        DEFAULT_MAX_DEPTH,
        DEFAULT_MIN_SAMPLES_LEAF,
        DEFAULT_MIN_SAMPLES_SPLIT,
        DEFAULT_N_ESTIMATORS,
        DEFAULT_RANDOM_STATE,
        validate_expected_features,
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
    from ml.training.random_forest import (
        DEFAULT_MAX_DEPTH,
        DEFAULT_MIN_SAMPLES_LEAF,
        DEFAULT_MIN_SAMPLES_SPLIT,
        DEFAULT_N_ESTIMATORS,
        DEFAULT_RANDOM_STATE,
        validate_expected_features,
    )


MODEL_LOGISTIC_REGRESSION = "Logistic Regression"
MODEL_RANDOM_FOREST = "Random Forest"
METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1"]


@dataclass(frozen=True)
class CVFoldResult:
    model_name: str
    fold: int
    train_recordings: int
    validation_recordings: int
    train_participants: int
    validation_participants: int
    participant_overlap: int
    train_participant_ids: tuple[str, ...]
    validation_participant_ids: tuple[str, ...]
    metrics: dict[str, object]
    y_pred: np.ndarray
    scaler: StandardScaler | None = None


@dataclass(frozen=True)
class CrossValidationResult:
    fold_results: list[CVFoldResult]
    aggregate_metrics: pd.DataFrame
    development_metadata: pd.DataFrame
    final_test_metadata: pd.DataFrame
    cv_strategy: str = "GroupKFold"


def _participants(metadata: pd.DataFrame) -> set[str]:
    if "participant_id" not in metadata.columns:
        raise ValueError("Missing required metadata columns: ['participant_id']")
    return set(metadata["participant_id"].astype(str))


def _validate_supported_labels(y: pd.Series, name: str) -> None:
    if y.isna().any():
        raise ValueError(f"{name} contains missing labels.")
    unsupported = sorted(set(y) - set(CLASS_LABELS))
    if unsupported:
        raise ValueError(f"{name} contains unsupported class labels: {unsupported}")


def validate_development_dataset(
    development: MLDataset,
    final_test_participants: set[str] | None = None,
) -> None:
    """Validate CV input and reject final-test participant leakage."""
    validate_expected_features(development.X, "development X")
    validate_finite_features(development.X, "development X")
    _validate_supported_labels(development.y, "development y")

    dev_participants = _participants(development.metadata)
    if len(dev_participants) < 2:
        raise ValueError("Development data must contain at least two participants for GroupKFold.")

    if final_test_participants:
        overlap = dev_participants & {str(participant) for participant in final_test_participants}
        if overlap:
            raise ValueError(
                "Development data contains final held-out test participants: "
                f"{sorted(overlap)}"
            )


def _make_logistic_regression(random_state: int) -> LogisticRegression:
    return LogisticRegression(random_state=random_state, max_iter=1000)


def _make_random_forest(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=DEFAULT_N_ESTIMATORS,
        max_depth=DEFAULT_MAX_DEPTH,
        min_samples_split=DEFAULT_MIN_SAMPLES_SPLIT,
        min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
        random_state=random_state,
    )


def _fold_participant_details(
    train_metadata: pd.DataFrame,
    validation_metadata: pd.DataFrame,
) -> tuple[tuple[str, ...], tuple[str, ...], int]:
    train_participants = _participants(train_metadata)
    validation_participants = _participants(validation_metadata)
    return (
        tuple(sorted(train_participants)),
        tuple(sorted(validation_participants)),
        len(train_participants & validation_participants),
    )


def _run_logistic_regression_fold(
    fold: int,
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
    train_metadata: pd.DataFrame,
    validation_metadata: pd.DataFrame,
    random_state: int,
) -> CVFoldResult:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    model = _make_logistic_regression(random_state)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_validation_scaled)
    train_participant_ids, validation_participant_ids, overlap = _fold_participant_details(
        train_metadata,
        validation_metadata,
    )
    if overlap:
        raise ValueError(f"Participant overlap detected inside CV fold {fold}.")

    return CVFoldResult(
        model_name=MODEL_LOGISTIC_REGRESSION,
        fold=fold,
        train_recordings=len(X_train),
        validation_recordings=len(X_validation),
        train_participants=len(train_participant_ids),
        validation_participants=len(validation_participant_ids),
        participant_overlap=overlap,
        train_participant_ids=train_participant_ids,
        validation_participant_ids=validation_participant_ids,
        metrics=evaluate_predictions(y_validation, y_pred),
        y_pred=y_pred,
        scaler=scaler,
    )


def _run_random_forest_fold(
    fold: int,
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
    train_metadata: pd.DataFrame,
    validation_metadata: pd.DataFrame,
    random_state: int,
) -> CVFoldResult:
    model = _make_random_forest(random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_validation)
    train_participant_ids, validation_participant_ids, overlap = _fold_participant_details(
        train_metadata,
        validation_metadata,
    )
    if overlap:
        raise ValueError(f"Participant overlap detected inside CV fold {fold}.")

    return CVFoldResult(
        model_name=MODEL_RANDOM_FOREST,
        fold=fold,
        train_recordings=len(X_train),
        validation_recordings=len(X_validation),
        train_participants=len(train_participant_ids),
        validation_participants=len(validation_participant_ids),
        participant_overlap=overlap,
        train_participant_ids=train_participant_ids,
        validation_participant_ids=validation_participant_ids,
        metrics=evaluate_predictions(y_validation, y_pred),
        y_pred=y_pred,
        scaler=None,
    )


def aggregate_fold_metrics(fold_results: list[CVFoldResult]) -> pd.DataFrame:
    """Aggregate mean and sample standard deviation of fold metrics by model."""
    rows = [
        {
            "model": result.model_name,
            **{metric: float(result.metrics[metric]) for metric in METRIC_COLUMNS},
        }
        for result in fold_results
    ]
    metrics = pd.DataFrame(rows)
    aggregate = metrics.groupby("model")[METRIC_COLUMNS].agg(["mean", "std"])
    aggregate.columns = [f"{stat}_{metric}" for metric, stat in aggregate.columns]
    return aggregate.reset_index()


def cross_validate_development_set(
    development: MLDataset,
    final_test_participants: set[str] | None = None,
    n_splits: int = 5,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> CrossValidationResult:
    """Run GroupKFold CV on development participants only."""
    validate_development_dataset(development, final_test_participants=final_test_participants)

    groups = development.metadata["participant_id"].astype(str)
    participant_count = groups.nunique()
    if n_splits > participant_count:
        raise ValueError(f"n_splits={n_splits} exceeds development participant count={participant_count}.")

    splitter = GroupKFold(n_splits=n_splits)
    fold_results: list[CVFoldResult] = []

    for fold, (train_idx, validation_idx) in enumerate(
        splitter.split(development.X, development.y, groups=groups),
        start=1,
    ):
        X_train = development.X.iloc[train_idx].copy()
        X_validation = development.X.iloc[validation_idx].copy()
        y_train = development.y.iloc[train_idx].copy()
        y_validation = development.y.iloc[validation_idx].copy()
        train_metadata = development.metadata.iloc[train_idx].copy()
        validation_metadata = development.metadata.iloc[validation_idx].copy()

        fold_results.append(
            _run_logistic_regression_fold(
                fold=fold,
                X_train=X_train,
                X_validation=X_validation,
                y_train=y_train,
                y_validation=y_validation,
                train_metadata=train_metadata,
                validation_metadata=validation_metadata,
                random_state=random_state,
            )
        )
        fold_results.append(
            _run_random_forest_fold(
                fold=fold,
                X_train=X_train,
                X_validation=X_validation,
                y_train=y_train,
                y_validation=y_validation,
                train_metadata=train_metadata,
                validation_metadata=validation_metadata,
                random_state=random_state,
            )
        )

    return CrossValidationResult(
        fold_results=fold_results,
        aggregate_metrics=aggregate_fold_metrics(fold_results),
        development_metadata=development.metadata.copy(),
        final_test_metadata=pd.DataFrame({"participant_id": sorted(final_test_participants or set())}),
    )


def make_development_dataset(split: ParticipantSplit) -> MLDataset:
    """Convert the existing participant-aware training side into the CV development set."""
    validate_participant_split(split)
    return MLDataset(
        X=split.X_train.copy(),
        y=split.y_train.copy(),
        metadata=split.train_metadata.copy(),
    )


def run_cross_validation(
    ml_dataset: MLDataset | None = None,
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratify: bool = True,
    n_splits: int = 5,
) -> tuple[CrossValidationResult, ParticipantSplit]:
    """Recover the existing final split, then run CV only on the development side."""
    dataset = build_ml_dataset() if ml_dataset is None else ml_dataset
    split = split_by_participant(
        dataset,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    development = make_development_dataset(split)
    final_test_participants = _participants(split.test_metadata)
    result = cross_validate_development_set(
        development,
        final_test_participants=final_test_participants,
        n_splits=n_splits,
        random_state=random_state,
    )
    return result, split


def _format_metric(value: object) -> str:
    return f"{float(value):.4f}"


def main() -> int:
    feature_table = build_feature_table()
    ml_dataset = build_ml_dataset(feature_table)
    result, split = run_cross_validation(
        ml_dataset=ml_dataset,
        test_size=0.2,
        random_state=DEFAULT_RANDOM_STATE,
        stratify=True,
        n_splits=5,
    )

    dev_participants = _participants(split.train_metadata)
    final_test_participants = _participants(split.test_metadata)
    cv_final_overlap = dev_participants & final_test_participants

    print("# NeuroTap Participant-Aware Cross-Validation")
    print("\n## Development Dataset")
    print(f"Recordings: {len(split.X_train)}")
    print(f"Participants: {len(dev_participants)}")
    print("Class counts:")
    print(split.y_train.value_counts().reindex(CLASS_LABELS, fill_value=0).to_string())
    print("\n## Final Held-Out Test Set")
    print(f"Recordings: {len(split.X_test)}")
    print(f"Participants: {len(final_test_participants)}")
    print("Class counts:")
    print(split.y_test.value_counts().reindex(CLASS_LABELS, fill_value=0).to_string())
    print(f"Final test participants used in CV: {len(cv_final_overlap)}")
    print("\n## Per-Fold Results")
    for fold in range(1, 6):
        print(f"\nFold {fold}")
        for model_name in [MODEL_LOGISTIC_REGRESSION, MODEL_RANDOM_FOREST]:
            fold_result = next(
                item for item in result.fold_results if item.fold == fold and item.model_name == model_name
            )
            print(
                f"{model_name}: "
                f"accuracy={_format_metric(fold_result.metrics['accuracy'])}, "
                f"precision={_format_metric(fold_result.metrics['precision'])}, "
                f"recall={_format_metric(fold_result.metrics['recall'])}, "
                f"f1={_format_metric(fold_result.metrics['f1'])}, "
                f"train_recordings={fold_result.train_recordings}, "
                f"validation_recordings={fold_result.validation_recordings}, "
                f"train_participants={fold_result.train_participants}, "
                f"validation_participants={fold_result.validation_participants}, "
                f"participant_overlap={fold_result.participant_overlap}"
            )

    print("\n## Aggregate Results")
    print("Model | Mean Accuracy | Std Accuracy | Mean Precision | Std Precision | Mean Recall | Std Recall | Mean F1 | Std F1")
    print("--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:")
    for _, row in result.aggregate_metrics.sort_values("model").iterrows():
        print(
            f"{row['model']} | "
            f"{row['mean_accuracy']:.4f} | {row['std_accuracy']:.4f} | "
            f"{row['mean_precision']:.4f} | {row['std_precision']:.4f} | "
            f"{row['mean_recall']:.4f} | {row['std_recall']:.4f} | "
            f"{row['mean_f1']:.4f} | {row['std_f1']:.4f}"
        )
    print("\nParticipant overlap for every fold:")
    print([item.participant_overlap for item in result.fold_results])
    print(f"Final-test participant overlap with CV participants: {len(cv_final_overlap)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
