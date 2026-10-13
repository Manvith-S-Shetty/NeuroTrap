"""Baseline touchscreen model training with participant-aware cross-validation.

Trains Logistic Regression and Random Forest baselines on the 27 touchscreen
features using ONLY the development set from the Step 6 participant-aware split.

This module does NOT:
- use or inspect the locked final-test participants/recordings
- tune hyperparameters
- perform feature selection
- remove features, outliers, or correlated features
- change labels
- perform imputation
- modify the feature extraction pipeline
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from ml.training.touchscreen_dataset import (  # type: ignore
        TouchscreenMLDataset,
        TouchscreenParticipantSplit,
        build_touchscreen_ml_dataset,
        split_touchscreen_by_participant,
        validate_touchscreen_ml_dataset,
    )
else:
    from ml.training.touchscreen_dataset import (
        TouchscreenMLDataset,
        TouchscreenParticipantSplit,
        build_touchscreen_ml_dataset,
        split_touchscreen_by_participant,
        validate_touchscreen_ml_dataset,
    )


# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

CLASS_LABELS = ["CNT", "PD"]
MODEL_LOGISTIC_REGRESSION = "Logistic Regression"
MODEL_RANDOM_FOREST = "Random Forest"
METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1"]

DEFAULT_RANDOM_STATE = 42
DEFAULT_N_SPLITS = 5
DEFAULT_TEST_SIZE = 0.2

# Logistic Regression hyperparameters (baseline — not tuned)
LR_C = 1.0
LR_MAX_ITER = 1000

# Random Forest hyperparameters (baseline — not tuned)
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_SPLIT = 2
RF_MIN_SAMPLES_LEAF = 1


# --------------------------------------------------------------------------- #
# Dataclasses                                                                 #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TouchscreenCVFoldResult:
    """Result for a single model in a single CV fold."""

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
    scaler: StandardScaler | None


@dataclass(frozen=True)
class TouchscreenCVResult:
    """Aggregate result of participant-aware cross-validation for both models."""

    fold_results: list[TouchscreenCVFoldResult]
    aggregate_metrics: pd.DataFrame
    development_metadata: pd.DataFrame
    final_test_participants: set[str]
    cv_strategy: str
    n_splits: int


# --------------------------------------------------------------------------- #
# Validation helpers                                                          #
# --------------------------------------------------------------------------- #

def validate_finite_features(X: pd.DataFrame, name: str) -> None:
    """Reject missing or infinite feature values before scaling/training."""
    if X.isna().any().any():
        cols = sorted(X.columns[X.isna().any()].tolist())
        raise ValueError(f"{name} contains NaN feature values in columns: {cols}")
    arr = X.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        bad = sorted(X.columns[~np.isfinite(arr).all(axis=0)].tolist())
        raise ValueError(f"{name} contains infinite feature values in columns: {bad}")


def _validate_labels(y: pd.Series, name: str) -> None:
    if y.isna().any():
        raise ValueError(f"{name} contains missing labels.")
    unsupported = sorted(set(y) - set(CLASS_LABELS))
    if unsupported:
        raise ValueError(f"{name} contains unsupported class labels: {unsupported}")


def validate_development_dataset(
    development: TouchscreenMLDataset,
    final_test_participants: set[str] | None = None,
) -> None:
    """Validate CV input. Reject any final-test participant leakage.

    Leakage detection runs first (before full feature-table validation) so
    that injecting a final-test participant raises the leakage error rather
    than an incidental duplicate-ID or inconsistent-label error.
    """
    if "participant_id" not in development.metadata.columns:
        raise ValueError("Development metadata is missing 'participant_id' column.")

    dev_p = set(development.metadata["participant_id"].astype(str))
    if len(dev_p) < 2:
        raise ValueError("Development data must contain at least two participants for GroupKFold.")

    if final_test_participants:
        leak = dev_p & {str(p) for p in final_test_participants}
        if leak:
            raise ValueError(
                f"Development data contains locked final-test participants: {sorted(leak)}"
            )

    validate_touchscreen_ml_dataset(development)
    validate_finite_features(development.X, "development X")
    _validate_labels(development.y, "development y")


# --------------------------------------------------------------------------- #
# Metric evaluation                                                           #
# --------------------------------------------------------------------------- #

def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, object]:
    """Calculate accuracy, precision, recall, and F1 with PD as the positive class."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, labels=CLASS_LABELS, pos_label="PD", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, labels=CLASS_LABELS, pos_label="PD", zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, labels=CLASS_LABELS, pos_label="PD", zero_division=0)
        ),
    }


# --------------------------------------------------------------------------- #
# Model factories (baseline configs — not tuned)                             #
# --------------------------------------------------------------------------- #

def _make_logistic_regression(random_state: int) -> LogisticRegression:
    return LogisticRegression(C=LR_C, max_iter=LR_MAX_ITER, random_state=random_state)


def _make_random_forest(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=random_state,
    )


# --------------------------------------------------------------------------- #
# Per-fold participant helpers                                                #
# --------------------------------------------------------------------------- #

def _fold_participant_details(
    train_meta: pd.DataFrame,
    val_meta: pd.DataFrame,
) -> tuple[tuple[str, ...], tuple[str, ...], int]:
    train_p = set(train_meta["participant_id"].astype(str))
    val_p = set(val_meta["participant_id"].astype(str))
    return (
        tuple(sorted(train_p)),
        tuple(sorted(val_p)),
        len(train_p & val_p),
    )


# --------------------------------------------------------------------------- #
# Per-fold runners                                                            #
# --------------------------------------------------------------------------- #

def _run_lr_fold(
    fold: int,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    train_meta: pd.DataFrame,
    val_meta: pd.DataFrame,
    random_state: int,
) -> TouchscreenCVFoldResult:
    # Scale is ONLY fitted on the training fold
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    model = _make_logistic_regression(random_state)
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_val_sc)

    train_p, val_p, overlap = _fold_participant_details(train_meta, val_meta)
    if overlap:
        raise ValueError(f"Participant overlap in CV fold {fold}: {overlap}")

    return TouchscreenCVFoldResult(
        model_name=MODEL_LOGISTIC_REGRESSION,
        fold=fold,
        train_recordings=len(X_train),
        validation_recordings=len(X_val),
        train_participants=len(train_p),
        validation_participants=len(val_p),
        participant_overlap=overlap,
        train_participant_ids=train_p,
        validation_participant_ids=val_p,
        metrics=evaluate_predictions(y_val, y_pred),
        y_pred=y_pred,
        scaler=scaler,
    )


def _run_rf_fold(
    fold: int,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    train_meta: pd.DataFrame,
    val_meta: pd.DataFrame,
    random_state: int,
) -> TouchscreenCVFoldResult:
    model = _make_random_forest(random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    train_p, val_p, overlap = _fold_participant_details(train_meta, val_meta)
    if overlap:
        raise ValueError(f"Participant overlap in CV fold {fold}: {overlap}")

    return TouchscreenCVFoldResult(
        model_name=MODEL_RANDOM_FOREST,
        fold=fold,
        train_recordings=len(X_train),
        validation_recordings=len(X_val),
        train_participants=len(train_p),
        validation_participants=len(val_p),
        participant_overlap=overlap,
        train_participant_ids=train_p,
        validation_participant_ids=val_p,
        metrics=evaluate_predictions(y_val, y_pred),
        y_pred=y_pred,
        scaler=None,
    )


# --------------------------------------------------------------------------- #
# Metric aggregation                                                          #
# --------------------------------------------------------------------------- #

def aggregate_fold_metrics(fold_results: list[TouchscreenCVFoldResult]) -> pd.DataFrame:
    """Compute mean ± std of fold metrics grouped by model name."""
    rows = [
        {
            "model": r.model_name,
            **{m: float(r.metrics[m]) for m in METRIC_COLUMNS},
        }
        for r in fold_results
    ]
    df = pd.DataFrame(rows)
    agg = df.groupby("model")[METRIC_COLUMNS].agg(["mean", "std"])
    agg.columns = [f"{stat}_{metric}" for metric, stat in agg.columns]
    return agg.reset_index()


# --------------------------------------------------------------------------- #
# Main cross-validation entry point                                           #
# --------------------------------------------------------------------------- #

def cross_validate_touchscreen_development(
    development: TouchscreenMLDataset,
    final_test_participants: set[str] | None = None,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> TouchscreenCVResult:
    """Run participant-aware GroupKFold CV on ONLY the development set.

    Both models use exactly the same folds. Scaling for Logistic Regression
    is fitted inside each fold on the training portion only.
    """
    validate_development_dataset(development, final_test_participants=final_test_participants)

    groups = development.metadata["participant_id"].astype(str)
    n_participants = groups.nunique()
    if n_splits > n_participants:
        raise ValueError(
            f"n_splits={n_splits} exceeds development participant count={n_participants}."
        )

    splitter = GroupKFold(n_splits=n_splits)
    fold_results: list[TouchscreenCVFoldResult] = []

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(development.X, development.y, groups=groups),
        start=1,
    ):
        X_train = development.X.iloc[train_idx].copy()
        X_val = development.X.iloc[val_idx].copy()
        y_train = development.y.iloc[train_idx].copy()
        y_val = development.y.iloc[val_idx].copy()
        train_meta = development.metadata.iloc[train_idx].copy()
        val_meta = development.metadata.iloc[val_idx].copy()

        fold_results.append(
            _run_lr_fold(fold, X_train, X_val, y_train, y_val, train_meta, val_meta, random_state)
        )
        fold_results.append(
            _run_rf_fold(fold, X_train, X_val, y_train, y_val, train_meta, val_meta, random_state)
        )

    return TouchscreenCVResult(
        fold_results=fold_results,
        aggregate_metrics=aggregate_fold_metrics(fold_results),
        development_metadata=development.metadata.copy(),
        final_test_participants=set(final_test_participants or set()),
        cv_strategy="GroupKFold",
        n_splits=n_splits,
    )


# --------------------------------------------------------------------------- #
# High-level pipeline                                                         #
# --------------------------------------------------------------------------- #

def run_touchscreen_baseline(
    ml_dataset: TouchscreenMLDataset | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratify: bool = True,
    n_splits: int = DEFAULT_N_SPLITS,
) -> tuple[TouchscreenCVResult, TouchscreenParticipantSplit]:
    """Build the touchscreen ML dataset, split by participant, then cross-validate development only."""
    dataset = build_touchscreen_ml_dataset() if ml_dataset is None else ml_dataset
    split = split_touchscreen_by_participant(
        dataset,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # Development set only — y_dev carries 'label' column, metadata with participant_id
    development = TouchscreenMLDataset(
        X=split.X_dev.copy(),
        y=split.y_dev.copy(),
        metadata=split.dev_metadata.copy(),
    )
    final_test_participants = split.final_test_participants

    cv_result = cross_validate_touchscreen_development(
        development,
        final_test_participants=final_test_participants,
        n_splits=n_splits,
        random_state=random_state,
    )
    return cv_result, split


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def _fmt(value: object) -> str:
    return f"{float(value):.4f}"


def main() -> int:
    print("Touchscreen Baseline Results")
    print("=" * 50)

    cv_result, split = run_touchscreen_baseline(
        test_size=DEFAULT_TEST_SIZE,
        random_state=DEFAULT_RANDOM_STATE,
        stratify=True,
        n_splits=DEFAULT_N_SPLITS,
    )

    dev_p = set(split.dev_metadata["participant_id"].astype(str))
    test_p = split.final_test_participants
    overlap = dev_p & test_p

    print("\nDataset:")
    print(f"  Development recordings: {len(split.X_dev)}")
    print(f"  Development participants: {len(dev_p)}")
    print(f"  Final-test recordings: {len(split.X_test)} (LOCKED — not used)")
    print(f"  Final-test participants: {len(test_p)} (LOCKED — not used)")
    print(f"  Participant overlap (dev/test): {len(overlap)}")
    print(f"  Features: {split.X_dev.shape[1]}")

    print("\nCross-validation:")
    print(f"  Strategy: {cv_result.cv_strategy}")
    print(f"  Folds: {cv_result.n_splits}")
    print(f"  Grouping: participant_id")

    print("\nPer-Fold Details:")
    for fold in range(1, cv_result.n_splits + 1):
        lr_fold = next(
            r for r in cv_result.fold_results
            if r.fold == fold and r.model_name == MODEL_LOGISTIC_REGRESSION
        )
        print(
            f"  Fold {fold}: "
            f"train_recordings={lr_fold.train_recordings}, "
            f"val_recordings={lr_fold.validation_recordings}, "
            f"train_participants={lr_fold.train_participants}, "
            f"val_participants={lr_fold.validation_participants}, "
            f"participant_overlap={lr_fold.participant_overlap}"
        )

    agg = cv_result.aggregate_metrics.set_index("model")

    for model_name in [MODEL_LOGISTIC_REGRESSION, MODEL_RANDOM_FOREST]:
        row = agg.loc[model_name]
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {row['mean_accuracy']:.4f} ± {row['std_accuracy']:.4f}")
        print(f"  Precision: {row['mean_precision']:.4f} ± {row['std_precision']:.4f}")
        print(f"  Recall:    {row['mean_recall']:.4f} ± {row['std_recall']:.4f}")
        print(f"  F1:        {row['mean_f1']:.4f} ± {row['std_f1']:.4f}")

    print("\nNOTE: Final-test set was not loaded, inspected, or used at any point.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
