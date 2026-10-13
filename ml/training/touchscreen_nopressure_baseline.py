"""Baseline pressure-independent touchscreen model training with participant-aware cross-validation.

Trains Logistic Regression and Random Forest baselines on the 19 pressure-independent
touchscreen features using ONLY the development set from the participant-aware split.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from ml.training.touchscreen_nopressure_dataset import (  # type: ignore
        TouchscreenNoPressureMLDataset,
        TouchscreenNoPressureParticipantSplit,
        build_touchscreen_nopressure_ml_dataset,
        split_touchscreen_nopressure_by_participant,
        validate_touchscreen_nopressure_ml_dataset,
    )
else:
    from ml.training.touchscreen_nopressure_dataset import (
        TouchscreenNoPressureMLDataset,
        TouchscreenNoPressureParticipantSplit,
        build_touchscreen_nopressure_ml_dataset,
        split_touchscreen_nopressure_by_participant,
        validate_touchscreen_nopressure_ml_dataset,
    )


CLASS_LABELS = ["CNT", "PD"]
MODEL_LOGISTIC_REGRESSION = "Logistic Regression"
MODEL_RANDOM_FOREST = "Random Forest"
METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

DEFAULT_RANDOM_STATE = 42
DEFAULT_N_SPLITS = 5
DEFAULT_TEST_SIZE = 0.2

# Baseline hyperparameter settings
LR_C = 1.0
LR_MAX_ITER = 1000

RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_SPLIT = 2
RF_MIN_SAMPLES_LEAF = 1


@dataclass(frozen=True)
class TouchscreenNoPressureCVFoldResult:
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
    metrics: dict[str, float]
    y_pred: np.ndarray
    y_prob: np.ndarray
    scaler: StandardScaler | None


@dataclass(frozen=True)
class TouchscreenNoPressureCVResult:
    """Aggregate result of participant-aware cross-validation."""

    fold_results: list[TouchscreenNoPressureCVFoldResult]

    def summary_dataframe(self) -> pd.DataFrame:
        rows = []
        for res in self.fold_results:
            row = {
                "model": res.model_name,
                "fold": res.fold,
                "train_recs": res.train_recordings,
                "val_recs": res.validation_recordings,
                "train_parts": res.train_participants,
                "val_parts": res.validation_participants,
                "overlap": res.participant_overlap,
            }
            row.update(res.metrics)
            rows.append(row)
        return pd.DataFrame(rows)

    def mean_metrics(self) -> pd.DataFrame:
        df = self.summary_dataframe()
        numeric_cols = [c for c in METRIC_COLUMNS if c in df.columns]
        means = df.groupby("model")[numeric_cols].mean()
        stds = df.groupby("model")[numeric_cols].std()
        combined = pd.DataFrame()
        for col in numeric_cols:
            combined[f"{col}_mean"] = means[col]
            combined[f"{col}_std"] = stds[col]
        return combined


def validate_development_dataset(dev_dataset: TouchscreenNoPressureMLDataset) -> None:
    """Validate that development dataset meets basic integrity checks."""
    validate_touchscreen_nopressure_ml_dataset(dev_dataset)
    if dev_dataset.X.empty or dev_dataset.y.empty or dev_dataset.metadata.empty:
        raise ValueError("Development dataset is empty.")


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict[str, float]:
    """Calculate accuracy, precision, recall, F1, and ROC-AUC metrics."""
    binary_true = (y_true == "PD").astype(int)
    binary_pred = (pd.Series(y_pred) == "PD").astype(int)

    metrics = {
        "accuracy": float(accuracy_score(binary_true, binary_pred)),
        "precision": float(precision_score(binary_true, binary_pred, zero_division=0)),
        "recall": float(recall_score(binary_true, binary_pred, zero_division=0)),
        "f1": float(f1_score(binary_true, binary_pred, zero_division=0)),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(binary_true, y_prob))
        except Exception:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


def train_evaluate_nopressure_baseline_cv(
    dev_dataset: TouchscreenNoPressureMLDataset,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> TouchscreenNoPressureCVResult:
    """Perform 5-fold participant-aware cross-validation on development dataset."""
    validate_development_dataset(dev_dataset)

    X = dev_dataset.X.reset_index(drop=True)
    y = dev_dataset.y.reset_index(drop=True)
    metadata = dev_dataset.metadata.reset_index(drop=True)
    groups = metadata["participant_id"].astype(str).to_numpy()

    gkf = GroupKFold(n_splits=n_splits)

    fold_results: list[TouchscreenNoPressureCVFoldResult] = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()
        meta_train, meta_val = metadata.iloc[train_idx], metadata.iloc[val_idx]

        train_parts = set(meta_train["participant_id"].astype(str))
        val_parts = set(meta_val["participant_id"].astype(str))
        overlap = len(train_parts & val_parts)
        if overlap > 0:
            raise ValueError(f"Fold {fold_idx}: Participant overlap detected ({overlap}).")

        # 1. Logistic Regression (Requires StandardScaler)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        lr = LogisticRegression(
            C=LR_C,
            max_iter=LR_MAX_ITER,
            random_state=random_state,
        )
        lr.fit(X_train_scaled, y_train)
        lr_pred = lr.predict(X_val_scaled)
        lr_prob = lr.predict_proba(X_val_scaled)[:, 1] if len(lr.classes_) == 2 else np.zeros(len(y_val))
        lr_metrics = evaluate_predictions(y_val, lr_pred, lr_prob)

        fold_results.append(
            TouchscreenNoPressureCVFoldResult(
                model_name=MODEL_LOGISTIC_REGRESSION,
                fold=fold_idx,
                train_recordings=len(train_idx),
                validation_recordings=len(val_idx),
                train_participants=len(train_parts),
                validation_participants=len(val_parts),
                participant_overlap=overlap,
                train_participant_ids=tuple(sorted(train_parts)),
                validation_participant_ids=tuple(sorted(val_parts)),
                metrics=lr_metrics,
                y_pred=lr_pred,
                y_prob=lr_prob,
                scaler=scaler,
            )
        )

        # 2. Random Forest (Unscaled)
        rf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            random_state=random_state,
        )
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_val)
        rf_prob = rf.predict_proba(X_val)[:, 1] if len(rf.classes_) == 2 else np.zeros(len(y_val))
        rf_metrics = evaluate_predictions(y_val, rf_pred, rf_prob)

        fold_results.append(
            TouchscreenNoPressureCVFoldResult(
                model_name=MODEL_RANDOM_FOREST,
                fold=fold_idx,
                train_recordings=len(train_idx),
                validation_recordings=len(val_idx),
                train_participants=len(train_parts),
                validation_participants=len(val_parts),
                participant_overlap=overlap,
                train_participant_ids=tuple(sorted(train_parts)),
                validation_participant_ids=tuple(sorted(val_parts)),
                metrics=rf_metrics,
                y_pred=rf_pred,
                y_prob=rf_prob,
                scaler=None,
            )
        )

    return TouchscreenNoPressureCVResult(fold_results=fold_results)


def main() -> int:
    print("# Pressure-Independent Touchscreen Baseline Model CV Evaluation\n")
    dataset = build_touchscreen_nopressure_ml_dataset()
    split = split_touchscreen_nopressure_by_participant(dataset, test_size=0.2, random_state=DEFAULT_RANDOM_STATE, stratify=True)

    dev_dataset = split.development_recordings
    print(f"Development recordings: {len(dev_dataset.X)}")
    print(f"Development participants: {len(split.development_participants)}\n")

    cv_result = train_evaluate_nopressure_baseline_cv(dev_dataset, n_splits=5, random_state=DEFAULT_RANDOM_STATE)

    print("## Per-Fold Performance")
    print(cv_result.summary_dataframe().to_string(index=False))

    print("\n## Aggregate Performance (Mean ± Std)")
    print(cv_result.mean_metrics().to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
