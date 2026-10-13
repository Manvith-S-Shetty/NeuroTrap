"""Controlled hyperparameter tuning and threshold selection for pressure-independent touchscreen models.

Uses ONLY the 269-recording / 26-participant development set from the participant-aware split.
The 69-recording / 7-participant final test set is NEVER loaded, inspected, or used here.
All model selection and threshold tuning are based on participant-aware CV F1.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from ml.training.touchscreen_nopressure_baseline import (  # type: ignore
        CLASS_LABELS,
        DEFAULT_RANDOM_STATE,
        MODEL_LOGISTIC_REGRESSION,
        MODEL_RANDOM_FOREST,
    )
    from ml.training.touchscreen_nopressure_dataset import (  # type: ignore
        TouchscreenNoPressureMLDataset,
        TouchscreenNoPressureParticipantSplit,
        build_touchscreen_nopressure_ml_dataset,
        split_touchscreen_nopressure_by_participant,
    )
else:
    from ml.training.touchscreen_nopressure_baseline import (
        CLASS_LABELS,
        DEFAULT_RANDOM_STATE,
        MODEL_LOGISTIC_REGRESSION,
        MODEL_RANDOM_FOREST,
    )
    from ml.training.touchscreen_nopressure_dataset import (
        TouchscreenNoPressureMLDataset,
        TouchscreenNoPressureParticipantSplit,
        build_touchscreen_nopressure_ml_dataset,
        split_touchscreen_nopressure_by_participant,
    )


LR_C_VALUES: list[float] = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
LR_CLASS_WEIGHTS: list[str | None] = [None, "balanced"]
LR_SOLVERS: list[str] = ["liblinear", "lbfgs"]

RF_N_ESTIMATORS: list[int] = [100, 200, 500]
RF_MAX_DEPTH: list[int | None] = [None, 3, 5, 10]
RF_MIN_SAMPLES_LEAF: list[int] = [1, 2, 5]
RF_CLASS_WEIGHTS: list[str | None] = [None, "balanced"]

METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
SELECTION_METRIC = "f1"
DEFAULT_N_SPLITS = 5
THRESHOLD_GRID = np.linspace(0.1, 0.9, 81)

OUTPUT_DIR = Path(__file__).resolve().parent / "touchscreen_nopressure_tuning_outputs"


@dataclass(frozen=True)
class TouchscreenNoPressureTuningResult:
    model_name: str
    params: dict[str, Any]
    mean_accuracy: float
    mean_precision: float
    mean_recall: float
    mean_f1: float
    mean_roc_auc: float
    optimal_threshold: float
    threshold_f1: float


def evaluate_threshold_performance(
    y_true: pd.Series, y_probs: np.ndarray, threshold: float
) -> dict[str, float]:
    binary_true = (y_true == "PD").astype(int)
    binary_pred = (y_probs >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(binary_true, binary_pred)),
        "precision": float(precision_score(binary_true, binary_pred, zero_division=0)),
        "recall": float(recall_score(binary_true, binary_pred, zero_division=0)),
        "f1": float(f1_score(binary_true, binary_pred, zero_division=0)),
    }


def find_optimal_threshold(y_true: pd.Series, y_probs: np.ndarray) -> tuple[float, float]:
    """Find the threshold (0.1 to 0.9) maximizing validation F1 score on out-of-fold probabilities."""
    best_thresh = 0.5
    best_f1 = -1.0
    for thresh in THRESHOLD_GRID:
        perf = evaluate_threshold_performance(y_true, y_probs, thresh)
        if perf["f1"] > best_f1:
            best_f1 = perf["f1"]
            best_thresh = float(thresh)
    return best_thresh, best_f1


def tune_nopressure_models(
    dev_dataset: TouchscreenNoPressureMLDataset,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[list[TouchscreenNoPressureTuningResult], TouchscreenNoPressureTuningResult]:
    """Grid search hyperparameter configurations on dev dataset using 5-fold GroupKFold."""
    X = dev_dataset.X.reset_index(drop=True)
    y = dev_dataset.y.reset_index(drop=True)
    metadata = dev_dataset.metadata.reset_index(drop=True)
    groups = metadata["participant_id"].astype(str).to_numpy()

    gkf = GroupKFold(n_splits=n_splits)
    results: list[TouchscreenNoPressureTuningResult] = []

    # 1. Logistic Regression Grid Search
    for c, cw, solver in product(LR_C_VALUES, LR_CLASS_WEIGHTS, LR_SOLVERS):
        if solver == "lbfgs" and c < 1e-4:
            continue
        params = {"C": c, "class_weight": cw, "solver": solver}

        oof_probs = np.zeros(len(y))
        fold_metrics = []

        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_va_sc = scaler.transform(X_va)

            model = LogisticRegression(
                C=c,
                class_weight=cw,
                solver=solver,
                max_iter=1000,
                random_state=random_state,
            )
            model.fit(X_tr_sc, y_tr)

            probs = model.predict_proba(X_va_sc)[:, 1] if len(model.classes_) == 2 else np.zeros(len(y_va))
            oof_probs[val_idx] = probs

            preds = (probs >= 0.5).astype(int)
            b_true = (y_va == "PD").astype(int)
            fold_metrics.append({
                "accuracy": accuracy_score(b_true, preds),
                "precision": precision_score(b_true, preds, zero_division=0),
                "recall": recall_score(b_true, preds, zero_division=0),
                "f1": f1_score(b_true, preds, zero_division=0),
                "roc_auc": roc_auc_score(b_true, probs) if len(np.unique(b_true)) > 1 else 0.5,
            })

        metrics_df = pd.DataFrame(fold_metrics)
        opt_thresh, thresh_f1 = find_optimal_threshold(y, oof_probs)

        results.append(
            TouchscreenNoPressureTuningResult(
                model_name=MODEL_LOGISTIC_REGRESSION,
                params=params,
                mean_accuracy=float(metrics_df["accuracy"].mean()),
                mean_precision=float(metrics_df["precision"].mean()),
                mean_recall=float(metrics_df["recall"].mean()),
                mean_f1=float(metrics_df["f1"].mean()),
                mean_roc_auc=float(metrics_df["roc_auc"].mean()),
                optimal_threshold=opt_thresh,
                threshold_f1=thresh_f1,
            )
        )

    # 2. Random Forest Grid Search
    for n_est, max_d, min_l, cw in product(RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_LEAF, RF_CLASS_WEIGHTS):
        params = {
            "n_estimators": n_est,
            "max_depth": max_d,
            "min_samples_leaf": min_l,
            "class_weight": cw,
        }

        oof_probs = np.zeros(len(y))
        fold_metrics = []

        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

            model = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=max_d,
                min_samples_leaf=min_l,
                class_weight=cw,
                random_state=random_state,
            )
            model.fit(X_tr, y_tr)

            probs = model.predict_proba(X_va)[:, 1] if len(model.classes_) == 2 else np.zeros(len(y_va))
            oof_probs[val_idx] = probs

            preds = (probs >= 0.5).astype(int)
            b_true = (y_va == "PD").astype(int)
            fold_metrics.append({
                "accuracy": accuracy_score(b_true, preds),
                "precision": precision_score(b_true, preds, zero_division=0),
                "recall": recall_score(b_true, preds, zero_division=0),
                "f1": f1_score(b_true, preds, zero_division=0),
                "roc_auc": roc_auc_score(b_true, probs) if len(np.unique(b_true)) > 1 else 0.5,
            })

        metrics_df = pd.DataFrame(fold_metrics)
        opt_thresh, thresh_f1 = find_optimal_threshold(y, oof_probs)

        results.append(
            TouchscreenNoPressureTuningResult(
                model_name=MODEL_RANDOM_FOREST,
                params=params,
                mean_accuracy=float(metrics_df["accuracy"].mean()),
                mean_precision=float(metrics_df["precision"].mean()),
                mean_recall=float(metrics_df["recall"].mean()),
                mean_f1=float(metrics_df["f1"].mean()),
                mean_roc_auc=float(metrics_df["roc_auc"].mean()),
                optimal_threshold=opt_thresh,
                threshold_f1=thresh_f1,
            )
        )

    # Rank results by mean_f1 (and threshold_f1)
    best_result = sorted(results, key=lambda r: (r.mean_f1, r.threshold_f1, r.mean_roc_auc), reverse=True)[0]
    return results, best_result


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = build_touchscreen_nopressure_ml_dataset()
    split = split_touchscreen_nopressure_by_participant(dataset, test_size=0.2, random_state=DEFAULT_RANDOM_STATE, stratify=True)

    print("# Pressure-Independent Touchscreen Hyperparameter & Threshold Tuning\n")
    print(f"Development recordings: {len(split.X_dev)}")
    print(f"Development participants: {len(split.development_participants)}\n")

    all_results, best = tune_nopressure_models(split.development_recordings, n_splits=5, random_state=DEFAULT_RANDOM_STATE)

    print("## Top 5 Models Evaluated (by CV F1 Score)")
    sorted_res = sorted(all_results, key=lambda r: (r.mean_f1, r.threshold_f1), reverse=True)[:5]
    for i, res in enumerate(sorted_res, 1):
        print(f"{i}. {res.model_name} | Params: {res.params}")
        print(f"   Mean F1 (0.5): {res.mean_f1:.4f} | Accuracy: {res.mean_accuracy:.4f} | ROC-AUC: {res.mean_roc_auc:.4f}")
        print(f"   Optimal Threshold: {res.optimal_threshold:.2f} (F1 at threshold: {res.threshold_f1:.4f})\n")

    print("## Selected Model & Configuration")
    print(f"Model Name: {best.model_name}")
    print(f"Hyperparameters: {best.params}")
    print(f"Selected Threshold: {best.optimal_threshold:.2f}")
    print(f"Development CV F1 Score: {best.mean_f1:.4f}")

    # Save summary report
    summary = {
        "selected_model": best.model_name,
        "selected_params": best.params,
        "selected_threshold": best.optimal_threshold,
        "dev_cv_f1": best.mean_f1,
        "dev_cv_accuracy": best.mean_accuracy,
        "dev_cv_roc_auc": best.mean_roc_auc,
        "total_models_evaluated": len(all_results),
    }
    with open(OUTPUT_DIR / "nopressure_tuning_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
