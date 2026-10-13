"""Final locked-test evaluation for NeuroTrap touchscreen model.

This module evaluates the primary model selected in Step 8:
Logistic Regression (C=0.001, class_weight=None, random_state=42)
trained on the COMPLETE 269-recording development set and evaluated
EXACTLY ONCE on the locked 69-recording / 7-participant final test set.

Integrity Rules Enforced:
- 0 participant overlap between development and final-test sets.
- Scaler is fitted ONLY on the development set.
- Model is fitted ONLY on the development set.
- Final test set is evaluated exactly once. No tuning after evaluation.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from ml.training.touchscreen_baseline import (  # type: ignore
        CLASS_LABELS,
        DEFAULT_RANDOM_STATE,
        validate_finite_features,
    )
    from ml.training.touchscreen_dataset import (  # type: ignore
        EXACT_FEATURE_COUNT,
        TouchscreenMLDataset,
        TouchscreenParticipantSplit,
        build_touchscreen_ml_dataset,
        split_touchscreen_by_participant,
        validate_touchscreen_ml_dataset,
    )
else:
    from ml.training.touchscreen_baseline import (
        CLASS_LABELS,
        DEFAULT_RANDOM_STATE,
        validate_finite_features,
    )
    from ml.training.touchscreen_dataset import (
        EXACT_FEATURE_COUNT,
        TouchscreenMLDataset,
        TouchscreenParticipantSplit,
        build_touchscreen_ml_dataset,
        split_touchscreen_by_participant,
        validate_touchscreen_ml_dataset,
    )


# --------------------------------------------------------------------------- #
# Constants & Fixed Model Configuration                                        #
# --------------------------------------------------------------------------- #

SELECTED_MODEL_NAME = "Logistic Regression"
SELECTED_C = 0.001
SELECTED_CLASS_WEIGHT = None

DEV_SGKF_CV_F1_MEAN = 0.8319
DEV_SGKF_CV_F1_STD = 0.1438

EXPECTED_DEV_RECORDINGS = 269
EXPECTED_DEV_PARTICIPANTS = 26
EXPECTED_TEST_RECORDINGS = 69
EXPECTED_TEST_PARTICIPANTS = 7
EXPECTED_TEST_CNT = 32
EXPECTED_TEST_PD = 37

OUTPUT_DIR = Path(__file__).resolve().parent / "touchscreen_final_evaluation_outputs"


@dataclass(frozen=True)
class TouchscreenParticipantSummary:
    participant_id: str
    actual_label: str
    recording_count: int
    CNT_predictions: int
    PD_predictions: int
    majority_prediction: str


@dataclass(frozen=True)
class TouchscreenFinalEvaluationResult:
    model_name: str
    model_params: dict[str, Any]
    dev_recordings: int
    dev_participants: int
    test_recordings: int
    test_participants: int
    participant_overlap: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]
    actual_class_distribution: dict[str, int]
    predicted_class_distribution: dict[str, int]
    correct_class_predictions: dict[str, int]
    participant_summaries: list[TouchscreenParticipantSummary]
    dev_cv_f1_mean: float
    dev_cv_f1_std: float
    f1_difference: float
    prediction_table: pd.DataFrame
    scaler: StandardScaler
    model: LogisticRegression


# --------------------------------------------------------------------------- #
# Integrity Validation Guard                                                   #
# --------------------------------------------------------------------------- #

def validate_touchscreen_final_split(split: TouchscreenParticipantSplit) -> None:
    """Strictly validate final test locked assumptions and integrity."""
    dev_p = set(split.dev_metadata["participant_id"].astype(str))
    test_p = set(split.test_metadata["participant_id"].astype(str))
    overlap = dev_p & test_p

    if len(overlap) > 0:
        raise ValueError(f"Data leakage detected! Participant overlap: {sorted(overlap)}")

    if len(split.X_dev) != EXPECTED_DEV_RECORDINGS:
        raise ValueError(f"Expected {EXPECTED_DEV_RECORDINGS} dev recordings, got {len(split.X_dev)}")

    if len(dev_p) != EXPECTED_DEV_PARTICIPANTS:
        raise ValueError(f"Expected {EXPECTED_DEV_PARTICIPANTS} dev participants, got {len(dev_p)}")

    if len(split.X_test) != EXPECTED_TEST_RECORDINGS:
        raise ValueError(f"Expected {EXPECTED_TEST_RECORDINGS} test recordings, got {len(split.X_test)}")

    if len(test_p) != EXPECTED_TEST_PARTICIPANTS:
        raise ValueError(f"Expected {EXPECTED_TEST_PARTICIPANTS} test participants, got {len(test_p)}")

    if len(split.X_dev.columns) != EXACT_FEATURE_COUNT:
        raise ValueError(f"Expected {EXACT_FEATURE_COUNT} feature columns, got {len(split.X_dev.columns)}")

    if len(split.X_test.columns) != EXACT_FEATURE_COUNT:
        raise ValueError(f"Expected {EXACT_FEATURE_COUNT} feature columns, got {len(split.X_test.columns)}")

    validate_finite_features(split.X_dev, "X_dev")
    validate_finite_features(split.X_test, "X_test")

    test_cnt = (split.y_test == "CNT").sum()
    test_pd = (split.y_test == "PD").sum()

    if test_cnt != EXPECTED_TEST_CNT or test_pd != EXPECTED_TEST_PD:
        raise ValueError(
            f"Expected {EXPECTED_TEST_CNT} CNT and {EXPECTED_TEST_PD} PD in final test, got {test_cnt} CNT / {test_pd} PD"
        )


# --------------------------------------------------------------------------- #
# Participant Level Summary Helper                                            #
# --------------------------------------------------------------------------- #

def build_participant_summaries(
    test_metadata: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> list[TouchscreenParticipantSummary]:
    """Generate participant-level prediction summary for reporting."""
    df = test_metadata.copy()
    df["actual_label"] = y_true.values
    df["prediction"] = y_pred

    summaries: list[TouchscreenParticipantSummary] = []
    for pid, group in df.groupby("participant_id", sort=True):
        actual = group["actual_label"].iloc[0]
        cnt_preds = int((group["prediction"] == "CNT").sum())
        pd_preds = int((group["prediction"] == "PD").sum())
        majority = "PD" if pd_preds >= cnt_preds else "CNT"

        summaries.append(
            TouchscreenParticipantSummary(
                participant_id=str(pid),
                actual_label=str(actual),
                recording_count=len(group),
                CNT_predictions=cnt_preds,
                PD_predictions=pd_preds,
                majority_prediction=majority,
            )
        )
    return summaries


# --------------------------------------------------------------------------- #
# Core Evaluation Function                                                    #
# --------------------------------------------------------------------------- #

def evaluate_touchscreen_final_model(
    split: TouchscreenParticipantSplit,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> TouchscreenFinalEvaluationResult:
    """Train selected model on dev set and evaluate once on held-out test set."""
    validate_touchscreen_final_split(split)

    # 1. Fit scaler ONLY on development set
    scaler = StandardScaler()
    X_dev_scaled = scaler.fit_transform(split.X_dev)
    X_test_scaled = scaler.transform(split.X_test)

    # 2. Train fixed selected Logistic Regression model ONLY on development set
    model = LogisticRegression(
        C=SELECTED_C,
        class_weight=SELECTED_CLASS_WEIGHT,
        max_iter=1000,
        random_state=random_state,
    )
    model.fit(X_dev_scaled, split.y_dev)

    # 3. Evaluate EXACTLY ONCE on test set
    y_pred = model.predict(X_test_scaled)

    # 4. Compute metrics
    acc = float(accuracy_score(split.y_test, y_pred))
    prec = float(precision_score(split.y_test, y_pred, pos_label="PD", zero_division=0))
    rec = float(recall_score(split.y_test, y_pred, pos_label="PD", zero_division=0))
    f1 = float(f1_score(split.y_test, y_pred, pos_label="PD", zero_division=0))

    cm = confusion_matrix(split.y_test, y_pred, labels=CLASS_LABELS).tolist()

    actual_counts = {label: int((split.y_test == label).sum()) for label in CLASS_LABELS}
    pred_counts = {label: int((y_pred == label).sum()) for label in CLASS_LABELS}

    # Confusion matrix: cm[0][0] = True CNT (Pred CNT, Act CNT), cm[1][1] = True PD (Pred PD, Act PD)
    correct_counts = {
        "CNT": int(cm[0][0]),
        "PD": int(cm[1][1]),
    }

    participant_summaries = build_participant_summaries(split.test_metadata, split.y_test, y_pred)

    f1_diff = f1 - DEV_SGKF_CV_F1_MEAN

    # Prediction audit dataframe
    pred_df = split.test_metadata.reset_index(drop=True).copy()
    pred_df["actual_label"] = split.y_test.reset_index(drop=True).values
    pred_df["predicted_label"] = y_pred

    return TouchscreenFinalEvaluationResult(
        model_name=SELECTED_MODEL_NAME,
        model_params={"C": SELECTED_C, "class_weight": SELECTED_CLASS_WEIGHT, "random_state": random_state},
        dev_recordings=len(split.X_dev),
        dev_participants=split.dev_metadata["participant_id"].nunique(),
        test_recordings=len(split.X_test),
        test_participants=split.test_metadata["participant_id"].nunique(),
        participant_overlap=0,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        confusion_matrix=cm,
        actual_class_distribution=actual_counts,
        predicted_class_distribution=pred_counts,
        correct_class_predictions=correct_counts,
        participant_summaries=participant_summaries,
        dev_cv_f1_mean=DEV_SGKF_CV_F1_MEAN,
        dev_cv_f1_std=DEV_SGKF_CV_F1_STD,
        f1_difference=f1_diff,
        prediction_table=pred_df,
        scaler=scaler,
        model=model,
    )


# --------------------------------------------------------------------------- #
# High-Level Pipeline & Output Persistence                                     #
# --------------------------------------------------------------------------- #

def run_touchscreen_final_evaluation(
    ml_dataset: TouchscreenMLDataset | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> TouchscreenFinalEvaluationResult:
    """Run end-to-end touchscreen final evaluation pipeline."""
    dataset = build_touchscreen_ml_dataset() if ml_dataset is None else ml_dataset
    split = split_touchscreen_by_participant(dataset, test_size=0.2, random_state=random_state, stratify=True)
    return evaluate_touchscreen_final_model(split, random_state=random_state)


def save_touchscreen_final_evaluation_results(
    result: TouchscreenFinalEvaluationResult,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Path]:
    """Save machine-readable evaluation results (JSON and CSV)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_dict = {
        "model_name": result.model_name,
        "model_params": result.model_params,
        "dev_recordings": result.dev_recordings,
        "dev_participants": result.dev_participants,
        "test_recordings": result.test_recordings,
        "test_participants": result.test_participants,
        "participant_overlap": result.participant_overlap,
        "metrics": {
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
        },
        "confusion_matrix": {
            "labels": CLASS_LABELS,
            "matrix": result.confusion_matrix,
        },
        "actual_class_distribution": result.actual_class_distribution,
        "predicted_class_distribution": result.predicted_class_distribution,
        "correct_class_predictions": result.correct_class_predictions,
        "dev_cv_comparison": {
            "dev_cv_f1_mean": result.dev_cv_f1_mean,
            "dev_cv_f1_std": result.dev_cv_f1_std,
            "final_test_f1": result.f1,
            "f1_difference": result.f1_difference,
        },
        "participant_summaries": [asdict(p) for p in result.participant_summaries],
    }

    json_path = output_dir / "final_evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    csv_path = output_dir / "final_predictions.csv"
    result.prediction_table.to_csv(csv_path, index=False)

    participant_csv_path = output_dir / "participant_summary.csv"
    pd.DataFrame([asdict(p) for p in result.participant_summaries]).to_csv(participant_csv_path, index=False)

    return {
        "json": json_path,
        "predictions_csv": csv_path,
        "participant_csv": participant_csv_path,
    }


# --------------------------------------------------------------------------- #
# CLI Execution                                                               #
# --------------------------------------------------------------------------- #

def main() -> int:
    result = run_touchscreen_final_evaluation()
    saved = save_touchscreen_final_evaluation_results(result)

    print("============================================================")
    print("NeuroTap Touchscreen Final Locked-Test Evaluation — Step 9")
    print("============================================================")
    print(f"\nModel Configuration:")
    print(f"  Name: {result.model_name}")
    print(f"  C: {result.model_params['C']}")
    print(f"  class_weight: {result.model_params['class_weight']}")
    print(f"  random_state: {result.model_params['random_state']}")

    print(f"\nDataset Sizes & Integrity:")
    print(f"  Development recordings: {result.dev_recordings}")
    print(f"  Development participants: {result.dev_participants}")
    print(f"  Final-test recordings: {result.test_recordings}")
    print(f"  Final-test participants: {result.test_participants}")
    print(f"  Participant leakage/overlap: {result.participant_overlap}")

    print(f"\nPerformance Metrics on Held-Out Final Test Set:")
    print(f"  Accuracy:  {result.accuracy:.4f}")
    print(f"  Precision: {result.precision:.4f}")
    print(f"  Recall:    {result.recall:.4f}")
    print(f"  F1 Score:  {result.f1:.4f}")

    print(f"\nClass Distribution & Predictions:")
    print(f"  Actual:    CNT={result.actual_class_distribution['CNT']}, PD={result.actual_class_distribution['PD']}")
    print(f"  Predicted: CNT={result.predicted_class_distribution['CNT']}, PD={result.predicted_class_distribution['PD']}")
    print(f"  Correct:   CNT={result.correct_class_predictions['CNT']}, PD={result.correct_class_predictions['PD']}")

    cm = result.confusion_matrix
    print(f"\nConfusion Matrix:")
    print(f"                 Pred CNT    Pred PD")
    print(f"Actual CNT          {cm[0][0]:<10}  {cm[0][1]:<10}")
    print(f"Actual PD           {cm[1][0]:<10}  {cm[1][1]:<10}")

    print(f"\nParticipant-Level Summary (7 Test Participants):")
    print("  Participant ID | Actual | Recordings | Pred CNT | Pred PD | Majority Pred")
    print("  -----------------------------------------------------------------------")
    for p in result.participant_summaries:
        print(f"  {p.participant_id:<14} | {p.actual_label:<6} | {p.recording_count:<10} | {p.CNT_predictions:<8} | {p.PD_predictions:<7} | {p.majority_prediction}")

    print(f"\nDevelopment CV vs Final Test Comparison:")
    print(f"  Development StratifiedGroupKFold Mean F1: {result.dev_cv_f1_mean:.4f} ± {result.dev_cv_f1_std:.4f}")
    print(f"  Final Held-Out Test F1:                 {result.f1:.4f}")
    print(f"  Difference (Final Test F1 - Dev CV F1): {result.f1_difference:+.4f}")

    print(f"\nSaved Reports:")
    print(f"  JSON: {saved['json']}")
    print(f"  Predictions CSV: {saved['predictions_csv']}")
    print(f"  Participant CSV: {saved['participant_csv']}")
    print("============================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
