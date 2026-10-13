"""Final locked-test evaluation for NeuroTap pressure-independent touchscreen model.

This module evaluates the primary 19-feature model selected in development:
Logistic Regression (C=0.001, class_weight=None, solver='liblinear', random_state=42)
trained on the COMPLETE 269-recording development set and evaluated EXACTLY ONCE
on the locked 69-recording / 7-participant final test set at decision threshold 0.50.

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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from ml.training.touchscreen_nopressure_baseline import (  # type: ignore
        CLASS_LABELS,
        DEFAULT_RANDOM_STATE,
    )
    from ml.training.touchscreen_nopressure_dataset import (  # type: ignore
        EXACT_NOPRESSURE_FEATURE_COUNT,
        TouchscreenNoPressureMLDataset,
        TouchscreenNoPressureParticipantSplit,
        build_touchscreen_nopressure_ml_dataset,
        split_touchscreen_nopressure_by_participant,
        validate_touchscreen_nopressure_ml_dataset,
    )
else:
    from ml.training.touchscreen_nopressure_baseline import (
        CLASS_LABELS,
        DEFAULT_RANDOM_STATE,
    )
    from ml.training.touchscreen_nopressure_dataset import (
        EXACT_NOPRESSURE_FEATURE_COUNT,
        TouchscreenNoPressureMLDataset,
        TouchscreenNoPressureParticipantSplit,
        build_touchscreen_nopressure_ml_dataset,
        split_touchscreen_nopressure_by_participant,
        validate_touchscreen_nopressure_ml_dataset,
    )


SELECTED_MODEL_NAME = "Logistic Regression"
SELECTED_C = 0.1
SELECTED_CLASS_WEIGHT = None
SELECTED_SOLVER = "liblinear"
SELECTED_THRESHOLD = 0.50

DEV_CV_F1_MEAN = 0.7450
DEV_CV_ACCURACY_MEAN = 0.7575

EXPECTED_DEV_RECORDINGS = 269
EXPECTED_DEV_PARTICIPANTS = 26
EXPECTED_TEST_RECORDINGS = 69
EXPECTED_TEST_PARTICIPANTS = 7
EXPECTED_TEST_PARTICIPANTS_LIST = ["S09", "S10", "S16", "S18", "S22", "S27", "S32"]
EXPECTED_TEST_CNT = 32
EXPECTED_TEST_PD = 37

OUTPUT_DIR = Path(__file__).resolve().parent / "touchscreen_nopressure_final_evaluation_outputs"


@dataclass(frozen=True)
class TouchscreenNoPressureParticipantSummary:
    participant_id: str
    actual_label: str
    recording_count: int
    CNT_predictions: int
    PD_predictions: int
    majority_prediction: str


@dataclass(frozen=True)
class TouchscreenNoPressureFinalEvaluationResult:
    model_name: str
    model_params: dict[str, Any]
    decision_threshold: float
    dev_recordings: int
    dev_participants: int
    test_recordings: int
    test_participants: int
    participant_overlap: int
    test_participant_ids: tuple[str, ...]
    accuracy: float
    precision: float
    recall_pd: float  # Sensitivity
    specificity_cnt: float
    f1_score: float
    roc_auc: float
    confusion_matrix: list[list[int]]
    participant_summaries: list[TouchscreenNoPressureParticipantSummary]
    scaler: StandardScaler
    model: LogisticRegression

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_params": self.model_params,
            "decision_threshold": self.decision_threshold,
            "feature_count": EXACT_NOPRESSURE_FEATURE_COUNT,
            "dev_recordings": self.dev_recordings,
            "dev_participants": self.dev_participants,
            "test_recordings": self.test_recordings,
            "test_participants": self.test_participants,
            "participant_overlap": self.participant_overlap,
            "test_participant_ids": list(self.test_participant_ids),
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall_pd_sensitivity": self.recall_pd,
            "specificity_cnt": self.specificity_cnt,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "confusion_matrix": self.confusion_matrix,
            "participant_summaries": [asdict(p) for p in self.participant_summaries],
        }


def validate_touchscreen_nopressure_final_split(split: TouchscreenNoPressureParticipantSplit) -> None:
    """Strictly validate split dimensions and participant isolation before evaluation."""
    dev_p = split.development_participants
    test_p = split.final_test_participants
    overlap = dev_p & test_p

    if len(overlap) > 0:
        raise ValueError(f"CRITICAL TEST LEAKAGE: Participant overlap detected: {sorted(overlap)}")

    if len(split.X_dev) != EXPECTED_DEV_RECORDINGS:
        raise ValueError(f"Expected {EXPECTED_DEV_RECORDINGS} dev recordings, got {len(split.X_dev)}")

    if len(dev_p) != EXPECTED_DEV_PARTICIPANTS:
        raise ValueError(f"Expected {EXPECTED_DEV_PARTICIPANTS} dev participants, got {len(dev_p)}")

    if len(split.X_test) != EXPECTED_TEST_RECORDINGS:
        raise ValueError(f"Expected {EXPECTED_TEST_RECORDINGS} test recordings, got {len(split.X_test)}")

    if len(test_p) != EXPECTED_TEST_PARTICIPANTS:
        raise ValueError(f"Expected {EXPECTED_TEST_PARTICIPANTS} test participants, got {len(test_p)}")

    if sorted(test_p) != sorted(EXPECTED_TEST_PARTICIPANTS_LIST):
        raise ValueError(f"Test participants mismatch! Expected {EXPECTED_TEST_PARTICIPANTS_LIST}, got {sorted(test_p)}")


def evaluate_touchscreen_nopressure_final_model(
    split: TouchscreenNoPressureParticipantSplit,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> TouchscreenNoPressureFinalEvaluationResult:
    """Train selected 19-feature model on dev set ONLY, evaluate ONCE on locked test set."""
    validate_touchscreen_nopressure_final_split(split)

    X_dev, y_dev = split.X_dev, split.y_dev
    X_test, y_test = split.X_test, split.y_test

    # 1. Fit StandardScaler ONLY on development data
    scaler = StandardScaler()
    X_dev_scaled = scaler.fit_transform(X_dev)
    X_test_scaled = scaler.transform(X_test)

    # 2. Fit Logistic Regression ONLY on development data
    model = LogisticRegression(
        C=SELECTED_C,
        class_weight=SELECTED_CLASS_WEIGHT,
        solver=SELECTED_SOLVER,
        max_iter=1000,
        random_state=random_state,
    )
    model.fit(X_dev_scaled, y_dev)

    # 3. Evaluate EXACTLY ONCE on locked final test set
    y_test_probs = model.predict_proba(X_test_scaled)[:, 1]  # P(PD)
    y_test_preds = np.where(y_test_probs >= SELECTED_THRESHOLD, "PD", "CNT")

    # 4. Compute metrics
    b_true = (y_test == "PD").astype(int)
    b_pred = (y_test_preds == "PD").astype(int)

    acc = float(accuracy_score(b_true, b_pred))
    prec = float(precision_score(b_true, b_pred, zero_division=0))
    rec_pd = float(recall_score(b_true, b_pred, zero_division=0))  # Sensitivity
    f1 = float(f1_score(b_true, b_pred, zero_division=0))
    auc = float(roc_auc_score(b_true, y_test_probs))

    # Confusion matrix: rows=actual, cols=pred -> [[TN, FP], [FN, TP]]
    cm_array = confusion_matrix(b_true, b_pred, labels=[0, 1])
    tn, fp, fn, tp = cm_array[0, 0], cm_array[0, 1], cm_array[1, 0], cm_array[1, 1]
    spec_cnt = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # 5. Participant-level summary on test set
    test_meta = split.test_metadata.copy()
    test_meta["pred_label"] = y_test_preds

    participant_summaries: list[TouchscreenNoPressureParticipantSummary] = []
    for pid, group in test_meta.groupby("participant_id"):
        actual_label = str(group["label"].iloc[0])
        cnt_preds = int((group["pred_label"] == "CNT").sum())
        pd_preds = int((group["pred_label"] == "PD").sum())
        maj_pred = "PD" if pd_preds >= cnt_preds else "CNT"

        participant_summaries.append(
            TouchscreenNoPressureParticipantSummary(
                participant_id=str(pid),
                actual_label=actual_label,
                recording_count=len(group),
                CNT_predictions=cnt_preds,
                PD_predictions=pd_preds,
                majority_prediction=maj_pred,
            )
        )

    return TouchscreenNoPressureFinalEvaluationResult(
        model_name=SELECTED_MODEL_NAME,
        model_params={
            "C": SELECTED_C,
            "class_weight": SELECTED_CLASS_WEIGHT,
            "solver": SELECTED_SOLVER,
            "random_state": random_state,
        },
        decision_threshold=SELECTED_THRESHOLD,
        dev_recordings=len(X_dev),
        dev_participants=len(split.development_participants),
        test_recordings=len(X_test),
        test_participants=len(split.final_test_participants),
        participant_overlap=len(split.development_participants & split.final_test_participants),
        test_participant_ids=tuple(sorted(split.final_test_participants)),
        accuracy=acc,
        precision=prec,
        recall_pd=rec_pd,
        specificity_cnt=spec_cnt,
        f1_score=f1,
        roc_auc=auc,
        confusion_matrix=cm_array.tolist(),
        participant_summaries=participant_summaries,
        scaler=scaler,
        model=model,
    )


def run_touchscreen_nopressure_final_evaluation() -> TouchscreenNoPressureFinalEvaluationResult:
    """Execute evaluation and write JSON report to output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = build_touchscreen_nopressure_ml_dataset()
    split = split_touchscreen_nopressure_by_participant(dataset, test_size=0.2, random_state=DEFAULT_RANDOM_STATE, stratify=True)

    result = evaluate_touchscreen_nopressure_final_model(split, random_state=DEFAULT_RANDOM_STATE)

    output_file = OUTPUT_DIR / "touchscreen_nopressure_final_evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    return result


def main() -> int:
    result = run_touchscreen_nopressure_final_evaluation()

    print("==========================================================")
    print("  FINAL LOCKED-TEST EVALUATION: 19-FEATURE NO-PRESSURE MODEL")
    print("==========================================================")
    print(f"Model Name:          {result.model_name}")
    print(f"Hyperparameters:     {result.model_params}")
    print(f"Decision Threshold:  {result.decision_threshold:.2f}")
    print(f"Development Set:     {result.dev_recordings} recordings ({result.dev_participants} participants)")
    print(f"Final Test Set:      {result.test_recordings} recordings ({result.test_participants} participants)")
    print(f"Participant Overlap: {result.participant_overlap}")
    print(f"Test Participant IDs:{list(result.test_participant_ids)}")
    print("----------------------------------------------------------")
    print(f"Accuracy:            {result.accuracy:.4f} ({result.accuracy * 100:.2f}%)")
    print(f"Precision (PD):      {result.precision:.4f}")
    print(f"Sensitivity (PD):    {result.recall_pd:.4f} (Recall)")
    print(f"Specificity (CNT):   {result.specificity_cnt:.4f}")
    print(f"F1 Score:            {result.f1_score:.4f}")
    print(f"ROC-AUC:             {result.roc_auc:.4f}")
    print("----------------------------------------------------------")
    print("Confusion Matrix [[TN (CNT), FP], [FN, TP (PD)]]:")
    print(f"  {result.confusion_matrix[0]}")
    print(f"  {result.confusion_matrix[1]}")
    print("----------------------------------------------------------")
    print("Participant-Level Test Performance (Majority Voting):")
    for p in result.participant_summaries:
        status = "CORRECT" if p.majority_prediction == p.actual_label else "MISCLASSIFIED"
        print(f"  Participant {p.participant_id}: Actual={p.actual_label}, Pred={p.majority_prediction} ({p.PD_predictions}/{p.recording_count} PD) -> {status}")
    print("==========================================================")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
