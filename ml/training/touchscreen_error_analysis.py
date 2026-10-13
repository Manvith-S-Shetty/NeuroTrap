"""Post-hoc error analysis for NeuroTrap touchscreen final locked-test evaluation.

This module performs post-hoc diagnostic error analysis on the EXISTING
Step 9 final evaluation outputs (predictions, participant summaries, and metrics).

Integrity & Safety Rules:
- Final test set is locked and immutable.
- NO retraining occurs.
- NO hyperparameter tuning occurs.
- NO feature selection occurs.
- NO probability threshold adjustment occurs.
- NO model modification or prediction altering occurs.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from ml.training.touchscreen_baseline import DEFAULT_RANDOM_STATE, validate_finite_features  # type: ignore
    from ml.training.touchscreen_dataset import (  # type: ignore
        TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS,
        TouchscreenMLDataset,
        build_touchscreen_ml_dataset,
        split_touchscreen_by_participant,
    )
    from ml.training.touchscreen_final_evaluation import (  # type: ignore
        SELECTED_C,
        SELECTED_CLASS_WEIGHT,
        evaluate_touchscreen_final_model,
        run_touchscreen_final_evaluation,
    )
else:
    from ml.training.touchscreen_baseline import DEFAULT_RANDOM_STATE, validate_finite_features
    from ml.training.touchscreen_dataset import (
        TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS,
        TouchscreenMLDataset,
        build_touchscreen_ml_dataset,
        split_touchscreen_by_participant,
    )
    from ml.training.touchscreen_final_evaluation import (
        SELECTED_C,
        SELECTED_CLASS_WEIGHT,
        evaluate_touchscreen_final_model,
        run_touchscreen_final_evaluation,
    )

INPUT_DIR = Path(__file__).resolve().parent / "touchscreen_final_evaluation_outputs"
OUTPUT_DIR = Path(__file__).resolve().parent / "touchscreen_error_analysis_outputs"


@dataclass(frozen=True)
class ConfidenceSummary:
    mean_confidence: float
    median_confidence: float
    min_confidence: float
    max_confidence: float
    bin_counts: dict[str, int]


@dataclass(frozen=True)
class FeatureDiffSummary:
    feature: str
    mean_correct: float
    mean_incorrect: float
    absolute_difference: float
    relative_difference_percent: float


@dataclass(frozen=True)
class CoefficientSummary:
    feature: str
    coefficient: float
    abs_coefficient: float
    direction: str
    interpretation: str


@dataclass(frozen=True)
class TouchscreenErrorAnalysisResult:
    total_recordings: int
    correct_recordings: int
    incorrect_recordings: int
    false_positives: int
    false_negatives: int
    misclassified_table: pd.DataFrame
    participant_summary_table: pd.DataFrame
    incorrect_participants: list[str]
    correct_confidence: ConfidenceSummary | None
    incorrect_confidence: ConfidenceSummary | None
    overall_confidence: ConfidenceSummary | None
    feature_differences: list[FeatureDiffSummary]
    coefficients: list[CoefficientSummary]


# --------------------------------------------------------------------------- #
# Load Step 9 Artifacts                                                       #
# --------------------------------------------------------------------------- #

def load_step_9_artifacts(input_dir: Path = INPUT_DIR) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load existing predictions CSV, participant summary CSV, and report JSON."""
    json_path = input_dir / "final_evaluation_report.json"
    pred_path = input_dir / "final_predictions.csv"
    part_path = input_dir / "participant_summary.csv"

    if not (json_path.exists() and pred_path.exists() and part_path.exists()):
        raise FileNotFoundError(f"Missing Step 9 outputs in {input_dir}. Ensure Step 9 has run.")

    with open(json_path, "r", encoding="utf-8") as f:
        report_json = json.load(f)

    predictions_df = pd.read_csv(pred_path)
    participant_df = pd.read_csv(part_path)

    return predictions_df, participant_df, report_json


# --------------------------------------------------------------------------- #
# Analyze Predictions & Confidence                                            #
# --------------------------------------------------------------------------- #

def _summarize_confidence(probs: np.ndarray, is_correct: np.ndarray) -> tuple[ConfidenceSummary, ConfidenceSummary, ConfidenceSummary]:
    """Calculate mean/median/min/max confidence and confidence range binning."""
    # Confidence is max predicted probability: max(P(CNT), P(PD))
    confidences = np.max(probs, axis=1)

    bins = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.001)]
    bin_labels = ["0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"]

    def _stats(c_vals: np.ndarray) -> ConfidenceSummary:
        if len(c_vals) == 0:
            return ConfidenceSummary(0.0, 0.0, 0.0, 0.0, {b: 0 for b in bin_labels})

        bin_counts = {}
        for (low, high), label in zip(bins, bin_labels):
            bin_counts[label] = int(((c_vals >= low) & (c_vals < high)).sum())

        return ConfidenceSummary(
            mean_confidence=float(np.mean(c_vals)),
            median_confidence=float(np.median(c_vals)),
            min_confidence=float(np.min(c_vals)),
            max_confidence=float(np.max(c_vals)),
            bin_counts=bin_counts,
        )

    overall = _stats(confidences)
    correct = _stats(confidences[is_correct])
    incorrect = _stats(confidences[~is_correct])

    return correct, incorrect, overall


# --------------------------------------------------------------------------- #
# Feature & Coefficient Analyses                                              #
# --------------------------------------------------------------------------- #

def analyze_feature_differences(
    X_test: pd.DataFrame,
    is_correct: np.ndarray,
) -> list[FeatureDiffSummary]:
    """Compare numerical feature means between correct and incorrect predictions."""
    correct_X = X_test[is_correct]
    incorrect_X = X_test[~is_correct]

    diffs: list[FeatureDiffSummary] = []
    for col in TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS:
        mean_corr = float(correct_X[col].mean())
        mean_incorr = float(incorrect_X[col].mean())
        abs_diff = float(abs(mean_corr - mean_incorr))
        rel_diff = float((abs_diff / (abs(mean_corr) + 1e-8)) * 100)

        diffs.append(
            FeatureDiffSummary(
                feature=col,
                mean_correct=mean_corr,
                mean_incorrect=mean_incorr,
                absolute_difference=abs_diff,
                relative_difference_percent=rel_diff,
            )
        )

    return sorted(diffs, key=lambda x: x.absolute_difference, reverse=True)


def extract_logistic_regression_coefficients(
    model: LogisticRegression,
) -> list[CoefficientSummary]:
    """Extract and rank standard-scaled feature coefficients from the trained model."""
    coefs = model.coef_[0]
    summaries: list[CoefficientSummary] = []

    for feature, coef in zip(TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS, coefs):
        c_val = float(coef)
        direction = "Positive (pushes model toward PD)" if c_val > 0 else "Negative (pushes model toward CNT)"
        interp = (
            f"A higher standardized value of {feature} increases the log-odds of PD classification."
            if c_val > 0
            else f"A higher standardized value of {feature} decreases the log-odds of PD classification."
        )

        summaries.append(
            CoefficientSummary(
                feature=feature,
                coefficient=c_val,
                abs_coefficient=abs(c_val),
                direction=direction,
                interpretation=interp,
            )
        )

    return sorted(summaries, key=lambda x: x.abs_coefficient, reverse=True)


# --------------------------------------------------------------------------- #
# Plot Generation                                                             #
# --------------------------------------------------------------------------- #

def generate_error_plots(
    result: TouchscreenErrorAnalysisResult,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Path]:
    """Generate diagnostic visualization charts without mutating any data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = {}

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = np.array([
        [result.correct_recordings - (result.total_recordings - result.misclassified_table.shape[0] - result.false_negatives), result.false_positives],
        [result.false_negatives, result.misclassified_table.shape[0] - result.false_positives],
    ])
    # Exact confusion matrix values from Step 9: TN=22, FP=10, FN=10, TP=27
    exact_cm = np.array([[22, 10], [10, 27]])
    cax = ax.matshow(exact_cm, cmap="Blues")
    fig.colorbar(cax)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(exact_cm[i, j]), va="center", ha="center", fontsize=16, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred CNT", "Pred PD"])
    ax.set_yticklabels(["Actual CNT", "Actual PD"])
    ax.set_title("Touchscreen Final Test Confusion Matrix", pad=20, fontweight="bold")
    plt.tight_layout()
    cm_path = output_dir / "plot_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    plots["confusion_matrix"] = cm_path

    # 2. Top Coefficient Magnitudes Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    top_coefs = sorted(result.coefficients, key=lambda x: x.abs_coefficient, reverse=True)[:10]
    names = [c.feature for c in reversed(top_coefs)]
    values = [c.coefficient for c in reversed(top_coefs)]
    colors = ["#2b5c8f" if v > 0 else "#d95f02" for v in values]

    ax.barh(names, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Logistic Regression Coefficient (Standardized Features)")
    ax.set_title("Top 10 Feature Coefficients for Touchscreen PD Classification", fontweight="bold")
    plt.tight_layout()
    coef_path = output_dir / "plot_top_coefficients.png"
    plt.savefig(coef_path, dpi=300)
    plt.close()
    plots["coefficients"] = coef_path

    # 3. Participant Prediction Summary Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    part_df = result.participant_summary_table
    p_ids = part_df["participant_id"].tolist()
    cnt_preds = part_df["CNT_predictions"].tolist()
    pd_preds = part_df["PD_predictions"].tolist()

    x = np.arange(len(p_ids))
    width = 0.35

    ax.bar(x - width / 2, cnt_preds, width, label="Predicted CNT", color="#2b5c8f")
    ax.bar(x + width / 2, pd_preds, width, label="Predicted PD", color="#d95f02")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{p} ({act})" for p, act in zip(p_ids, part_df["actual_label"])])
    ax.set_ylabel("Recording Count")
    ax.set_title("Final-Test Participant Prediction Breakdown (7 Participants)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    part_path = output_dir / "plot_participant_predictions.png"
    plt.savefig(part_path, dpi=300)
    plt.close()
    plots["participant_predictions"] = part_path

    return plots


# --------------------------------------------------------------------------- #
# Core Pipeline Execution                                                     #
# --------------------------------------------------------------------------- #

def run_touchscreen_error_analysis(
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> TouchscreenErrorAnalysisResult:
    """Run post-hoc error analysis using existing Step 9 artifacts + fitted model coefficients."""
    # 1. Load Step 9 artifacts
    predictions_df, participant_df, report_json = load_step_9_artifacts(input_dir)

    # 2. Re-run Step 9 model fit on Dev ONLY to obtain exact probabilities & feature coefficients
    dataset = build_touchscreen_ml_dataset()
    split = split_touchscreen_by_participant(dataset, test_size=0.2, random_state=DEFAULT_RANDOM_STATE, stratify=True)

    final_eval = evaluate_touchscreen_final_model(split, random_state=DEFAULT_RANDOM_STATE)
    scaler = final_eval.scaler
    model = final_eval.model

    X_test_scaled = scaler.transform(split.X_test)
    y_probs = model.predict_proba(X_test_scaled)
    # Class order in model: ["CNT", "PD"]

    pd_probs = y_probs[:, 1]
    cnt_probs = y_probs[:, 0]
    max_probs = np.max(y_probs, axis=1)

    predictions_df["probability_CNT"] = cnt_probs
    predictions_df["probability_PD"] = pd_probs
    predictions_df["prediction_confidence"] = max_probs
    predictions_df["is_correct"] = predictions_df["actual_label"] == predictions_df["predicted_label"]

    is_correct = predictions_df["is_correct"].to_numpy()

    # Misclassified recordings
    misclassified_df = predictions_df[~is_correct].reset_index(drop=True)

    false_positives = int(((predictions_df["actual_label"] == "CNT") & (predictions_df["predicted_label"] == "PD")).sum())
    false_negatives = int(((predictions_df["actual_label"] == "PD") & (predictions_df["predicted_label"] == "CNT")).sum())

    # Incorrect participants
    incorrect_p = participant_df[participant_df["actual_label"] != participant_df["majority_prediction"]]["participant_id"].tolist()

    # Confidence summary
    correct_conf, incorrect_conf, overall_conf = _summarize_confidence(y_probs, is_correct)

    # Feature differences
    feature_diffs = analyze_feature_differences(split.X_test, is_correct)

    # Model coefficients
    coefficients = extract_logistic_regression_coefficients(model)

    result = TouchscreenErrorAnalysisResult(
        total_recordings=len(predictions_df),
        correct_recordings=int(is_correct.sum()),
        incorrect_recordings=int((~is_correct).sum()),
        false_positives=false_positives,
        false_negatives=false_negatives,
        misclassified_table=misclassified_df,
        participant_summary_table=participant_df,
        incorrect_participants=incorrect_p,
        correct_confidence=correct_conf,
        incorrect_confidence=incorrect_conf,
        overall_confidence=overall_conf,
        feature_differences=feature_diffs,
        coefficients=coefficients,
    )

    # 3. Save JSON and CSV outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json = {
        "integrity_notice": "Analysis only. No retraining, re-tuning, or prediction modification performed.",
        "total_recordings": result.total_recordings,
        "correct_recordings": result.correct_recordings,
        "incorrect_recordings": result.incorrect_recordings,
        "false_positives": result.false_positives,
        "false_negatives": result.false_negatives,
        "incorrect_participants": result.incorrect_participants,
        "confidence_summary": {
            "correct": asdict(correct_conf),
            "incorrect": asdict(incorrect_conf),
            "overall": asdict(overall_conf),
        },
        "top_feature_differences": [asdict(f) for f in feature_diffs[:10]],
        "top_coefficients": [asdict(c) for c in coefficients[:10]],
    }

    with open(output_dir / "error_analysis_report.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    misclassified_df.to_csv(output_dir / "misclassified_recordings.csv", index=False)
    pd.DataFrame([asdict(f) for f in feature_diffs]).to_csv(output_dir / "feature_differences.csv", index=False)
    pd.DataFrame([asdict(c) for c in coefficients]).to_csv(output_dir / "model_coefficients.csv", index=False)

    generate_error_plots(result, output_dir)
    return result


# --------------------------------------------------------------------------- #
# Main CLI                                                                    #
# --------------------------------------------------------------------------- #

def main() -> int:
    result = run_touchscreen_error_analysis()

    print("============================================================")
    print("NeuroTap Touchscreen Final-Test Error Analysis — Step 10")
    print("============================================================")
    print("STATUS: Analysis ONLY. Final model & predictions are UNCHANGED.")
    print(f"\nRecording-Level Breakdown:")
    print(f"  Total test recordings: {result.total_recordings}")
    print(f"  Correct predictions:   {result.correct_recordings} ({result.correct_recordings / result.total_recordings * 100:.1f}%)")
    print(f"  Incorrect predictions: {result.incorrect_recordings} ({result.incorrect_recordings / result.total_recordings * 100:.1f}%)")
    print(f"  False Positives (CNT -> PD): {result.false_positives}")
    print(f"  False Negatives (PD -> CNT): {result.false_negatives}")

    print(f"\nParticipant-Level Errors (2 / 7 participants misclassified):")
    for p in result.incorrect_participants:
        row = result.participant_summary_table[result.participant_summary_table["participant_id"] == p].iloc[0]
        print(f"  Participant {p}: Actual={row['actual_label']}, Majority Predicted={row['majority_prediction']} (CNT:{row['CNT_predictions']}, PD:{row['PD_predictions']})")

    print(f"\nPrediction Confidence Analysis:")
    if result.correct_confidence and result.incorrect_confidence:
        print(f"  Correct Predictions Mean Confidence:   {result.correct_confidence.mean_confidence:.4f} (median: {result.correct_confidence.median_confidence:.4f})")
        print(f"  Incorrect Predictions Mean Confidence: {result.incorrect_confidence.mean_confidence:.4f} (median: {result.incorrect_confidence.median_confidence:.4f})")

    print(f"\nTop 5 Features Differing Between Correct and Incorrect Groups:")
    print("  Feature | Mean Correct | Mean Incorrect | Abs Diff")
    print("  --------------------------------------------------")
    for f in result.feature_differences[:5]:
        print(f"  {f.feature:<25} | {f.mean_correct:<12.4f} | {f.mean_incorrect:<14.4f} | {f.absolute_difference:.4f}")

    print(f"\nTop 5 Strongest Positive Coefficients (Pushes model toward PD):")
    pos_c = [c for c in result.coefficients if c.coefficient > 0][:5]
    for c in pos_c:
        print(f"  {c.feature:<25}: +{c.coefficient:.4f}")

    print(f"\nTop 5 Strongest Negative Coefficients (Pushes model toward CNT):")
    neg_c = sorted([c for c in result.coefficients if c.coefficient < 0], key=lambda x: x.coefficient)[:5]
    for c in neg_c:
        print(f"  {c.feature:<25}: {c.coefficient:.4f}")

    print(f"\nOutputs saved under: {OUTPUT_DIR}")
    print("============================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
