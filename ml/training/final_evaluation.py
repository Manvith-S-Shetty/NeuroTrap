"""Final held-out evaluation for the selected NeuroTap models.

This is a one-shot evaluation stage. It trains selected configurations on the
development set and evaluates once on the final held-out test set.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from ml.training.baseline import CLASS_LABELS, evaluate_predictions, validate_finite_features  # type: ignore
    from ml.training.dataset import (  # type: ignore
        MLDataset,
        ParticipantSplit,
        build_feature_table,
        build_ml_dataset,
        split_by_participant,
        validate_participant_split,
    )
    from ml.training.random_forest import DEFAULT_RANDOM_STATE, validate_expected_features  # type: ignore
else:
    from ml.training.baseline import CLASS_LABELS, evaluate_predictions, validate_finite_features
    from ml.training.dataset import (
        MLDataset,
        ParticipantSplit,
        build_feature_table,
        build_ml_dataset,
        split_by_participant,
        validate_participant_split,
    )
    from ml.training.random_forest import DEFAULT_RANDOM_STATE, validate_expected_features


SELECTED_LOGISTIC_C = 100.0
SELECTED_RF_N_ESTIMATORS = 200
SELECTED_RF_MAX_DEPTH = 5
SELECTED_RF_MIN_SAMPLES_SPLIT = 2
SELECTED_RF_MIN_SAMPLES_LEAF = 1

DEVELOPMENT_CV_RESULTS = {
    "Tuned Logistic Regression": {"cv_mean_f1": 0.6413, "cv_std_f1": 0.0872},
    "Tuned Random Forest": {"cv_mean_f1": 0.5975, "cv_std_f1": 0.1539},
}


@dataclass(frozen=True)
class FinalModelResult:
    model_name: str
    model: LogisticRegression | RandomForestClassifier
    metrics: dict[str, object]
    y_pred: np.ndarray
    scaler: StandardScaler | None = None


@dataclass(frozen=True)
class FinalEvaluationResult:
    split: ParticipantSplit
    logistic_regression: FinalModelResult
    random_forest: FinalModelResult
    predictions: pd.DataFrame


def _validate_supported_labels(y: pd.Series, name: str) -> None:
    if y.isna().any():
        raise ValueError(f"{name} contains missing labels.")
    unsupported = sorted(set(y) - set(CLASS_LABELS))
    if unsupported:
        raise ValueError(f"{name} contains unsupported class labels: {unsupported}")


def validate_final_split(split: ParticipantSplit) -> None:
    """Validate final train/test boundary and feature matrices."""
    validate_participant_split(split)
    validate_expected_features(split.X_train, "X_train")
    validate_expected_features(split.X_test, "X_test")
    validate_finite_features(split.X_train, "X_train")
    validate_finite_features(split.X_test, "X_test")
    _validate_supported_labels(split.y_train, "y_train")
    _validate_supported_labels(split.y_test, "y_test")


def train_selected_logistic_regression(
    split: ParticipantSplit,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> FinalModelResult:
    """Fit scaler on development data only, then train selected Logistic Regression."""
    validate_final_split(split)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(split.X_train)
    X_test_scaled = scaler.transform(split.X_test)

    model = LogisticRegression(C=SELECTED_LOGISTIC_C, random_state=random_state, max_iter=1000)
    model.fit(X_train_scaled, split.y_train)
    y_pred = model.predict(X_test_scaled)
    metrics = evaluate_predictions(split.y_test, y_pred)
    return FinalModelResult(
        model_name="Tuned Logistic Regression",
        model=model,
        metrics=metrics,
        y_pred=y_pred,
        scaler=scaler,
    )


def train_selected_random_forest(
    split: ParticipantSplit,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> FinalModelResult:
    """Train selected Random Forest on unscaled development features."""
    validate_final_split(split)
    model = RandomForestClassifier(
        n_estimators=SELECTED_RF_N_ESTIMATORS,
        max_depth=SELECTED_RF_MAX_DEPTH,
        min_samples_split=SELECTED_RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=SELECTED_RF_MIN_SAMPLES_LEAF,
        random_state=random_state,
    )
    model.fit(split.X_train, split.y_train)
    y_pred = model.predict(split.X_test)
    metrics = evaluate_predictions(split.y_test, y_pred)
    return FinalModelResult(
        model_name="Tuned Random Forest",
        model=model,
        metrics=metrics,
        y_pred=y_pred,
        scaler=None,
    )


def build_prediction_audit_table(
    split: ParticipantSplit,
    logistic_predictions: np.ndarray,
    random_forest_predictions: np.ndarray,
) -> pd.DataFrame:
    """Return final-test metadata aligned with true and predicted labels."""
    if len(logistic_predictions) != len(split.test_metadata):
        raise ValueError("Logistic Regression predictions are not aligned with test metadata length.")
    if len(random_forest_predictions) != len(split.test_metadata):
        raise ValueError("Random Forest predictions are not aligned with test metadata length.")

    audit = split.test_metadata.reset_index(drop=True).copy()
    audit["true_label"] = split.y_test.reset_index(drop=True).to_numpy()
    audit["logistic_regression_prediction"] = logistic_predictions
    audit["random_forest_prediction"] = random_forest_predictions
    return audit


def evaluate_selected_models(split: ParticipantSplit, random_state: int = DEFAULT_RANDOM_STATE) -> FinalEvaluationResult:
    """Train selected models on development data and evaluate once on final test data."""
    validate_final_split(split)
    logistic = train_selected_logistic_regression(split, random_state=random_state)
    forest = train_selected_random_forest(split, random_state=random_state)
    predictions = build_prediction_audit_table(split, logistic.y_pred, forest.y_pred)
    return FinalEvaluationResult(
        split=split,
        logistic_regression=logistic,
        random_forest=forest,
        predictions=predictions,
    )


def run_final_evaluation(
    ml_dataset: MLDataset | None = None,
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratify: bool = True,
) -> FinalEvaluationResult:
    """Recover the existing split and run final held-out evaluation once."""
    dataset = build_ml_dataset() if ml_dataset is None else ml_dataset
    split = split_by_participant(dataset, test_size=test_size, random_state=random_state, stratify=stratify)
    return evaluate_selected_models(split, random_state=random_state)


def participant_overlap(split: ParticipantSplit) -> int:
    train = set(split.train_metadata["participant_id"].astype(str))
    test = set(split.test_metadata["participant_id"].astype(str))
    return len(train & test)


def _print_model_result(result: FinalModelResult) -> None:
    print(f"Accuracy: {result.metrics['accuracy']:.4f}")
    print(f"Precision: {result.metrics['precision']:.4f}")
    print(f"Recall: {result.metrics['recall']:.4f}")
    print(f"F1: {result.metrics['f1']:.4f}")
    print("Confusion matrix labels: CNT, PD")
    print(result.metrics["confusion_matrix"])


def main() -> int:
    feature_table = build_feature_table()
    ml_dataset = build_ml_dataset(feature_table)
    result = run_final_evaluation(ml_dataset=ml_dataset, test_size=0.2, random_state=DEFAULT_RANDOM_STATE, stratify=True)
    split = result.split

    print("# NeuroTap Final Held-Out Evaluation")
    print("\n## Dataset")
    print(f"Development recordings: {len(split.X_train)}")
    print(f"Development participants: {split.train_metadata['participant_id'].nunique()}")
    print(f"Final test recordings: {len(split.X_test)}")
    print(f"Final test participants: {split.test_metadata['participant_id'].nunique()}")
    print(f"Participant overlap: {participant_overlap(split)}")
    print("Final test class counts:")
    print(split.y_test.value_counts().reindex(CLASS_LABELS, fill_value=0).to_string())

    print("\n## Logistic Regression")
    print(f"C: {SELECTED_LOGISTIC_C}")
    print(f"random_state: {DEFAULT_RANDOM_STATE}")
    print("scaler: StandardScaler fit on development data only")
    _print_model_result(result.logistic_regression)

    print("\n## Random Forest")
    print(f"n_estimators: {SELECTED_RF_N_ESTIMATORS}")
    print(f"max_depth: {SELECTED_RF_MAX_DEPTH}")
    print(f"min_samples_split: {SELECTED_RF_MIN_SAMPLES_SPLIT}")
    print(f"min_samples_leaf: {SELECTED_RF_MIN_SAMPLES_LEAF}")
    print(f"random_state: {DEFAULT_RANDOM_STATE}")
    print("scaler: none")
    _print_model_result(result.random_forest)

    print("\n## Final Test Comparison")
    print("Model | Accuracy | Precision | Recall | F1")
    print("--- | ---: | ---: | ---: | ---:")
    for model_result in [result.logistic_regression, result.random_forest]:
        print(
            f"{model_result.model_name} | "
            f"{model_result.metrics['accuracy']:.4f} | "
            f"{model_result.metrics['precision']:.4f} | "
            f"{model_result.metrics['recall']:.4f} | "
            f"{model_result.metrics['f1']:.4f}"
        )

    print("\n## Development CV vs Final Test")
    print("Model | CV Mean F1 | CV Std F1 | Final Test F1")
    print("--- | ---: | ---: | ---:")
    for model_result in [result.logistic_regression, result.random_forest]:
        cv = DEVELOPMENT_CV_RESULTS[model_result.model_name]
        print(
            f"{model_result.model_name} | "
            f"{cv['cv_mean_f1']:.4f} | "
            f"{cv['cv_std_f1']:.4f} | "
            f"{model_result.metrics['f1']:.4f}"
        )

    print("\n## Prediction Audit Table")
    print(result.predictions.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
