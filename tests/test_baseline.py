import numpy as np
import pandas as pd
import pytest

from ml.training.baseline import (
    CLASS_LABELS,
    evaluate_predictions,
    train_baseline_from_split,
    validate_finite_features,
    validate_split_for_baseline,
)
from ml.training.dataset import ParticipantSplit


def make_split() -> ParticipantSplit:
    columns = ["feature_a", "feature_b"]
    X_train = pd.DataFrame(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [2.0, 2.0],
            [2.2, 2.1],
        ],
        columns=columns,
    )
    X_test = pd.DataFrame(
        [
            [0.1, 0.2],
            [2.1, 2.2],
        ],
        columns=columns,
    )
    y_train = pd.Series(["CNT", "CNT", "PD", "PD"], name="label")
    y_test = pd.Series(["CNT", "PD"], name="label")
    train_metadata = pd.DataFrame(
        {
            "participant_id": ["p1", "p2", "p3", "p4"],
            "recording_id": ["r1", "r2", "r3", "r4"],
        }
    )
    test_metadata = pd.DataFrame(
        {
            "participant_id": ["p5", "p6"],
            "recording_id": ["r5", "r6"],
        }
    )
    return ParticipantSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_metadata=train_metadata,
        test_metadata=test_metadata,
    )


def test_scaler_is_fitted_using_training_data_only() -> None:
    split = make_split()

    result = train_baseline_from_split(split, random_state=7)

    assert result.scaler.mean_ == pytest.approx(split.X_train.mean().to_numpy())
    assert result.scaler.mean_ != pytest.approx(
        pd.concat([split.X_train, split.X_test], ignore_index=True).mean().to_numpy()
    )


def test_model_trains_successfully_on_synthetic_data() -> None:
    split = make_split()

    result = train_baseline_from_split(split)

    assert result.model.classes_.tolist() == CLASS_LABELS
    assert hasattr(result.model, "coef_")


def test_prediction_length_matches_test_set() -> None:
    split = make_split()

    result = train_baseline_from_split(split)

    assert len(result.y_pred) == len(split.X_test)


def test_predictions_contain_only_supported_classes() -> None:
    result = train_baseline_from_split(make_split())

    assert set(result.y_pred).issubset(set(CLASS_LABELS))


def test_metrics_are_calculated_correctly() -> None:
    y_true = pd.Series(["CNT", "PD", "PD", "CNT"])
    y_pred = np.array(["CNT", "PD", "CNT", "PD"])

    metrics = evaluate_predictions(y_true, y_pred)

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["confusion_matrix"].tolist() == [[1, 1], [1, 1]]


def test_participant_overlap_is_rejected() -> None:
    split = make_split()
    split = ParticipantSplit(
        X_train=split.X_train,
        X_test=split.X_test,
        y_train=split.y_train,
        y_test=split.y_test,
        train_metadata=split.train_metadata,
        test_metadata=pd.DataFrame({"participant_id": ["p1", "p6"], "recording_id": ["r5", "r6"]}),
    )

    with pytest.raises(ValueError, match="Participant overlap"):
        validate_split_for_baseline(split)


def test_nan_feature_values_raise_clear_error() -> None:
    X = pd.DataFrame({"feature_a": [1.0, np.nan], "feature_b": [2.0, 3.0]})

    with pytest.raises(ValueError, match="NaN feature values"):
        validate_finite_features(X, "X_train")


def test_infinite_feature_values_raise_clear_error() -> None:
    X = pd.DataFrame({"feature_a": [1.0, np.inf], "feature_b": [2.0, 3.0]})

    with pytest.raises(ValueError, match="infinite feature values"):
        validate_finite_features(X, "X_train")


def test_repeated_execution_with_same_random_state_is_reproducible() -> None:
    split = make_split()

    first = train_baseline_from_split(split, random_state=11)
    second = train_baseline_from_split(split, random_state=11)

    assert first.y_pred.tolist() == second.y_pred.tolist()
    assert first.metrics["accuracy"] == second.metrics["accuracy"]
    assert first.model.coef_ == pytest.approx(second.model.coef_)
