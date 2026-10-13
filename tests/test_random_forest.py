import numpy as np
import pandas as pd
import pytest

from ml.training.dataset import FEATURE_COLUMNS, ParticipantSplit
from ml.training.random_forest import (
    CLASS_LABELS,
    evaluate_predictions,
    train_random_forest_from_split,
    validate_expected_features,
    validate_split_for_random_forest,
)


def make_feature_frame(rows: int, offset: float = 0.0) -> pd.DataFrame:
    data = []
    for row_idx in range(rows):
        data.append(
            {
                column: float(offset + row_idx + column_idx / 100)
                for column_idx, column in enumerate(FEATURE_COLUMNS)
            }
        )
    return pd.DataFrame(data, columns=FEATURE_COLUMNS)


def make_split() -> ParticipantSplit:
    X_train = pd.concat(
        [
            make_feature_frame(4, offset=0.0),
            make_feature_frame(4, offset=10.0),
        ],
        ignore_index=True,
    )
    X_test = pd.concat(
        [
            make_feature_frame(2, offset=1.0),
            make_feature_frame(2, offset=11.0),
        ],
        ignore_index=True,
    )
    y_train = pd.Series(["CNT"] * 4 + ["PD"] * 4, name="label")
    y_test = pd.Series(["CNT", "CNT", "PD", "PD"], name="label")
    train_metadata = pd.DataFrame(
        {
            "participant_id": [f"train_p{idx}" for idx in range(8)],
            "recording_id": [f"train_r{idx}" for idx in range(8)],
        }
    )
    test_metadata = pd.DataFrame(
        {
            "participant_id": [f"test_p{idx}" for idx in range(4)],
            "recording_id": [f"test_r{idx}" for idx in range(4)],
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


def test_model_trains_successfully_on_synthetic_data() -> None:
    result = train_random_forest_from_split(make_split(), random_state=7, n_estimators=25)

    assert result.model.classes_.tolist() == CLASS_LABELS
    assert len(result.model.estimators_) == 25


def test_predictions_have_correct_length() -> None:
    split = make_split()

    result = train_random_forest_from_split(split, random_state=7, n_estimators=25)

    assert len(result.y_pred) == len(split.X_test)


def test_predictions_contain_only_supported_classes() -> None:
    result = train_random_forest_from_split(make_split(), random_state=7, n_estimators=25)

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


def test_reproducibility_with_same_random_state() -> None:
    split = make_split()

    first = train_random_forest_from_split(split, random_state=13, n_estimators=30)
    second = train_random_forest_from_split(split, random_state=13, n_estimators=30)

    assert first.y_pred.tolist() == second.y_pred.tolist()
    assert first.metrics["accuracy"] == second.metrics["accuracy"]
    assert first.model.feature_importances_ == pytest.approx(second.model.feature_importances_)


def test_nan_feature_values_raise_clear_error() -> None:
    split = make_split()
    X_train = split.X_train.copy()
    X_train.loc[0, "hold_mean"] = np.nan
    bad_split = ParticipantSplit(
        X_train=X_train,
        X_test=split.X_test,
        y_train=split.y_train,
        y_test=split.y_test,
        train_metadata=split.train_metadata,
        test_metadata=split.test_metadata,
    )

    with pytest.raises(ValueError, match="NaN feature values"):
        validate_split_for_random_forest(bad_split)


def test_infinite_feature_values_raise_clear_error() -> None:
    split = make_split()
    X_test = split.X_test.copy()
    X_test.loc[0, "hold_mean"] = np.inf
    bad_split = ParticipantSplit(
        X_train=split.X_train,
        X_test=X_test,
        y_train=split.y_train,
        y_test=split.y_test,
        train_metadata=split.train_metadata,
        test_metadata=split.test_metadata,
    )

    with pytest.raises(ValueError, match="infinite feature values"):
        validate_split_for_random_forest(bad_split)


def test_participant_overlap_is_rejected() -> None:
    split = make_split()
    test_metadata = split.test_metadata.copy()
    test_metadata.loc[0, "participant_id"] = "train_p0"
    bad_split = ParticipantSplit(
        X_train=split.X_train,
        X_test=split.X_test,
        y_train=split.y_train,
        y_test=split.y_test,
        train_metadata=split.train_metadata,
        test_metadata=test_metadata,
    )

    with pytest.raises(ValueError, match="Participant overlap"):
        validate_split_for_random_forest(bad_split)


def test_missing_expected_feature_is_rejected() -> None:
    X = make_feature_frame(2).drop(columns=["hold_mean"])

    with pytest.raises(ValueError, match="missing expected feature columns"):
        validate_expected_features(X, "X_train")


def test_unexpected_feature_is_rejected() -> None:
    X = make_feature_frame(2)
    X["participant_id"] = ["p1", "p2"]

    with pytest.raises(ValueError, match="unexpected feature columns"):
        validate_expected_features(X, "X_train")


def test_feature_order_is_validated() -> None:
    X = make_feature_frame(2)
    reordered = X[list(reversed(FEATURE_COLUMNS))]

    with pytest.raises(ValueError, match="expected order"):
        validate_expected_features(reordered, "X_train")
