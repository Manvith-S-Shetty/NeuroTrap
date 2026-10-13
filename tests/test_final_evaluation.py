import numpy as np
import pandas as pd
import pytest

from ml.training.dataset import FEATURE_COLUMNS, ParticipantSplit
from ml.training.final_evaluation import (
    SELECTED_LOGISTIC_C,
    SELECTED_RF_MAX_DEPTH,
    SELECTED_RF_MIN_SAMPLES_LEAF,
    SELECTED_RF_MIN_SAMPLES_SPLIT,
    SELECTED_RF_N_ESTIMATORS,
    build_prediction_audit_table,
    evaluate_predictions,
    evaluate_selected_models,
    participant_overlap,
    train_selected_logistic_regression,
    train_selected_random_forest,
    validate_final_split,
)


def make_feature_frame(rows: int, offset: float = 0.0) -> pd.DataFrame:
    data = []
    for row_idx in range(rows):
        row = {}
        base = offset + row_idx
        for column_idx, column in enumerate(FEATURE_COLUMNS):
            row[column] = float(base + column_idx / 100)
        data.append(row)
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
            make_feature_frame(2, offset=20.0),
            make_feature_frame(2, offset=30.0),
        ],
        ignore_index=True,
    )
    y_train = pd.Series(["CNT"] * 4 + ["PD"] * 4, name="label")
    y_test = pd.Series(["CNT", "CNT", "PD", "PD"], name="label")
    train_metadata = pd.DataFrame(
        {
            "participant_id": [f"train_p{idx}" for idx in range(8)],
            "recording_id": [f"train_r{idx}" for idx in range(8)],
            "raw_file": [f"train_r{idx}.csv" for idx in range(8)],
        }
    )
    test_metadata = pd.DataFrame(
        {
            "participant_id": [f"test_p{idx}" for idx in range(4)],
            "recording_id": [f"test_r{idx}" for idx in range(4)],
            "raw_file": [f"test_r{idx}.csv" for idx in range(4)],
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


def test_final_test_participants_are_distinct_from_development_participants() -> None:
    split = make_split()

    validate_final_split(split)

    assert participant_overlap(split) == 0


def test_scaler_is_fitted_only_on_development_data() -> None:
    split = make_split()

    result = train_selected_logistic_regression(split, random_state=42)

    assert result.scaler is not None
    assert result.scaler.mean_ == pytest.approx(split.X_train.mean().to_numpy())
    assert result.scaler.mean_ != pytest.approx(pd.concat([split.X_train, split.X_test]).mean().to_numpy())


def test_final_test_data_does_not_affect_scaler_fitting() -> None:
    split = make_split()
    changed_test = split.X_test + 10000
    changed_split = ParticipantSplit(
        X_train=split.X_train,
        X_test=changed_test,
        y_train=split.y_train,
        y_test=split.y_test,
        train_metadata=split.train_metadata,
        test_metadata=split.test_metadata,
    )

    original = train_selected_logistic_regression(split, random_state=42)
    changed = train_selected_logistic_regression(changed_split, random_state=42)

    assert original.scaler is not None
    assert changed.scaler is not None
    assert original.scaler.mean_ == pytest.approx(changed.scaler.mean_)


def test_selected_logistic_regression_c_is_used() -> None:
    result = train_selected_logistic_regression(make_split(), random_state=42)

    assert result.model.C == SELECTED_LOGISTIC_C


def test_selected_random_forest_configuration_is_used() -> None:
    result = train_selected_random_forest(make_split(), random_state=42)

    assert result.model.n_estimators == SELECTED_RF_N_ESTIMATORS
    assert result.model.max_depth == SELECTED_RF_MAX_DEPTH
    assert result.model.min_samples_split == SELECTED_RF_MIN_SAMPLES_SPLIT
    assert result.model.min_samples_leaf == SELECTED_RF_MIN_SAMPLES_LEAF
    assert result.scaler is None


def test_predictions_have_correct_length() -> None:
    split = make_split()

    result = evaluate_selected_models(split, random_state=42)

    assert len(result.logistic_regression.y_pred) == len(split.X_test)
    assert len(result.random_forest.y_pred) == len(split.X_test)


def test_predictions_contain_only_supported_labels() -> None:
    result = evaluate_selected_models(make_split(), random_state=42)

    assert set(result.logistic_regression.y_pred).issubset({"CNT", "PD"})
    assert set(result.random_forest.y_pred).issubset({"CNT", "PD"})


def test_metrics_are_calculated_correctly() -> None:
    y_true = pd.Series(["CNT", "PD", "PD", "CNT"])
    y_pred = np.array(["CNT", "PD", "CNT", "PD"])

    metrics = evaluate_predictions(y_true, y_pred)

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["confusion_matrix"].tolist() == [[1, 1], [1, 1]]


def test_participant_overlap_validation_works() -> None:
    split = make_split()
    bad_metadata = split.test_metadata.copy()
    bad_metadata.loc[0, "participant_id"] = "train_p0"
    bad_split = ParticipantSplit(
        X_train=split.X_train,
        X_test=split.X_test,
        y_train=split.y_train,
        y_test=split.y_test,
        train_metadata=split.train_metadata,
        test_metadata=bad_metadata,
    )

    with pytest.raises(ValueError, match="Participant overlap"):
        validate_final_split(bad_split)


def test_nan_validation_works() -> None:
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
        validate_final_split(bad_split)


def test_inf_validation_works() -> None:
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
        validate_final_split(bad_split)


def test_reproducibility_with_random_state_42() -> None:
    first = evaluate_selected_models(make_split(), random_state=42)
    second = evaluate_selected_models(make_split(), random_state=42)

    assert first.logistic_regression.y_pred.tolist() == second.logistic_regression.y_pred.tolist()
    assert first.random_forest.y_pred.tolist() == second.random_forest.y_pred.tolist()
    assert first.logistic_regression.metrics["f1"] == second.logistic_regression.metrics["f1"]
    assert first.random_forest.metrics["f1"] == second.random_forest.metrics["f1"]


def test_test_metadata_remains_aligned_with_predictions() -> None:
    split = make_split()
    logistic_predictions = np.array(["CNT", "CNT", "PD", "PD"])
    forest_predictions = np.array(["PD", "CNT", "PD", "CNT"])

    audit = build_prediction_audit_table(split, logistic_predictions, forest_predictions)

    assert audit["recording_id"].tolist() == split.test_metadata["recording_id"].tolist()
    assert audit["true_label"].tolist() == split.y_test.tolist()
    assert audit["logistic_regression_prediction"].tolist() == logistic_predictions.tolist()
    assert audit["random_forest_prediction"].tolist() == forest_predictions.tolist()


def test_prediction_metadata_length_mismatch_is_rejected() -> None:
    split = make_split()

    with pytest.raises(ValueError, match="metadata length"):
        build_prediction_audit_table(split, np.array(["CNT"]), np.array(["CNT", "PD", "PD", "CNT"]))
