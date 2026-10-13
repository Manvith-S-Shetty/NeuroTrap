import numpy as np
import pandas as pd
import pytest

from ml.training.cross_validation import (
    MODEL_LOGISTIC_REGRESSION,
    MODEL_RANDOM_FOREST,
    CVFoldResult,
    aggregate_fold_metrics,
    cross_validate_development_set,
    make_development_dataset,
)
from ml.training.dataset import FEATURE_COLUMNS, MLDataset, ParticipantSplit


def make_feature_frame(participant_index: int) -> dict[str, float]:
    base = 0.0 if participant_index % 2 == 0 else 10.0
    return {
        column: float(base + participant_index / 10 + column_index / 1000)
        for column_index, column in enumerate(FEATURE_COLUMNS)
    }


def make_development_data(participant_count: int = 10) -> MLDataset:
    rows = [make_feature_frame(idx) for idx in range(participant_count)]
    labels = ["CNT" if idx % 2 == 0 else "PD" for idx in range(participant_count)]
    metadata = pd.DataFrame(
        {
            "participant_id": [f"p{idx}" for idx in range(participant_count)],
            "recording_id": [f"r{idx}" for idx in range(participant_count)],
        }
    )
    return MLDataset(
        X=pd.DataFrame(rows, columns=FEATURE_COLUMNS),
        y=pd.Series(labels, name="label"),
        metadata=metadata,
    )


def test_group_kfold_is_reported_as_cv_strategy() -> None:
    result = cross_validate_development_set(make_development_data(), n_splits=5, random_state=7)

    assert result.cv_strategy == "GroupKFold"


def test_same_participant_never_appears_in_train_and_validation() -> None:
    result = cross_validate_development_set(make_development_data(), n_splits=5, random_state=7)

    for fold in result.fold_results:
        assert set(fold.train_participant_ids).isdisjoint(set(fold.validation_participant_ids))
        assert fold.participant_overlap == 0


def test_every_participant_is_used_as_validation_once_per_model() -> None:
    development = make_development_data()

    result = cross_validate_development_set(development, n_splits=5, random_state=7)

    logistic_validation_ids = [
        participant_id
        for fold in result.fold_results
        if fold.model_name == MODEL_LOGISTIC_REGRESSION
        for participant_id in fold.validation_participant_ids
    ]
    assert set(logistic_validation_ids) == set(development.metadata["participant_id"])
    assert len(logistic_validation_ids) == len(development.metadata["participant_id"])


def test_exactly_five_folds_are_produced_for_each_model() -> None:
    result = cross_validate_development_set(make_development_data(), n_splits=5, random_state=7)

    logistic_folds = [fold.fold for fold in result.fold_results if fold.model_name == MODEL_LOGISTIC_REGRESSION]
    random_forest_folds = [fold.fold for fold in result.fold_results if fold.model_name == MODEL_RANDOM_FOREST]
    assert logistic_folds == [1, 2, 3, 4, 5]
    assert random_forest_folds == [1, 2, 3, 4, 5]


def test_logistic_regression_scaler_is_fit_only_on_fold_training_data() -> None:
    development = make_development_data()

    result = cross_validate_development_set(development, n_splits=5, random_state=7)

    first_logistic_fold = next(
        fold for fold in result.fold_results if fold.model_name == MODEL_LOGISTIC_REGRESSION and fold.fold == 1
    )
    train_mask = development.metadata["participant_id"].isin(first_logistic_fold.train_participant_ids)
    expected_mean = development.X.loc[train_mask].mean().to_numpy()
    combined_mean = development.X.mean().to_numpy()

    assert first_logistic_fold.scaler is not None
    assert first_logistic_fold.scaler.mean_ == pytest.approx(expected_mean)
    assert first_logistic_fold.scaler.mean_ != pytest.approx(combined_mean)


def test_random_forest_does_not_use_scaling() -> None:
    result = cross_validate_development_set(make_development_data(), n_splits=5, random_state=7)

    random_forest_folds = [fold for fold in result.fold_results if fold.model_name == MODEL_RANDOM_FOREST]
    assert all(fold.scaler is None for fold in random_forest_folds)


def test_no_final_test_participant_enters_development_cv_when_supplied() -> None:
    development = make_development_data()

    with pytest.raises(ValueError, match="final held-out test participants"):
        cross_validate_development_set(development, final_test_participants={"p0"}, n_splits=5, random_state=7)


def test_metrics_are_returned_for_every_fold() -> None:
    result = cross_validate_development_set(make_development_data(), n_splits=5, random_state=7)

    assert len(result.fold_results) == 10
    for fold in result.fold_results:
        assert {"accuracy", "precision", "recall", "f1"}.issubset(fold.metrics)


def test_aggregate_mean_and_std_metrics_are_correct() -> None:
    fold_results = [
        CVFoldResult(
            model_name="Example",
            fold=1,
            train_recordings=4,
            validation_recordings=2,
            train_participants=4,
            validation_participants=2,
            participant_overlap=0,
            train_participant_ids=("p1", "p2", "p3", "p4"),
            validation_participant_ids=("p5", "p6"),
            metrics={"accuracy": 0.2, "precision": 0.4, "recall": 0.6, "f1": 0.8},
            y_pred=np.array(["CNT", "PD"]),
        ),
        CVFoldResult(
            model_name="Example",
            fold=2,
            train_recordings=4,
            validation_recordings=2,
            train_participants=4,
            validation_participants=2,
            participant_overlap=0,
            train_participant_ids=("p5", "p6", "p7", "p8"),
            validation_participant_ids=("p1", "p2"),
            metrics={"accuracy": 0.4, "precision": 0.6, "recall": 0.8, "f1": 1.0},
            y_pred=np.array(["CNT", "PD"]),
        ),
    ]

    aggregate = aggregate_fold_metrics(fold_results)

    row = aggregate.loc[aggregate["model"] == "Example"].iloc[0]
    assert row["mean_accuracy"] == pytest.approx(0.3)
    assert row["std_accuracy"] == pytest.approx(pd.Series([0.2, 0.4]).std())
    assert row["mean_f1"] == pytest.approx(0.9)
    assert row["std_f1"] == pytest.approx(pd.Series([0.8, 1.0]).std())


def test_reproducibility_with_fixed_model_settings() -> None:
    development = make_development_data()

    first = cross_validate_development_set(development, n_splits=5, random_state=42)
    second = cross_validate_development_set(development, n_splits=5, random_state=42)

    first_predictions = [(fold.model_name, fold.fold, fold.y_pred.tolist()) for fold in first.fold_results]
    second_predictions = [(fold.model_name, fold.fold, fold.y_pred.tolist()) for fold in second.fold_results]
    assert first_predictions == second_predictions
    pd.testing.assert_frame_equal(first.aggregate_metrics, second.aggregate_metrics)


def test_nan_validation() -> None:
    development = make_development_data()
    development.X.loc[0, "hold_mean"] = np.nan

    with pytest.raises(ValueError, match="NaN feature values"):
        cross_validate_development_set(development, n_splits=5, random_state=7)


def test_inf_validation() -> None:
    development = make_development_data()
    development.X.loc[0, "hold_mean"] = np.inf

    with pytest.raises(ValueError, match="infinite feature values"):
        cross_validate_development_set(development, n_splits=5, random_state=7)


def test_invalid_participant_overlap_raises_clear_error() -> None:
    development = make_development_data(6)
    split = ParticipantSplit(
        X_train=development.X.iloc[:4].copy(),
        X_test=development.X.iloc[4:].copy(),
        y_train=development.y.iloc[:4].copy(),
        y_test=development.y.iloc[4:].copy(),
        train_metadata=development.metadata.iloc[:4].copy(),
        test_metadata=pd.DataFrame({"participant_id": ["p0", "p5"], "recording_id": ["r4", "r5"]}),
    )

    with pytest.raises(ValueError, match="Participant overlap"):
        make_development_dataset(split)


def test_expected_feature_validation() -> None:
    development = make_development_data()
    development = MLDataset(
        X=development.X.drop(columns=["hold_mean"]),
        y=development.y,
        metadata=development.metadata,
    )

    with pytest.raises(ValueError, match="missing expected feature columns"):
        cross_validate_development_set(development, n_splits=5, random_state=7)
