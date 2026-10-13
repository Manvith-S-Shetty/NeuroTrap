import numpy as np
import pandas as pd
import pytest

from ml.training.dataset import FEATURE_COLUMNS, MLDataset
from ml.training.tuning import (
    LOGISTIC_C_VALUES,
    MODEL_LOGISTIC_REGRESSION,
    MODEL_RANDOM_FOREST,
    RANDOM_FOREST_MAX_DEPTH,
    RANDOM_FOREST_MIN_SAMPLES_LEAF,
    RANDOM_FOREST_N_ESTIMATORS,
    TuningFoldResult,
    aggregate_tuning_results,
    random_forest_grid,
    select_best_configuration,
    tune_development_set,
)


def make_feature_frame(participant_index: int) -> dict[str, float]:
    base = 0.0 if participant_index % 2 == 0 else 10.0
    return {
        column: float(base + participant_index / 10 + column_index / 1000)
        for column_index, column in enumerate(FEATURE_COLUMNS)
    }


def make_development_data(participant_count: int = 10) -> MLDataset:
    return MLDataset(
        X=pd.DataFrame(
            [make_feature_frame(idx) for idx in range(participant_count)],
            columns=FEATURE_COLUMNS,
        ),
        y=pd.Series(["CNT" if idx % 2 == 0 else "PD" for idx in range(participant_count)], name="label"),
        metadata=pd.DataFrame(
            {
                "participant_id": [f"p{idx}" for idx in range(participant_count)],
                "recording_id": [f"r{idx}" for idx in range(participant_count)],
            }
        ),
    )


def small_rf_grid() -> list[dict[str, object]]:
    return [
        {"n_estimators": 5, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1},
        {"n_estimators": 5, "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 2},
    ]


def test_all_intended_logistic_c_values_are_defined() -> None:
    assert LOGISTIC_C_VALUES == [0.01, 0.1, 1.0, 10.0, 100.0]


def test_all_intended_random_forest_configurations_are_defined() -> None:
    configs = random_forest_grid()

    assert RANDOM_FOREST_N_ESTIMATORS == [100, 200, 500]
    assert RANDOM_FOREST_MAX_DEPTH == [None, 3, 5, 10]
    assert RANDOM_FOREST_MIN_SAMPLES_LEAF == [1, 2, 4]
    assert len(configs) == 36
    assert {
        (config["n_estimators"], config["max_depth"], config["min_samples_leaf"])
        for config in configs
    } == {
        (n_estimators, max_depth, min_samples_leaf)
        for n_estimators in RANDOM_FOREST_N_ESTIMATORS
        for max_depth in RANDOM_FOREST_MAX_DEPTH
        for min_samples_leaf in RANDOM_FOREST_MIN_SAMPLES_LEAF
    }


def test_group_kfold_is_used() -> None:
    result = tune_development_set(
        make_development_data(),
        n_splits=5,
        logistic_c_values=[1.0],
        rf_grid=small_rf_grid(),
    )

    assert result.cv_strategy == "GroupKFold"


def test_no_participant_overlap_within_folds() -> None:
    result = tune_development_set(
        make_development_data(),
        n_splits=5,
        logistic_c_values=[1.0],
        rf_grid=small_rf_grid(),
    )

    for fold in result.fold_results:
        assert fold.participant_overlap == 0
        assert set(fold.train_participant_ids).isdisjoint(fold.validation_participant_ids)


def test_final_test_participants_are_excluded() -> None:
    development = make_development_data()

    with pytest.raises(ValueError, match="final held-out test participants"):
        tune_development_set(
            development,
            final_test_participants={"p0"},
            n_splits=5,
            logistic_c_values=[1.0],
            rf_grid=small_rf_grid(),
        )


def test_logistic_scaler_is_fit_independently_inside_each_fold() -> None:
    development = make_development_data()

    result = tune_development_set(
        development,
        n_splits=5,
        logistic_c_values=[0.1, 1.0],
        rf_grid=small_rf_grid(),
    )

    logistic_folds = [fold for fold in result.fold_results if fold.model_name == MODEL_LOGISTIC_REGRESSION]
    for fold in logistic_folds:
        train_mask = development.metadata["participant_id"].isin(fold.train_participant_ids)
        expected_mean = development.X.loc[train_mask].mean().to_numpy()
        assert fold.scaler_mean is not None
        assert fold.scaler_mean == pytest.approx(expected_mean)


def test_random_forest_uses_no_scaler() -> None:
    result = tune_development_set(
        make_development_data(),
        n_splits=5,
        logistic_c_values=[1.0],
        rf_grid=small_rf_grid(),
    )

    forest_folds = [fold for fold in result.fold_results if fold.model_name == MODEL_RANDOM_FOREST]
    assert forest_folds
    assert all(fold.scaler_mean is None for fold in forest_folds)


def test_mean_std_aggregation_is_correct() -> None:
    fold_results = [
        TuningFoldResult(
            model_name="Example",
            params={"C": 1.0},
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
        TuningFoldResult(
            model_name="Example",
            params={"C": 1.0},
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

    aggregate = aggregate_tuning_results(fold_results)

    row = aggregate.iloc[0]
    assert row["mean_f1"] == pytest.approx(0.9)
    assert row["std_f1"] == pytest.approx(pd.Series([0.8, 1.0]).std())
    assert row["mean_accuracy"] == pytest.approx(0.3)


def test_best_configuration_uses_highest_mean_f1() -> None:
    config_results = pd.DataFrame(
        [
            {"model": MODEL_LOGISTIC_REGRESSION, "params": {"C": 0.1}, "mean_f1": 0.6, "std_f1": 0.2},
            {"model": MODEL_LOGISTIC_REGRESSION, "params": {"C": 1.0}, "mean_f1": 0.8, "std_f1": 0.3},
        ]
    )

    best = select_best_configuration(config_results, MODEL_LOGISTIC_REGRESSION)

    assert best["params"] == {"C": 1.0}


def test_tie_breaking_uses_lower_std_then_simpler_configuration() -> None:
    lower_std_results = pd.DataFrame(
        [
            {"model": MODEL_LOGISTIC_REGRESSION, "params": {"C": 1.0}, "mean_f1": 0.8, "std_f1": 0.2},
            {"model": MODEL_LOGISTIC_REGRESSION, "params": {"C": 10.0}, "mean_f1": 0.8, "std_f1": 0.1},
        ]
    )
    simpler_results = pd.DataFrame(
        [
            {"model": MODEL_LOGISTIC_REGRESSION, "params": {"C": 10.0}, "mean_f1": 0.8, "std_f1": 0.1},
            {"model": MODEL_LOGISTIC_REGRESSION, "params": {"C": 0.1}, "mean_f1": 0.8, "std_f1": 0.1},
        ]
    )

    assert select_best_configuration(lower_std_results, MODEL_LOGISTIC_REGRESSION)["params"] == {"C": 10.0}
    assert select_best_configuration(simpler_results, MODEL_LOGISTIC_REGRESSION)["params"] == {"C": 0.1}


def test_reproducibility_with_random_state_42() -> None:
    development = make_development_data()

    first = tune_development_set(
        development,
        n_splits=5,
        random_state=42,
        logistic_c_values=[1.0],
        rf_grid=small_rf_grid(),
    )
    second = tune_development_set(
        development,
        n_splits=5,
        random_state=42,
        logistic_c_values=[1.0],
        rf_grid=small_rf_grid(),
    )

    first_predictions = [(fold.model_name, fold.params, fold.fold, fold.y_pred.tolist()) for fold in first.fold_results]
    second_predictions = [(fold.model_name, fold.params, fold.fold, fold.y_pred.tolist()) for fold in second.fold_results]
    assert first_predictions == second_predictions
    pd.testing.assert_frame_equal(first.config_results, second.config_results)


def test_nan_data_is_rejected() -> None:
    development = make_development_data()
    development.X.loc[0, "hold_mean"] = np.nan

    with pytest.raises(ValueError, match="NaN feature values"):
        tune_development_set(development, n_splits=5, logistic_c_values=[1.0], rf_grid=small_rf_grid())


def test_inf_data_is_rejected() -> None:
    development = make_development_data()
    development.X.loc[0, "hold_mean"] = np.inf

    with pytest.raises(ValueError, match="infinite feature values"):
        tune_development_set(development, n_splits=5, logistic_c_values=[1.0], rf_grid=small_rf_grid())


def test_missing_participant_id_is_rejected() -> None:
    development = make_development_data()
    development = MLDataset(X=development.X, y=development.y, metadata=development.metadata.drop(columns=["participant_id"]))

    with pytest.raises(ValueError, match="participant_id"):
        tune_development_set(development, n_splits=5, logistic_c_values=[1.0], rf_grid=small_rf_grid())


def test_unsupported_labels_are_rejected() -> None:
    development = make_development_data()
    development.y.iloc[0] = "UNKNOWN"

    with pytest.raises(ValueError, match="unsupported class labels"):
        tune_development_set(development, n_splits=5, logistic_c_values=[1.0], rf_grid=small_rf_grid())
