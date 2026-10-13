"""Controlled participant-aware hyperparameter tuning for NeuroTap.

Tuning uses only the existing training/development participants. The final
held-out test participants are not used for model selection or evaluation here.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from ml.training.baseline import evaluate_predictions  # type: ignore
    from ml.training.cross_validation import (  # type: ignore
        METRIC_COLUMNS,
        MODEL_LOGISTIC_REGRESSION,
        MODEL_RANDOM_FOREST,
        _participants,
        make_development_dataset,
        validate_development_dataset,
    )
    from ml.training.dataset import MLDataset, ParticipantSplit, build_feature_table, build_ml_dataset, split_by_participant  # type: ignore
    from ml.training.random_forest import (  # type: ignore
        DEFAULT_MAX_DEPTH,
        DEFAULT_MIN_SAMPLES_LEAF,
        DEFAULT_MIN_SAMPLES_SPLIT,
        DEFAULT_N_ESTIMATORS,
        DEFAULT_RANDOM_STATE,
    )
else:
    from ml.training.baseline import evaluate_predictions
    from ml.training.cross_validation import (
        METRIC_COLUMNS,
        MODEL_LOGISTIC_REGRESSION,
        MODEL_RANDOM_FOREST,
        _participants,
        make_development_dataset,
        validate_development_dataset,
    )
    from ml.training.dataset import MLDataset, ParticipantSplit, build_feature_table, build_ml_dataset, split_by_participant
    from ml.training.random_forest import (
        DEFAULT_MAX_DEPTH,
        DEFAULT_MIN_SAMPLES_LEAF,
        DEFAULT_MIN_SAMPLES_SPLIT,
        DEFAULT_N_ESTIMATORS,
        DEFAULT_RANDOM_STATE,
    )


LOGISTIC_C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]
RANDOM_FOREST_N_ESTIMATORS = [100, 200, 500]
RANDOM_FOREST_MAX_DEPTH = [None, 3, 5, 10]
RANDOM_FOREST_MIN_SAMPLES_LEAF = [1, 2, 4]


@dataclass(frozen=True)
class TuningFoldResult:
    model_name: str
    params: dict[str, Any]
    fold: int
    train_recordings: int
    validation_recordings: int
    train_participants: int
    validation_participants: int
    participant_overlap: int
    train_participant_ids: tuple[str, ...]
    validation_participant_ids: tuple[str, ...]
    metrics: dict[str, object]
    y_pred: np.ndarray
    scaler_mean: np.ndarray | None = None


@dataclass(frozen=True)
class TuningResult:
    fold_results: list[TuningFoldResult]
    config_results: pd.DataFrame
    best_logistic_regression: pd.Series
    best_random_forest: pd.Series
    development_metadata: pd.DataFrame
    final_test_metadata: pd.DataFrame
    cv_strategy: str = "GroupKFold"
    selection_metric: str = "mean_f1"


def random_forest_grid() -> list[dict[str, Any]]:
    """Return the controlled Random Forest grid."""
    return [
        {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": DEFAULT_MIN_SAMPLES_SPLIT,
            "min_samples_leaf": min_samples_leaf,
        }
        for n_estimators, max_depth, min_samples_leaf in product(
            RANDOM_FOREST_N_ESTIMATORS,
            RANDOM_FOREST_MAX_DEPTH,
            RANDOM_FOREST_MIN_SAMPLES_LEAF,
        )
    ]


def _fold_participant_details(
    train_metadata: pd.DataFrame,
    validation_metadata: pd.DataFrame,
) -> tuple[tuple[str, ...], tuple[str, ...], int]:
    train_participants = _participants(train_metadata)
    validation_participants = _participants(validation_metadata)
    return (
        tuple(sorted(train_participants)),
        tuple(sorted(validation_participants)),
        len(train_participants & validation_participants),
    )


def _params_key(params: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((key, str(value)) for key, value in params.items()))


def _run_logistic_tuning_fold(
    C: float,
    fold: int,
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
    train_metadata: pd.DataFrame,
    validation_metadata: pd.DataFrame,
    random_state: int,
) -> TuningFoldResult:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    model = LogisticRegression(C=C, random_state=random_state, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_validation_scaled)
    train_ids, validation_ids, overlap = _fold_participant_details(train_metadata, validation_metadata)
    if overlap:
        raise ValueError(f"Participant overlap detected inside tuning fold {fold}.")
    return TuningFoldResult(
        model_name=MODEL_LOGISTIC_REGRESSION,
        params={"C": C},
        fold=fold,
        train_recordings=len(X_train),
        validation_recordings=len(X_validation),
        train_participants=len(train_ids),
        validation_participants=len(validation_ids),
        participant_overlap=overlap,
        train_participant_ids=train_ids,
        validation_participant_ids=validation_ids,
        metrics=evaluate_predictions(y_validation, y_pred),
        y_pred=y_pred,
        scaler_mean=scaler.mean_.copy(),
    )


def _run_random_forest_tuning_fold(
    params: dict[str, Any],
    fold: int,
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
    train_metadata: pd.DataFrame,
    validation_metadata: pd.DataFrame,
    random_state: int,
) -> TuningFoldResult:
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_validation)
    train_ids, validation_ids, overlap = _fold_participant_details(train_metadata, validation_metadata)
    if overlap:
        raise ValueError(f"Participant overlap detected inside tuning fold {fold}.")
    return TuningFoldResult(
        model_name=MODEL_RANDOM_FOREST,
        params=params.copy(),
        fold=fold,
        train_recordings=len(X_train),
        validation_recordings=len(X_validation),
        train_participants=len(train_ids),
        validation_participants=len(validation_ids),
        participant_overlap=overlap,
        train_participant_ids=train_ids,
        validation_participant_ids=validation_ids,
        metrics=evaluate_predictions(y_validation, y_pred),
        y_pred=y_pred,
        scaler_mean=None,
    )


def aggregate_tuning_results(fold_results: list[TuningFoldResult]) -> pd.DataFrame:
    """Aggregate fold metrics per model/configuration."""
    rows = []
    for result in fold_results:
        rows.append(
            {
                "model": result.model_name,
                "params_key": _params_key(result.params),
                "params": result.params,
                **{metric: float(result.metrics[metric]) for metric in METRIC_COLUMNS},
            }
        )
    metrics = pd.DataFrame(rows)
    grouped = metrics.groupby(["model", "params_key"], sort=False)
    aggregate = grouped[METRIC_COLUMNS].agg(["mean", "std"])
    aggregate.columns = [f"{stat}_{metric}" for metric, stat in aggregate.columns]
    aggregate = aggregate.reset_index()
    aggregate["params"] = grouped["params"].first().reset_index(drop=True)
    return aggregate[
        [
            "model",
            "params",
            "mean_f1",
            "std_f1",
            "mean_accuracy",
            "mean_precision",
            "mean_recall",
            "std_accuracy",
            "std_precision",
            "std_recall",
        ]
    ]


def _logistic_simplicity_key(params: dict[str, Any]) -> tuple[float]:
    return (float(params["C"]),)


def _random_forest_simplicity_key(params: dict[str, Any]) -> tuple[int, int, int]:
    max_depth = params["max_depth"]
    depth_rank = 10_000 if max_depth is None else int(max_depth)
    return (depth_rank, int(params["n_estimators"]), -int(params["min_samples_leaf"]))


def select_best_configuration(config_results: pd.DataFrame, model_name: str) -> pd.Series:
    """Select by mean F1, then lower std F1, then simpler configuration."""
    subset = config_results[config_results["model"] == model_name].copy()
    if subset.empty:
        raise ValueError(f"No tuning results found for model: {model_name}")

    if model_name == MODEL_LOGISTIC_REGRESSION:
        subset["simplicity_key"] = subset["params"].map(_logistic_simplicity_key)
    elif model_name == MODEL_RANDOM_FOREST:
        subset["simplicity_key"] = subset["params"].map(_random_forest_simplicity_key)
    else:
        subset["simplicity_key"] = subset["params"].map(str)

    subset = subset.sort_values(
        by=["mean_f1", "std_f1", "simplicity_key"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    return subset.iloc[0].drop(labels=["simplicity_key"])


def tune_development_set(
    development: MLDataset,
    final_test_participants: set[str] | None = None,
    n_splits: int = 5,
    random_state: int = DEFAULT_RANDOM_STATE,
    logistic_c_values: list[float] | None = None,
    rf_grid: list[dict[str, Any]] | None = None,
) -> TuningResult:
    """Tune fixed grids with GroupKFold on development participants only."""
    validate_development_dataset(development, final_test_participants=final_test_participants)
    c_values = LOGISTIC_C_VALUES if logistic_c_values is None else logistic_c_values
    forest_grid = random_forest_grid() if rf_grid is None else rf_grid

    groups = development.metadata["participant_id"].astype(str)
    if n_splits > groups.nunique():
        raise ValueError(f"n_splits={n_splits} exceeds development participant count={groups.nunique()}.")

    splitter = GroupKFold(n_splits=n_splits)
    fold_results: list[TuningFoldResult] = []

    for fold, (train_idx, validation_idx) in enumerate(splitter.split(development.X, development.y, groups), start=1):
        X_train = development.X.iloc[train_idx].copy()
        X_validation = development.X.iloc[validation_idx].copy()
        y_train = development.y.iloc[train_idx].copy()
        y_validation = development.y.iloc[validation_idx].copy()
        train_metadata = development.metadata.iloc[train_idx].copy()
        validation_metadata = development.metadata.iloc[validation_idx].copy()

        for C in c_values:
            fold_results.append(
                _run_logistic_tuning_fold(
                    C=C,
                    fold=fold,
                    X_train=X_train,
                    X_validation=X_validation,
                    y_train=y_train,
                    y_validation=y_validation,
                    train_metadata=train_metadata,
                    validation_metadata=validation_metadata,
                    random_state=random_state,
                )
            )

        for params in forest_grid:
            fold_results.append(
                _run_random_forest_tuning_fold(
                    params=params,
                    fold=fold,
                    X_train=X_train,
                    X_validation=X_validation,
                    y_train=y_train,
                    y_validation=y_validation,
                    train_metadata=train_metadata,
                    validation_metadata=validation_metadata,
                    random_state=random_state,
                )
            )

    config_results = aggregate_tuning_results(fold_results)
    final_test_metadata = pd.DataFrame({"participant_id": sorted(final_test_participants or set())})
    return TuningResult(
        fold_results=fold_results,
        config_results=config_results,
        best_logistic_regression=select_best_configuration(config_results, MODEL_LOGISTIC_REGRESSION),
        best_random_forest=select_best_configuration(config_results, MODEL_RANDOM_FOREST),
        development_metadata=development.metadata.copy(),
        final_test_metadata=final_test_metadata,
    )


def run_tuning(
    ml_dataset: MLDataset | None = None,
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratify: bool = True,
    n_splits: int = 5,
) -> tuple[TuningResult, ParticipantSplit]:
    """Recover the existing final split, then tune only on the development side."""
    dataset = build_ml_dataset() if ml_dataset is None else ml_dataset
    split = split_by_participant(dataset, test_size=test_size, random_state=random_state, stratify=stratify)
    development = make_development_dataset(split)
    result = tune_development_set(
        development,
        final_test_participants=_participants(split.test_metadata),
        n_splits=n_splits,
        random_state=random_state,
    )
    return result, split


def _params_label(params: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in params.items())


def _print_model_results(config_results: pd.DataFrame, model_name: str) -> None:
    rows = config_results[config_results["model"] == model_name].copy()
    for _, row in rows.iterrows():
        print(
            f"{_params_label(row['params'])}: "
            f"mean_f1={row['mean_f1']:.4f}, "
            f"std_f1={row['std_f1']:.4f}, "
            f"mean_accuracy={row['mean_accuracy']:.4f}, "
            f"mean_precision={row['mean_precision']:.4f}, "
            f"mean_recall={row['mean_recall']:.4f}"
        )


def main() -> int:
    feature_table = build_feature_table()
    ml_dataset = build_ml_dataset(feature_table)
    result, split = run_tuning(ml_dataset=ml_dataset, test_size=0.2, random_state=DEFAULT_RANDOM_STATE, stratify=True)

    dev_participants = _participants(split.train_metadata)
    final_test_participants = _participants(split.test_metadata)
    participant_overlap = dev_participants & final_test_participants

    print("# NeuroTap Controlled Hyperparameter Tuning")
    print("\n## Data Boundary")
    print(f"Development recordings: {len(split.X_train)}")
    print(f"Development participants: {len(dev_participants)}")
    print(f"Final held-out test recordings: {len(split.X_test)}")
    print(f"Final held-out test participants: {len(final_test_participants)}")
    print(f"Final-test participant overlap with tuning data: {len(participant_overlap)}")
    print("\n## Logistic Regression")
    _print_model_results(result.config_results, MODEL_LOGISTIC_REGRESSION)
    print(f"Best C: {result.best_logistic_regression['params']['C']}")
    print("\n## Random Forest")
    _print_model_results(result.config_results, MODEL_RANDOM_FOREST)
    print(f"Best configuration: {_params_label(result.best_random_forest['params'])}")
    print("\n## Participant Overlap Inside CV Folds")
    overlaps = [fold.participant_overlap for fold in result.fold_results]
    print(overlaps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
