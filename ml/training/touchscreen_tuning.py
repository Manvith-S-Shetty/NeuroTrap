"""Controlled hyperparameter tuning for NeuroTap touchscreen models.

Uses ONLY the 269-recording / 26-participant development set from Step 6.
The 69-recording / 7-participant final test set is NEVER loaded, inspected,
or used here. All model selection is based on participant-aware CV F1.

Does NOT perform feature selection, outlier removal, imputation, or label
modification.
"""

from __future__ import annotations

import json
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

try:
    from sklearn.model_selection import StratifiedGroupKFold as _StratifiedGroupKFold
    STRATIFIED_GROUP_KFOLD_AVAILABLE = True
except ImportError:
    _StratifiedGroupKFold = None  # type: ignore[assignment]
    STRATIFIED_GROUP_KFOLD_AVAILABLE = False


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from ml.training.touchscreen_baseline import (  # type: ignore
        CLASS_LABELS,
        MODEL_LOGISTIC_REGRESSION,
        MODEL_RANDOM_FOREST,
        evaluate_predictions,
        validate_development_dataset,
        validate_finite_features,
    )
    from ml.training.touchscreen_dataset import (  # type: ignore
        TouchscreenMLDataset,
        TouchscreenParticipantSplit,
        build_touchscreen_ml_dataset,
        split_touchscreen_by_participant,
    )
else:
    from ml.training.touchscreen_baseline import (
        CLASS_LABELS,
        MODEL_LOGISTIC_REGRESSION,
        MODEL_RANDOM_FOREST,
        evaluate_predictions,
        validate_development_dataset,
        validate_finite_features,
    )
    from ml.training.touchscreen_dataset import (
        TouchscreenMLDataset,
        TouchscreenParticipantSplit,
        build_touchscreen_ml_dataset,
        split_touchscreen_by_participant,
    )


# --------------------------------------------------------------------------- #
# Search spaces (Step 8 specification)                                        #
# --------------------------------------------------------------------------- #

LR_C_VALUES: list[float] = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
LR_CLASS_WEIGHTS: list[str | None] = [None, "balanced"]

RF_N_ESTIMATORS: list[int] = [100, 200, 500]
RF_MAX_DEPTH: list[int | None] = [None, 3, 5, 10]
RF_MIN_SAMPLES_LEAF: list[int] = [1, 2, 5]
RF_CLASS_WEIGHTS: list[str | None] = [None, "balanced"]

METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1"]
SELECTION_METRIC = "mean_f1"
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_SPLITS = 5
DEFAULT_TEST_SIZE = 0.2

OUTPUT_DIR = Path(__file__).resolve().parent / "touchscreen_tuning_outputs"


# --------------------------------------------------------------------------- #
# Grid builders                                                               #
# --------------------------------------------------------------------------- #

def logistic_regression_grid() -> list[dict[str, Any]]:
    """All (C, class_weight) combinations for the LR search."""
    return [
        {"C": c, "class_weight": cw}
        for c, cw in product(LR_C_VALUES, LR_CLASS_WEIGHTS)
    ]


def random_forest_grid() -> list[dict[str, Any]]:
    """All (n_estimators, max_depth, min_samples_leaf, class_weight) combinations."""
    return [
        {
            "n_estimators": n,
            "max_depth": d,
            "min_samples_leaf": msl,
            "class_weight": cw,
        }
        for n, d, msl, cw in product(
            RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_LEAF, RF_CLASS_WEIGHTS
        )
    ]


# --------------------------------------------------------------------------- #
# Dataclasses                                                                 #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TouchscreenTuningFoldResult:
    model_name: str
    params: dict[str, Any]
    cv_strategy: str
    fold: int
    train_recordings: int
    validation_recordings: int
    train_participants: int
    validation_participants: int
    participant_overlap: int
    train_participant_ids: tuple[str, ...]
    validation_participant_ids: tuple[str, ...]
    train_class_counts: dict[str, int]
    validation_class_counts: dict[str, int]
    metrics: dict[str, object]
    y_pred: np.ndarray
    scaler_mean: np.ndarray | None


@dataclass(frozen=True)
class TouchscreenTuningResult:
    fold_results: list[TouchscreenTuningFoldResult]
    config_results: pd.DataFrame
    best_logistic_regression: pd.Series
    best_random_forest: pd.Series
    development_metadata: pd.DataFrame
    final_test_participants: set[str]
    cv_strategy: str
    n_splits: int
    selection_metric: str = SELECTION_METRIC


# --------------------------------------------------------------------------- #
# Participant / fold helpers                                                  #
# --------------------------------------------------------------------------- #

def _fold_participant_details(
    train_meta: pd.DataFrame,
    val_meta: pd.DataFrame,
) -> tuple[tuple[str, ...], tuple[str, ...], int]:
    train_p = set(train_meta["participant_id"].astype(str))
    val_p = set(val_meta["participant_id"].astype(str))
    return (
        tuple(sorted(train_p)),
        tuple(sorted(val_p)),
        len(train_p & val_p),
    )


def _class_counts(y: pd.Series) -> dict[str, int]:
    counts = y.value_counts()
    return {label: int(counts.get(label, 0)) for label in CLASS_LABELS}


def _params_key(params: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((k, str(v)) for k, v in params.items()))


# --------------------------------------------------------------------------- #
# Per-fold runners                                                            #
# --------------------------------------------------------------------------- #

def _run_lr_fold(
    params: dict[str, Any],
    cv_strategy: str,
    fold: int,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    train_meta: pd.DataFrame,
    val_meta: pd.DataFrame,
    random_state: int,
) -> TouchscreenTuningFoldResult:
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    model = LogisticRegression(
        C=params["C"],
        class_weight=params["class_weight"],
        max_iter=1000,
        random_state=random_state,
    )
    model.fit(X_tr_sc, y_train)
    y_pred = model.predict(X_val_sc)

    train_p, val_p, overlap = _fold_participant_details(train_meta, val_meta)
    if overlap:
        raise ValueError(f"Participant overlap in tuning fold {fold} ({cv_strategy})")

    return TouchscreenTuningFoldResult(
        model_name=MODEL_LOGISTIC_REGRESSION,
        params=params.copy(),
        cv_strategy=cv_strategy,
        fold=fold,
        train_recordings=len(X_train),
        validation_recordings=len(X_val),
        train_participants=len(train_p),
        validation_participants=len(val_p),
        participant_overlap=overlap,
        train_participant_ids=train_p,
        validation_participant_ids=val_p,
        train_class_counts=_class_counts(y_train),
        validation_class_counts=_class_counts(y_val),
        metrics=evaluate_predictions(y_val, y_pred),
        y_pred=y_pred,
        scaler_mean=scaler.mean_.copy(),
    )


def _run_rf_fold(
    params: dict[str, Any],
    cv_strategy: str,
    fold: int,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    train_meta: pd.DataFrame,
    val_meta: pd.DataFrame,
    random_state: int,
) -> TouchscreenTuningFoldResult:
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        class_weight=params["class_weight"],
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    train_p, val_p, overlap = _fold_participant_details(train_meta, val_meta)
    if overlap:
        raise ValueError(f"Participant overlap in tuning fold {fold} ({cv_strategy})")

    return TouchscreenTuningFoldResult(
        model_name=MODEL_RANDOM_FOREST,
        params=params.copy(),
        cv_strategy=cv_strategy,
        fold=fold,
        train_recordings=len(X_train),
        validation_recordings=len(X_val),
        train_participants=len(train_p),
        validation_participants=len(val_p),
        participant_overlap=overlap,
        train_participant_ids=train_p,
        validation_participant_ids=val_p,
        train_class_counts=_class_counts(y_train),
        validation_class_counts=_class_counts(y_val),
        metrics=evaluate_predictions(y_val, y_pred),
        y_pred=y_pred,
        scaler_mean=None,
    )


# --------------------------------------------------------------------------- #
# Aggregation & selection                                                     #
# --------------------------------------------------------------------------- #

def aggregate_tuning_results(
    fold_results: list[TouchscreenTuningFoldResult],
) -> pd.DataFrame:
    """Aggregate mean ± std per (model, params, cv_strategy) configuration."""
    rows = [
        {
            "model": r.model_name,
            "cv_strategy": r.cv_strategy,
            "params_key": _params_key(r.params),
            "params": r.params,
            **{m: float(r.metrics[m]) for m in METRIC_COLUMNS},
        }
        for r in fold_results
    ]
    df = pd.DataFrame(rows)
    grouped = df.groupby(["model", "cv_strategy", "params_key"], sort=False)
    agg = grouped[METRIC_COLUMNS].agg(["mean", "std"])
    agg.columns = [f"{stat}_{metric}" for metric, stat in agg.columns]
    agg = agg.reset_index()
    agg["params"] = grouped["params"].first().reset_index(drop=True)
    col_order = [
        "model", "cv_strategy", "params",
        "mean_f1", "std_f1",
        "mean_accuracy", "std_accuracy",
        "mean_precision", "std_precision",
        "mean_recall", "std_recall",
    ]
    return agg[col_order]


def _lr_simplicity_key(params: dict[str, Any]) -> tuple[float, int]:
    cw_rank = 0 if params["class_weight"] is None else 1
    return (float(params["C"]), cw_rank)


def _rf_simplicity_key(params: dict[str, Any]) -> tuple[int, int, int, int]:
    d = params["max_depth"]
    depth_rank = 10_000 if d is None else int(d)
    cw_rank = 0 if params["class_weight"] is None else 1
    return (depth_rank, int(params["n_estimators"]), -int(params["min_samples_leaf"]), cw_rank)


def select_best_configuration(
    config_results: pd.DataFrame,
    model_name: str,
    cv_strategy: str,
) -> pd.Series:
    """Select by mean_f1 DESC, std_f1 ASC, then simplest config."""
    subset = config_results[
        (config_results["model"] == model_name)
        & (config_results["cv_strategy"] == cv_strategy)
    ].copy()
    if subset.empty:
        raise ValueError(f"No results for model={model_name!r}, cv_strategy={cv_strategy!r}")

    if model_name == MODEL_LOGISTIC_REGRESSION:
        subset["_sk"] = subset["params"].map(_lr_simplicity_key)
    else:
        subset["_sk"] = subset["params"].map(_rf_simplicity_key)

    subset = subset.sort_values(
        by=["mean_f1", "std_f1", "_sk"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    return subset.iloc[0].drop(labels=["_sk"])


# --------------------------------------------------------------------------- #
# Splitter factory                                                            #
# --------------------------------------------------------------------------- #

def _make_splitter(cv_strategy: str, n_splits: int):
    if cv_strategy == "GroupKFold":
        return GroupKFold(n_splits=n_splits)
    elif cv_strategy == "StratifiedGroupKFold":
        if not STRATIFIED_GROUP_KFOLD_AVAILABLE:
            raise ImportError("StratifiedGroupKFold is not available in this scikit-learn version.")
        return _StratifiedGroupKFold(n_splits=n_splits)
    else:
        raise ValueError(f"Unknown cv_strategy: {cv_strategy!r}")


# --------------------------------------------------------------------------- #
# Core tuning function                                                        #
# --------------------------------------------------------------------------- #

def tune_touchscreen_development(
    development: TouchscreenMLDataset,
    final_test_participants: set[str] | None = None,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
    cv_strategy: str = "GroupKFold",
    lr_grid: list[dict[str, Any]] | None = None,
    rf_grid: list[dict[str, Any]] | None = None,
) -> TouchscreenTuningResult:
    """Tune LR and RF grids with participant-aware CV on development data ONLY.

    The final test set is never loaded or referenced here.
    Scaling (LR) is fitted inside each fold on training data only.
    """
    validate_development_dataset(development, final_test_participants=final_test_participants)

    lr_configs = logistic_regression_grid() if lr_grid is None else lr_grid
    rf_configs = random_forest_grid() if rf_grid is None else rf_grid

    groups = development.metadata["participant_id"].astype(str)
    n_participants = groups.nunique()
    if n_splits > n_participants:
        raise ValueError(
            f"n_splits={n_splits} exceeds development participant count={n_participants}."
        )

    splitter = _make_splitter(cv_strategy, n_splits)
    fold_results: list[TouchscreenTuningFoldResult] = []

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(development.X, development.y, groups=groups),
        start=1,
    ):
        X_train = development.X.iloc[train_idx].copy()
        X_val = development.X.iloc[val_idx].copy()
        y_train = development.y.iloc[train_idx].copy()
        y_val = development.y.iloc[val_idx].copy()
        train_meta = development.metadata.iloc[train_idx].copy()
        val_meta = development.metadata.iloc[val_idx].copy()

        for params in lr_configs:
            fold_results.append(
                _run_lr_fold(
                    params, cv_strategy, fold,
                    X_train, X_val, y_train, y_val,
                    train_meta, val_meta, random_state,
                )
            )

        for params in rf_configs:
            fold_results.append(
                _run_rf_fold(
                    params, cv_strategy, fold,
                    X_train, X_val, y_train, y_val,
                    train_meta, val_meta, random_state,
                )
            )

    config_results = aggregate_tuning_results(fold_results)

    return TouchscreenTuningResult(
        fold_results=fold_results,
        config_results=config_results,
        best_logistic_regression=select_best_configuration(
            config_results, MODEL_LOGISTIC_REGRESSION, cv_strategy
        ),
        best_random_forest=select_best_configuration(
            config_results, MODEL_RANDOM_FOREST, cv_strategy
        ),
        development_metadata=development.metadata.copy(),
        final_test_participants=set(final_test_participants or set()),
        cv_strategy=cv_strategy,
        n_splits=n_splits,
    )


# --------------------------------------------------------------------------- #
# High-level pipeline                                                         #
# --------------------------------------------------------------------------- #

def run_touchscreen_tuning(
    ml_dataset: TouchscreenMLDataset | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratify: bool = True,
    n_splits: int = DEFAULT_N_SPLITS,
) -> tuple[TouchscreenTuningResult, TouchscreenTuningResult | None, TouchscreenParticipantSplit]:
    """Build dataset, split by participant, then run tuning on development only.

    Returns:
        (group_kfold_result, stratified_group_kfold_result_or_None, split)
    """
    dataset = build_touchscreen_ml_dataset() if ml_dataset is None else ml_dataset
    split = split_touchscreen_by_participant(
        dataset, test_size=test_size, random_state=random_state, stratify=stratify
    )
    development = TouchscreenMLDataset(
        X=split.X_dev.copy(),
        y=split.y_dev.copy(),
        metadata=split.dev_metadata.copy(),
    )
    final_test_p = split.final_test_participants

    gkf_result = tune_touchscreen_development(
        development,
        final_test_participants=final_test_p,
        n_splits=n_splits,
        random_state=random_state,
        cv_strategy="GroupKFold",
    )

    sgkf_result: TouchscreenTuningResult | None = None
    if STRATIFIED_GROUP_KFOLD_AVAILABLE:
        sgkf_result = tune_touchscreen_development(
            development,
            final_test_participants=final_test_p,
            n_splits=n_splits,
            random_state=random_state,
            cv_strategy="StratifiedGroupKFold",
        )

    return gkf_result, sgkf_result, split


# --------------------------------------------------------------------------- #
# Persistence                                                                 #
# --------------------------------------------------------------------------- #

def _params_to_json_safe(params: dict[str, Any]) -> dict[str, Any]:
    return {k: (None if v is None else v) for k, v in params.items()}


def save_tuning_results(
    result: TouchscreenTuningResult,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Path]:
    """Save fold results and config summary as JSON/CSV under output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = result.cv_strategy.lower()

    # 1. Per-fold results as CSV
    fold_rows = []
    for r in result.fold_results:
        row: dict[str, Any] = {
            "model": r.model_name,
            "cv_strategy": r.cv_strategy,
            "fold": r.fold,
            "train_recordings": r.train_recordings,
            "validation_recordings": r.validation_recordings,
            "train_participants": r.train_participants,
            "validation_participants": r.validation_participants,
            "participant_overlap": r.participant_overlap,
            **{f"param_{k}": v for k, v in r.params.items()},
            **{m: float(r.metrics[m]) for m in METRIC_COLUMNS},
        }
        fold_rows.append(row)
    fold_df = pd.DataFrame(fold_rows)
    fold_csv = output_dir / f"fold_results_{suffix}.csv"
    fold_df.to_csv(fold_csv, index=False)

    # 2. Config summary as CSV
    cfg_df = result.config_results.copy()
    cfg_df["params_json"] = cfg_df["params"].apply(
        lambda p: json.dumps(_params_to_json_safe(p))
    )
    cfg_df = cfg_df.drop(columns=["params"])
    config_csv = output_dir / f"config_results_{suffix}.csv"
    cfg_df.to_csv(config_csv, index=False)

    # 3. Best configs as JSON
    best = {
        "cv_strategy": result.cv_strategy,
        "selection_metric": result.selection_metric,
        "best_logistic_regression": {
            "params": _params_to_json_safe(result.best_logistic_regression["params"]),
            "mean_f1": float(result.best_logistic_regression["mean_f1"]),
            "std_f1": float(result.best_logistic_regression["std_f1"]),
        },
        "best_random_forest": {
            "params": _params_to_json_safe(result.best_random_forest["params"]),
            "mean_f1": float(result.best_random_forest["mean_f1"]),
            "std_f1": float(result.best_random_forest["std_f1"]),
        },
    }
    best_json = output_dir / f"best_configs_{suffix}.json"
    best_json.write_text(json.dumps(best, indent=2))

    return {"fold_csv": fold_csv, "config_csv": config_csv, "best_json": best_json}


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def _fmt(v: object) -> str:
    return f"{float(v):.4f}"


def _print_top_configs(config_results: pd.DataFrame, model_name: str, cv_strategy: str, top_n: int = 5) -> None:
    subset = config_results[
        (config_results["model"] == model_name)
        & (config_results["cv_strategy"] == cv_strategy)
    ].sort_values("mean_f1", ascending=False).head(top_n)
    for _, row in subset.iterrows():
        p = row["params"]
        param_str = ", ".join(f"{k}={v}" for k, v in p.items())
        print(
            f"  {param_str}: "
            f"F1={_fmt(row['mean_f1'])}±{_fmt(row['std_f1'])}, "
            f"Acc={_fmt(row['mean_accuracy'])}, "
            f"Pre={_fmt(row['mean_precision'])}, "
            f"Rec={_fmt(row['mean_recall'])}"
        )


def _print_per_fold_robustness(
    fold_results: list[TouchscreenTuningFoldResult],
    model_name: str,
    cv_strategy: str,
    best_params: dict[str, Any],
) -> None:
    best_key = _params_key(best_params)
    folds = [
        r for r in fold_results
        if r.model_name == model_name
        and r.cv_strategy == cv_strategy
        and _params_key(r.params) == best_key
    ]
    for r in sorted(folds, key=lambda x: x.fold):
        print(
            f"  Fold {r.fold}: "
            f"F1={_fmt(r.metrics['f1'])}, "
            f"Pre={_fmt(r.metrics['precision'])}, "
            f"Rec={_fmt(r.metrics['recall'])}, "
            f"train={r.train_recordings}rec/{r.train_participants}p, "
            f"val={r.validation_recordings}rec/{r.validation_participants}p "
            f"(PD={r.validation_class_counts.get('PD', 0)}, "
            f"CNT={r.validation_class_counts.get('CNT', 0)})"
        )


def main() -> int:
    print("Touchscreen Model Tuning — Step 8")
    print("=" * 60)

    gkf_result, sgkf_result, split = run_touchscreen_tuning(
        test_size=DEFAULT_TEST_SIZE,
        random_state=DEFAULT_RANDOM_STATE,
        n_splits=DEFAULT_N_SPLITS,
    )

    dev_p = set(split.dev_metadata["participant_id"].astype(str))
    test_p = split.final_test_participants

    print(f"\nDataset:")
    print(f"  Development: {len(split.X_dev)} recordings / {len(dev_p)} participants")
    print(f"  Final-test (LOCKED): {len(split.X_test)} recordings / {len(test_p)} participants")
    print(f"  Participant overlap: {len(dev_p & test_p)}")
    print(f"  Features: {split.X_dev.shape[1]}")

    # Step 7 baseline reference (GroupKFold C=1.0, class_weight=None, RF n=200 default)
    BASELINE_LR_F1 = 0.8075
    BASELINE_RF_F1 = 0.7741

    for label, result in [("GroupKFold", gkf_result), ("StratifiedGroupKFold", sgkf_result)]:
        if result is None:
            print(f"\n{label}: not available in this scikit-learn version.")
            continue

        print(f"\n{'='*60}")
        print(f"CV Strategy: {label}")
        print(f"  Folds: {result.n_splits}   |   Groups: participant_id")

        # Fold details for baseline LR config
        lr_best_p = result.best_logistic_regression["params"]
        rf_best_p = result.best_random_forest["params"]

        print(f"\nLogistic Regression — Top 5 configurations (by mean F1):")
        _print_top_configs(result.config_results, MODEL_LOGISTIC_REGRESSION, label)

        best_lr = result.best_logistic_regression
        lr_improvement = float(best_lr["mean_f1"]) - BASELINE_LR_F1
        print(f"\n  >> Best LR: {lr_best_p}")
        print(f"     mean F1={_fmt(best_lr['mean_f1'])} ± {_fmt(best_lr['std_f1'])}")
        print(f"     Improvement over Step 7 baseline: {lr_improvement:+.4f}")

        print(f"\nLogistic Regression — Per-fold robustness (best config):")
        _print_per_fold_robustness(result.fold_results, MODEL_LOGISTIC_REGRESSION, label, lr_best_p)

        print(f"\nRandom Forest — Top 5 configurations (by mean F1):")
        _print_top_configs(result.config_results, MODEL_RANDOM_FOREST, label)

        best_rf = result.best_random_forest
        rf_improvement = float(best_rf["mean_f1"]) - BASELINE_RF_F1
        print(f"\n  >> Best RF: {rf_best_p}")
        print(f"     mean F1={_fmt(best_rf['mean_f1'])} ± {_fmt(best_rf['std_f1'])}")
        print(f"     Improvement over Step 7 baseline: {rf_improvement:+.4f}")

        print(f"\nRandom Forest — Per-fold robustness (best config):")
        _print_per_fold_robustness(result.fold_results, MODEL_RANDOM_FOREST, label, rf_best_p)

        # Save outputs
        saved = save_tuning_results(result)
        print(f"\nOutputs saved:")
        for name, path in saved.items():
            print(f"  {name}: {path}")

    print(f"\n{'='*60}")
    print("NOTE: The final-test set was NEVER loaded, inspected, or evaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
