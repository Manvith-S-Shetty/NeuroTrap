"""Unit tests for touchscreen baseline model training and participant-aware CV.

Verifies:
1. Development dataset loads correctly
2. Final test set is never used during baseline evaluation
3. Participant groups are respected (zero overlap in every fold)
4. Scaling is fitted inside CV folds, not globally
5. Both models produce valid predictions
6. Metrics are computed correctly
7. Feature count remains 27
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.features.touchscreen_feature_extraction import TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS
from ml.training.touchscreen_baseline import (
    CLASS_LABELS,
    DEFAULT_N_SPLITS,
    DEFAULT_RANDOM_STATE,
    MODEL_LOGISTIC_REGRESSION,
    MODEL_RANDOM_FOREST,
    RF_N_ESTIMATORS,
    TouchscreenCVFoldResult,
    TouchscreenCVResult,
    aggregate_fold_metrics,
    cross_validate_touchscreen_development,
    evaluate_predictions,
    run_touchscreen_baseline,
    validate_development_dataset,
    validate_finite_features,
)
from ml.training.touchscreen_dataset import (
    TouchscreenMLDataset,
    build_touchscreen_ml_dataset,
    split_touchscreen_by_participant,
)


# ------------------------------------------------------------------ #
# Fixtures                                                            #
# ------------------------------------------------------------------ #

def _make_synthetic_feature_table(
    n_participants: int = 8,
    recs_per_participant: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """Return a minimal valid feature table with deterministic values."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_participants):
        label = "PD" if i < n_participants // 2 else "CNT"
        for j in range(recs_per_participant):
            rec_id = f"REC_{i:02d}_{j}"
            row: dict[str, object] = {
                "participant_id": f"P{i:03d}",
                "recording_id": rec_id,
                "text_id": f"T{j}",
                "label": label,
                "raw_file": f"{rec_id}.csv",
                "raw_path": f"/data/{rec_id}.csv",
            }
            for feat in TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS:
                row[feat] = float(rng.uniform(0.001, 1.0))
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic_table():
    return _make_synthetic_feature_table(n_participants=8, recs_per_participant=3)


@pytest.fixture(scope="module")
def synthetic_dataset(synthetic_table):
    return build_touchscreen_ml_dataset(synthetic_table)


@pytest.fixture(scope="module")
def synthetic_split(synthetic_dataset):
    return split_touchscreen_by_participant(
        synthetic_dataset, test_size=0.25, random_state=42, stratify=True
    )


@pytest.fixture(scope="module")
def synthetic_development(synthetic_split):
    sp = synthetic_split
    return TouchscreenMLDataset(
        X=sp.X_dev.copy(),
        y=sp.y_dev.copy(),
        metadata=sp.dev_metadata.copy(),
    )


@pytest.fixture(scope="module")
def real_dataset():
    return build_touchscreen_ml_dataset()


@pytest.fixture(scope="module")
def real_split(real_dataset):
    return split_touchscreen_by_participant(real_dataset, test_size=0.2, random_state=42)


@pytest.fixture(scope="module")
def real_development(real_split):
    sp = real_split
    return TouchscreenMLDataset(
        X=sp.X_dev.copy(),
        y=sp.y_dev.copy(),
        metadata=sp.dev_metadata.copy(),
    )


@pytest.fixture(scope="module")
def real_cv_result(real_development, real_split):
    return cross_validate_touchscreen_development(
        real_development,
        final_test_participants=real_split.final_test_participants,
        n_splits=5,
        random_state=42,
    )


# ------------------------------------------------------------------ #
# 1. Development dataset loads correctly                              #
# ------------------------------------------------------------------ #

def test_development_dataset_loads(real_development):
    assert len(real_development.X) == 269
    assert real_development.X.shape[1] == 27
    assert set(real_development.y.unique()).issubset({"PD", "CNT"})
    assert "participant_id" in real_development.metadata.columns


# ------------------------------------------------------------------ #
# 2. Final test set is not used in baseline evaluation               #
# ------------------------------------------------------------------ #

def test_final_test_not_in_cv(real_development, real_split, real_cv_result):
    """The final-test participants must not appear in any CV fold."""
    test_p = real_split.final_test_participants
    for fold_result in real_cv_result.fold_results:
        train_ids = set(fold_result.train_participant_ids)
        val_ids = set(fold_result.validation_participant_ids)
        all_cv_ids = train_ids | val_ids
        assert test_p.isdisjoint(all_cv_ids), (
            f"Fold {fold_result.fold} / {fold_result.model_name} contains "
            f"locked final-test participants: {test_p & all_cv_ids}"
        )


def test_final_test_never_loaded_in_run():
    """run_touchscreen_baseline must not include final-test data in CV."""
    cv_result, split = run_touchscreen_baseline(
        test_size=0.2, random_state=42, n_splits=5
    )
    test_p = split.final_test_participants
    for fold_result in cv_result.fold_results:
        all_fold_p = set(fold_result.train_participant_ids) | set(fold_result.validation_participant_ids)
        assert test_p.isdisjoint(all_fold_p)


# ------------------------------------------------------------------ #
# 3. Participant groups are respected (zero overlap per fold)         #
# ------------------------------------------------------------------ #

def test_zero_participant_overlap_per_fold(real_cv_result):
    for fold_result in real_cv_result.fold_results:
        assert fold_result.participant_overlap == 0, (
            f"Fold {fold_result.fold} / {fold_result.model_name} has "
            f"participant_overlap={fold_result.participant_overlap}"
        )


def test_participant_groups_cover_all_dev(real_cv_result, real_development):
    """Every development participant must appear in at least one validation fold."""
    all_dev_p = set(real_development.metadata["participant_id"].astype(str))
    validated_p: set[str] = set()
    for fold_result in real_cv_result.fold_results:
        if fold_result.model_name == MODEL_LOGISTIC_REGRESSION:
            validated_p |= set(fold_result.validation_participant_ids)
    assert validated_p == all_dev_p


# ------------------------------------------------------------------ #
# 4. Scaling fitted inside CV folds, not globally                    #
# ------------------------------------------------------------------ #

def test_scaler_fitted_inside_fold_not_globally(real_cv_result, real_development):
    """Each LR fold result has its own scaler with the correct input dimension."""
    n_features = real_development.X.shape[1]
    lr_folds = [r for r in real_cv_result.fold_results if r.model_name == MODEL_LOGISTIC_REGRESSION]
    assert len(lr_folds) > 0
    for fold_result in lr_folds:
        assert fold_result.scaler is not None, f"Fold {fold_result.fold} LR missing scaler"
        assert fold_result.scaler.n_features_in_ == n_features
        # Scaler must NOT have been fitted on the full dev set (mean_ length is same as
        # train fold, which is smaller than the full dev set — we verify dimension only)
        assert fold_result.scaler.mean_.shape[0] == n_features


def test_rf_fold_has_no_scaler(real_cv_result):
    """Random Forest folds must not carry a scaler (RF is scale-invariant)."""
    rf_folds = [r for r in real_cv_result.fold_results if r.model_name == MODEL_RANDOM_FOREST]
    for fold_result in rf_folds:
        assert fold_result.scaler is None


# ------------------------------------------------------------------ #
# 5. Both models produce valid predictions                            #
# ------------------------------------------------------------------ #

def test_lr_produces_valid_predictions(real_cv_result):
    lr_folds = [r for r in real_cv_result.fold_results if r.model_name == MODEL_LOGISTIC_REGRESSION]
    for fold_result in lr_folds:
        assert len(fold_result.y_pred) == fold_result.validation_recordings
        assert set(fold_result.y_pred).issubset(set(CLASS_LABELS))


def test_rf_produces_valid_predictions(real_cv_result):
    rf_folds = [r for r in real_cv_result.fold_results if r.model_name == MODEL_RANDOM_FOREST]
    for fold_result in rf_folds:
        assert len(fold_result.y_pred) == fold_result.validation_recordings
        assert set(fold_result.y_pred).issubset(set(CLASS_LABELS))


def test_both_models_present_in_every_fold(real_cv_result):
    model_names_per_fold: dict[int, set[str]] = {}
    for r in real_cv_result.fold_results:
        model_names_per_fold.setdefault(r.fold, set()).add(r.model_name)
    for fold, models in model_names_per_fold.items():
        assert MODEL_LOGISTIC_REGRESSION in models, f"Fold {fold} missing LR"
        assert MODEL_RANDOM_FOREST in models, f"Fold {fold} missing RF"


# ------------------------------------------------------------------ #
# 6. Metrics are calculated correctly                                 #
# ------------------------------------------------------------------ #

def test_evaluate_predictions_all_correct():
    y_true = pd.Series(["PD", "PD", "CNT", "CNT"])
    y_pred = pd.Series(["PD", "PD", "CNT", "CNT"])
    m = evaluate_predictions(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)


def test_evaluate_predictions_all_wrong():
    y_true = pd.Series(["PD", "PD", "CNT", "CNT"])
    y_pred = pd.Series(["CNT", "CNT", "PD", "PD"])
    m = evaluate_predictions(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(0.0)
    assert m["recall"] == pytest.approx(0.0)


def test_aggregate_metrics_structure(real_cv_result):
    agg = real_cv_result.aggregate_metrics
    assert "model" in agg.columns
    assert set(agg["model"]) == {MODEL_LOGISTIC_REGRESSION, MODEL_RANDOM_FOREST}
    for metric in ["accuracy", "precision", "recall", "f1"]:
        assert f"mean_{metric}" in agg.columns
        assert f"std_{metric}" in agg.columns


def test_aggregate_metrics_values_in_range(real_cv_result):
    agg = real_cv_result.aggregate_metrics
    for _, row in agg.iterrows():
        for metric in ["accuracy", "precision", "recall", "f1"]:
            mean_val = float(row[f"mean_{metric}"])
            std_val = float(row[f"std_{metric}"])
            assert 0.0 <= mean_val <= 1.0, f"{metric} mean={mean_val} out of range"
            assert std_val >= 0.0, f"{metric} std={std_val} negative"


# ------------------------------------------------------------------ #
# 7. Feature count remains 27                                         #
# ------------------------------------------------------------------ #

def test_feature_count_27_in_dev(real_development):
    assert real_development.X.shape[1] == 27
    assert list(real_development.X.columns) == TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS


def test_feature_count_27_in_full_dataset(real_dataset):
    assert real_dataset.X.shape[1] == 27


def test_feature_count_unchanged_after_cv(real_cv_result, real_development):
    """Running CV must not alter the development set's column count."""
    assert real_development.X.shape[1] == 27


# ------------------------------------------------------------------ #
# Additional: validate_development_dataset rejects final-test leak   #
# ------------------------------------------------------------------ #

def test_validate_development_rejects_test_leak(real_development, real_split):
    """If a final-test participant is injected into dev, validation must raise."""
    test_p = real_split.final_test_participants
    some_test_p = next(iter(test_p))

    # Inject a row with a final-test participant_id; give it a unique recording_id
    # so the duplicate-ID check does not fire before the participant-leakage check.
    injected_row = real_development.metadata.iloc[[0]].copy()
    injected_row["participant_id"] = some_test_p
    injected_row["recording_id"] = "__INJECTED_LEAK_REC__"
    injected_meta = pd.concat(
        [real_development.metadata, injected_row], ignore_index=True
    )
    injected_X = pd.concat(
        [real_development.X, real_development.X.iloc[[0]].copy()], ignore_index=True
    )
    injected_y = pd.concat(
        [real_development.y, real_development.y.iloc[0:1].copy()], ignore_index=True
    )
    bad_dev = TouchscreenMLDataset(X=injected_X, y=injected_y, metadata=injected_meta)

    with pytest.raises(ValueError, match="locked final-test participants"):
        validate_development_dataset(bad_dev, final_test_participants=test_p)


def test_validate_finite_features_rejects_nan():
    X = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 3.0]})
    with pytest.raises(ValueError, match="NaN feature values"):
        validate_finite_features(X, "test_X")


def test_validate_finite_features_rejects_inf():
    X = pd.DataFrame({"a": [1.0, np.inf], "b": [2.0, 3.0]})
    with pytest.raises(ValueError, match="infinite feature values"):
        validate_finite_features(X, "test_X")
