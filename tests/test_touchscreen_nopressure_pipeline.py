"""Isolated tests for the 19-feature pressure-independent touchscreen ML pipeline.

Verifies:
1. Exactly 19 features are produced.
2. Feature names match the canonical 19-feature list.
3. Feature order is deterministic.
4. No pressure feature is present.
5. Hold statistics match the established implementation.
6. Inter-tap statistics match the established implementation.
7. Recording-level features match the established implementation.
8. Participant split contains exactly 269 dev / 69 test recordings.
9. There is zero participant overlap.
10. Locked test participant IDs are unchanged.
11. Test data is not used during training/tuning.
12. Model artifact contains exactly 19 features.
13. Threshold is selected without using final test data.
14. Final evaluation uses exactly 69 recordings.
"""

from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from ml.features.touchscreen_nopressure_feature_extraction import (
    ALL_TOUCHSCREEN_NOPRESSURE_OUTPUT_COLUMNS,
    HOLD_FEATURES,
    INTER_TAP_FEATURES,
    RECORDING_LEVEL_FEATURES,
    TOUCHSCREEN_NOPRESSURE_NUMERICAL_FEATURE_COLUMNS,
    extract_all_touchscreen_nopressure_features,
    extract_touchscreen_nopressure_recording_features,
)
from ml.models.export_touchscreen_nopressure_artifacts import export_touchscreen_nopressure_artifacts
from ml.training.touchscreen_nopressure_baseline import (
    DEFAULT_N_SPLITS,
    DEFAULT_RANDOM_STATE,
    train_evaluate_nopressure_baseline_cv,
)
from ml.training.touchscreen_nopressure_dataset import (
    EXACT_NOPRESSURE_FEATURE_COUNT,
    EXPECTED_TEST_PARTICIPANTS_LIST,
    build_touchscreen_nopressure_ml_dataset,
    split_touchscreen_nopressure_by_participant,
)
from ml.training.touchscreen_nopressure_final_evaluation import (
    evaluate_touchscreen_nopressure_final_model,
    run_touchscreen_nopressure_final_evaluation,
)
from ml.training.touchscreen_nopressure_tuning import tune_nopressure_models


CANONICAL_19_FEATURES = [
    "hold_mean",
    "hold_std",
    "hold_variance",
    "hold_median",
    "hold_min",
    "hold_max",
    "hold_skewness",
    "hold_kurtosis",
    "inter_tap_mean",
    "inter_tap_std",
    "inter_tap_variance",
    "inter_tap_median",
    "inter_tap_min",
    "inter_tap_max",
    "inter_tap_skewness",
    "inter_tap_kurtosis",
    "event_count",
    "recording_duration",
    "typing_rate",
]

EXPECTED_TEST_PIDS = ["S09", "S10", "S16", "S18", "S22", "S27", "S32"]


def test_1_exactly_19_features_produced():
    """Requirement 1: Exactly 19 features are produced."""
    assert len(TOUCHSCREEN_NOPRESSURE_NUMERICAL_FEATURE_COLUMNS) == 19
    assert EXACT_NOPRESSURE_FEATURE_COUNT == 19


def test_2_feature_names_match_canonical_list():
    """Requirement 2: Feature names match the canonical 19-feature list."""
    assert list(TOUCHSCREEN_NOPRESSURE_NUMERICAL_FEATURE_COLUMNS) == CANONICAL_19_FEATURES


def test_3_feature_order_is_deterministic():
    """Requirement 3: Feature order is deterministic."""
    expected_order = HOLD_FEATURES + INTER_TAP_FEATURES + RECORDING_LEVEL_FEATURES
    assert list(TOUCHSCREEN_NOPRESSURE_NUMERICAL_FEATURE_COLUMNS) == expected_order
    assert list(TOUCHSCREEN_NOPRESSURE_NUMERICAL_FEATURE_COLUMNS) == CANONICAL_19_FEATURES


def test_4_no_pressure_feature_present():
    """Requirement 4: No pressure feature is present."""
    for feat in TOUCHSCREEN_NOPRESSURE_NUMERICAL_FEATURE_COLUMNS:
        assert "pressure" not in feat.lower(), f"Pressure feature found: {feat}"


def test_5_hold_statistics_match_implementation():
    """Requirement 5: Hold statistics match established implementation."""
    dummy_recording = pd.DataFrame(
        {
            "participant_id": ["S01"] * 5,
            "recording_id": ["R01"] * 5,
            "text_id": ["T1"] * 5,
            "label": ["CNT"] * 5,
            "press_time_ms": [100.0, 500.0, 1000.0, 1500.0, 2000.0],
            "release_time_ms": [200.0, 650.0, 1200.0, 1750.0, 2300.0],
            "hold_time_ms": [100.0, 150.0, 200.0, 250.0, 300.0],
        }
    )
    feats = extract_touchscreen_nopressure_recording_features(dummy_recording)

    hold_vals_sec = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
    assert pytest.approx(feats["hold_mean"], rel=1e-5) == float(hold_vals_sec.mean())
    assert pytest.approx(feats["hold_std"], rel=1e-5) == float(pd.Series(hold_vals_sec).std())
    assert pytest.approx(feats["hold_variance"], rel=1e-5) == float(pd.Series(hold_vals_sec).var())
    assert pytest.approx(feats["hold_median"], rel=1e-5) == float(np.median(hold_vals_sec))
    assert pytest.approx(feats["hold_min"], rel=1e-5) == 0.1
    assert pytest.approx(feats["hold_max"], rel=1e-5) == 0.3


def test_6_inter_tap_statistics_match_implementation():
    """Requirement 6: Inter-tap statistics match established implementation."""
    dummy_recording = pd.DataFrame(
        {
            "participant_id": ["S01"] * 5,
            "recording_id": ["R01"] * 5,
            "text_id": ["T1"] * 5,
            "label": ["CNT"] * 5,
            "press_time_ms": [1000.0, 1500.0, 2100.0, 2800.0, 3600.0],
            "release_time_ms": [1100.0, 1600.0, 2200.0, 2900.0, 3700.0],
            "hold_time_ms": [100.0, 100.0, 100.0, 100.0, 100.0],
        }
    )
    feats = extract_touchscreen_nopressure_recording_features(dummy_recording)

    # Intervals: 500ms, 600ms, 700ms, 800ms -> in sec: 0.5, 0.6, 0.7, 0.8
    expected_intervals = np.array([0.5, 0.6, 0.7, 0.8])
    assert pytest.approx(feats["inter_tap_mean"], rel=1e-5) == float(expected_intervals.mean())
    assert pytest.approx(feats["inter_tap_std"], rel=1e-5) == float(pd.Series(expected_intervals).std())
    assert pytest.approx(feats["inter_tap_min"], rel=1e-5) == 0.5
    assert pytest.approx(feats["inter_tap_max"], rel=1e-5) == 0.8


def test_7_recording_level_features_match_implementation():
    """Requirement 7: Recording-level features match established implementation."""
    dummy_recording = pd.DataFrame(
        {
            "participant_id": ["S01"] * 4,
            "recording_id": ["R01"] * 4,
            "text_id": ["T1"] * 4,
            "label": ["CNT"] * 4,
            "press_time_ms": [1000.0, 2000.0, 3000.0, 4000.0],
            "release_time_ms": [1200.0, 2200.0, 3200.0, 5000.0],
            "hold_time_ms": [200.0, 200.0, 200.0, 1000.0],
        }
    )
    feats = extract_touchscreen_nopressure_recording_features(dummy_recording)

    assert feats["event_count"] == 4
    # duration = (5000 - 1000) / 1000 = 4.0 sec
    assert feats["recording_duration"] == 4.0
    # typing_rate = 4 / 4.0 = 1.0 event/sec
    assert feats["typing_rate"] == 1.0


def test_8_participant_split_counts():
    """Requirement 8: Participant split contains exactly 269 dev / 69 test recordings."""
    dataset = build_touchscreen_nopressure_ml_dataset()
    split = split_touchscreen_nopressure_by_participant(dataset, test_size=0.2, random_state=42, stratify=True)

    assert len(dataset.X) == 338
    assert len(split.X_dev) == 269
    assert len(split.X_test) == 69
    assert len(split.development_participants) == 26
    assert len(split.final_test_participants) == 7


def test_9_zero_participant_overlap():
    """Requirement 9: There is zero participant overlap between dev and test."""
    dataset = build_touchscreen_nopressure_ml_dataset()
    split = split_touchscreen_nopressure_by_participant(dataset, test_size=0.2, random_state=42, stratify=True)

    dev_pids = split.development_participants
    test_pids = split.final_test_participants
    overlap = dev_pids & test_pids

    assert len(overlap) == 0, f"Participant overlap detected: {overlap}"


def test_10_locked_test_participant_ids():
    """Requirement 10: Locked test participant IDs are unchanged."""
    dataset = build_touchscreen_nopressure_ml_dataset()
    split = split_touchscreen_nopressure_by_participant(dataset, test_size=0.2, random_state=42, stratify=True)

    actual_test_pids = sorted(split.final_test_participants)
    assert actual_test_pids == EXPECTED_TEST_PIDS
    assert EXPECTED_TEST_PARTICIPANTS_LIST == EXPECTED_TEST_PIDS


def test_11_test_data_not_used_during_training_tuning():
    """Requirement 11: Test data is not used during training/tuning."""
    dataset = build_touchscreen_nopressure_ml_dataset()
    split = split_touchscreen_nopressure_by_participant(dataset, test_size=0.2, random_state=42, stratify=True)

    dev_ds = split.development_recordings
    assert len(dev_ds.X) == 269

    # Verify CV baseline uses only dev_ds
    cv_res = train_evaluate_nopressure_baseline_cv(dev_ds, n_splits=5, random_state=42)
    assert len(cv_res.fold_results) == 10  # 5 folds x 2 models

    for fold in cv_res.fold_results:
        # Check no test participant was in train or val set
        train_val_parts = set(fold.train_participant_ids) | set(fold.validation_participant_ids)
        assert len(train_val_parts & set(EXPECTED_TEST_PIDS)) == 0


def test_12_model_artifact_contains_exactly_19_features():
    """Requirement 12: Model artifact contains exactly 19 features."""
    paths = export_touchscreen_nopressure_artifacts()

    # Check joblib feature names
    feat_names = joblib.load(paths["feature_names"])
    assert len(feat_names) == 19
    assert feat_names == CANONICAL_19_FEATURES

    # Check metadata joblib
    meta = joblib.load(paths["metadata"])
    assert meta["feature_count"] == 19
    assert meta["pressure_features_excluded"] is True
    assert meta["development_recordings"] == 269
    assert meta["final_test_recordings"] == 69

    # Check mobile JSON artifact
    with open(paths["mobile_json"], "r") as f:
        mobile_data = json.load(f)

    assert mobile_data["feature_count"] == 19
    assert mobile_data["feature_names"] == CANONICAL_19_FEATURES
    assert len(mobile_data["scaler"]["means"]) == 19
    assert len(mobile_data["scaler"]["scales"]) == 19
    assert len(mobile_data["logistic_regression"]["coefficients"]) == 19
    assert mobile_data["pressure_features_excluded"] is True


def test_13_threshold_selected_without_test_data():
    """Requirement 13: Threshold is selected without using final test data."""
    dataset = build_touchscreen_nopressure_ml_dataset()
    split = split_touchscreen_nopressure_by_participant(dataset, test_size=0.2, random_state=42, stratify=True)

    # Tuning accepts development_recordings only
    results, best = tune_nopressure_models(split.development_recordings, n_splits=5, random_state=42)
    assert best.optimal_threshold >= 0.1
    assert best.optimal_threshold <= 0.9


def test_14_final_evaluation_uses_exactly_69_recordings():
    """Requirement 14: Final evaluation uses exactly 69 recordings."""
    res = run_touchscreen_nopressure_final_evaluation()

    assert res.test_recordings == 69
    assert res.test_participants == 7
    assert sorted(res.test_participant_ids) == EXPECTED_TEST_PIDS
    assert res.dev_recordings == 269
    assert res.dev_participants == 26
    assert res.participant_overlap == 0
