"""Unit tests for touchscreen feature extraction module."""

import numpy as np
import pandas as pd
import pytest

from ml.features.touchscreen_feature_extraction import (
    HOLD_FEATURES,
    INTER_TAP_FEATURES,
    PRESSURE_FEATURES,
    RECORDING_LEVEL_FEATURES,
    TOUCHSCREEN_METADATA_COLUMNS,
    TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS,
    calculate_inter_tap_intervals_sec,
    extract_all_touchscreen_features,
    extract_touchscreen_recording_features,
    get_valid_hold_times_sec,
    get_valid_pressures,
)
from ml.preprocessing.touchscreen.data_loader import (
    DEFAULT_TOUCHSCREEN_PACKAGE_ROOT,
    load_all_touchscreen_recordings,
)


@pytest.fixture
def sample_recording_df() -> pd.DataFrame:
    """Fixture providing a clean multi-event touchscreen recording DataFrame."""
    return pd.DataFrame(
        [
            {
                "participant_id": "S01",
                "recording_id": "S01_TEX01",
                "text_id": "TEX01",
                "row_number": 1,
                "press_time_ms": 1000,
                "release_time_ms": 1100,
                "hold_time_ms": 100,
                "pressure": 0.5,
                "label": "PD",
                "raw_file": "S01_TEX01.txt",
                "raw_path": "/path/S01_TEX01.txt",
            },
            {
                "participant_id": "S01",
                "recording_id": "S01_TEX01",
                "text_id": "TEX01",
                "row_number": 2,
                "press_time_ms": 2000,
                "release_time_ms": 2200,
                "hold_time_ms": 200,
                "pressure": 0.6,
                "label": "PD",
                "raw_file": "S01_TEX01.txt",
                "raw_path": "/path/S01_TEX01.txt",
            },
            {
                "participant_id": "S01",
                "recording_id": "S01_TEX01",
                "text_id": "TEX01",
                "row_number": 3,
                "press_time_ms": 3500,
                "release_time_ms": 3800,
                "hold_time_ms": 300,
                "pressure": 0.7,
                "label": "PD",
                "raw_file": "S01_TEX01.txt",
                "raw_path": "/path/S01_TEX01.txt",
            },
            {
                "participant_id": "S01",
                "recording_id": "S01_TEX01",
                "text_id": "TEX01",
                "row_number": 4,
                "press_time_ms": 5000,
                "release_time_ms": 5400,
                "hold_time_ms": 400,
                "pressure": 0.8,
                "label": "PD",
                "raw_file": "S01_TEX01.txt",
                "raw_path": "/path/S01_TEX01.txt",
            },
        ]
    )


def test_exact_27_feature_columns_and_order():
    """Verify that feature extraction defines exactly 27 numerical features in expected order."""
    assert len(TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS) == 27
    assert len(HOLD_FEATURES) == 8
    assert len(INTER_TAP_FEATURES) == 8
    assert len(PRESSURE_FEATURES) == 8
    assert len(RECORDING_LEVEL_FEATURES) == 3

    expected_columns = [
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
        "pressure_mean",
        "pressure_std",
        "pressure_variance",
        "pressure_median",
        "pressure_min",
        "pressure_max",
        "pressure_skewness",
        "pressure_kurtosis",
        "event_count",
        "recording_duration",
        "typing_rate",
    ]
    assert TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS == expected_columns


def test_hold_time_calculations(sample_recording_df: pd.DataFrame):
    """Test valid hold time extraction in seconds and aggregate statistics."""
    hold_sec = get_valid_hold_times_sec(sample_recording_df)
    assert list(hold_sec) == [0.1, 0.2, 0.3, 0.4]

    feats = extract_touchscreen_recording_features(sample_recording_df)
    assert feats["hold_mean"] == pytest.approx(0.25)
    assert feats["hold_median"] == pytest.approx(0.25)
    assert feats["hold_min"] == pytest.approx(0.1)
    assert feats["hold_max"] == pytest.approx(0.4)


def test_inter_tap_calculations(sample_recording_df: pd.DataFrame):
    """Test inter-tap press[i+1] - press[i] calculation in seconds."""
    inter_tap_sec = calculate_inter_tap_intervals_sec(sample_recording_df)
    # press: 1000, 2000, 3500, 5000 -> diffs ms: 1000, 1500, 1500 -> in sec: 1.0, 1.5, 1.5
    assert list(inter_tap_sec) == [1.0, 1.5, 1.5]

    feats = extract_touchscreen_recording_features(sample_recording_df)
    assert feats["inter_tap_mean"] == pytest.approx(1.3333333333333333)
    assert feats["inter_tap_min"] == pytest.approx(1.0)
    assert feats["inter_tap_max"] == pytest.approx(1.5)


def test_pressure_calculations(sample_recording_df: pd.DataFrame):
    """Test pressure extraction and statistics."""
    pressures = get_valid_pressures(sample_recording_df)
    assert list(pressures) == [0.5, 0.6, 0.7, 0.8]

    feats = extract_touchscreen_recording_features(sample_recording_df)
    assert feats["pressure_mean"] == pytest.approx(0.65)
    assert feats["pressure_min"] == pytest.approx(0.5)
    assert feats["pressure_max"] == pytest.approx(0.8)


def test_recording_duration_and_typing_rate(sample_recording_df: pd.DataFrame):
    """Test recording duration (max release - min press in sec) and typing rate."""
    feats = extract_touchscreen_recording_features(sample_recording_df)
    # min press = 1000, max release = 5400 -> duration ms = 4400 -> 4.4 sec
    assert feats["recording_duration"] == pytest.approx(4.4)
    # event_count = 4 -> rate = 4 / 4.4 = 0.9090909...
    assert feats["typing_rate"] == pytest.approx(4.0 / 4.4)
    assert feats["event_count"] == 4.0


def test_non_monotonic_input_ordering():
    """Test that inter-tap calculation sorts a copied view by press_time_ms without failing."""
    unordered_df = pd.DataFrame(
        [
            {
                "participant_id": "S01",
                "recording_id": "S01_TEX01",
                "text_id": "TEX01",
                "row_number": 1,
                "press_time_ms": 3000,
                "release_time_ms": 3100,
                "hold_time_ms": 100,
                "pressure": 0.5,
                "label": "PD",
            },
            {
                "participant_id": "S01",
                "recording_id": "S01_TEX01",
                "text_id": "TEX01",
                "row_number": 2,
                "press_time_ms": 1000,
                "release_time_ms": 1100,
                "hold_time_ms": 100,
                "pressure": 0.5,
                "label": "PD",
            },
            {
                "participant_id": "S01",
                "recording_id": "S01_TEX01",
                "text_id": "TEX01",
                "row_number": 3,
                "press_time_ms": 2000,
                "release_time_ms": 2100,
                "hold_time_ms": 100,
                "pressure": 0.5,
                "label": "PD",
            },
        ]
    )

    inter_tap_sec = calculate_inter_tap_intervals_sec(unordered_df)
    # Sorted press times: 1000, 2000, 3000 -> diffs: 1.0, 1.0 sec
    assert list(inter_tap_sec) == [1.0, 1.0]


def test_invalid_events_filtered_out():
    """Test that invalid events (negative hold, release < press) are filtered from hold stats."""
    mixed_df = pd.DataFrame(
        [
            {
                "participant_id": "S01",
                "recording_id": "S01_TEX01",
                "text_id": "TEX01",
                "row_number": 1,
                "press_time_ms": 1000,
                "release_time_ms": 1100,
                "hold_time_ms": 100,
                "pressure": 0.5,
                "label": "PD",
            },
            {
                "participant_id": "S01",
                "recording_id": "S01_TEX01",
                "text_id": "TEX01",
                "row_number": 2,
                "press_time_ms": 2000,
                "release_time_ms": 1900,  # Invalid: release < press
                "hold_time_ms": -100,     # Invalid: negative hold
                "pressure": 0.5,
                "label": "PD",
            },
        ]
    )

    hold_sec = get_valid_hold_times_sec(mixed_df)
    assert len(hold_sec) == 1
    assert list(hold_sec) == [0.1]


def test_insufficient_observations_returns_nan():
    """Test that insufficient observations return NaN for higher-order statistics."""
    single_event_df = pd.DataFrame(
        [
            {
                "participant_id": "S01",
                "recording_id": "S01_TEX01",
                "text_id": "TEX01",
                "row_number": 1,
                "press_time_ms": 1000,
                "release_time_ms": 1100,
                "hold_time_ms": 100,
                "pressure": 0.5,
                "label": "PD",
            }
        ]
    )

    feats = extract_touchscreen_recording_features(single_event_df)

    assert feats["hold_mean"] == pytest.approx(0.1)
    assert np.isnan(feats["hold_std"])
    assert np.isnan(feats["hold_skewness"])
    assert np.isnan(feats["hold_kurtosis"])

    # Single event has no inter-tap intervals
    assert np.isnan(feats["inter_tap_mean"])
    assert np.isnan(feats["inter_tap_min"])

    assert feats["event_count"] == 1.0
    assert feats["recording_duration"] == pytest.approx(0.1)
    assert feats["typing_rate"] == pytest.approx(1.0 / 0.1)


def test_zero_duration_recording_returns_nan_typing_rate():
    """Test that a zero duration recording returns NaN typing rate instead of infinity."""
    zero_duration_df = pd.DataFrame(
        [
            {
                "participant_id": "S01",
                "recording_id": "S01_TEX01",
                "text_id": "TEX01",
                "row_number": 1,
                "press_time_ms": 1000,
                "release_time_ms": 1000,  # duration = 0
                "hold_time_ms": 0,
                "pressure": 0.5,
                "label": "PD",
            }
        ]
    )

    feats = extract_touchscreen_recording_features(zero_duration_df)
    assert feats["recording_duration"] == 0.0
    assert np.isnan(feats["typing_rate"])


def test_input_dataframe_immutability(sample_recording_df: pd.DataFrame):
    """Test that input DataFrame is not mutated during feature extraction."""
    original_copy = sample_recording_df.copy(deep=True)
    _ = extract_touchscreen_recording_features(sample_recording_df)
    pd.testing.assert_frame_equal(sample_recording_df, original_copy)


def test_real_dataset_integration():
    """Integration test checking full dataset feature extraction."""
    if not DEFAULT_TOUCHSCREEN_PACKAGE_ROOT.exists():
        pytest.skip("Touchscreen raw dataset package root not available.")

    events_df, recs_df, malformed_df = load_all_touchscreen_recordings()
    X, meta, y = extract_all_touchscreen_features(events_df)

    assert X.shape == (338, 27)
    assert meta.shape == (338, 6)
    assert len(y) == 338
    assert meta["participant_id"].nunique() == 33
    assert set(y.unique()) == {"PD", "CNT"}
    assert X.isna().sum().sum() == 0
