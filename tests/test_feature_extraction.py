import math

import pandas as pd
import pytest

from ml.features.feature_extraction import (
    FEATURE_COLUMNS,
    calculate_flight_times,
    calculate_inter_key_intervals,
    chronological_valid_view,
    extract_features,
    extract_recording_features,
    hold_valid_view,
    press_time_valid_view,
)


def make_recording(rows: list[dict[str, object]] | None = None) -> pd.DataFrame:
    base = {
        "dataset": "MIT-TEST",
        "participant_id": "1",
        "recording_id": "rec1",
        "session_id": "001",
        "repetition_id": "001",
        "experiment_id": "014",
        "raw_file": "rec1.csv",
        "gt": False,
        "label": "CNT",
    }
    default_rows = [
        {"row_number": 1, "key": "a", "hold_time": 0.2, "release_time": 1.2, "press_time": 1.0},
        {"row_number": 2, "key": "b", "hold_time": 0.3, "release_time": 1.7, "press_time": 1.4},
        {"row_number": 3, "key": "c", "hold_time": 0.5, "release_time": 2.4, "press_time": 1.9},
    ]
    payload = rows if rows is not None else default_rows
    return pd.DataFrame([{**base, **row} for row in payload])


def test_hold_statistics() -> None:
    features = extract_recording_features(make_recording())

    assert features["hold_mean"] == pytest.approx(1.0 / 3.0)
    assert features["hold_std"] == pytest.approx(0.15275252316519466)
    assert features["hold_variance"] == pytest.approx(0.02333333333333333)
    assert features["hold_median"] == pytest.approx(0.3)
    assert features["hold_min"] == pytest.approx(0.2)
    assert features["hold_max"] == pytest.approx(0.5)


def test_hold_skewness_and_fisher_excess_kurtosis() -> None:
    recording = make_recording(
        [
            {"row_number": 1, "key": "a", "hold_time": 1.0, "release_time": 2.0, "press_time": 1.0},
            {"row_number": 2, "key": "b", "hold_time": 2.0, "release_time": 4.0, "press_time": 2.0},
            {"row_number": 3, "key": "c", "hold_time": 3.0, "release_time": 6.0, "press_time": 3.0},
            {"row_number": 4, "key": "d", "hold_time": 4.0, "release_time": 8.0, "press_time": 4.0},
        ]
    )

    features = extract_recording_features(recording)

    assert features["hold_skewness"] == pytest.approx(0.0)
    assert features["hold_kurtosis"] == pytest.approx(-1.2)


def test_flight_calculation_after_press_time_sorting() -> None:
    recording = make_recording(
        [
            {"row_number": 1, "key": "b", "hold_time": 0.2, "release_time": 2.2, "press_time": 2.0},
            {"row_number": 2, "key": "a", "hold_time": 0.1, "release_time": 1.1, "press_time": 1.0},
            {"row_number": 3, "key": "c", "hold_time": 0.3, "release_time": 3.3, "press_time": 3.0},
        ]
    )

    chronological = chronological_valid_view(recording)
    flight_times = calculate_flight_times(chronological)
    features = extract_recording_features(recording)

    assert list(chronological["key"]) == ["a", "b", "c"]
    assert list(flight_times) == pytest.approx([0.9, 0.8])
    assert features["flight_mean"] == pytest.approx(0.85)
    assert features["flight_min"] == pytest.approx(0.8)
    assert features["flight_max"] == pytest.approx(0.9)


def test_flight_skewness_and_fisher_excess_kurtosis() -> None:
    recording = make_recording(
        [
            {"row_number": 1, "key": "a", "hold_time": 0.1, "release_time": 1.5, "press_time": 1.0},
            {"row_number": 2, "key": "b", "hold_time": 0.1, "release_time": 3.5, "press_time": 2.5},
            {"row_number": 3, "key": "c", "hold_time": 0.1, "release_time": 7.5, "press_time": 5.5},
            {"row_number": 4, "key": "d", "hold_time": 0.1, "release_time": 13.5, "press_time": 10.5},
            {"row_number": 5, "key": "e", "hold_time": 0.1, "release_time": 21.5, "press_time": 17.5},
        ]
    )

    features = extract_recording_features(recording)

    assert features["flight_mean"] == pytest.approx(2.5)
    assert features["flight_skewness"] == pytest.approx(0.0)
    assert features["flight_kurtosis"] == pytest.approx(-1.2)


def test_inter_key_interval_calculation() -> None:
    recording = make_recording(
        [
            {"row_number": 1, "key": "a", "hold_time": 0.1, "release_time": 1.1, "press_time": 1.0},
            {"row_number": 2, "key": "b", "hold_time": 0.1, "release_time": 2.1, "press_time": 2.0},
            {"row_number": 3, "key": "c", "hold_time": 0.1, "release_time": 4.1, "press_time": 4.0},
            {"row_number": 4, "key": "d", "hold_time": 0.1, "release_time": 7.1, "press_time": 7.0},
            {"row_number": 5, "key": "e", "hold_time": 0.1, "release_time": 11.1, "press_time": 11.0},
        ]
    )

    intervals = calculate_inter_key_intervals(press_time_valid_view(recording))
    features = extract_recording_features(recording)

    assert list(intervals) == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert features["inter_key_mean"] == pytest.approx(2.5)
    assert features["inter_key_std"] == pytest.approx(1.2909944487358056)
    assert features["inter_key_variance"] == pytest.approx(1.6666666666666667)
    assert features["inter_key_median"] == pytest.approx(2.5)
    assert features["inter_key_min"] == pytest.approx(1.0)
    assert features["inter_key_max"] == pytest.approx(4.0)
    assert features["inter_key_skewness"] == pytest.approx(0.0)
    assert features["inter_key_kurtosis"] == pytest.approx(-1.2)


def test_inter_key_intervals_after_unsorted_input() -> None:
    recording = make_recording(
        [
            {"row_number": 1, "key": "c", "hold_time": 0.1, "release_time": 4.1, "press_time": 4.0},
            {"row_number": 2, "key": "a", "hold_time": 0.1, "release_time": 1.1, "press_time": 1.0},
            {"row_number": 3, "key": "b", "hold_time": 0.1, "release_time": 2.1, "press_time": 2.0},
        ]
    )

    sorted_view = press_time_valid_view(recording)
    intervals = calculate_inter_key_intervals(sorted_view)

    assert list(sorted_view["key"]) == ["a", "b", "c"]
    assert list(intervals) == pytest.approx([1.0, 2.0])


def test_invalid_negative_hold_excluded_from_hold_features() -> None:
    recording = make_recording(
        [
            {"row_number": 1, "key": "a", "hold_time": 0.2, "release_time": 1.2, "press_time": 1.0},
            {"row_number": 2, "key": "b", "hold_time": -0.1, "release_time": 1.3, "press_time": 1.4},
        ]
    )

    hold_view = hold_valid_view(recording)
    features = extract_recording_features(recording)

    assert list(hold_view["key"]) == ["a"]
    assert features["hold_mean"] == pytest.approx(0.2)
    assert math.isnan(features["hold_std"])


def test_non_positive_press_time_excluded_from_flight_calculations() -> None:
    recording = make_recording(
        [
            {"row_number": 1, "key": "bad", "hold_time": 11.0, "release_time": 1.0, "press_time": -10.0},
            {"row_number": 2, "key": "a", "hold_time": 0.2, "release_time": 2.2, "press_time": 2.0},
            {"row_number": 3, "key": "b", "hold_time": 0.3, "release_time": 3.3, "press_time": 3.0},
        ]
    )

    chronological = chronological_valid_view(recording)
    features = extract_recording_features(recording)

    assert list(chronological["key"]) == ["a", "b"]
    assert features["flight_mean"] == pytest.approx(0.8)
    assert features["recording_duration"] == pytest.approx(1.3)
    assert features["inter_key_mean"] == pytest.approx(1.0)


def test_invalid_non_positive_timestamps_excluded_from_inter_key_features() -> None:
    recording = make_recording(
        [
            {"row_number": 1, "key": "bad", "hold_time": 0.1, "release_time": 0.1, "press_time": 0.0},
            {"row_number": 2, "key": "a", "hold_time": 0.1, "release_time": 1.1, "press_time": 1.0},
            {"row_number": 3, "key": "b", "hold_time": 0.1, "release_time": 3.1, "press_time": 3.0},
        ]
    )

    press_view = press_time_valid_view(recording)
    features = extract_recording_features(recording)

    assert list(press_view["key"]) == ["a", "b"]
    assert features["inter_key_mean"] == pytest.approx(2.0)


def test_missing_key_does_not_break_timing_features() -> None:
    recording = make_recording(
        [
            {"row_number": 1, "key": None, "hold_time": 0.2, "release_time": 1.2, "press_time": 1.0},
            {"row_number": 2, "key": "b", "hold_time": 0.3, "release_time": 2.3, "press_time": 2.0},
        ]
    )

    features = extract_recording_features(recording)

    assert features["key_count"] == 2
    assert features["hold_mean"] == pytest.approx(0.25)
    assert features["flight_mean"] == pytest.approx(0.8)
    assert features["inter_key_mean"] == pytest.approx(1.0)


def test_long_positive_timestamp_holds_are_retained() -> None:
    recording = make_recording(
        [
            {"row_number": 1, "key": "a", "hold_time": 6.0, "release_time": 7.0, "press_time": 1.0},
            {"row_number": 2, "key": "b", "hold_time": 0.2, "release_time": 8.2, "press_time": 8.0},
        ]
    )

    features = extract_recording_features(recording)

    assert features["hold_max"] == pytest.approx(6.0)
    assert features["hold_mean"] == pytest.approx(3.1)


def test_zero_duration_recording_does_not_cause_division_error() -> None:
    recording = make_recording(
        [
            {"row_number": 1, "key": "a", "hold_time": 0.0, "release_time": 1.0, "press_time": 1.0},
        ]
    )

    features = extract_recording_features(recording)

    assert features["recording_duration"] == pytest.approx(0.0)
    assert math.isnan(features["typing_rate"])
    assert math.isnan(features["flight_mean"])
    assert math.isnan(features["inter_key_mean"])


def test_insufficient_observations_return_nan_for_shape_and_interval_stats() -> None:
    recording = make_recording(
        [
            {"row_number": 1, "key": "a", "hold_time": 0.2, "release_time": 1.2, "press_time": 1.0},
        ]
    )

    features = extract_recording_features(recording)

    assert math.isnan(features["hold_std"])
    assert math.isnan(features["hold_variance"])
    assert math.isnan(features["hold_skewness"])
    assert math.isnan(features["hold_kurtosis"])
    assert math.isnan(features["flight_skewness"])
    assert math.isnan(features["flight_kurtosis"])
    assert math.isnan(features["inter_key_skewness"])
    assert math.isnan(features["inter_key_kurtosis"])


def test_metadata_is_preserved() -> None:
    features = extract_recording_features(make_recording())

    assert features["dataset"] == "MIT-TEST"
    assert features["participant_id"] == "1"
    assert features["recording_id"] == "rec1"
    assert features["session_id"] == "001"
    assert features["repetition_id"] == "001"
    assert features["experiment_id"] == "014"
    assert features["raw_file"] == "rec1.csv"
    assert features["gt"] is False
    assert features["label"] == "CNT"


def test_raw_input_is_not_mutated() -> None:
    recording = make_recording()
    original = recording.copy(deep=True)

    extract_recording_features(recording)
    extract_features(recording)

    pd.testing.assert_frame_equal(recording, original)


def test_extract_features_returns_one_row_per_recording() -> None:
    rec1 = make_recording()
    rec2 = make_recording().assign(
        participant_id="2",
        recording_id="rec2",
        raw_file="rec2.csv",
        label="PD",
        gt=True,
    )

    features = extract_features(pd.concat([rec1, rec2], ignore_index=True))

    assert len(features) == 2
    assert list(features["recording_id"]) == ["rec1", "rec2"]
    for column in FEATURE_COLUMNS:
        assert column in features.columns
