import numpy as np
import pandas as pd
import pytest

from ml.inference.pipeline import (
    extract_features_from_events,
    normalize_new_session,
    predict_from_events,
    synthetic_typing_session,
    validate_new_session,
)
from ml.training.dataset import FEATURE_COLUMNS, ParticipantSplit
from ml.training.model_artifact import load_model_artifact, predict_single_recording, train_artifact_from_split


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
    return ParticipantSplit(
        X_train=pd.concat(
            [make_feature_frame(4, offset=0.0), make_feature_frame(4, offset=10.0)],
            ignore_index=True,
        ),
        X_test=pd.concat(
            [make_feature_frame(2, offset=20.0), make_feature_frame(2, offset=30.0)],
            ignore_index=True,
        ),
        y_train=pd.Series(["CNT"] * 4 + ["PD"] * 4, name="label"),
        y_test=pd.Series(["CNT", "CNT", "PD", "PD"], name="label"),
        train_metadata=pd.DataFrame(
            {
                "participant_id": [f"train_p{idx}" for idx in range(8)],
                "recording_id": [f"train_r{idx}" for idx in range(8)],
            }
        ),
        test_metadata=pd.DataFrame(
            {
                "participant_id": [f"test_p{idx}" for idx in range(4)],
                "recording_id": [f"test_r{idx}" for idx in range(4)],
            }
        ),
    )


def test_synthetic_raw_events_convert_to_27_features() -> None:
    features, validation = extract_features_from_events(synthetic_typing_session())

    assert features.shape == (1, 27)
    assert validation.total_events == 5


def test_feature_names_exactly_match_trained_artifact(tmp_path) -> None:
    artifact = train_artifact_from_split(make_split(), output_dir=tmp_path)
    features, _ = extract_features_from_events(synthetic_typing_session())

    assert list(features.columns) == artifact.feature_names


def test_feature_order_exactly_matches() -> None:
    features, _ = extract_features_from_events(synthetic_typing_session())

    assert list(features.columns) == FEATURE_COLUMNS


def test_saved_scaler_and_model_are_loaded_not_retrained(tmp_path) -> None:
    split = make_split()
    train_artifact_from_split(split, output_dir=tmp_path)
    loaded = load_model_artifact(tmp_path)
    scaler_mean = loaded.scaler.mean_.copy()
    model_coef = loaded.model.coef_.copy()

    predict_from_events(synthetic_typing_session(), artifact=loaded)

    assert loaded.scaler.mean_ == pytest.approx(scaler_mean)
    assert loaded.model.coef_ == pytest.approx(model_coef)


def test_inference_produces_prediction_and_probabilities(tmp_path) -> None:
    artifact = train_artifact_from_split(make_split(), output_dir=tmp_path)

    result = predict_from_events(synthetic_typing_session(), artifact=artifact)

    assert result.predicted_class in {"CNT", "PD"}
    assert set(result.probabilities) == {"CNT", "PD"}
    assert sum(result.probabilities.values()) == pytest.approx(1.0)


def test_inference_does_not_mutate_input_events(tmp_path) -> None:
    artifact = train_artifact_from_split(make_split(), output_dir=tmp_path)
    events = pd.DataFrame(synthetic_typing_session())
    original = events.copy(deep=True)

    predict_from_events(events, artifact=artifact)

    pd.testing.assert_frame_equal(events, original)


def test_invalid_events_are_reported_and_existing_rules_are_used(tmp_path) -> None:
    artifact = train_artifact_from_split(make_split(), output_dir=tmp_path)
    events = synthetic_typing_session()
    events.append({"key": None, "hold_time": -0.5, "press_time": 2.0, "release_time": 1.9})

    result = predict_from_events(events, artifact=artifact)

    assert result.validation.missing_key == 1
    assert result.validation.negative_hold_duration == 1
    assert result.validation.release_before_press == 1
    assert result.validation.valid_hold_events == 5
    assert result.predicted_class in {"CNT", "PD"}


def test_empty_input_is_rejected_cleanly() -> None:
    with pytest.raises(ValueError, match="empty"):
        extract_features_from_events([])


def test_missing_required_fields_are_rejected() -> None:
    events = [{"key": "n", "press_time": 1.0, "release_time": 1.1}]

    with pytest.raises(ValueError, match="missing required fields"):
        extract_features_from_events(events)


def test_invalid_timestamps_are_reported_when_enough_valid_events_remain(tmp_path) -> None:
    artifact = train_artifact_from_split(make_split(), output_dir=tmp_path)
    events = synthetic_typing_session()
    events.append({"key": "x", "hold_time": 0.1, "press_time": 0.0, "release_time": 0.1})

    result = predict_from_events(events, artifact=artifact)

    assert result.validation.non_positive_timestamp == 1
    assert result.validation.valid_chronological_events == 5


def test_insufficient_valid_timing_events_are_rejected() -> None:
    events = [
        {"key": "n", "hold_time": 0.1, "press_time": 0.0, "release_time": 0.1},
        {"key": "e", "hold_time": 0.1, "press_time": -1.0, "release_time": -0.9},
    ]

    with pytest.raises(ValueError, match="insufficient valid timing events"):
        extract_features_from_events(events)


def test_model_receives_exactly_27_features(tmp_path, monkeypatch) -> None:
    artifact = train_artifact_from_split(make_split(), output_dir=tmp_path)
    captured = {}

    def fake_predict_single_recording(feature_vector, loaded_artifact):
        captured["shape"] = feature_vector.shape
        captured["columns"] = list(feature_vector.columns)
        return predict_single_recording(feature_vector, loaded_artifact)

    monkeypatch.setattr("ml.inference.pipeline.predict_single_recording", fake_predict_single_recording)

    predict_from_events(synthetic_typing_session(), artifact=artifact)

    assert captured["shape"] == (1, 27)
    assert captured["columns"] == FEATURE_COLUMNS


def test_inference_result_is_deterministic_for_same_input(tmp_path) -> None:
    artifact = train_artifact_from_split(make_split(), output_dir=tmp_path)

    first = predict_from_events(synthetic_typing_session(), artifact=artifact)
    second = predict_from_events(synthetic_typing_session(), artifact=artifact)

    assert first.predicted_class == second.predicted_class
    assert first.probabilities == second.probabilities
    pd.testing.assert_frame_equal(first.features, second.features)


def test_normalized_session_contains_inference_metadata_only_in_memory() -> None:
    normalized = normalize_new_session(synthetic_typing_session())

    assert normalized["dataset"].iloc[0] == "NEW_SESSION"
    assert normalized["recording_id"].iloc[0] == "inference_recording"
    assert normalized["label"].iloc[0] == "UNKNOWN"
