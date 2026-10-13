"""Realistic raw-session inference pipeline for NeuroTap.

This module transforms a new raw typing session into the existing 27 recording
features and runs inference with the saved model artifact. It never trains or
refits preprocessing.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from ml.features.feature_extraction import (  # type: ignore
        FEATURE_COLUMNS,
        chronological_valid_view,
        extract_recording_features,
        hold_valid_view,
        press_time_valid_view,
    )
    from ml.training.model_artifact import (  # type: ignore
        DEFAULT_ARTIFACT_DIR,
        ModelArtifact,
        load_model_artifact,
        predict_single_recording,
    )
else:
    from ml.features.feature_extraction import (
        FEATURE_COLUMNS,
        chronological_valid_view,
        extract_recording_features,
        hold_valid_view,
        press_time_valid_view,
    )
    from ml.training.model_artifact import (
        DEFAULT_ARTIFACT_DIR,
        ModelArtifact,
        load_model_artifact,
        predict_single_recording,
    )


REQUIRED_EVENT_FIELDS = ["key", "hold_time", "press_time", "release_time"]
INFERENCE_METADATA = {
    "dataset": "NEW_SESSION",
    "participant_id": "inference_participant",
    "recording_id": "inference_recording",
    "session_id": "inference_session",
    "repetition_id": "inference_repetition",
    "experiment_id": "inference",
    "raw_file": "in_memory_session",
    "gt": False,
    "label": "UNKNOWN",
}


@dataclass(frozen=True)
class SessionValidationReport:
    total_events: int
    missing_key: int
    missing_timing_values: int
    non_positive_timestamp: int
    negative_hold_duration: int
    release_before_press: int
    valid_hold_events: int
    valid_chronological_events: int
    valid_press_events: int
    enough_flight_events: bool
    enough_inter_key_events: bool


@dataclass(frozen=True)
class InferenceResult:
    predicted_class: str
    probabilities: dict[str, float]
    features: pd.DataFrame
    validation: SessionValidationReport


def synthetic_typing_session() -> list[dict[str, float | str]]:
    """Return a deterministic typing session that does not use training/test data."""
    return [
        {"key": "n", "hold_time": 0.09, "press_time": 1.00, "release_time": 1.09},
        {"key": "e", "hold_time": 0.08, "press_time": 1.18, "release_time": 1.26},
        {"key": "u", "hold_time": 0.10, "press_time": 1.34, "release_time": 1.44},
        {"key": "r", "hold_time": 0.07, "press_time": 1.56, "release_time": 1.63},
        {"key": "o", "hold_time": 0.11, "press_time": 1.74, "release_time": 1.85},
    ]


def _as_event_frame(events: Iterable[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(events, pd.DataFrame):
        frame = events.copy(deep=True)
    else:
        frame = pd.DataFrame(list(events))

    if frame.empty:
        raise ValueError("Inference session is empty.")

    missing = sorted(set(REQUIRED_EVENT_FIELDS) - set(frame.columns))
    if missing:
        raise ValueError(f"Inference events are missing required fields: {missing}")

    return frame[REQUIRED_EVENT_FIELDS].copy()


def normalize_new_session(events: Iterable[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    """Convert raw session events to the normalized shape expected by feature extraction."""
    frame = _as_event_frame(events)
    normalized = frame.copy(deep=True)
    normalized["row_number"] = range(1, len(normalized) + 1)

    for column in ["hold_time", "press_time", "release_time"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    for column, value in INFERENCE_METADATA.items():
        normalized[column] = value

    return normalized


def validate_new_session(normalized: pd.DataFrame) -> SessionValidationReport:
    """Report inference-session issues without changing event records."""
    missing_key = int(normalized["key"].isna().sum())
    missing_timing_values = int(normalized[["hold_time", "press_time", "release_time"]].isna().any(axis=1).sum())
    non_positive_timestamp = int(
        (
            normalized["press_time"].notna()
            & normalized["release_time"].notna()
            & ((normalized["press_time"] <= 0) | (normalized["release_time"] <= 0))
        ).sum()
    )
    negative_hold_duration = int((normalized["hold_time"].notna() & (normalized["hold_time"] < 0)).sum())
    release_before_press = int(
        (
            normalized["press_time"].notna()
            & normalized["release_time"].notna()
            & (normalized["release_time"] < normalized["press_time"])
        ).sum()
    )

    hold_view = hold_valid_view(normalized)
    chronological_view = chronological_valid_view(normalized)
    press_view = press_time_valid_view(normalized)
    report = SessionValidationReport(
        total_events=len(normalized),
        missing_key=missing_key,
        missing_timing_values=missing_timing_values,
        non_positive_timestamp=non_positive_timestamp,
        negative_hold_duration=negative_hold_duration,
        release_before_press=release_before_press,
        valid_hold_events=len(hold_view),
        valid_chronological_events=len(chronological_view),
        valid_press_events=len(press_view),
        enough_flight_events=len(chronological_view) >= 2,
        enough_inter_key_events=len(press_view) >= 2,
    )
    if not report.enough_flight_events:
        raise ValueError("Inference session has insufficient valid timing events for flight-time features.")
    if not report.enough_inter_key_events:
        raise ValueError("Inference session has insufficient valid press-time events for inter-key features.")
    return report


def extract_features_from_events(events: Iterable[dict[str, Any]] | pd.DataFrame) -> tuple[pd.DataFrame, SessionValidationReport]:
    """Extract exactly the trained 27-feature vector from one new raw session."""
    normalized = normalize_new_session(events)
    validation = validate_new_session(normalized)
    feature_row = extract_recording_features(normalized)
    features = pd.DataFrame([{column: feature_row[column] for column in FEATURE_COLUMNS}], columns=FEATURE_COLUMNS)
    if list(features.columns) != FEATURE_COLUMNS:
        raise ValueError("Extracted feature order does not match the trained feature order.")
    return features, validation


def predict_from_events(
    events: Iterable[dict[str, Any]] | pd.DataFrame,
    artifact: ModelArtifact | None = None,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
) -> InferenceResult:
    """Run raw events -> features -> saved scaler/model -> prediction."""
    loaded_artifact = load_model_artifact(artifact_dir) if artifact is None else artifact
    features, validation = extract_features_from_events(events)
    prediction = predict_single_recording(features, loaded_artifact)
    return InferenceResult(
        predicted_class=str(prediction["predicted_class"]),
        probabilities=dict(prediction.get("probabilities", {})),
        features=features,
        validation=validation,
    )


def main() -> int:
    result = predict_from_events(synthetic_typing_session())
    print("# NeuroTap Synthetic Inference Example")
    print(f"Extracted feature count: {len(result.features.columns)}")
    print(f"Feature order matches trained artifact: {list(result.features.columns) == FEATURE_COLUMNS}")
    print(f"Total events: {result.validation.total_events}")
    print(f"Valid hold events: {result.validation.valid_hold_events}")
    print(f"Valid chronological events: {result.validation.valid_chronological_events}")
    print(f"Valid press events: {result.validation.valid_press_events}")
    print(f"Predicted class: {result.predicted_class}")
    print(f"Probabilities: {result.probabilities}")
    print("\n## Extracted Features")
    print(result.features.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
