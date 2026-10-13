"""Recording-level feature extraction for NeuroTap.

This first feature version operates on the normalized DataFrame produced by
ml.preprocessing.data_loader. It preserves raw inputs and applies
feature-specific validity rules on copied views.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1] / "preprocessing"))
    from data_loader import load_all_datasets  # type: ignore
else:
    from ml.preprocessing.data_loader import load_all_datasets


METADATA_COLUMNS = [
    "dataset",
    "participant_id",
    "recording_id",
    "session_id",
    "repetition_id",
    "experiment_id",
    "raw_file",
    "gt",
    "label",
]
HOLD_FEATURES = [
    "hold_mean",
    "hold_std",
    "hold_variance",
    "hold_median",
    "hold_min",
    "hold_max",
    "hold_skewness",
    "hold_kurtosis",
]
FLIGHT_FEATURES = [
    "flight_mean",
    "flight_std",
    "flight_variance",
    "flight_median",
    "flight_min",
    "flight_max",
    "flight_skewness",
    "flight_kurtosis",
]
INTER_KEY_FEATURES = [
    "inter_key_mean",
    "inter_key_std",
    "inter_key_variance",
    "inter_key_median",
    "inter_key_min",
    "inter_key_max",
    "inter_key_skewness",
    "inter_key_kurtosis",
]
ACTIVITY_FEATURES = ["key_count", "recording_duration", "typing_rate"]
FEATURE_COLUMNS = HOLD_FEATURES + FLIGHT_FEATURES + INTER_KEY_FEATURES + ACTIVITY_FEATURES
OUTPUT_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS


def _validate_required_columns(data: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = set(columns) - set(data.columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {sorted(missing)}")


def hold_valid_view(recording: pd.DataFrame) -> pd.DataFrame:
    """Return rows valid for hold-time features without mutating input."""
    _validate_required_columns(recording, ["hold_time", "release_time", "press_time"])
    mask = (
        recording["hold_time"].notna()
        & (recording["hold_time"] >= 0)
        & ~(
            recording["release_time"].notna()
            & recording["press_time"].notna()
            & (recording["release_time"] < recording["press_time"])
        )
    )
    return recording.loc[mask].copy()


def chronological_valid_view(recording: pd.DataFrame) -> pd.DataFrame:
    """Return rows valid for chronological features, sorted by press_time."""
    _validate_required_columns(recording, ["press_time", "release_time"])
    mask = (
        recording["press_time"].notna()
        & recording["release_time"].notna()
        & (recording["press_time"] > 0)
        & (recording["release_time"] > 0)
        & (recording["release_time"] >= recording["press_time"])
    )
    return (
        recording.loc[mask]
        .copy()
        .sort_values(["press_time", "release_time", "row_number"], kind="mergesort")
    )


def press_time_valid_view(recording: pd.DataFrame) -> pd.DataFrame:
    """Return rows valid for press-to-press timing, sorted by press_time."""
    _validate_required_columns(recording, ["press_time"])
    mask = recording["press_time"].notna() & (recording["press_time"] > 0)
    return recording.loc[mask].copy().sort_values(["press_time", "row_number"], kind="mergesort")


def _numeric_stats(series: pd.Series, prefix: str) -> dict[str, float]:
    """Return standard first-pass aggregate stats in seconds."""
    if series.empty:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_variance": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
            f"{prefix}_skewness": float("nan"),
            f"{prefix}_kurtosis": float("nan"),
        }

    return {
        f"{prefix}_mean": float(series.mean()),
        f"{prefix}_std": float(series.std()),
        f"{prefix}_variance": float(series.var()),
        f"{prefix}_median": float(series.median()),
        f"{prefix}_min": float(series.min()),
        f"{prefix}_max": float(series.max()),
        f"{prefix}_skewness": float(series.skew()),
        f"{prefix}_kurtosis": float(series.kurt()),
    }


def calculate_flight_times(chronological: pd.DataFrame) -> pd.Series:
    """Calculate press_time[i+1] - release_time[i] from a sorted valid view."""
    _validate_required_columns(chronological, ["press_time", "release_time"])
    if len(chronological) < 2:
        return pd.Series(dtype="float64")

    next_press = chronological["press_time"].shift(-1)
    flight_times = next_press - chronological["release_time"]
    return flight_times.iloc[:-1].reset_index(drop=True)


def calculate_inter_key_intervals(press_time_sorted: pd.DataFrame) -> pd.Series:
    """Calculate positive press_time[i+1] - press_time[i] intervals."""
    _validate_required_columns(press_time_sorted, ["press_time"])
    if len(press_time_sorted) < 2:
        return pd.Series(dtype="float64")

    intervals = press_time_sorted["press_time"].diff().iloc[1:]
    return intervals[intervals > 0].reset_index(drop=True)


def extract_recording_features(recording: pd.DataFrame) -> dict[str, object]:
    """Extract one feature vector from one normalized recording DataFrame."""
    _validate_required_columns(recording, OUTPUT_COLUMNS[:0] + METADATA_COLUMNS + ["row_number", "hold_time", "release_time", "press_time"])
    if recording.empty:
        raise ValueError("Cannot extract features from an empty recording.")

    metadata: dict[str, object] = {}
    for column in METADATA_COLUMNS:
        values = recording[column].drop_duplicates()
        if len(values) != 1:
            raise ValueError(f"Recording has inconsistent metadata column {column!r}.")
        metadata[column] = bool(values.iloc[0]) if column == "gt" else values.iloc[0]

    hold_view = hold_valid_view(recording)
    chronological = chronological_valid_view(recording)
    press_time_sorted = press_time_valid_view(recording)
    flight_times = calculate_flight_times(chronological)
    inter_key_intervals = calculate_inter_key_intervals(press_time_sorted)

    features: dict[str, object] = {}
    features.update(metadata)
    features.update(_numeric_stats(hold_view["hold_time"], "hold"))
    features.update(_numeric_stats(flight_times, "flight"))
    features.update(_numeric_stats(inter_key_intervals, "inter_key"))

    key_count = int(len(recording))
    if chronological.empty:
        recording_duration = float("nan")
    else:
        recording_duration = float(chronological["release_time"].max() - chronological["press_time"].min())

    typing_rate = float("nan")
    if pd.notna(recording_duration) and recording_duration > 0:
        typing_rate = key_count / recording_duration

    features["key_count"] = key_count
    features["recording_duration"] = recording_duration
    features["typing_rate"] = typing_rate
    return features


def extract_features(data: pd.DataFrame) -> pd.DataFrame:
    """Extract one recording-level feature row per recording."""
    required = METADATA_COLUMNS + ["row_number", "hold_time", "release_time", "press_time"]
    _validate_required_columns(data, required)

    if data.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    rows = [
        extract_recording_features(recording)
        for _, recording in data.groupby(["dataset", "recording_id"], sort=False)
    ]
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def summarize_features(features: pd.DataFrame) -> pd.DataFrame:
    """Return a compact dataset-level feature extraction summary."""
    if features.empty:
        return pd.DataFrame(columns=["dataset", "feature_rows", "participants", "recordings"])

    return (
        features.groupby("dataset", as_index=False)
        .agg(
            feature_rows=("recording_id", "size"),
            participants=("participant_id", "nunique"),
            recordings=("recording_id", "nunique"),
        )
        .sort_values("dataset")
    )


def main() -> int:
    data = load_all_datasets()
    features = extract_features(data)
    nan_counts = features[FEATURE_COLUMNS].isna().sum()
    nan_counts = nan_counts[nan_counts > 0]

    print("# NeuroTap Feature Extraction")
    print(f"Input rows: {len(data)}")
    print(f"Feature rows: {len(features)}")
    print(f"Feature columns: {len(features.columns)}")
    print(f"Feature names: {FEATURE_COLUMNS}")
    print("\n## Dataset Summary")
    print(summarize_features(features).to_string(index=False))
    print("\n## NaN Feature Counts")
    if nan_counts.empty:
        print("No NaN feature values detected.")
    else:
        print(nan_counts.to_string())
    print("\n## Representative Feature Rows")
    print(features.head(6).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
