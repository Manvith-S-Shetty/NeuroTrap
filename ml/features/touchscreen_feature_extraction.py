"""Recording-level feature extraction for touchscreen typing dataset.

Converts each touchscreen recording into a 27-numerical-feature vector without
modifying input data or fabricating key identities.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1] / "preprocessing" / "touchscreen"))
    from data_loader import load_all_touchscreen_recordings  # type: ignore
else:
    from ml.preprocessing.touchscreen.data_loader import load_all_touchscreen_recordings


TOUCHSCREEN_METADATA_COLUMNS = [
    "participant_id",
    "recording_id",
    "text_id",
    "label",
    "raw_file",
    "raw_path",
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

INTER_TAP_FEATURES = [
    "inter_tap_mean",
    "inter_tap_std",
    "inter_tap_variance",
    "inter_tap_median",
    "inter_tap_min",
    "inter_tap_max",
    "inter_tap_skewness",
    "inter_tap_kurtosis",
]

PRESSURE_FEATURES = [
    "pressure_mean",
    "pressure_std",
    "pressure_variance",
    "pressure_median",
    "pressure_min",
    "pressure_max",
    "pressure_skewness",
    "pressure_kurtosis",
]

RECORDING_LEVEL_FEATURES = [
    "event_count",
    "recording_duration",
    "typing_rate",
]

TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS = (
    HOLD_FEATURES + INTER_TAP_FEATURES + PRESSURE_FEATURES + RECORDING_LEVEL_FEATURES
)

ALL_TOUCHSCREEN_OUTPUT_COLUMNS = TOUCHSCREEN_METADATA_COLUMNS + TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS


def _validate_required_columns(data: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = set(columns) - set(data.columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {sorted(missing)}")


def _numeric_stats(series: pd.Series, prefix: str) -> dict[str, float]:
    """Calculate 8 standard aggregate stats for a numerical series.

    Returns NaNs for statistics with insufficient observations.
    - skewness uses pandas Series.skew() (Fisher-Pearson moment coefficient, requires N >= 3)
    - kurtosis uses pandas Series.kurt() (Fisher excess kurtosis, requires N >= 4)
    """
    clean_series = series.dropna()
    clean_series = clean_series[np.isfinite(clean_series)]

    if clean_series.empty:
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
        f"{prefix}_mean": float(clean_series.mean()),
        f"{prefix}_std": float(clean_series.std()) if len(clean_series) >= 2 else float("nan"),
        f"{prefix}_variance": float(clean_series.var()) if len(clean_series) >= 2 else float("nan"),
        f"{prefix}_median": float(clean_series.median()),
        f"{prefix}_min": float(clean_series.min()),
        f"{prefix}_max": float(clean_series.max()),
        f"{prefix}_skewness": float(clean_series.skew()) if len(clean_series) >= 3 else float("nan"),
        f"{prefix}_kurtosis": float(clean_series.kurt()) if len(clean_series) >= 4 else float("nan"),
    }


def get_valid_hold_times_sec(recording: pd.DataFrame) -> pd.Series:
    """Return valid hold durations in seconds without mutating input."""
    _validate_required_columns(recording, ["hold_time_ms", "release_time_ms", "press_time_ms"])

    mask = (
        recording["hold_time_ms"].notna()
        & np.isfinite(recording["hold_time_ms"])
        & (recording["hold_time_ms"] >= 0)
        & (recording["release_time_ms"] >= recording["press_time_ms"])
    )
    return recording.loc[mask, "hold_time_ms"].copy() / 1000.0


def calculate_inter_tap_intervals_sec(recording: pd.DataFrame) -> pd.Series:
    """Calculate positive inter-tap intervals press_time[i+1] - press_time[i] in seconds."""
    _validate_required_columns(recording, ["press_time_ms"])

    mask = recording["press_time_ms"].notna() & np.isfinite(recording["press_time_ms"]) & (recording["press_time_ms"] > 0)
    valid_presses = recording.loc[mask].copy()

    if len(valid_presses) < 2:
        return pd.Series(dtype="float64")

    # Sort a copied view by press_time_ms and row_number
    sort_cols = ["press_time_ms"]
    if "row_number" in valid_presses.columns:
        sort_cols.append("row_number")

    sorted_view = valid_presses.sort_values(sort_cols, kind="mergesort")
    diffs_ms = sorted_view["press_time_ms"].diff().dropna()

    # Retain strictly positive finite intervals
    positive_diffs_ms = diffs_ms[(diffs_ms > 0) & np.isfinite(diffs_ms)]
    return positive_diffs_ms / 1000.0


def get_valid_pressures(recording: pd.DataFrame) -> pd.Series:
    """Return valid numeric pressure values without clipping or mutating input."""
    _validate_required_columns(recording, ["pressure"])

    mask = recording["pressure"].notna() & np.isfinite(recording["pressure"])
    return recording.loc[mask, "pressure"].copy()


def extract_touchscreen_recording_features(recording: pd.DataFrame) -> dict[str, float | str | int]:
    """Extract metadata and 27 numerical features for a single touchscreen recording."""
    _validate_required_columns(
        recording,
        [
            "participant_id",
            "recording_id",
            "text_id",
            "press_time_ms",
            "release_time_ms",
            "hold_time_ms",
            "pressure",
            "label",
        ],
    )

    first_row = recording.iloc[0]
    meta = {
        "participant_id": str(first_row["participant_id"]),
        "recording_id": str(first_row["recording_id"]),
        "text_id": str(first_row["text_id"]),
        "label": str(first_row["label"]),
        "raw_file": str(first_row.get("raw_file", "")),
        "raw_path": str(first_row.get("raw_path", "")),
    }

    # 1. Hold-time features (in seconds)
    hold_sec = get_valid_hold_times_sec(recording)
    hold_stats = _numeric_stats(hold_sec, "hold")

    # 2. Inter-tap features (in seconds)
    inter_tap_sec = calculate_inter_tap_intervals_sec(recording)
    inter_tap_stats = _numeric_stats(inter_tap_sec, "inter_tap")

    # 3. Pressure features
    pressures = get_valid_pressures(recording)
    pressure_stats = _numeric_stats(pressures, "pressure")

    # 4. Recording-level activity features
    # Event count: number of valid chronological events
    valid_chrono_mask = (
        recording["press_time_ms"].notna()
        & recording["release_time_ms"].notna()
        & np.isfinite(recording["press_time_ms"])
        & np.isfinite(recording["release_time_ms"])
        & (recording["press_time_ms"] > 0)
        & (recording["release_time_ms"] >= recording["press_time_ms"])
    )
    valid_events_df = recording.loc[valid_chrono_mask]

    event_count = len(valid_events_df)

    if event_count > 0:
        min_press = valid_events_df["press_time_ms"].min()
        max_release = valid_events_df["release_time_ms"].max()
        duration_ms = max_release - min_press

        if duration_ms > 0 and np.isfinite(duration_ms):
            duration_sec = float(duration_ms / 1000.0)
            typing_rate = float(event_count / duration_sec)
        else:
            duration_sec = 0.0 if duration_ms == 0 else float("nan")
            typing_rate = float("nan")
    else:
        duration_sec = float("nan")
        typing_rate = float("nan")

    rec_stats = {
        "event_count": float(event_count),
        "recording_duration": duration_sec,
        "typing_rate": typing_rate,
    }

    result = {}
    result.update(meta)
    result.update(hold_stats)
    result.update(inter_tap_stats)
    result.update(pressure_stats)
    result.update(rec_stats)

    return result


def extract_all_touchscreen_features(
    events_df: pd.DataFrame,
    recordings_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Extract features for all recordings in events_df without mutating input.

    Returns:
        X: DataFrame of exactly 27 numerical features (N x 27).
        metadata: DataFrame of metadata columns (N x 6).
        y: Series of target labels ('PD' or 'CNT').
    """
    _validate_required_columns(events_df, ["recording_id"])

    # Group events by recording_id maintaining appearance order
    grouped = events_df.groupby("recording_id", sort=False)

    feature_rows: list[dict] = []
    for _, recording_data in grouped:
        if recording_data.empty:
            continue
        feat_dict = extract_touchscreen_recording_features(recording_data)
        feature_rows.append(feat_dict)

    if not feature_rows:
        empty_x = pd.DataFrame(columns=TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS)
        empty_meta = pd.DataFrame(columns=TOUCHSCREEN_METADATA_COLUMNS)
        empty_y = pd.Series(dtype="object", name="label")
        return empty_x, empty_meta, empty_y

    all_features_df = pd.DataFrame(feature_rows)

    metadata_df = all_features_df[TOUCHSCREEN_METADATA_COLUMNS].copy()
    X_df = all_features_df[TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS].copy()
    y_series = all_features_df["label"].copy()

    return X_df, metadata_df, y_series
