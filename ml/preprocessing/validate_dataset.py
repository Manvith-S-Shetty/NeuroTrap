"""Read-only validation for the normalized NeuroTap dataset."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from data_loader import TIMING_COLUMNS, load_all_datasets  # type: ignore
else:
    from .data_loader import TIMING_COLUMNS, load_all_datasets


DEFAULT_LONG_HOLD_THRESHOLD_SECONDS = 5.0


@dataclass(frozen=True)
class ValidationReport:
    issues: pd.DataFrame

    @property
    def summary(self) -> pd.DataFrame:
        if self.issues.empty:
            return pd.DataFrame(columns=["issue_type", "count"])
        return (
            self.issues.groupby("issue_type")
            .size()
            .rename("count")
            .reset_index()
            .sort_values(["issue_type"])
        )


def _issue_rows(data: pd.DataFrame, mask: pd.Series, issue_type: str) -> pd.DataFrame:
    columns = [
        "dataset",
        "participant_id",
        "recording_id",
        "session_id",
        "raw_file",
        "row_number",
        "key",
        "hold_time",
        "press_time",
        "release_time",
        "gt",
        "label",
    ]
    issues = data.loc[mask, columns].copy()
    issues.insert(0, "issue_type", issue_type)
    return issues


def validate_normalized_data(
    data: pd.DataFrame,
    long_hold_threshold_seconds: float = DEFAULT_LONG_HOLD_THRESHOLD_SECONDS,
) -> ValidationReport:
    """Report suspicious records without dropping or modifying the input data."""
    issue_frames: list[pd.DataFrame] = []

    missing_key = data["key"].isna() | (data["key"].astype("string").str.len() == 0)
    issue_frames.append(_issue_rows(data, missing_key, "missing_key"))

    missing_timing = data[TIMING_COLUMNS].isna().any(axis=1)
    issue_frames.append(_issue_rows(data, missing_timing, "missing_timing"))

    negative_hold = data["hold_time"] < 0
    issue_frames.append(_issue_rows(data, negative_hold, "negative_hold_duration"))

    release_before_press = data["release_time"] < data["press_time"]
    issue_frames.append(_issue_rows(data, release_before_press, "release_before_press"))

    non_positive_timestamps = (data["press_time"] <= 0) | (data["release_time"] <= 0)
    issue_frames.append(_issue_rows(data, non_positive_timestamps, "non_positive_timestamp"))

    suspicious_long_hold = data["hold_time"] >= long_hold_threshold_seconds
    issue_frames.append(_issue_rows(data, suspicious_long_hold, "suspicious_long_hold"))

    previous_press = data.groupby(["dataset", "recording_id"], sort=False)["press_time"].shift(1)
    non_monotonic_press = data["press_time"] < previous_press
    issue_frames.append(_issue_rows(data, non_monotonic_press, "non_monotonic_press_time"))

    duplicate_columns = [
        "dataset",
        "participant_id",
        "recording_id",
        "key",
        "hold_time",
        "press_time",
        "release_time",
    ]
    duplicate_rows = data.duplicated(subset=duplicate_columns, keep=False)
    issue_frames.append(_issue_rows(data, duplicate_rows, "duplicate_row"))

    issues = pd.concat(issue_frames, ignore_index=True)
    if issues.empty:
        return ValidationReport(issues=pd.DataFrame(columns=["issue_type"]))
    return ValidationReport(issues=issues)


def main() -> int:
    data = load_all_datasets()
    report = validate_normalized_data(data)

    print("# NeuroTap Dataset Validation")
    print(f"Rows validated: {len(data)}")
    print(f"Suspicious issue flags: {len(report.issues)}")
    print("\n## Issue Summary")
    if report.summary.empty:
        print("No validation issues detected.")
    else:
        print(report.summary.to_string(index=False))

    print("\n## Dataset-Level Issue Summary")
    if report.issues.empty:
        print("No validation issues detected.")
    else:
        dataset_summary = (
            report.issues.groupby(["dataset", "issue_type"])
            .size()
            .rename("count")
            .reset_index()
            .sort_values(["dataset", "issue_type"])
        )
        print(dataset_summary.to_string(index=False))

    print("\n## Representative Issue Rows")
    if report.issues.empty:
        print("No validation issues detected.")
    else:
        print(report.issues.head(12).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
