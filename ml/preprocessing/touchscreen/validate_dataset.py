"""Read-only validation module for normalized touchscreen dataset."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


DEFAULT_FEW_EVENTS_THRESHOLD = 10


@dataclass(frozen=True)
class TouchscreenValidationReport:
    issues: pd.DataFrame
    malformed_lines: pd.DataFrame
    recording_issues: pd.DataFrame

    @property
    def summary(self) -> pd.DataFrame:
        """Combine all issue categories into a sorted frequency summary."""
        all_issues: list[pd.Series] = []

        if not self.issues.empty:
            all_issues.append(self.issues["issue_type"])
        if not self.malformed_lines.empty:
            all_issues.append(pd.Series(["malformed_line"] * len(self.malformed_lines)))
        if not self.recording_issues.empty:
            all_issues.append(self.recording_issues["issue_type"])

        if not all_issues:
            return pd.DataFrame(columns=["issue_type", "count"])

        combined = pd.concat(all_issues, ignore_index=True)
        return (
            combined.groupby(combined)
            .size()
            .rename("count")
            .reset_index()
            .rename(columns={"index": "issue_type"})
            .sort_values(["issue_type"])
        )


def _event_issue_rows(data: pd.DataFrame, mask: pd.Series, issue_type: str, details: str = "") -> pd.DataFrame:
    columns = [
        "dataset",
        "participant_id",
        "recording_id",
        "raw_file",
        "row_number",
        "press_time_ms",
        "release_time_ms",
        "hold_time_ms",
        "pressure",
        "label",
    ]
    issues = data.loc[mask, columns].copy()
    issues.insert(0, "issue_type", issue_type)
    issues["details"] = details
    return issues


def validate_touchscreen_data(
    events_df: pd.DataFrame,
    recordings_df: pd.DataFrame,
    malformed_df: pd.DataFrame | None = None,
    min_events_threshold: int = DEFAULT_FEW_EVENTS_THRESHOLD,
) -> TouchscreenValidationReport:
    """Report suspiciously formatted or irregular touchscreen records without modifying input data."""
    event_issue_frames: list[pd.DataFrame] = []

    if not events_df.empty:
        # 1. Missing press timestamp
        missing_press = events_df["press_time_ms"].isna()
        event_issue_frames.append(_event_issue_rows(events_df, missing_press, "missing_press_timestamp"))

        # 2. Missing release timestamp
        missing_release = events_df["release_time_ms"].isna()
        event_issue_frames.append(_event_issue_rows(events_df, missing_release, "missing_release_timestamp"))

        # 3. Missing/non-numeric pressure
        missing_pressure = events_df["pressure"].isna()
        event_issue_frames.append(_event_issue_rows(events_df, missing_pressure, "missing_pressure"))

        # 4. Negative hold duration
        negative_hold = events_df["hold_time_ms"] < 0
        event_issue_frames.append(_event_issue_rows(events_df, negative_hold, "negative_hold_duration"))

        # 5. Release timestamp earlier than press timestamp
        release_before_press = events_df["release_time_ms"] < events_df["press_time_ms"]
        event_issue_frames.append(_event_issue_rows(events_df, release_before_press, "release_before_press"))

        # 6. Pressure out of range [0, 1]
        pressure_out_of_range = (events_df["pressure"] < 0.0) | (events_df["pressure"] > 1.0)
        event_issue_frames.append(_event_issue_rows(events_df, pressure_out_of_range, "pressure_out_of_range"))

        # 7. Non-monotonic press timestamp within each recording
        prev_press = events_df.groupby(["dataset", "recording_id"], sort=False)["press_time_ms"].shift(1)
        non_monotonic_press = events_df["press_time_ms"] < prev_press
        event_issue_frames.append(_event_issue_rows(events_df, non_monotonic_press, "non_monotonic_press_timestamp"))

        # 8. Unsupported label
        unsupported_label = ~events_df["label"].isin(["PD", "CNT"])
        event_issue_frames.append(_event_issue_rows(events_df, unsupported_label, "unsupported_label"))

    issues_df = pd.concat(event_issue_frames, ignore_index=True) if event_issue_frames else pd.DataFrame()

    # Recording-level issues
    recording_issue_frames: list[pd.DataFrame] = []

    if not recordings_df.empty:
        # Empty recordings
        empty_recs = recordings_df["valid_events"] == 0
        if empty_recs.any():
            empty_df = recordings_df[empty_recs].copy()
            empty_df.insert(0, "issue_type", "empty_recording")
            empty_df["details"] = "0 valid events"
            recording_issue_frames.append(empty_df)

        # Few events recordings
        few_events = (recordings_df["valid_events"] > 0) & (recordings_df["valid_events"] < min_events_threshold)
        if few_events.any():
            few_df = recordings_df[few_events].copy()
            few_df.insert(0, "issue_type", "very_few_events")
            few_df["details"] = few_df["valid_events"].apply(lambda count: f"Only {count} valid events (< {min_events_threshold})")
            recording_issue_frames.append(few_df)

    rec_issues_df = pd.concat(recording_issue_frames, ignore_index=True) if recording_issue_frames else pd.DataFrame()
    malformed_lines_df = malformed_df.copy() if malformed_df is not None else pd.DataFrame()

    return TouchscreenValidationReport(
        issues=issues_df,
        malformed_lines=malformed_lines_df,
        recording_issues=rec_issues_df,
    )
