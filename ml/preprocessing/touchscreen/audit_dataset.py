"""Reusable structural audit script for the touchscreen dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from ml.preprocessing.touchscreen.data_loader import (
        DEFAULT_TOUCHSCREEN_PACKAGE_ROOT,
        load_all_touchscreen_recordings,
    )
    from ml.preprocessing.touchscreen.validate_dataset import validate_touchscreen_data
else:
    from .data_loader import DEFAULT_TOUCHSCREEN_PACKAGE_ROOT, load_all_touchscreen_recordings
    from .validate_dataset import validate_touchscreen_data


def run_audit(package_root: Path | None = None, print_report: bool = True) -> dict:
    """Perform a structural audit of the touchscreen dataset and return metrics dictionary."""
    root = package_root or DEFAULT_TOUCHSCREEN_PACKAGE_ROOT

    events_df, recordings_df, malformed_df = load_all_touchscreen_recordings(root)
    validation_report = validate_touchscreen_data(events_df, recordings_df, malformed_df)

    subject_count = len(recordings_df["participant_id"].unique())
    recording_count = len(recordings_df)
    total_physical_lines = int(recordings_df["total_physical_lines"].sum())
    blank_lines = int(recordings_df["blank_lines"].sum())
    valid_event_count = len(events_df)
    malformed_non_blank_lines = len(malformed_df)

    # Event metrics per recording
    events_per_rec = recordings_df["valid_events"]
    rec_min = int(events_per_rec.min())
    rec_max = int(events_per_rec.max())
    rec_mean = float(events_per_rec.mean())
    rec_median = float(events_per_rec.median())

    # Physical lines per recording
    phys_lines = recordings_df["total_physical_lines"]
    phys_mean = float(phys_lines.mean())

    # Recordings per subject
    recs_per_subj = recordings_df.groupby("participant_id").size()
    subj_recs_min = int(recs_per_subj.min())
    subj_recs_max = int(recs_per_subj.max())
    subj_recs_mean = float(recs_per_subj.mean())
    subj_recs_median = float(recs_per_subj.median())

    # Recordings per label
    recs_by_label = recordings_df.groupby("label").size().to_dict()
    pd_recording_count = int(recs_by_label.get("PD", 0))
    cnt_recording_count = int(recs_by_label.get("CNT", 0))

    # Subjects per label
    subjs_by_label = recordings_df.groupby("label")["participant_id"].nunique().to_dict()
    pd_subject_count = int(subjs_by_label.get("PD", 0))
    cnt_subject_count = int(subjs_by_label.get("CNT", 0))

    # Issues breakdown
    issues_df = validation_report.issues
    negative_hold_events = int((issues_df["issue_type"] == "negative_hold_duration").sum()) if not issues_df.empty else 0
    pressure_out_of_range_events = int((issues_df["issue_type"] == "pressure_out_of_range").sum()) if not issues_df.empty else 0
    non_monotonic_press_timestamps = int((issues_df["issue_type"] == "non_monotonic_press_timestamp").sum()) if not issues_df.empty else 0

    # Format issue examples
    issue_examples: list[dict] = []
    if not issues_df.empty:
        for issue_type, group in issues_df.groupby("issue_type"):
            for _, row in group.iterrows():
                issue_examples.append(
                    {
                        "issue_type": issue_type,
                        "recording_id": row["recording_id"],
                        "raw_file": row["raw_file"],
                        "row_number": int(row["row_number"]),
                        "press_time_ms": row["press_time_ms"],
                        "release_time_ms": row["release_time_ms"],
                        "hold_time_ms": row["hold_time_ms"],
                        "pressure": row["pressure"],
                    }
                )

    metrics = {
        "subject_count": subject_count,
        "recording_count": recording_count,
        "total_physical_lines": total_physical_lines,
        "blank_lines": blank_lines,
        "valid_event_count": valid_event_count,
        "malformed_non_blank_lines": malformed_non_blank_lines,
        "negative_hold_events": negative_hold_events,
        "pressure_out_of_range_events": pressure_out_of_range_events,
        "non_monotonic_press_timestamps": non_monotonic_press_timestamps,
        "mean_physical_lines_per_file": phys_mean,
        "events_per_recording": {
            "min": rec_min,
            "max": rec_max,
            "mean": rec_mean,
            "median": rec_median,
        },
        "recordings_per_subject": {
            "min": subj_recs_min,
            "max": subj_recs_max,
            "mean": subj_recs_mean,
            "median": subj_recs_median,
        },
        "recordings_per_label": {
            "PD": pd_recording_count,
            "CNT": cnt_recording_count,
        },
        "subjects_per_label": {
            "PD": pd_subject_count,
            "CNT": cnt_subject_count,
        },
        "issue_examples": issue_examples,
    }

    if print_report:
        _print_audit_report(metrics)

    return metrics


def _print_audit_report(m: dict) -> None:
    """Print clean formatted audit results to console."""
    print("=" * 65)
    print("NEUROTRAP TOUCHSCREEN DATASET AUDIT REPORT")
    print("=" * 65)
    print(f"Subjects Total           : {m['subject_count']} (PD: {m['subjects_per_label']['PD']}, CNT: {m['subjects_per_label']['CNT']})")
    print(f"Recordings Total         : {m['recording_count']} (PD: {m['recordings_per_label']['PD']}, CNT: {m['recordings_per_label']['CNT']})")
    print(f"Total Physical Lines     : {m['total_physical_lines']}")
    print(f"Blank Lines              : {m['blank_lines']}")
    print(f"Valid Event Records      : {m['valid_event_count']}")
    print(f"Malformed Non-Blank Lines: {m['malformed_non_blank_lines']}")
    print(f"Mean Lines per File      : {m['mean_physical_lines_per_file']:.2f}")
    print("-" * 65)
    print("EVENTS PER RECORDING STATS:")
    print(f"  Min   : {m['events_per_recording']['min']}")
    print(f"  Max   : {m['events_per_recording']['max']}")
    print(f"  Mean  : {m['events_per_recording']['mean']:.2f}")
    print(f"  Median: {m['events_per_recording']['median']:.2f}")
    print("-" * 65)
    print("RECORDINGS PER SUBJECT STATS:")
    print(f"  Min   : {m['recordings_per_subject']['min']}")
    print(f"  Max   : {m['recordings_per_subject']['max']}")
    print(f"  Mean  : {m['recordings_per_subject']['mean']:.2f}")
    print(f"  Median: {m['recordings_per_subject']['median']:.2f}")
    print("-" * 65)
    print("DATA QUALITY ANOMALIES REPORTED:")
    print(f"  Negative Hold Duration Events      : {m['negative_hold_events']}")
    print(f"  Pressure Out of Range [0,1] Events : {m['pressure_out_of_range_events']}")
    print(f"  Non-Monotonic Press Timestamps     : {m['non_monotonic_press_timestamps']}")
    print("-" * 65)

    if m["issue_examples"]:
        print("EXAMPLE ANOMALY RECORDS:")
        for idx, ex in enumerate(m["issue_examples"], start=1):
            print(
                f"  [{idx}] Type: {ex['issue_type']} | File: {ex['raw_file']} | "
                f"Line: {ex['row_number']} | Press: {ex['press_time_ms']} | "
                f"Release: {ex['release_time_ms']} | Pressure: {ex['pressure']}"
            )
        print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the NeuroTrap Touchscreen Dataset.")
    parser.add_argument("--root", type=str, help="Path to raw touchscreen dataset package root.", default=None)
    args = parser.parse_args()

    root_path = Path(args.root) if args.root else None
    run_audit(package_root=root_path, print_report=True)


if __name__ == "__main__":
    main()
