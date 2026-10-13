"""Unit tests for the touchscreen dataset loader and validation modules."""

from pathlib import Path

import pandas as pd
import pytest

from ml.preprocessing.touchscreen.data_loader import (
    DEFAULT_TOUCHSCREEN_PACKAGE_ROOT,
    DatasetMappingError,
    TouchscreenRecordingMetadata,
    load_all_touchscreen_recordings,
    load_demographics_metadata,
    parse_touchscreen_file,
    parse_touchscreen_filename,
)
from ml.preprocessing.touchscreen.validate_dataset import (
    TouchscreenValidationReport,
    validate_touchscreen_data,
)


def test_parse_touchscreen_filename_variants():
    """Test filename parsing supporting TEX01, TEX1, and TEX010."""
    res1 = parse_touchscreen_filename("S01_TEX01.txt")
    assert res1["participant_id"] == "S01"
    assert res1["text_id"] == "TEX01"
    assert res1["recording_id"] == "S01_TEX01"

    res2 = parse_touchscreen_filename("S06_TEX1.txt")
    assert res2["participant_id"] == "S06"
    assert res2["text_id"] == "TEX1"
    assert res2["recording_id"] == "S06_TEX1"

    res3 = parse_touchscreen_filename("S05_TEX010.txt")
    assert res3["participant_id"] == "S05"
    assert res3["text_id"] == "TEX010"
    assert res3["recording_id"] == "S05_TEX010"


def test_parse_touchscreen_filename_invalid():
    """Test that invalid filename patterns raise DatasetMappingError."""
    with pytest.raises(DatasetMappingError):
        parse_touchscreen_filename("invalid_filename.txt")

    with pytest.raises(DatasetMappingError):
        parse_touchscreen_filename("S01.txt")


def test_parse_raw_events_valid_and_blank(tmp_path: Path):
    """Test raw event parsing, hold-time calculation, pressure parsing, and blank line handling."""
    content = (
        "Press,1000,Release 1090,0.575\n"
        "\n"
        "  \n"
        "Press,2000,Release 2150,0.800\n"
    )
    raw_file = tmp_path / "S01_TEX01.txt"
    raw_file.write_text(content, encoding="utf-8")

    meta = TouchscreenRecordingMetadata(
        dataset="i-PROGNOSIS",
        participant_id="S01",
        recording_id="S01_TEX01",
        text_id="TEX01",
        raw_file="S01_TEX01.txt",
        raw_path=str(raw_file),
        label="PD",
    )

    events, malformed, stats = parse_touchscreen_file(raw_file, meta)

    assert stats["total_physical_lines"] == 4
    assert stats["blank_lines"] == 2
    assert stats["malformed_lines"] == 0
    assert stats["valid_events"] == 2
    assert len(malformed) == 0

    assert events[0]["press_time_ms"] == 1000
    assert events[0]["release_time_ms"] == 1090
    assert events[0]["hold_time_ms"] == 90
    assert events[0]["pressure"] == 0.575

    assert events[1]["press_time_ms"] == 2000
    assert events[1]["release_time_ms"] == 2150
    assert events[1]["hold_time_ms"] == 150
    assert events[1]["pressure"] == 0.8


def test_parse_raw_events_malformed_lines(tmp_path: Path):
    """Test that malformed non-blank lines are captured and recorded rather than discarded silently."""
    content = (
        "Press,1000,Release 1090,0.575\n"
        "This is a bad line with no commas\n"
        "Press,INVALID,Release 2000,0.5\n"
    )
    raw_file = tmp_path / "S01_TEX02.txt"
    raw_file.write_text(content, encoding="utf-8")

    meta = TouchscreenRecordingMetadata(
        dataset="i-PROGNOSIS",
        participant_id="S01",
        recording_id="S01_TEX02",
        text_id="TEX02",
        raw_file="S01_TEX02.txt",
        raw_path=str(raw_file),
        label="PD",
    )

    events, malformed, stats = parse_touchscreen_file(raw_file, meta)

    assert stats["valid_events"] == 1
    assert stats["malformed_lines"] == 2
    assert len(malformed) == 2
    assert malformed[0]["row_number"] == 2
    assert "bad line" in malformed[0]["raw_line"]


def test_validation_negative_hold_and_pressure_out_of_range():
    """Test validation detection of negative hold duration and pressure out of range."""
    events_data = [
        {
            "dataset": "i-PROGNOSIS",
            "participant_id": "S01",
            "recording_id": "S01_TEX01",
            "text_id": "TEX01",
            "raw_file": "S01_TEX01.txt",
            "raw_path": "/path/S01_TEX01.txt",
            "row_number": 1,
            "press_time_ms": 1000,
            "release_time_ms": 900,  # Negative hold (-100)
            "hold_time_ms": -100,
            "pressure": 0.5,
            "label": "PD",
        },
        {
            "dataset": "i-PROGNOSIS",
            "participant_id": "S01",
            "recording_id": "S01_TEX01",
            "text_id": "TEX01",
            "raw_file": "S01_TEX01.txt",
            "raw_path": "/path/S01_TEX01.txt",
            "row_number": 2,
            "press_time_ms": 2000,
            "release_time_ms": 2100,
            "hold_time_ms": 100,
            "pressure": 1.05,  # Out of range > 1.0
            "label": "PD",
        },
    ]

    events_df = pd.DataFrame(events_data)
    recordings_df = pd.DataFrame(
        [
            {
                "dataset": "i-PROGNOSIS",
                "participant_id": "S01",
                "recording_id": "S01_TEX01",
                "text_id": "TEX01",
                "raw_file": "S01_TEX01.txt",
                "raw_path": "/path/S01_TEX01.txt",
                "label": "PD",
                "total_physical_lines": 2,
                "blank_lines": 0,
                "malformed_lines": 0,
                "valid_events": 2,
            }
        ]
    )

    report = validate_touchscreen_data(events_df, recordings_df)

    issue_types = report.issues["issue_type"].tolist()
    assert "negative_hold_duration" in issue_types
    assert "release_before_press" in issue_types
    assert "pressure_out_of_range" in issue_types


def test_validation_non_monotonic_timestamps():
    """Test validation of non-monotonic press timestamps within a recording."""
    events_data = [
        {
            "dataset": "i-PROGNOSIS",
            "participant_id": "S01",
            "recording_id": "S01_TEX01",
            "text_id": "TEX01",
            "raw_file": "S01_TEX01.txt",
            "raw_path": "/path/S01_TEX01.txt",
            "row_number": 1,
            "press_time_ms": 5000,
            "release_time_ms": 5100,
            "hold_time_ms": 100,
            "pressure": 0.5,
            "label": "PD",
        },
        {
            "dataset": "i-PROGNOSIS",
            "participant_id": "S01",
            "recording_id": "S01_TEX01",
            "text_id": "TEX01",
            "raw_file": "S01_TEX01.txt",
            "raw_path": "/path/S01_TEX01.txt",
            "row_number": 2,
            "press_time_ms": 4000,  # Non-monotonic (4000 < 5000)
            "release_time_ms": 4100,
            "hold_time_ms": 100,
            "pressure": 0.5,
            "label": "PD",
        },
    ]

    events_df = pd.DataFrame(events_data)
    recordings_df = pd.DataFrame()
    report = validate_touchscreen_data(events_df, recordings_df)

    issue_types = report.issues["issue_type"].tolist()
    assert "non_monotonic_press_timestamp" in issue_types


def test_metadata_label_mapping_and_validation(tmp_path: Path):
    """Test metadata loading, subject folder mapping, and label normalization."""
    # Create fake excel structure using openpyxl / pandas
    excel_path = tmp_path / "Demographics_Clinical_Characteristics.xlsx"
    df = pd.DataFrame(
        {
            "Subject ID": [1, 2],
            "Group": ["Control", "PD"],
        }
    )
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Demographics_Clinical", index=False)

    metadata = load_demographics_metadata(tmp_path)
    assert metadata["S01"] == "CNT"
    assert metadata["S02"] == "PD"


def test_subject_metadata_mismatch_raises_error(tmp_path: Path):
    """Test that a subject directory missing from metadata raises DatasetMappingError."""
    excel_path = tmp_path / "Demographics_Clinical_Characteristics.xlsx"
    df = pd.DataFrame({"Subject ID": [1], "Group": ["Control"]})
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Demographics_Clinical", index=False)

    data_dir = tmp_path / "Data"
    data_dir.mkdir(parents=True)
    (data_dir / "S01").mkdir()
    (data_dir / "S02").mkdir()  # S02 not in Excel

    with pytest.raises(DatasetMappingError):
        load_all_touchscreen_recordings(tmp_path)


def test_real_dataset_discovery_and_immutability():
    """Integration test verifying full touchscreen dataset discovery and raw file immutability."""
    if not DEFAULT_TOUCHSCREEN_PACKAGE_ROOT.exists():
        pytest.skip("Touchscreen raw dataset package root not available.")

    events_df, recordings_df, malformed_df = load_all_touchscreen_recordings()

    assert len(recordings_df) == 338
    assert len(recordings_df["participant_id"].unique()) == 33
    assert len(events_df) == 26343
    assert set(recordings_df["label"].unique()) == {"PD", "CNT"}

    report = validate_touchscreen_data(events_df, recordings_df, malformed_df)
    assert not report.summary.empty
