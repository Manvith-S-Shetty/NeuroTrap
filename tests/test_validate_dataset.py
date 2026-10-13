import pandas as pd

from ml.preprocessing.validate_dataset import validate_normalized_data


def base_rows() -> list[dict[str, object]]:
    return [
        {
            "dataset": "MIT-TEST",
            "participant_id": "1",
            "recording_id": "rec1",
            "session_id": "001",
            "repetition_id": "001",
            "experiment_id": "014",
            "recording_timestamp_unix": 1,
            "raw_file": "rec1.csv",
            "raw_path": "data/rec1.csv",
            "source_file_column": "file_1",
            "row_number": 1,
            "key": "a",
            "hold_time": 0.1,
            "release_time": 1.1,
            "press_time": 1.0,
            "gt": False,
            "label": "CNT",
        },
        {
            "dataset": "MIT-TEST",
            "participant_id": "1",
            "recording_id": "rec1",
            "session_id": "001",
            "repetition_id": "001",
            "experiment_id": "014",
            "recording_timestamp_unix": 1,
            "raw_file": "rec1.csv",
            "raw_path": "data/rec1.csv",
            "source_file_column": "file_1",
            "row_number": 2,
            "key": "",
            "hold_time": -0.2,
            "release_time": -0.2,
            "press_time": 0.4,
            "gt": False,
            "label": "CNT",
        },
        {
            "dataset": "MIT-TEST",
            "participant_id": "1",
            "recording_id": "rec1",
            "session_id": "001",
            "repetition_id": "001",
            "experiment_id": "014",
            "recording_timestamp_unix": 1,
            "raw_file": "rec1.csv",
            "raw_path": "data/rec1.csv",
            "source_file_column": "file_1",
            "row_number": 3,
            "key": "b",
            "hold_time": 6.0,
            "release_time": 6.5,
            "press_time": 0.5,
            "gt": False,
            "label": "CNT",
        },
    ]


def test_validation_reports_suspicious_rows_without_removing_them() -> None:
    data = pd.DataFrame(base_rows())

    report = validate_normalized_data(data)
    summary = dict(zip(report.summary["issue_type"], report.summary["count"]))

    assert len(data) == 3
    assert summary["missing_key"] == 1
    assert summary["negative_hold_duration"] == 1
    assert summary["release_before_press"] == 1
    assert summary["non_positive_timestamp"] == 1
    assert summary["suspicious_long_hold"] == 1
    assert summary["non_monotonic_press_time"] == 1


def test_validation_reports_duplicate_rows() -> None:
    rows = base_rows()
    rows.append(rows[0].copy())
    data = pd.DataFrame(rows)

    report = validate_normalized_data(data)
    summary = dict(zip(report.summary["issue_type"], report.summary["count"]))

    assert summary["duplicate_row"] == 2
