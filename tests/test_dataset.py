import pandas as pd
import pytest

from ml.training.dataset import (
    FEATURE_COLUMNS,
    METADATA_COLUMNS,
    TARGET_COLUMN,
    build_ml_dataset,
    split_by_participant,
    validate_feature_table,
)


def make_feature_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    participants = [
        ("p1", "CNT", "MIT-CS1PD", 2),
        ("p2", "PD", "MIT-CS1PD", 2),
        ("p3", "CNT", "MIT-CS2PD", 1),
        ("p4", "PD", "MIT-CS2PD", 1),
        ("p5", "CNT", "MIT-CS2PD", 1),
        ("p6", "PD", "MIT-CS2PD", 1),
    ]
    recording_index = 0
    for participant_id, label, dataset, recordings in participants:
        for session in range(recordings):
            recording_index += 1
            row = {
                "dataset": dataset,
                "participant_id": participant_id,
                "recording_id": f"rec{recording_index}",
                "session_id": f"{session + 1:03d}",
                "repetition_id": f"{session + 1:03d}",
                "experiment_id": "014",
                "raw_file": f"rec{recording_index}.csv",
                "gt": label == "PD",
                "label": label,
            }
            for idx, column in enumerate(FEATURE_COLUMNS):
                row[column] = float(recording_index + idx / 100)
            rows.append(row)
    return pd.DataFrame(rows)


def test_build_ml_dataset_uses_correct_x_feature_columns() -> None:
    table = make_feature_table()

    ml_dataset = build_ml_dataset(table)

    assert list(ml_dataset.X.columns) == FEATURE_COLUMNS
    assert ml_dataset.X.shape == (8, len(FEATURE_COLUMNS))


def test_build_ml_dataset_uses_label_as_y() -> None:
    table = make_feature_table()

    ml_dataset = build_ml_dataset(table)

    assert ml_dataset.y.name == TARGET_COLUMN
    assert list(ml_dataset.y) == list(table["label"])


def test_metadata_is_excluded_from_x() -> None:
    ml_dataset = build_ml_dataset(make_feature_table())

    forbidden = set(METADATA_COLUMNS + [TARGET_COLUMN, "raw_path", "recording_timestamp_unix"])
    assert not forbidden & set(ml_dataset.X.columns)
    assert set(METADATA_COLUMNS).issubset(ml_dataset.metadata.columns)


def test_participant_aware_split_keeps_participants_isolated() -> None:
    ml_dataset = build_ml_dataset(make_feature_table())

    split = split_by_participant(ml_dataset, test_size=0.33, random_state=7, stratify=True)

    train_participants = set(split.train_metadata["participant_id"])
    test_participants = set(split.test_metadata["participant_id"])
    assert train_participants.isdisjoint(test_participants)
    assert len(split.X_train) + len(split.X_test) == len(ml_dataset.X)
    assert len(split.y_train) + len(split.y_test) == len(ml_dataset.y)


def test_split_is_reproducible_with_same_random_state() -> None:
    ml_dataset = build_ml_dataset(make_feature_table())

    first = split_by_participant(ml_dataset, test_size=0.33, random_state=3, stratify=True)
    second = split_by_participant(ml_dataset, test_size=0.33, random_state=3, stratify=True)

    assert set(first.test_metadata["participant_id"]) == set(second.test_metadata["participant_id"])
    pd.testing.assert_frame_equal(first.X_test, second.X_test)
    pd.testing.assert_series_equal(first.y_test, second.y_test)


def test_different_random_state_can_produce_different_participant_split() -> None:
    ml_dataset = build_ml_dataset(make_feature_table())

    first = split_by_participant(ml_dataset, test_size=0.5, random_state=1, stratify=False)
    second = split_by_participant(ml_dataset, test_size=0.5, random_state=2, stratify=False)

    assert set(first.test_metadata["participant_id"]) != set(second.test_metadata["participant_id"])


def test_missing_required_feature_raises_clear_error() -> None:
    table = make_feature_table().drop(columns=["hold_mean"])

    with pytest.raises(ValueError, match="Missing required feature columns"):
        validate_feature_table(table)


def test_missing_participant_id_raises_clear_error() -> None:
    table = make_feature_table().drop(columns=["participant_id"])

    with pytest.raises(ValueError, match="Missing required metadata columns"):
        validate_feature_table(table)


def test_missing_label_raises_clear_error() -> None:
    table = make_feature_table().drop(columns=["label"])

    with pytest.raises(ValueError, match="Missing target label column"):
        validate_feature_table(table)


def test_non_numeric_feature_entering_x_raises_clear_error() -> None:
    table = make_feature_table()
    table["hold_mean"] = "not numeric"

    with pytest.raises(ValueError, match="Non-numeric columns"):
        build_ml_dataset(table)
