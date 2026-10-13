"""Unit tests for touchscreen ML dataset construction and participant-aware splitting.

Covers 18 required test assertions:
1. exact 27 feature columns
2. numerical feature validation
3. label validation
4. participant metadata validation
5. duplicate recording detection
6. inconsistent participant-label detection
7. participant-aware split
8. zero participant overlap
9. every recording assigned exactly once
10. every participant assigned exactly once
11. reproducibility with random_state=42
12. class distribution reporting
13. input immutability
14. real dataset integration
15. expected 338 recordings
16. expected 33 participants
17. expected 179 PD recordings
18. expected 159 CNT recordings
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from ml.features.touchscreen_feature_extraction import (
    TOUCHSCREEN_METADATA_COLUMNS,
    TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS,
)
from ml.training.touchscreen_dataset import (
    EXACT_FEATURE_COUNT,
    TouchscreenMLDataset,
    TouchscreenParticipantSplit,
    build_touchscreen_ml_dataset,
    split_touchscreen_by_participant,
    validate_touchscreen_feature_table,
    validate_touchscreen_ml_dataset,
    validate_touchscreen_participant_split,
)


@pytest.fixture
def sample_touchscreen_feature_table() -> pd.DataFrame:
    """Create a valid synthetic touchscreen feature table with 10 recordings across 5 participants."""
    rows = []
    # 3 PD participants (P001: 2 recs, P002: 2 recs, P003: 2 recs)
    # 2 CNT participants (P004: 2 recs, P005: 2 recs)
    participants = [
        ("P001", "PD", 2),
        ("P002", "PD", 2),
        ("P003", "PD", 2),
        ("P004", "CNT", 2),
        ("P005", "CNT", 2),
    ]
    rec_counter = 1
    for p_id, label, count in participants:
        for i in range(count):
            rec_id = f"REC_{rec_counter:03d}"
            rec_counter += 1
            row = {
                "participant_id": p_id,
                "recording_id": rec_id,
                "text_id": f"TXT_{i+1}",
                "label": label,
                "raw_file": f"{rec_id}.csv",
                "raw_path": f"/path/to/{rec_id}.csv",
            }
            # Fill 27 numerical features with distinct valid values
            for idx, feature in enumerate(TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS):
                row[feature] = float((rec_counter * 10) + idx + 0.5)
            rows.append(row)

    return pd.DataFrame(rows)


@pytest.fixture
def sample_ml_dataset(sample_touchscreen_feature_table: pd.DataFrame) -> TouchscreenMLDataset:
    return build_touchscreen_ml_dataset(sample_touchscreen_feature_table)


# 1. Exact 27 feature columns
def test_exact_27_feature_columns(sample_ml_dataset: TouchscreenMLDataset) -> None:
    assert len(TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS) == 27
    assert len(sample_ml_dataset.X.columns) == 27
    assert list(sample_ml_dataset.X.columns) == TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS


# 2. Numerical feature validation
def test_numerical_feature_validation(sample_touchscreen_feature_table: pd.DataFrame) -> None:
    bad_table = sample_touchscreen_feature_table.copy()
    bad_table["hold_mean"] = bad_table["hold_mean"].astype(str)
    with pytest.raises(ValueError, match="Non-numeric columns"):
        build_touchscreen_ml_dataset(bad_table)

    nan_table = sample_touchscreen_feature_table.copy()
    nan_table.loc[0, "hold_std"] = np.nan
    with pytest.raises(ValueError, match="missing \\(NaN\\) values"):
        build_touchscreen_ml_dataset(nan_table)

    inf_table = sample_touchscreen_feature_table.copy()
    inf_table.loc[0, "pressure_mean"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        build_touchscreen_ml_dataset(inf_table)


# 3. Label validation
def test_label_validation(sample_touchscreen_feature_table: pd.DataFrame) -> None:
    bad_labels = sample_touchscreen_feature_table.copy()
    bad_labels.loc[0, "label"] = "UNKNOWN"
    with pytest.raises(ValueError, match="unsupported values"):
        build_touchscreen_ml_dataset(bad_labels)

    missing_labels = sample_touchscreen_feature_table.copy()
    missing_labels.loc[0, "label"] = None
    with pytest.raises(ValueError, match="Target labels contain missing values"):
        build_touchscreen_ml_dataset(missing_labels)


# 4. Participant metadata validation
def test_participant_metadata_validation(sample_touchscreen_feature_table: pd.DataFrame) -> None:
    missing_p_table = sample_touchscreen_feature_table.drop(columns=["participant_id"])
    with pytest.raises(ValueError, match="Missing required metadata columns"):
        validate_touchscreen_feature_table(missing_p_table)

    nan_p_table = sample_touchscreen_feature_table.copy()
    nan_p_table.loc[0, "participant_id"] = None
    with pytest.raises(ValueError, match="participant_id column contains missing values"):
        validate_touchscreen_feature_table(nan_p_table)


# 5. Duplicate recording detection
def test_duplicate_recording_detection(sample_touchscreen_feature_table: pd.DataFrame) -> None:
    dup_table = sample_touchscreen_feature_table.copy()
    dup_table.loc[1, "recording_id"] = dup_table.loc[0, "recording_id"]
    with pytest.raises(ValueError, match="Duplicate recording_id values found"):
        build_touchscreen_ml_dataset(dup_table)


# 6. Inconsistent participant-label detection
def test_inconsistent_participant_label_detection(sample_touchscreen_feature_table: pd.DataFrame) -> None:
    inconsistent_table = sample_touchscreen_feature_table.copy()
    # P001 has 2 recordings. Change label of second recording to CNT
    inconsistent_table.loc[1, "label"] = "CNT"
    with pytest.raises(ValueError, match="Participants have inconsistent target labels"):
        build_touchscreen_ml_dataset(inconsistent_table)


# 7. Participant-aware split
def test_participant_aware_split(sample_ml_dataset: TouchscreenMLDataset) -> None:
    split = split_touchscreen_by_participant(sample_ml_dataset, test_size=0.2, random_state=42, stratify=True)
    assert isinstance(split, TouchscreenParticipantSplit)
    assert len(split.X_dev) > 0
    assert len(split.X_test) > 0


# 8. Zero participant overlap
def test_zero_participant_overlap(sample_ml_dataset: TouchscreenMLDataset) -> None:
    split = split_touchscreen_by_participant(sample_ml_dataset, test_size=0.4, random_state=42, stratify=True)
    dev_p = set(split.dev_metadata["participant_id"].astype(str))
    test_p = set(split.test_metadata["participant_id"].astype(str))
    overlap = dev_p & test_p
    assert len(overlap) == 0, f"Participant overlap found: {overlap}"
    assert dev_p.isdisjoint(test_p)


# 9. Every recording assigned exactly once
def test_every_recording_assigned_exactly_once(sample_ml_dataset: TouchscreenMLDataset) -> None:
    split = split_touchscreen_by_participant(sample_ml_dataset, test_size=0.4, random_state=42, stratify=True)
    dev_recs = list(split.dev_metadata["recording_id"])
    test_recs = list(split.test_metadata["recording_id"])

    assert set(dev_recs).isdisjoint(set(test_recs))
    total_assigned = len(dev_recs) + len(test_recs)
    assert total_assigned == len(sample_ml_dataset.metadata)
    assert set(dev_recs) | set(test_recs) == set(sample_ml_dataset.metadata["recording_id"])


# 10. Every participant assigned exactly once
def test_every_participant_assigned_exactly_once(sample_ml_dataset: TouchscreenMLDataset) -> None:
    split = split_touchscreen_by_participant(sample_ml_dataset, test_size=0.4, random_state=42, stratify=True)
    dev_p = set(split.development_participants)
    test_p = set(split.final_test_participants)

    all_p = set(sample_ml_dataset.metadata["participant_id"].astype(str))
    assert dev_p | test_p == all_p
    assert len(dev_p) + len(test_p) == len(all_p)


# 11. Reproducibility with random_state=42
def test_reproducibility_with_random_state_42(sample_ml_dataset: TouchscreenMLDataset) -> None:
    split1 = split_touchscreen_by_participant(sample_ml_dataset, test_size=0.4, random_state=42, stratify=True)
    split2 = split_touchscreen_by_participant(sample_ml_dataset, test_size=0.4, random_state=42, stratify=True)

    pd.testing.assert_frame_equal(split1.X_dev, split2.X_dev)
    pd.testing.assert_frame_equal(split1.X_test, split2.X_test)
    pd.testing.assert_series_equal(split1.y_dev, split2.y_dev)
    pd.testing.assert_series_equal(split1.y_test, split2.y_test)
    pd.testing.assert_frame_equal(split1.dev_metadata, split2.dev_metadata)
    pd.testing.assert_frame_equal(split1.test_metadata, split2.test_metadata)


# 12. Class distribution reporting
def test_class_distribution_reporting(sample_ml_dataset: TouchscreenMLDataset) -> None:
    split = split_touchscreen_by_participant(sample_ml_dataset, test_size=0.4, random_state=42, stratify=True)
    dev_counts = split.y_dev.value_counts().to_dict()
    test_counts = split.y_test.value_counts().to_dict()

    assert "PD" in dev_counts and "CNT" in dev_counts
    assert "PD" in test_counts and "CNT" in test_counts


# 13. Input immutability
def test_input_immutability(sample_touchscreen_feature_table: pd.DataFrame) -> None:
    table_copy = sample_touchscreen_feature_table.copy(deep=True)
    ml_dataset = build_touchscreen_ml_dataset(sample_touchscreen_feature_table)

    pd.testing.assert_frame_equal(sample_touchscreen_feature_table, table_copy)

    X_copy = ml_dataset.X.copy(deep=True)
    metadata_copy = ml_dataset.metadata.copy(deep=True)
    split_touchscreen_by_participant(ml_dataset, test_size=0.2, random_state=42)

    pd.testing.assert_frame_equal(ml_dataset.X, X_copy)
    pd.testing.assert_frame_equal(ml_dataset.metadata, metadata_copy)


# 14. Real dataset integration
def test_real_dataset_integration() -> None:
    real_dataset = build_touchscreen_ml_dataset()
    validate_touchscreen_ml_dataset(real_dataset)

    split = split_touchscreen_by_participant(real_dataset, test_size=0.2, random_state=42, stratify=True)
    validate_touchscreen_participant_split(split)


# 15. Expected 338 recordings
def test_expected_338_recordings() -> None:
    real_dataset = build_touchscreen_ml_dataset()
    assert len(real_dataset.X) == 338
    assert len(real_dataset.y) == 338
    assert len(real_dataset.metadata) == 338


# 16. Expected 33 participants
def test_expected_33_participants() -> None:
    real_dataset = build_touchscreen_ml_dataset()
    assert real_dataset.metadata["participant_id"].nunique() == 33


# 17. Expected 179 PD recordings
def test_expected_179_pd_recordings() -> None:
    real_dataset = build_touchscreen_ml_dataset()
    assert (real_dataset.y == "PD").sum() == 179


# 18. Expected 159 CNT recordings
def test_expected_159_cnt_recordings() -> None:
    real_dataset = build_touchscreen_ml_dataset()
    assert (real_dataset.y == "CNT").sum() == 159
