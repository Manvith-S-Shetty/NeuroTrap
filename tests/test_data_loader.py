from pathlib import Path

import pandas as pd
import pytest

from ml.preprocessing.data_loader import (
    DatasetConfig,
    DatasetMappingError,
    build_recording_index,
    label_from_gt,
    load_dataset,
    parse_raw_filename,
)


def make_dataset(tmp_path: Path) -> DatasetConfig:
    root = tmp_path / "MIT-CSXPD"
    data_dir = root / "data_MIT-CSXPD"
    data_dir.mkdir(parents=True)
    gt_file = root / "GT_DataPD_MIT-CSXPD.csv"
    gt_file.write_text(
        "pID,gt,updrs108,afTap,sTap,nqScore,typingSpeed,file_1\n"
        "60,False,2,,162.25,0.1,60.5,1402932300.060_001_014.csv\n",
        encoding="utf-8",
    )
    (data_dir / "1402932300.060_001_014.csv").write_text(
        '"a",0.10,1.20,1.10\n"space",0.20,1.60,1.40\n',
        encoding="utf-8",
    )
    return DatasetConfig(
        name="MIT-CSXPD",
        root=root,
        data_dir=data_dir,
        ground_truth_file=gt_file,
        file_columns=("file_1",),
    )


def test_parse_raw_filename_extracts_metadata() -> None:
    parsed = parse_raw_filename("1402932300.060_001_014.csv")

    assert parsed["recording_timestamp_unix"] == 1402932300
    assert parsed["participant_id"] == "60"
    assert parsed["repetition_id"] == "001"
    assert parsed["experiment_id"] == "014"


def test_label_from_gt_uses_documented_encoding() -> None:
    assert label_from_gt(True) == "PD"
    assert label_from_gt(False) == "CNT"
    assert label_from_gt("True") == "PD"
    assert label_from_gt("False") == "CNT"


def test_build_recording_index_maps_ground_truth_to_file(tmp_path: Path) -> None:
    config = make_dataset(tmp_path)

    index = build_recording_index(config)

    assert index.to_dict("records")[0] | {} == {
        "dataset": "MIT-CSXPD",
        "participant_id": "60",
        "recording_id": "1402932300.060_001_014",
        "session_id": "001",
        "repetition_id": "001",
        "experiment_id": "014",
        "recording_timestamp_unix": 1402932300,
        "raw_file": "1402932300.060_001_014.csv",
        "raw_path": str(config.data_dir / "1402932300.060_001_014.csv"),
        "source_file_column": "file_1",
        "gt": False,
        "label": "CNT",
    }


def test_load_dataset_preserves_raw_rows_and_metadata(tmp_path: Path) -> None:
    config = make_dataset(tmp_path)

    data = load_dataset(config)

    assert len(data) == 2
    assert list(data["key"]) == ["a", "space"]
    assert list(data["row_number"]) == [1, 2]
    assert set(data["dataset"]) == {"MIT-CSXPD"}
    assert set(data["participant_id"]) == {"60"}
    assert set(data["label"]) == {"CNT"}
    assert pd.api.types.is_float_dtype(data["hold_time"])
    assert pd.api.types.is_float_dtype(data["press_time"])
    assert pd.api.types.is_float_dtype(data["release_time"])


def test_build_recording_index_rejects_uncertain_participant_mapping(tmp_path: Path) -> None:
    config = make_dataset(tmp_path)
    config.ground_truth_file.write_text(
        "pID,gt,file_1\n"
        "61,False,1402932300.060_001_014.csv\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetMappingError, match="participant mismatch"):
        build_recording_index(config)
