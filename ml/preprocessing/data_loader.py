"""Read-only loader for the NeuroTap MIT-CS1PD/MIT-CS2PD raw datasets.

The loader uses the canonical top-level MIT-CS1PD and MIT-CS2PD directories and
ignores the duplicate neuroQWERTY tree. It preserves every raw row and adds
metadata needed for participant-safe downstream processing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE_ROOT = PROJECT_ROOT / "ml" / "data" / "raw" / "neuroqwerty-mit-csxpd-dataset-1.0.0"
RAW_COLUMNS = ["key", "hold_time", "release_time", "press_time"]
TIMING_COLUMNS = ["hold_time", "release_time", "press_time"]
NORMALIZED_COLUMNS = [
    "dataset",
    "participant_id",
    "recording_id",
    "session_id",
    "repetition_id",
    "experiment_id",
    "recording_timestamp_unix",
    "raw_file",
    "raw_path",
    "source_file_column",
    "row_number",
    "key",
    "hold_time",
    "release_time",
    "press_time",
    "gt",
    "label",
]
FILENAME_RE = re.compile(r"^(?P<timestamp>\d+)\.(?P<pid>\d+)_(?P<rep>\d+)_(?P<exp>\d+)\.csv$")


class DatasetMappingError(ValueError):
    """Raised when raw files cannot be mapped to participants or labels safely."""


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: Path
    data_dir: Path
    ground_truth_file: Path
    file_columns: tuple[str, ...]


@dataclass(frozen=True)
class RecordingMetadata:
    dataset: str
    participant_id: str
    recording_id: str
    session_id: str
    repetition_id: str
    experiment_id: str
    recording_timestamp_unix: int
    raw_file: str
    raw_path: str
    source_file_column: str
    gt: bool
    label: str


def get_dataset_configs(package_root: Path | None = None) -> dict[str, DatasetConfig]:
    """Return canonical dataset locations using project-relative defaults."""
    root = package_root or DEFAULT_PACKAGE_ROOT
    return {
        "MIT-CS1PD": DatasetConfig(
            name="MIT-CS1PD",
            root=root / "MIT-CS1PD",
            data_dir=root / "MIT-CS1PD" / "data_MIT-CS1PD",
            ground_truth_file=root / "MIT-CS1PD" / "GT_DataPD_MIT-CS1PD.csv",
            file_columns=("file_1", "file_2"),
        ),
        "MIT-CS2PD": DatasetConfig(
            name="MIT-CS2PD",
            root=root / "MIT-CS2PD",
            data_dir=root / "MIT-CS2PD" / "data_MIT-CS2PD",
            ground_truth_file=root / "MIT-CS2PD" / "GT_DataPD_MIT-CS2PD.csv",
            file_columns=("file_1",),
        ),
    }


def parse_raw_filename(filename: str) -> dict[str, str | int]:
    """Parse timestamp.pID_repID_expID.csv filenames used by the official loader."""
    match = FILENAME_RE.match(filename)
    if not match:
        raise DatasetMappingError(f"Raw filename does not match expected pattern: {filename}")

    return {
        "recording_timestamp_unix": int(match.group("timestamp")),
        "participant_id": str(int(match.group("pid"))),
        "repetition_id": match.group("rep"),
        "experiment_id": match.group("exp"),
    }


def label_from_gt(gt_value: object) -> str:
    """Convert the documented ground-truth encoding to a readable label."""
    if isinstance(gt_value, bool):
        return "PD" if gt_value else "CNT"

    normalized = str(gt_value).strip().lower()
    if normalized == "true":
        return "PD"
    if normalized == "false":
        return "CNT"
    raise DatasetMappingError(f"Unsupported gt label value: {gt_value!r}")


def load_ground_truth(config: DatasetConfig) -> pd.DataFrame:
    """Load a dataset ground-truth CSV without changing the source file."""
    if not config.ground_truth_file.exists():
        raise FileNotFoundError(f"Missing ground-truth file: {config.ground_truth_file}")

    ground_truth = pd.read_csv(config.ground_truth_file, dtype={"pID": "string"})
    required_columns = {"pID", "gt", *config.file_columns}
    missing_columns = required_columns - set(ground_truth.columns)
    if missing_columns:
        raise DatasetMappingError(
            f"{config.name} ground truth is missing required columns: {sorted(missing_columns)}"
        )

    ground_truth["participant_id"] = ground_truth["pID"].astype(str).map(lambda value: str(int(value)))
    ground_truth["gt"] = ground_truth["gt"].map(
        lambda value: value if isinstance(value, bool) else str(value).strip().lower() == "true"
    )
    ground_truth["label"] = ground_truth["gt"].map(label_from_gt)
    return ground_truth


def build_recording_index(config: DatasetConfig) -> pd.DataFrame:
    """Map each canonical raw recording to participant, session, and label metadata."""
    if not config.data_dir.exists():
        raise FileNotFoundError(f"Missing canonical data directory: {config.data_dir}")

    ground_truth = load_ground_truth(config)
    actual_files = {path.name: path for path in config.data_dir.glob("*.csv")}
    records: list[RecordingMetadata] = []
    referenced_files: set[str] = set()

    for _, row in ground_truth.iterrows():
        participant_id = str(row["participant_id"])
        for file_column in config.file_columns:
            raw_file = row[file_column]
            if pd.isna(raw_file) or str(raw_file).strip() == "":
                continue

            raw_file = str(raw_file)
            referenced_files.add(raw_file)
            raw_path = actual_files.get(raw_file)
            if raw_path is None:
                raise DatasetMappingError(
                    f"{config.name} ground truth references missing raw file {raw_file}"
                )

            parsed = parse_raw_filename(raw_file)
            filename_participant_id = str(parsed["participant_id"])
            if filename_participant_id != participant_id:
                raise DatasetMappingError(
                    f"{config.name} participant mismatch for {raw_file}: "
                    f"ground truth pID={participant_id}, filename pID={filename_participant_id}"
                )

            repetition_id = str(parsed["repetition_id"])
            experiment_id = str(parsed["experiment_id"])
            recording_id = Path(raw_file).stem
            session_id = repetition_id
            records.append(
                RecordingMetadata(
                    dataset=config.name,
                    participant_id=participant_id,
                    recording_id=recording_id,
                    session_id=session_id,
                    repetition_id=repetition_id,
                    experiment_id=experiment_id,
                    recording_timestamp_unix=int(parsed["recording_timestamp_unix"]),
                    raw_file=raw_file,
                    raw_path=raw_path.relative_to(PROJECT_ROOT).as_posix()
                    if raw_path.is_relative_to(PROJECT_ROOT)
                    else str(raw_path),
                    source_file_column=file_column,
                    gt=bool(row["gt"]),
                    label=str(row["label"]),
                )
            )

    unreferenced_files = sorted(set(actual_files) - referenced_files)
    if unreferenced_files:
        raise DatasetMappingError(
            f"{config.name} has raw files not referenced by ground truth: {unreferenced_files}"
        )

    return pd.DataFrame([record.__dict__ for record in records])


def load_raw_recording(path: Path) -> pd.DataFrame:
    """Load one headerless raw CSV and assign official positional column names."""
    raw = pd.read_csv(path, header=None, names=RAW_COLUMNS)
    raw.insert(0, "row_number", range(1, len(raw) + 1))

    for column in TIMING_COLUMNS:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    return raw


def load_dataset(config: DatasetConfig) -> pd.DataFrame:
    """Load one canonical dataset into the normalized row-level representation."""
    recording_index = build_recording_index(config)
    frames: list[pd.DataFrame] = []

    for metadata in recording_index.to_dict("records"):
        raw_path = config.data_dir / metadata["raw_file"]
        raw = load_raw_recording(raw_path)
        for column in [
            "dataset",
            "participant_id",
            "recording_id",
            "session_id",
            "repetition_id",
            "experiment_id",
            "recording_timestamp_unix",
            "raw_file",
            "raw_path",
            "source_file_column",
            "gt",
            "label",
        ]:
            raw[column] = metadata[column]
        frames.append(raw[NORMALIZED_COLUMNS])

    if not frames:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def load_all_datasets(package_root: Path | None = None) -> pd.DataFrame:
    """Load MIT-CS1PD and MIT-CS2PD canonical data, excluding duplicate copies."""
    configs = get_dataset_configs(package_root)
    frames = [load_dataset(config) for config in configs.values()]
    return pd.concat(frames, ignore_index=True)


def summarize_loaded_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return dataset-level counts for loaded normalized records."""
    if data.empty:
        return pd.DataFrame(columns=["dataset", "participants", "recordings", "records"])

    return (
        data.groupby("dataset", as_index=False)
        .agg(
            participants=("participant_id", "nunique"),
            recordings=("recording_id", "nunique"),
            records=("recording_id", "size"),
        )
        .sort_values("dataset")
    )


def main() -> int:
    data = load_all_datasets()
    summary = summarize_loaded_data(data)
    label_summary = (
        data[["dataset", "participant_id", "label"]]
        .drop_duplicates()
        .groupby(["dataset", "label"])
        .size()
        .rename("participants")
        .reset_index()
        .sort_values(["dataset", "label"])
    )

    print("# NeuroTap Normalized Data Load")
    print(f"Package root: {DEFAULT_PACKAGE_ROOT.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Normalized rows: {len(data)}")
    print(f"Normalized columns: {list(data.columns)}")
    print("\n## Dataset Summary")
    print(summary.to_string(index=False))
    print("\n## Participant Label Summary")
    print(label_summary.to_string(index=False))
    print("\n## Representative Rows")
    print(data.head(6).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
