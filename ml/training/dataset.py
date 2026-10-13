"""ML dataset construction and participant-aware splitting for NeuroTap.

This module consumes the recording-level feature table. It does not train a
model, scale features, select features, export artifacts, or modify raw data.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from ml.features.feature_extraction import extract_features  # type: ignore
    from ml.preprocessing.data_loader import load_all_datasets  # type: ignore
else:
    from ml.features.feature_extraction import extract_features
    from ml.preprocessing.data_loader import load_all_datasets


FEATURE_COLUMNS = [
    "hold_mean",
    "hold_std",
    "hold_variance",
    "hold_median",
    "hold_min",
    "hold_max",
    "hold_skewness",
    "hold_kurtosis",
    "flight_mean",
    "flight_std",
    "flight_variance",
    "flight_median",
    "flight_min",
    "flight_max",
    "flight_skewness",
    "flight_kurtosis",
    "inter_key_mean",
    "inter_key_std",
    "inter_key_variance",
    "inter_key_median",
    "inter_key_min",
    "inter_key_max",
    "inter_key_skewness",
    "inter_key_kurtosis",
    "key_count",
    "recording_duration",
    "typing_rate",
]
METADATA_COLUMNS = [
    "dataset",
    "participant_id",
    "recording_id",
    "session_id",
    "repetition_id",
    "experiment_id",
    "raw_file",
    "gt",
]
TARGET_COLUMN = "label"
ALLOWED_LABELS = {"CNT", "PD"}


@dataclass(frozen=True)
class MLDataset:
    """Separated ML dataset components."""

    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame


@dataclass(frozen=True)
class ParticipantSplit:
    """Participant-aware train/test split outputs."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_metadata: pd.DataFrame
    test_metadata: pd.DataFrame


def _missing_columns(data: pd.DataFrame, required_columns: list[str]) -> list[str]:
    return sorted(set(required_columns) - set(data.columns))


def validate_feature_table(feature_table: pd.DataFrame) -> None:
    """Validate the recording-level feature table before ML separation."""
    missing_features = _missing_columns(feature_table, FEATURE_COLUMNS)
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    missing_metadata = _missing_columns(feature_table, METADATA_COLUMNS)
    if missing_metadata:
        raise ValueError(f"Missing required metadata columns: {missing_metadata}")

    if TARGET_COLUMN not in feature_table.columns:
        raise ValueError(f"Missing target label column: {TARGET_COLUMN}")

    if feature_table[TARGET_COLUMN].isna().any():
        raise ValueError("Target labels contain missing values.")

    unknown_labels = sorted(set(feature_table[TARGET_COLUMN]) - ALLOWED_LABELS)
    if unknown_labels:
        raise ValueError(f"Target labels contain unsupported values: {unknown_labels}")

    non_numeric_features = [
        column for column in FEATURE_COLUMNS if not pd.api.types.is_numeric_dtype(feature_table[column])
    ]
    if non_numeric_features:
        raise ValueError(f"Non-numeric columns found in ML feature matrix: {non_numeric_features}")


def build_feature_table(raw_normalized: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the recording-level ML table from the existing feature extractor."""
    normalized = load_all_datasets() if raw_normalized is None else raw_normalized
    return extract_features(normalized)


def build_ml_dataset(feature_table: pd.DataFrame | None = None) -> MLDataset:
    """Separate X, y, and metadata from a recording-level feature table."""
    table = build_feature_table() if feature_table is None else feature_table.copy(deep=True)
    validate_feature_table(table)
    return MLDataset(
        X=table[FEATURE_COLUMNS].copy(),
        y=table[TARGET_COLUMN].copy(),
        metadata=table[METADATA_COLUMNS].copy(),
    )


def _participant_label_table(y: pd.Series, metadata: pd.DataFrame) -> pd.DataFrame:
    participant_labels = pd.concat(
        [metadata[["participant_id"]].reset_index(drop=True), y.rename(TARGET_COLUMN).reset_index(drop=True)],
        axis=1,
    ).drop_duplicates()

    counts = participant_labels.groupby("participant_id")[TARGET_COLUMN].nunique()
    inconsistent = sorted(counts[counts > 1].index.astype(str))
    if inconsistent:
        raise ValueError(f"Participants have inconsistent target labels: {inconsistent}")

    return participant_labels.sort_values("participant_id").reset_index(drop=True)


def _choose_test_participants(
    participant_labels: pd.DataFrame,
    test_size: float,
    random_state: int,
    stratify: bool,
) -> set[str]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    shuffled = participant_labels.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if not stratify:
        test_count = max(1, round(len(shuffled) * test_size))
        test_count = min(test_count, len(shuffled) - 1)
        return set(shuffled.head(test_count)["participant_id"].astype(str))

    selected: list[str] = []
    for _, group in shuffled.groupby(TARGET_COLUMN, sort=False):
        class_test_count = max(1, round(len(group) * test_size))
        if len(group) > 1:
            class_test_count = min(class_test_count, len(group) - 1)
        selected.extend(group.head(class_test_count)["participant_id"].astype(str).tolist())

    if not selected:
        raise ValueError("Unable to select test participants.")
    if len(selected) >= len(shuffled):
        raise ValueError("Stratified split selected all participants for test.")
    return set(selected)


def split_by_participant(
    ml_dataset: MLDataset,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> ParticipantSplit:
    """Split recordings by participant so no participant appears in both sets."""
    if "participant_id" not in ml_dataset.metadata.columns:
        raise ValueError("Missing required metadata columns: ['participant_id']")

    participant_labels = _participant_label_table(ml_dataset.y, ml_dataset.metadata)
    test_participants = _choose_test_participants(
        participant_labels=participant_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    is_test = ml_dataset.metadata["participant_id"].astype(str).isin(test_participants)
    is_train = ~is_test

    train_metadata = ml_dataset.metadata.loc[is_train].copy()
    test_metadata = ml_dataset.metadata.loc[is_test].copy()
    overlap = set(train_metadata["participant_id"].astype(str)) & set(test_metadata["participant_id"].astype(str))
    if overlap:
        raise ValueError(f"Participant overlap detected between train and test: {sorted(overlap)}")
    if train_metadata.empty or test_metadata.empty:
        raise ValueError("Participant split produced an empty train or test set.")

    return ParticipantSplit(
        X_train=ml_dataset.X.loc[is_train].copy(),
        X_test=ml_dataset.X.loc[is_test].copy(),
        y_train=ml_dataset.y.loc[is_train].copy(),
        y_test=ml_dataset.y.loc[is_test].copy(),
        train_metadata=train_metadata,
        test_metadata=test_metadata,
    )


def validate_participant_split(split: ParticipantSplit) -> None:
    """Validate participant isolation and numeric feature matrices."""
    overlap = set(split.train_metadata["participant_id"].astype(str)) & set(
        split.test_metadata["participant_id"].astype(str)
    )
    if overlap:
        raise ValueError(f"Participant overlap detected between train and test: {sorted(overlap)}")

    for name, frame in {"X_train": split.X_train, "X_test": split.X_test}.items():
        non_numeric = [column for column in frame.columns if not pd.api.types.is_numeric_dtype(frame[column])]
        if non_numeric:
            raise ValueError(f"Non-numeric columns found in {name}: {non_numeric}")


def main() -> int:
    feature_table = build_feature_table()
    ml_dataset = build_ml_dataset(feature_table)
    split = split_by_participant(ml_dataset, test_size=0.2, random_state=42, stratify=True)
    validate_participant_split(split)

    train_participants = split.train_metadata["participant_id"].nunique()
    test_participants = split.test_metadata["participant_id"].nunique()
    participant_overlap = set(split.train_metadata["participant_id"]) & set(split.test_metadata["participant_id"])

    print("# NeuroTap ML Dataset")
    print(f"Total recordings: {len(feature_table)}")
    print(f"Total participants: {ml_dataset.metadata['participant_id'].nunique()}")
    print(f"X shape: {ml_dataset.X.shape}")
    print(f"y shape: {ml_dataset.y.shape}")
    print(f"Feature names: {FEATURE_COLUMNS}")
    print("\n## Class Counts")
    print(ml_dataset.y.value_counts().sort_index().to_string())
    print("\n## Participant-Aware Split")
    print(f"Train recordings: {len(split.X_train)}")
    print(f"Test recordings: {len(split.X_test)}")
    print(f"Train participants: {train_participants}")
    print(f"Test participants: {test_participants}")
    print(f"Participant overlap: {len(participant_overlap)}")
    print("\n## Train Class Counts")
    print(split.y_train.value_counts().sort_index().to_string())
    print("\n## Test Class Counts")
    print(split.y_test.value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
