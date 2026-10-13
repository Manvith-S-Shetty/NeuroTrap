"""ML dataset construction and participant-aware splitting for NeuroTap touchscreen modality.

This module consumes the 27 recording-level touchscreen features. It separates
X, y, and metadata and performs participant-aware splitting (development / final-test).

This module does NOT train models, scale features, normalize features, impute values,
select features, remove correlated features, or remove outliers.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from features.touchscreen_feature_extraction import (  # type: ignore
        TOUCHSCREEN_METADATA_COLUMNS,
        TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS,
        extract_all_touchscreen_features,
    )
    from preprocessing.touchscreen.data_loader import (  # type: ignore
        DEFAULT_TOUCHSCREEN_PACKAGE_ROOT,
        load_all_touchscreen_recordings,
    )
else:
    from ..features.touchscreen_feature_extraction import (
        TOUCHSCREEN_METADATA_COLUMNS,
        TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS,
        extract_all_touchscreen_features,
    )
    from ..preprocessing.touchscreen.data_loader import (
        DEFAULT_TOUCHSCREEN_PACKAGE_ROOT,
        load_all_touchscreen_recordings,
    )


TARGET_COLUMN = "label"
ALLOWED_LABELS = {"CNT", "PD"}
EXACT_FEATURE_COUNT = 27


@dataclass(frozen=True)
class TouchscreenMLDataset:
    """Separated ML dataset components for touchscreen data."""

    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame


@dataclass(frozen=True)
class TouchscreenParticipantSplit:
    """Participant-aware development/final-test split outputs."""

    X_dev: pd.DataFrame
    X_test: pd.DataFrame
    y_dev: pd.Series
    y_test: pd.Series
    dev_metadata: pd.DataFrame
    test_metadata: pd.DataFrame

    @property
    def development_recordings(self) -> TouchscreenMLDataset:
        return TouchscreenMLDataset(
            X=self.X_dev.copy(),
            y=self.y_dev.copy(),
            metadata=self.dev_metadata.copy(),
        )

    @property
    def final_test_recordings(self) -> TouchscreenMLDataset:
        return TouchscreenMLDataset(
            X=self.X_test.copy(),
            y=self.y_test.copy(),
            metadata=self.test_metadata.copy(),
        )

    @property
    def development_participants(self) -> set[str]:
        return set(self.dev_metadata["participant_id"].astype(str))

    @property
    def final_test_participants(self) -> set[str]:
        return set(self.test_metadata["participant_id"].astype(str))

    @property
    def dev_recordings(self) -> TouchscreenMLDataset:
        return self.development_recordings

    @property
    def dev_participants(self) -> set[str]:
        return self.development_participants

    @property
    def test_recordings(self) -> TouchscreenMLDataset:
        return self.final_test_recordings

    @property
    def test_participants(self) -> set[str]:
        return self.final_test_participants


def _missing_columns(data: pd.DataFrame, required_columns: list[str]) -> list[str]:
    return sorted(set(required_columns) - set(data.columns))


def validate_touchscreen_feature_table(feature_table: pd.DataFrame) -> None:
    """Validate the recording-level touchscreen feature table before ML separation."""
    missing_features = _missing_columns(feature_table, TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS)
    if missing_features:
        raise ValueError(f"Missing required touchscreen feature columns: {missing_features}")

    missing_metadata = _missing_columns(feature_table, TOUCHSCREEN_METADATA_COLUMNS)
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
        col for col in TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS if not pd.api.types.is_numeric_dtype(feature_table[col])
    ]
    if non_numeric_features:
        raise ValueError(f"Non-numeric columns found in ML feature matrix: {non_numeric_features}")

    X_mat = feature_table[TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS]
    if X_mat.isna().any().any():
        raise ValueError("Feature matrix contains missing (NaN) values.")

    if np.isinf(X_mat.to_numpy()).any():
        raise ValueError("Feature matrix contains infinite (+inf or -inf) values.")

    if feature_table["participant_id"].isna().any():
        raise ValueError("participant_id column contains missing values.")

    if feature_table["recording_id"].isna().any():
        raise ValueError("recording_id column contains missing values.")

    if feature_table["recording_id"].duplicated().any():
        duplicated_ids = sorted(feature_table.loc[feature_table["recording_id"].duplicated(), "recording_id"].unique())
        raise ValueError(f"Duplicate recording_id values found: {duplicated_ids}")

    # Participant label consistency: each participant must have exactly one unique label
    participant_labels = feature_table.groupby("participant_id")[TARGET_COLUMN].nunique()
    inconsistent = sorted(participant_labels[participant_labels > 1].index.astype(str))
    if inconsistent:
        raise ValueError(f"Participants have inconsistent target labels across recordings: {inconsistent}")

    # Single participant per recording
    rec_participants = feature_table.groupby("recording_id")["participant_id"].nunique()
    multi_participant_recs = sorted(rec_participants[rec_participants > 1].index.astype(str))
    if multi_participant_recs:
        raise ValueError(f"Recordings belong to multiple participants: {multi_participant_recs}")


def validate_touchscreen_ml_dataset(ml_dataset: TouchscreenMLDataset) -> None:
    """Validate an assembled TouchscreenMLDataset object."""
    if len(ml_dataset.X.columns) != EXACT_FEATURE_COUNT:
        raise ValueError(f"Expected exactly {EXACT_FEATURE_COUNT} feature columns, got {len(ml_dataset.X.columns)}")

    missing_cols = set(TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS) - set(ml_dataset.X.columns)
    if missing_cols:
        raise ValueError(f"Missing required touchscreen feature columns: {sorted(missing_cols)}")

    missing_meta = set(TOUCHSCREEN_METADATA_COLUMNS) - set(ml_dataset.metadata.columns)
    if missing_meta:
        raise ValueError(f"Missing required metadata columns: {sorted(missing_meta)}")

    if len(ml_dataset.X) != len(ml_dataset.y) or len(ml_dataset.X) != len(ml_dataset.metadata):
        raise ValueError("Length mismatch among X, y, and metadata.")

    recombined = pd.concat([ml_dataset.metadata, ml_dataset.X], axis=1)
    validate_touchscreen_feature_table(recombined)


def build_touchscreen_feature_table(package_root: Path | None = None) -> pd.DataFrame:
    """Extract 27 touchscreen features and metadata from raw touchscreen recordings."""
    root = package_root or DEFAULT_TOUCHSCREEN_PACKAGE_ROOT
    events_df, _, _ = load_all_touchscreen_recordings(root)
    X, metadata, y = extract_all_touchscreen_features(events_df)
    feature_table = pd.concat([metadata, X], axis=1)
    return feature_table


def build_touchscreen_ml_dataset(
    feature_table: pd.DataFrame | None = None,
    package_root: Path | None = None,
) -> TouchscreenMLDataset:
    """Build and separate X, y, and metadata for touchscreen ML readiness."""
    if feature_table is None:
        table = build_touchscreen_feature_table(package_root)
    else:
        table = feature_table.copy(deep=True)

    validate_touchscreen_feature_table(table)

    return TouchscreenMLDataset(
        X=table[TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS].copy(),
        y=table[TARGET_COLUMN].copy(),
        metadata=table[TOUCHSCREEN_METADATA_COLUMNS].copy(),
    )


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


def split_touchscreen_by_participant(
    ml_dataset: TouchscreenMLDataset,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> TouchscreenParticipantSplit:
    """Perform participant-aware development / final-test split.

    Guarantees:
    - Zero participant overlap between development and final-test sets.
    - Every recording is assigned to exactly one split.
    - Class balance is preserved via stratified participant sampling when stratify=True.
    - Input objects remain immutable.
    """
    validate_touchscreen_ml_dataset(ml_dataset)

    participant_labels = (
        ml_dataset.metadata[["participant_id", TARGET_COLUMN]]
        .drop_duplicates()
        .sort_values("participant_id")
        .reset_index(drop=True)
    )

    counts = participant_labels.groupby("participant_id")[TARGET_COLUMN].nunique()
    inconsistent = sorted(counts[counts > 1].index.astype(str))
    if inconsistent:
        raise ValueError(f"Participants have inconsistent target labels: {inconsistent}")

    test_participants = _choose_test_participants(
        participant_labels=participant_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    is_test = ml_dataset.metadata["participant_id"].astype(str).isin(test_participants)
    is_dev = ~is_test

    dev_metadata = ml_dataset.metadata.loc[is_dev].copy().reset_index(drop=True)
    test_metadata = ml_dataset.metadata.loc[is_test].copy().reset_index(drop=True)

    dev_p = set(dev_metadata["participant_id"].astype(str))
    test_p = set(test_metadata["participant_id"].astype(str))
    overlap = dev_p & test_p
    if overlap:
        raise ValueError(f"Participant overlap detected between dev and test sets: {sorted(overlap)}")

    if dev_metadata.empty or test_metadata.empty:
        raise ValueError("Participant split produced an empty development or test set.")

    split = TouchscreenParticipantSplit(
        X_dev=ml_dataset.X.loc[is_dev].copy().reset_index(drop=True),
        X_test=ml_dataset.X.loc[is_test].copy().reset_index(drop=True),
        y_dev=ml_dataset.y.loc[is_dev].copy().reset_index(drop=True),
        y_test=ml_dataset.y.loc[is_test].copy().reset_index(drop=True),
        dev_metadata=dev_metadata,
        test_metadata=test_metadata,
    )

    validate_touchscreen_participant_split(split)
    return split


def validate_touchscreen_participant_split(split: TouchscreenParticipantSplit) -> None:
    """Validate participant isolation and completeness of participant-aware split."""
    dev_participants = set(split.dev_metadata["participant_id"].astype(str))
    test_participants = set(split.test_metadata["participant_id"].astype(str))

    overlap = dev_participants & test_participants
    if overlap:
        raise ValueError(f"Participant overlap detected between dev and test: {sorted(overlap)}")

    dev_recs = set(split.dev_metadata["recording_id"].astype(str))
    test_recs = set(split.test_metadata["recording_id"].astype(str))
    rec_overlap = dev_recs & test_recs
    if rec_overlap:
        raise ValueError(f"Recording overlap detected between dev and test: {sorted(rec_overlap)}")

    for name, frame in [("X_dev", split.X_dev), ("X_test", split.X_test)]:
        missing_cols = set(TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS) - set(frame.columns)
        if missing_cols:
            raise ValueError(f"{name} missing required feature columns: {sorted(missing_cols)}")
        non_numeric = [col for col in frame.columns if not pd.api.types.is_numeric_dtype(frame[col])]
        if non_numeric:
            raise ValueError(f"Non-numeric columns found in {name}: {non_numeric}")
        if frame.isna().any().any():
            raise ValueError(f"NaN values found in {name}")
        if np.isinf(frame.to_numpy()).any():
            raise ValueError(f"Infinite values found in {name}")


def main() -> int:
    print("# Touchscreen ML Dataset & Participant-Aware Split Summary\n")
    feature_table = build_touchscreen_feature_table()
    ml_dataset = build_touchscreen_ml_dataset(feature_table)
    split = split_touchscreen_by_participant(ml_dataset, test_size=0.2, random_state=42, stratify=True)

    dev_p_count = len(split.development_participants)
    test_p_count = len(split.final_test_participants)
    overlap = split.development_participants & split.final_test_participants

    print(f"Total recordings: {len(feature_table)}")
    print(f"Total participants: {ml_dataset.metadata['participant_id'].nunique()}")
    print(f"X shape: {ml_dataset.X.shape}")
    print(f"y shape: {ml_dataset.y.shape}")
    print(f"Feature count: {len(ml_dataset.X.columns)}")
    print("\n## Overall Class Balance")
    print(ml_dataset.y.value_counts().sort_index().to_string())

    print("\n## Participant-Aware Split (80/20)")
    print(f"Development recordings: {len(split.X_dev)}")
    print(f"Development participants: {dev_p_count}")
    print(f"Final-test recordings: {len(split.X_test)}")
    print(f"Final-test participants: {test_p_count}")
    print(f"Participant overlap: {len(overlap)}")

    print("\n## Development Set Class Distribution (Recordings)")
    print(split.y_dev.value_counts().sort_index().to_string())
    print("\n## Development Set Class Distribution (Participants)")
    dev_p_labels = split.dev_metadata.groupby("participant_id")["label"].first()
    print(dev_p_labels.value_counts().sort_index().to_string())

    print("\n## Final-Test Set Class Distribution (Recordings)")
    print(split.y_test.value_counts().sort_index().to_string())
    print("\n## Final-Test Set Class Distribution (Participants)")
    test_p_labels = split.test_metadata.groupby("participant_id")["label"].first()
    print(test_p_labels.value_counts().sort_index().to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
