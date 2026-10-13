"""Read-only data loader for the i-PROGNOSIS touchscreen typing dataset.

Parses raw .txt recording files and Excel clinical metadata without modifying
source files. Exposes normalized event-level and recording-level DataFrames.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TOUCHSCREEN_PACKAGE_ROOT = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "raw"
    / "touch_datasets"
    / "touchscreen_typing_pd"
    / "i-PROGNOSIS_DS2.5_Sub1_KeystrokeTimingPressureData"
    / "i-PROGNOSIS_DS2.5_Sub1_KeystrokeTimingPressureData"
)

RAW_EVENT_RE = re.compile(
    r"^Press,\s*(?P<press_ms>-?\d+(?:\.\d+)?),\s*Release\s+(?P<release_ms>-?\d+(?:\.\d+)?),\s*(?P<pressure>-?\d+(?:\.\d+)?)$",
    re.IGNORECASE,
)
FILENAME_RE = re.compile(r"^S(?P<subject_num>\d+)_(?P<text_id>TEX\d+)\.txt$", re.IGNORECASE)

LABEL_MAP = {
    "Control": "CNT",
    "PD": "PD",
}

NORMALIZED_EVENT_COLUMNS = [
    "dataset",
    "participant_id",
    "recording_id",
    "text_id",
    "raw_file",
    "raw_path",
    "row_number",
    "press_time_ms",
    "release_time_ms",
    "hold_time_ms",
    "pressure",
    "label",
]

RECORDING_METADATA_COLUMNS = [
    "dataset",
    "participant_id",
    "recording_id",
    "text_id",
    "raw_file",
    "raw_path",
    "label",
    "total_physical_lines",
    "blank_lines",
    "malformed_lines",
    "valid_events",
]


class DatasetMappingError(ValueError):
    """Raised when touchscreen raw files or subject folders cannot be mapped safely."""


@dataclass(frozen=True)
class TouchscreenRecordingMetadata:
    dataset: str
    participant_id: str
    recording_id: str
    text_id: str
    raw_file: str
    raw_path: str
    label: str


def parse_touchscreen_filename(filename: str) -> dict[str, str]:
    """Parse S##_TEX*.txt filenames, supporting TEX01, TEX1, TEX010, etc."""
    match = FILENAME_RE.match(filename)
    if not match:
        raise DatasetMappingError(f"Touchscreen filename does not match expected S##_TEX*.txt pattern: {filename}")

    subject_num = int(match.group("subject_num"))
    participant_id = f"S{subject_num:02d}"
    text_id = match.group("text_id")
    recording_id = f"{participant_id}_{text_id}"

    return {
        "participant_id": participant_id,
        "text_id": text_id,
        "recording_id": recording_id,
    }


def load_demographics_metadata(package_root: Path | None = None) -> dict[str, str]:
    """Load Excel metadata mapping Subject ID (S##) -> normalized label ('PD' or 'CNT')."""
    root = package_root or DEFAULT_TOUCHSCREEN_PACKAGE_ROOT
    excel_path = root / "Demographics_Clinical_Characteristics.xlsx"

    if not excel_path.exists():
        raise DatasetMappingError(f"Clinical metadata file not found at: {excel_path}")

    try:
        df = pd.read_excel(excel_path, sheet_name="Demographics_Clinical")
    except Exception as exc:
        raise DatasetMappingError(f"Failed to load Excel sheet 'Demographics_Clinical' from {excel_path}: {exc}") from exc

    if "Subject ID" not in df.columns or "Group" not in df.columns:
        raise DatasetMappingError(f"Excel metadata missing required columns 'Subject ID' or 'Group': {excel_path}")

    metadata_map: dict[str, str] = {}
    for _, row in df.iterrows():
        subject_val = row["Subject ID"]
        group_val = str(row["Group"]).strip()

        if pd.isna(subject_val):
            continue

        try:
            subject_num = int(subject_val)
            participant_id = f"S{subject_num:02d}"
        except (ValueError, TypeError):
            raise DatasetMappingError(f"Invalid Subject ID in metadata: {subject_val}")

        if group_val not in LABEL_MAP:
            raise DatasetMappingError(f"Unsupported group label '{group_val}' for subject {participant_id}")

        metadata_map[participant_id] = LABEL_MAP[group_val]

    return metadata_map


def parse_touchscreen_file(
    file_path: Path,
    metadata: TouchscreenRecordingMetadata,
) -> tuple[list[dict], list[dict], dict[str, int]]:
    """Parse raw .txt event lines into normalized events, malformed lines, and line stats."""
    content = file_path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()

    total_lines = len(lines)
    blank_lines = 0
    malformed_records: list[dict] = []
    events: list[dict] = []

    for row_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            blank_lines += 1
            continue

        match = RAW_EVENT_RE.match(stripped)
        if not match:
            malformed_records.append(
                {
                    "dataset": metadata.dataset,
                    "participant_id": metadata.participant_id,
                    "recording_id": metadata.recording_id,
                    "raw_file": metadata.raw_file,
                    "raw_path": metadata.raw_path,
                    "row_number": row_number,
                    "raw_line": line,
                    "label": metadata.label,
                }
            )
            continue

        try:
            press_ms = float(match.group("press_ms"))
            release_ms = float(match.group("release_ms"))
            pressure = float(match.group("pressure"))
            hold_ms = release_ms - press_ms

            # Convert integer timestamps where exact
            if press_ms.is_integer():
                press_ms = int(press_ms)
            if release_ms.is_integer():
                release_ms = int(release_ms)
            if hold_ms.is_integer():
                hold_ms = int(hold_ms)

        except (ValueError, TypeError):
            malformed_records.append(
                {
                    "dataset": metadata.dataset,
                    "participant_id": metadata.participant_id,
                    "recording_id": metadata.recording_id,
                    "raw_file": metadata.raw_file,
                    "raw_path": metadata.raw_path,
                    "row_number": row_number,
                    "raw_line": line,
                    "label": metadata.label,
                }
            )
            continue

        events.append(
            {
                "dataset": metadata.dataset,
                "participant_id": metadata.participant_id,
                "recording_id": metadata.recording_id,
                "text_id": metadata.text_id,
                "raw_file": metadata.raw_file,
                "raw_path": metadata.raw_path,
                "row_number": row_number,
                "press_time_ms": press_ms,
                "release_time_ms": release_ms,
                "hold_time_ms": hold_ms,
                "pressure": pressure,
                "label": metadata.label,
            }
        )

    stats = {
        "total_physical_lines": total_lines,
        "blank_lines": blank_lines,
        "malformed_lines": len(malformed_records),
        "valid_events": len(events),
    }

    return events, malformed_records, stats


def load_all_touchscreen_recordings(
    package_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all touchscreen recordings and metadata without mutating raw files.

    Returns:
        events_df: Normalized event-level DataFrame.
        recordings_df: Recording-level summary DataFrame.
        malformed_df: DataFrame of non-blank malformed lines (if any).
    """
    root = package_root or DEFAULT_TOUCHSCREEN_PACKAGE_ROOT
    data_dir = root / "Data"

    if not data_dir.exists():
        raise DatasetMappingError(f"Touchscreen Data directory not found at: {data_dir}")

    metadata_map = load_demographics_metadata(root)

    # Discover subject directories
    subject_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("S")])

    if not subject_dirs:
        raise DatasetMappingError(f"No subject directories (S##) found under: {data_dir}")

    # Validate that every subject folder S## has a matching metadata entry
    for subj_dir in subject_dirs:
        if subj_dir.name not in metadata_map:
            raise DatasetMappingError(f"Subject folder '{subj_dir.name}' has no matching metadata in Excel file")

    all_events: list[dict] = []
    all_malformed: list[dict] = []
    recording_summaries: list[dict] = []

    for subj_dir in subject_dirs:
        participant_id = subj_dir.name
        label = metadata_map[participant_id]

        txt_files = sorted(list(subj_dir.glob("*.txt")))
        for txt_file in txt_files:
            file_meta = parse_touchscreen_filename(txt_file.name)
            rec_meta = TouchscreenRecordingMetadata(
                dataset="i-PROGNOSIS",
                participant_id=file_meta["participant_id"],
                recording_id=file_meta["recording_id"],
                text_id=file_meta["text_id"],
                raw_file=txt_file.name,
                raw_path=str(txt_file.resolve()),
                label=label,
            )

            events, malformed, stats = parse_touchscreen_file(txt_file, rec_meta)

            all_events.extend(events)
            all_malformed.extend(malformed)
            recording_summaries.append(
                {
                    "dataset": rec_meta.dataset,
                    "participant_id": rec_meta.participant_id,
                    "recording_id": rec_meta.recording_id,
                    "text_id": rec_meta.text_id,
                    "raw_file": rec_meta.raw_file,
                    "raw_path": rec_meta.raw_path,
                    "label": rec_meta.label,
                    "total_physical_lines": stats["total_physical_lines"],
                    "blank_lines": stats["blank_lines"],
                    "malformed_lines": stats["malformed_lines"],
                    "valid_events": stats["valid_events"],
                }
            )

    events_df = pd.DataFrame(all_events, columns=NORMALIZED_EVENT_COLUMNS)
    recordings_df = pd.DataFrame(recording_summaries, columns=RECORDING_METADATA_COLUMNS)
    malformed_df = pd.DataFrame(all_malformed)

    return events_df, recordings_df, malformed_df
