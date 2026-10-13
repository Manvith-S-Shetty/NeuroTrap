"""Read-only inspection for the NeuroTap neuroQWERTY MIT-CSxPD dataset.

Run from the project root:

    python ml/preprocessing/inspect_dataset.py

The script inspects canonical MIT-CS1PD and MIT-CS2PD data under
ml/data/raw/neuroqwerty-mit-csxpd-dataset-1.0.0. It does not modify, copy,
rename, clean, or train on any dataset files.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "ml" / "data" / "raw" / "neuroqwerty-mit-csxpd-dataset-1.0.0"
DATASETS = {
    "MIT-CS1PD": {
        "root": PACKAGE_ROOT / "MIT-CS1PD",
        "data_dir": PACKAGE_ROOT / "MIT-CS1PD" / "data_MIT-CS1PD",
        "gt_file": PACKAGE_ROOT / "MIT-CS1PD" / "GT_DataPD_MIT-CS1PD.csv",
        "gt_files": ["file_1", "file_2"],
    },
    "MIT-CS2PD": {
        "root": PACKAGE_ROOT / "MIT-CS2PD",
        "data_dir": PACKAGE_ROOT / "MIT-CS2PD" / "data_MIT-CS2PD",
        "gt_file": PACKAGE_ROOT / "MIT-CS2PD" / "GT_DataPD_MIT-CS2PD.csv",
        "gt_files": ["file_1"],
    },
}
RAW_COLUMNS = ["key", "hold_time", "release_time", "press_time"]
FILENAME_RE = re.compile(r"^(?P<timestamp>\d+)\.(?P<pid>\d+)_(?P<rep>\d+)_(?P<exp>\d+)\.csv$")


@dataclass
class DatasetStats:
    name: str
    data_files: list[Path]
    gt_rows: list[dict[str, str]]
    raw_schema_counts: Counter[int] = field(default_factory=Counter)
    raw_event_count: int = 0
    per_file_event_counts: dict[str, int] = field(default_factory=dict)
    missing_by_column: Counter[str] = field(default_factory=Counter)
    duplicate_rows_by_file: dict[str, int] = field(default_factory=dict)
    global_duplicate_rows: int = 0
    malformed_rows: list[dict[str, object]] = field(default_factory=list)
    unusual_rows: list[dict[str, object]] = field(default_factory=list)
    sample_rows: dict[str, list[dict[str, object]]] = field(default_factory=dict)
    key_counts: Counter[str] = field(default_factory=Counter)
    participant_ids_from_filenames: set[str] = field(default_factory=set)
    participant_ids_from_gt: set[str] = field(default_factory=set)
    files_by_participant: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    reps_by_participant: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ground_truth(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_raw_rows(path: Path) -> Iterable[tuple[int, list[str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for line_number, row in enumerate(reader, start=1):
            yield line_number, row


def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def inspect_dataset(name: str, config: dict[str, object]) -> DatasetStats:
    data_dir = config["data_dir"]
    gt_file = config["gt_file"]
    assert isinstance(data_dir, Path)
    assert isinstance(gt_file, Path)

    data_files = sorted(data_dir.glob("*.csv"))
    gt_rows = read_ground_truth(gt_file)
    stats = DatasetStats(name=name, data_files=data_files, gt_rows=gt_rows)
    stats.participant_ids_from_gt = {row["pID"] for row in gt_rows}
    global_hashes: Counter[str] = Counter()

    for path in data_files:
        filename = path.name
        match = FILENAME_RE.match(filename)
        if match:
            participant_id = str(int(match.group("pid")))
            stats.participant_ids_from_filenames.add(participant_id)
            stats.files_by_participant[participant_id].append(filename)
            stats.reps_by_participant[participant_id].add(match.group("rep"))
        else:
            stats.unusual_rows.append(
                {"file": rel(path), "line": None, "issue": "filename does not match expected timestamp.pID_rep_exp.csv pattern"}
            )

        file_hashes: Counter[str] = Counter()
        previous_press_time: float | None = None
        file_count = 0

        for line_number, row in read_raw_rows(path):
            file_count += 1
            stats.raw_event_count += 1
            stats.raw_schema_counts[len(row)] += 1

            row_hash = json.dumps(row, ensure_ascii=False)
            file_hashes[row_hash] += 1
            global_hashes[row_hash] += 1

            if len(row) != 4:
                stats.malformed_rows.append(
                    {"file": rel(path), "line": line_number, "issue": f"expected 4 fields, found {len(row)}", "row": row}
                )
                continue

            record = dict(zip(RAW_COLUMNS, row))
            key = record["key"]
            hold_time = parse_float(record["hold_time"])
            release_time = parse_float(record["release_time"])
            press_time = parse_float(record["press_time"])
            stats.key_counts[key] += 1

            for column, value in record.items():
                if value == "":
                    stats.missing_by_column[column] += 1

            if len(stats.sample_rows.get(filename, [])) < 3:
                stats.sample_rows.setdefault(filename, []).append(
                    {
                        "key": key,
                        "hold_time": record["hold_time"],
                        "release_time": record["release_time"],
                        "press_time": record["press_time"],
                    }
                )

            numeric_values = {
                "hold_time": hold_time,
                "release_time": release_time,
                "press_time": press_time,
            }
            for column, value in numeric_values.items():
                if value is None:
                    stats.malformed_rows.append(
                        {"file": rel(path), "line": line_number, "issue": f"{column} is not numeric", "value": record[column]}
                    )

            if hold_time is not None and hold_time < 0:
                stats.unusual_rows.append({"file": rel(path), "line": line_number, "issue": "negative hold_time", "row": record})
            if hold_time is not None and hold_time >= 5:
                stats.unusual_rows.append({"file": rel(path), "line": line_number, "issue": "hold_time >= 5 seconds", "row": record})
            if None not in (hold_time, release_time, press_time):
                assert hold_time is not None
                assert release_time is not None
                assert press_time is not None
                if abs((release_time - press_time) - hold_time) > 0.0011:
                    stats.unusual_rows.append(
                        {"file": rel(path), "line": line_number, "issue": "release_time - press_time differs from hold_time", "row": record}
                    )
                if press_time <= 0 or release_time <= 0:
                    stats.unusual_rows.append(
                        {"file": rel(path), "line": line_number, "issue": "non-positive timestamp", "row": record}
                    )
                if release_time < press_time:
                    stats.unusual_rows.append(
                        {"file": rel(path), "line": line_number, "issue": "release_time before press_time", "row": record}
                    )
                if previous_press_time is not None and press_time < previous_press_time:
                    stats.unusual_rows.append(
                        {"file": rel(path), "line": line_number, "issue": "press_time decreased within file", "row": record}
                    )
                previous_press_time = press_time

        stats.per_file_event_counts[filename] = file_count
        stats.duplicate_rows_by_file[filename] = sum(count - 1 for count in file_hashes.values() if count > 1)

    stats.global_duplicate_rows = sum(count - 1 for count in global_hashes.values() if count > 1)
    return stats


def find_duplicate_tree_files() -> list[dict[str, str]]:
    duplicate_checks: list[dict[str, str]] = []
    neuro_root = PACKAGE_ROOT / "neuroQWERTY"
    for canonical in [PACKAGE_ROOT / "nqDataLoader.py", PACKAGE_ROOT / "readme.ipynb"]:
        duplicate = neuro_root / canonical.name
        if duplicate.exists():
            duplicate_checks.append(
                {
                    "canonical": rel(canonical),
                    "duplicate": rel(duplicate),
                    "same_sha256": str(sha256(canonical) == sha256(duplicate)),
                }
            )
    for dataset in DATASETS:
        canonical_root = PACKAGE_ROOT / dataset
        duplicate_root = neuro_root / dataset
        if not duplicate_root.exists():
            continue
        for canonical in sorted(canonical_root.rglob("*")):
            if not canonical.is_file():
                continue
            duplicate = duplicate_root / canonical.relative_to(canonical_root)
            if duplicate.exists():
                duplicate_checks.append(
                    {
                        "canonical": rel(canonical),
                        "duplicate": rel(duplicate),
                        "same_sha256": str(sha256(canonical) == sha256(duplicate)),
                    }
                )
    return duplicate_checks


def label_distribution(rows: list[dict[str, str]]) -> Counter[str]:
    return Counter(row["gt"] for row in rows)


def gt_schema(rows: list[dict[str, str]]) -> list[str]:
    return list(rows[0].keys()) if rows else []


def gt_missing(rows: list[dict[str, str]]) -> Counter[str]:
    missing: Counter[str] = Counter()
    for row in rows:
        for key, value in row.items():
            if value == "":
                missing[key] += 1
    return missing


def gt_duplicate_rows(rows: list[dict[str, str]]) -> int:
    hashes = Counter(json.dumps(row, sort_keys=True) for row in rows)
    return sum(count - 1 for count in hashes.values() if count > 1)


def gt_file_mismatches(name: str, config: dict[str, object], stats: DatasetStats) -> dict[str, list[str]]:
    gt_files = config["gt_files"]
    assert isinstance(gt_files, list)
    referenced = {
        row[column]
        for row in stats.gt_rows
        for column in gt_files
        if row.get(column)
    }
    actual = {path.name for path in stats.data_files}
    return {
        "referenced_not_found": sorted(referenced - actual),
        "actual_not_referenced": sorted(actual - referenced),
    }


def print_dataset(stats: DatasetStats, config: dict[str, object]) -> None:
    print(f"\n## {stats.name}")
    print(f"Data directory: {rel(config['data_dir'])}")
    print(f"Ground truth: {rel(config['gt_file'])}")
    print(f"Raw CSV files: {len(stats.data_files)}")
    print(f"Ground-truth rows / participants: {len(stats.gt_rows)}")
    print(f"Participant IDs from ground truth: {len(stats.participant_ids_from_gt)}")
    print(f"Participant IDs from filenames: {len(stats.participant_ids_from_filenames)}")
    print(f"Total raw event records: {stats.raw_event_count}")
    print(f"Raw schema: {RAW_COLUMNS} (headerless CSV; 4 fields per row)")
    print(f"Raw field-count distribution: {dict(stats.raw_schema_counts)}")
    print(f"Ground-truth schema: {gt_schema(stats.gt_rows)}")
    print(f"Label distribution: {dict(label_distribution(stats.gt_rows))}")
    print(f"Ground-truth missing values: {dict(gt_missing(stats.gt_rows))}")
    print(f"Raw missing values: {dict(stats.missing_by_column)}")
    print(f"Ground-truth duplicate rows: {gt_duplicate_rows(stats.gt_rows)}")
    print(f"Raw duplicate rows within files: {sum(stats.duplicate_rows_by_file.values())}")
    print(f"Raw duplicate rows across dataset: {stats.global_duplicate_rows}")
    print(f"Malformed raw rows: {len(stats.malformed_rows)}")
    print(f"Unusual raw rows: {len(stats.unusual_rows)}")
    print(f"File reference mismatches: {gt_file_mismatches(stats.name, config, stats)}")
    repeated = {pid: files for pid, files in stats.files_by_participant.items() if len(files) > 1}
    print(f"Participants with multiple raw files: {len(repeated)}")
    print(f"Top keys/buttons: {dict(stats.key_counts.most_common(15))}")
    if stats.malformed_rows:
        print(f"Malformed issue types: {dict(Counter(str(row['issue']) for row in stats.malformed_rows))}")
    if stats.unusual_rows:
        print(f"Unusual issue types: {dict(Counter(str(row['issue']) for row in stats.unusual_rows))}")
    print("All canonical raw files:")
    for path in stats.data_files:
        print(f"- {path.name}: {stats.per_file_event_counts[path.name]} records")
    print("Representative samples:")
    for filename in list(stats.sample_rows)[:3]:
        print(f"- {filename}: {json.dumps(stats.sample_rows[filename], ensure_ascii=False)}")
    if stats.malformed_rows:
        print(f"Malformed row examples: {json.dumps(stats.malformed_rows[:5], ensure_ascii=False)}")
    if stats.unusual_rows:
        print(f"Unusual row examples: {json.dumps(stats.unusual_rows[:5], ensure_ascii=False)}")


def print_package_structure() -> None:
    print("# NeuroTap Dataset Inspection")
    print(f"Package root: {rel(PACKAGE_ROOT)}")
    print("\n## Top-Level Package Entries")
    for path in sorted(PACKAGE_ROOT.iterdir()):
        kind = "directory" if path.is_dir() else "file"
        size = "" if path.is_dir() else f", {path.stat().st_size} bytes"
        print(f"- {path.name}: {kind}{size}")


def main() -> int:
    if not PACKAGE_ROOT.exists():
        print(f"Dataset package not found: {rel(PACKAGE_ROOT)}")
        return 1

    print_package_structure()

    print("\n## Duplicate Copy Check")
    duplicate_checks = find_duplicate_tree_files()
    same_count = sum(item["same_sha256"] == "True" for item in duplicate_checks)
    print(f"Compared files with neuroQWERTY duplicates: {len(duplicate_checks)}")
    print(f"Byte-identical duplicate matches: {same_count}")
    print(f"Non-identical duplicate matches: {len(duplicate_checks) - same_count}")
    for item in duplicate_checks[:8]:
        print(f"- {item['canonical']} == {item['duplicate']}: {item['same_sha256']}")

    for name, config in DATASETS.items():
        stats = inspect_dataset(name, config)
        print_dataset(stats, config)

    print("\n## Interpretation Notes")
    print("- Canonical counted data directories are MIT-CS1PD/data_MIT-CS1PD and MIT-CS2PD/data_MIT-CS2PD.")
    print("- neuroQWERTY contains duplicate package content and is not counted as additional observations.")
    print("- Raw CSV files are headerless. One row represents one recorded key/button action with key, hold time, release timestamp, and press timestamp.")
    print("- Dataset documentation reports acquisition temporal resolution as 3/0.28 mean/std milliseconds.")
    print("- The numeric raw timing values are represented in seconds; hold_time equals release_time - press_time within rounding tolerance.")
    print("- Ground-truth gt encodes CNT/control as False and PD as True.")
    print("- Do not split train/test by row; split by participant to avoid leakage across rows and CS1PD repeated sessions.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
