# NeuroTap Dataset Investigation

This is a read-only investigation of `ml/data/raw/neuroqwerty-mit-csxpd-dataset-1.0.0/`. No raw dataset files were modified, renamed, cleaned, copied, or used for model training.

## 1. Dataset Package

Top-level package entries:

- `MIT-CS1PD/`
- `MIT-CS2PD/`
- `neuroQWERTY/`
- `nqDataLoader.py`
- `readme.ipynb`
- `SHA256SUMS.txt`

The package documentation in `readme.ipynb` describes two datasets from the neuroQWERTY Parkinson's typing study:

- `MIT-CS1PD`: early Parkinson's disease dataset.
- `MIT-CS2PD`: de-novo Parkinson's disease dataset.

The official loader in `nqDataLoader.py` treats the raw PD data files as headerless CSV files and loads positional fields as key, hold time, release timestamp, and press timestamp.

## 2. Directory Structure

Relevant canonical structure:

```text
ml/data/raw/neuroqwerty-mit-csxpd-dataset-1.0.0/
+-- MIT-CS1PD/
|   +-- GT_DataPD_MIT-CS1PD.csv
|   +-- data_MIT-CS1PD/
|       +-- 62 raw CSV files
+-- MIT-CS2PD/
|   +-- GT_DataPD_MIT-CS2PD.csv
|   +-- data_MIT-CS2PD/
|       +-- 54 raw CSV files
+-- neuroQWERTY/
|   +-- MIT-CS1PD/
|   +-- MIT-CS2PD/
|   +-- nqDataLoader.py
|   +-- readme.ipynb
+-- nqDataLoader.py
+-- readme.ipynb
+-- SHA256SUMS.txt
```

## 3. Canonical Data vs Duplicates

Canonical counted data directories:

- `MIT-CS1PD/data_MIT-CS1PD/`
- `MIT-CS2PD/data_MIT-CS2PD/`

`neuroQWERTY/` contains duplicate package content. The inspection script compared 120 files between the top-level package and the nested `neuroQWERTY/` copy. All 120 matched by SHA-256 hash, with 0 non-identical matches. The nested `neuroQWERTY/` copies are therefore not counted as additional observations.

## 4. MIT-CS1PD

Canonical raw data directory: `MIT-CS1PD/data_MIT-CS1PD/`

Ground-truth file: `MIT-CS1PD/GT_DataPD_MIT-CS1PD.csv`

Raw CSV files: 62

Ground-truth rows / participants: 31

Participant IDs from ground truth: 31

Participant IDs from filenames: 31

Total raw event records: 92,632

Ground-truth schema:

```text
pID, gt, updrs108, afTap, sTap, nqScore, typingSpeed, file_1, file_2
```

Raw CSV schema:

```text
key, hold_time, release_time, press_time
```

The raw CSV files are headerless; these names are assigned from the official loader and observed positional meaning.

Representative raw records:

```csv
"Shift_L",0.5609,2.6035,2.0426
"d",0.0895,2.6979,2.6084
"e",0.0954,2.9675,2.8721
```

All canonical MIT-CS1PD raw files:

```text
1401114972.068_001_014.csv
1401117235.067_001_014.csv
1401978395.067_003_014.csv
1401980765.068_003_014.csv
1402930351.011_001_014.csv
1402932300.060_001_014.csv
1403706430.011_003_014.csv
1403708258.060_003_014.csv
1403792224.074_001_014.csv
1403795516.072_001_014.csv
1403797570.077_001_014.csv
1404311419.070_001_014.csv
1404313658.081_001_014.csv
1404743687.070_003_014.csv
1404745424.081_003_014.csv
1404747197.074_003_014.csv
1404819923.080_001_014.csv
1404828566.083_001_014.csv
1404830509.082_001_014.csv
1404832986.072_003_014.csv
1404834530.077_003_014.csv
1407163319.084_001_014.csv
1407165254.085_001_014.csv
1407421738.075_001_014.csv
1407423449.076_001_014.csv
1407425493.082_003_014.csv
1407427375.083_003_014.csv
1408029544.097_001_014.csv
1408372533.088_001_014.csv
1408374282.089_001_014.csv
1408376132.093_001_014.csv
1408378153.094_001_014.csv
1408630965.080_003_014.csv
1408633211.097_003_014.csv
1408976968.086_001_014.csv
1408979136.087_001_014.csv
1408980899.085_003_014.csv
1408982860.084_003_014.csv
1409235978.088_003_014.csv
1409238087.089_003_014.csv
1409754834.076_003_014.csv
1409756763.075_003_014.csv
1410272175.092_001_014.csv
1410274781.086_003_014.csv
1410276815.087_003_014.csv
1410446321.071_001_014.csv
1410448218.095_001_014.csv
1410450204.078_001_014.csv
1410451895.079_001_014.csv
1411655869.071_003_014.csv
1411657572.095_003_014.csv
1412086661.092_003_014.csv
1412088758.078_003_014.csv
1412090543.079_003_014.csv
1412692310.094_003_014.csv
1412694503.093_003_014.csv
1418744709.098_001_014.csv
1418746466.099_001_014.csv
1418748996.073_001_014.csv
1419866851.098_003_014.csv
1419868806.099_003_014.csv
1419870664.073_003_014.csv
```

## 5. MIT-CS2PD

Canonical raw data directory: `MIT-CS2PD/data_MIT-CS2PD/`

Ground-truth file: `MIT-CS2PD/GT_DataPD_MIT-CS2PD.csv`

Raw CSV files: 54

Ground-truth rows / participants: 54

Participant IDs from ground truth: 54

Participant IDs from filenames: 54

Total raw event records: 84,881

Ground-truth schema:

```text
pID, gt, updrs108, afTap, sTap, nqScore, typingSpeed, file_1
```

Raw CSV schema:

```text
key, hold_time, release_time, press_time
```

Representative raw records:

```csv
"space",0.3179,1.0920,0.7741
"q",0.1892,2.4189,2.2297
"u",0.1641,2.7386,2.5745
```

All canonical MIT-CS2PD raw files:

```text
1424946827.1000_001_014.csv
1426676689.1002_001_014.csv
1427279751.1001_001_014.csv
1429091219.1006_001_014.csv
1429176842.1008_001_014.csv
1429866367.1004_001_014.csv
1430134526.1005_001_014.csv
1435046994.1010_001_014.csv
1436192511.1009_001_014.csv
1436195831.1011_001_014.csv
1437580607.1014_001_014.csv
1437583257.1012_001_014.csv
1437670749.1016_001_014.csv
1440577460.1022_001_014.csv
1441110114.1013_001_014.csv
1441279015.1015_001_014.csv
1441893019.1017_001_014.csv
1442299892.1019_001_014.csv
1442506646.1024_001_014.csv
1443090933.1028_001_014.csv
1443094060.1030_001_014.csv
1443534559.1021_001_014.csv
1444035286.1020_001_014.csv
1444400554.1023_001_014.csv
1444893029.1025_001_014.csv
1445956280.1029_001_014.csv
1446028987.1032_001_014.csv
1446626569.1034_001_014.csv
1449158503.1031_001_014.csv
1449842818.1033_001_014.csv
1451987802.1042_001_014.csv
1452598928.1035_001_014.csv
1453990802.1037_001_014.csv
1454061159.1050_001_014.csv
1456139318.1052_001_014.csv
1456752658.1041_001_014.csv
1457345994.1056_001_014.csv
1457347484.1039_001_014.csv
1458555711.1043_001_014.csv
1458557031.1062_001_014.csv
1458723488.1064_001_014.csv
1459162558.1047_001_014.csv
1459181723.1045_001_014.csv
1460104760.1066_001_014.csv
1460559248.1068_001_014.csv
1461063976.1049_001_014.csv
1461082447.1053_001_014.csv
1461084495.1051_001_014.csv
1461089132.1057_001_014.csv
1461091652.1055_001_014.csv
1461677130.1059_001_014.csv
1463130212.1070_001_014.csv
1463172280.1061_001_014.csv
1463511198.1063_001_014.csv
```

## 6. Ground Truth / Labels

The ground-truth `gt` column encodes the two classes exactly as documented:

- `False`: CNT / healthy control
- `True`: PD / Parkinson's disease group

MIT-CS1PD class distribution:

- Participants: `True` 18, `False` 13
- Raw files: `True` 36, `False` 26
- Raw event records mapped through ground truth: `True` 50,480, `False` 42,152

MIT-CS2PD class distribution:

- Participants: `True` 24, `False` 30
- Raw files: `True` 24, `False` 30
- Raw event records mapped through ground truth: `True` 34,925, `False` 49,956

The ground-truth files map participants to raw data files:

- MIT-CS1PD uses `file_1` and `file_2`; both columns map to files in `data_MIT-CS1PD/`.
- MIT-CS2PD uses `file_1`; it maps to files in `data_MIT-CS2PD/`.

No ground-truth file references were missing from the canonical raw directories, and no canonical raw files were unreferenced by ground truth.

## 7. Raw Record Structure

One raw CSV row represents one recorded key/button action. Each raw row has exactly 4 fields across all canonical raw files:

```text
field 1: key/button name
field 2: hold/dwell duration
field 3: release timestamp
field 4: press timestamp
```

The official loader confirms this positional interpretation:

- `dataKeys = data['f0']`
- `dataHT = data['f1']`
- `dataTimeStart = data['f3']`
- `dataTimeEnd = data['f2']`

There is no separate event-type column such as `press` or `release`. Press and release are represented by separate timestamp fields in the same row.

Observed key/button values include ordinary letters, `space`, `BackSpace`, `Shift_L`, punctuation names such as `parenright`, and other keyboard/meta names. Mouse-related keys are handled by the official loader's filtering patterns, but the current investigation did not perform filtering.

## 8. Timestamp Representation

Timestamp-related raw fields:

- `press_time`
- `release_time`
- `hold_time`

Timing values are decimal numeric values represented in seconds in the raw CSV files. This is supported by the observed values and by the relationship:

```text
hold_time ~= release_time - press_time
```

The package documentation reports keyboard acquisition temporal resolution as mean/std `3/0.28` milliseconds. This describes acquisition resolution, not the raw CSV unit.

The filename prefix is also a timestamp-like value. Example:

```text
1402932300.060_001_014.csv
```

The official loader parses filenames as:

```text
timestamp.pID_repID_expID.csv
```

It converts the integer `timestamp` portion with `datetime.datetime.fromtimestamp(int(timeStamp))`, so the filename timestamp is Unix epoch seconds.

## 9. Participant and Session Structure

Participant identifiers appear in:

- Ground-truth column `pID`.
- Raw filenames as the middle numeric component after the dot.

Filename structure:

```text
timestamp.pID_repID_expID.csv
```

Examples:

- `1402932300.060_001_014.csv`: participant `60`, repetition/session `001`, experiment `014`.
- `1403708258.060_003_014.csv`: participant `60`, repetition/session `003`, experiment `014`.
- `1424946827.1000_001_014.csv`: participant `1000`, repetition/session `001`, experiment `014`.

MIT-CS1PD has 31 participants and 62 raw files. Every participant has two raw files, represented in ground truth as `file_1` and `file_2`. The observed repetition IDs are `001` and `003`.

MIT-CS2PD has 54 participants and 54 raw files. Every participant has one raw file, represented in ground truth as `file_1`. The observed repetition ID is `001`.

## 10. Dataset Statistics

Combined canonical raw data only:

- Participants: 85
- Raw session/data files: 116
- Raw event records: 177,513
- Canonical ground-truth files: 2

MIT-CS1PD:

- Participants: 31
- Raw session/data files: 62
- Raw event records: 92,632
- Raw schema field-count distribution: 92,632 rows with 4 fields
- Participants with multiple raw files: 31

MIT-CS2PD:

- Participants: 54
- Raw session/data files: 54
- Raw event records: 84,881
- Raw schema field-count distribution: 84,881 rows with 4 fields
- Participants with multiple raw files: 0

## 11. Missing Values and Duplicates

MIT-CS1PD:

- Ground-truth missing values: `afTap` has 9 missing values.
- Raw missing values: `key` has 8 missing values.
- Ground-truth duplicate rows: 0
- Raw duplicate rows within files: 0
- Raw duplicate rows across canonical MIT-CS1PD: 0
- Malformed raw rows: 0

MIT-CS2PD:

- Ground-truth missing values: none detected.
- Raw missing values: `key` has 1 missing value.
- Ground-truth duplicate rows: 0
- Raw duplicate rows within files: 0
- Raw duplicate rows across canonical MIT-CS2PD: 0
- Malformed raw rows: 0

No rows with the wrong number of CSV fields were detected in either canonical dataset.

## 12. Data Quality Issues

The raw data contains unusual timing records that matter for later preprocessing decisions. These were reported but not cleaned.

MIT-CS1PD unusual raw rows:

- Total unusual row flags: 2,557
- `hold_time >= 5 seconds`: 480
- `non-positive timestamp`: 341
- `press_time decreased within file`: 1,736

MIT-CS2PD unusual raw rows:

- Total unusual row flags: 2,941
- `press_time decreased within file`: 1,768
- `hold_time >= 5 seconds`: 693
- `non-positive timestamp`: 478
- `negative hold_time`: 1
- `release_time before press_time`: 1

Representative unusual records:

```csv
"e",1401114972.0134,0.0652,-1401114971.9481
"s",1426676688.8819,0.0352,-1426676688.8467
```

These records appear to preserve the arithmetic relationship between hold time, release time, and press time, but contain non-positive press timestamps and very large hold durations. The official loader's `sanityCheck` removes records with non-positive timestamps, negative hold times, hold times `>= 5`, and non-consecutive press-time order. That filtering behavior is documented here for interpretation only; no filtering was applied in this investigation.

Some press times decrease within a file. In the official loader, non-consecutive start times are considered bad labels. Later feature extraction should explicitly decide whether to preserve, filter, or otherwise handle such records.

## 13. Data Leakage Considerations

Do not split train/test by raw row. Rows from the same typing session are highly dependent.

Do not split train/test by file alone for MIT-CS1PD. Every CS1PD participant has two files, so file-level splitting can place one session from a participant in train and the other in test.

Use participant-level splitting as the minimum leakage-safe split unit.

Do not count the nested `neuroQWERTY/` copy as additional data. It is byte-identical duplicate content and would duplicate observations.

Ground-truth columns such as `updrs108`, `afTap`, `sTap`, `nqScore`, and `typingSpeed` are metadata/clinical or derived measures. They should not be treated as raw typing-event inputs for a NeuroTap prototype unless a future task explicitly defines that modeling setup. `nqScore` and `typingSpeed` in particular may encode information derived from typing behavior and could create leakage or circular evaluation if mixed incorrectly with raw-event features.

## 14. Implications for NeuroTap

The raw dataset is suitable for a later typing-dynamics feature extraction stage, but preprocessing decisions must be explicit and reproducible.

Important implications:

- Treat `MIT-CS1PD/data_MIT-CS1PD/` and `MIT-CS2PD/data_MIT-CS2PD/` as canonical raw event sources.
- Treat one row as one key/button action with key, hold time, release time, and press time.
- Use `pID` and filename-derived participant IDs to group records by participant.
- Use participant-level splits to avoid leakage.
- Handle CS1PD as repeated-session data.
- Do not use `neuroQWERTY/` nested copies as extra observations.
- Decide later how to handle non-positive timestamps, long holds, negative hold time, release-before-press, missing keys, and non-monotonic press times.
- Keep NeuroTap output framed as a non-diagnostic screening/risk indicator; no medical claims are made from this investigation.

## 15. Loading and Validation Design

The read-only loading layer is implemented in `ml/preprocessing/data_loader.py`. It locates only the canonical top-level dataset directories:

- `ml/data/raw/neuroqwerty-mit-csxpd-dataset-1.0.0/MIT-CS1PD/`
- `ml/data/raw/neuroqwerty-mit-csxpd-dataset-1.0.0/MIT-CS2PD/`

It intentionally ignores the nested `neuroQWERTY/` duplicate tree so duplicate observations are not loaded.

Normalized row-level schema:

```text
dataset
participant_id
recording_id
session_id
repetition_id
experiment_id
recording_timestamp_unix
raw_file
raw_path
source_file_column
row_number
key
hold_time
release_time
press_time
gt
label
```

Participant mapping uses the official ground-truth file references and the documented filename structure:

```text
timestamp.pID_repID_expID.csv
```

For each ground-truth row, the loader follows `file_1` and `file_2` for MIT-CS1PD and `file_1` for MIT-CS2PD. It verifies that each referenced raw file exists and that the filename-derived participant ID matches the ground-truth `pID`. If a file is missing, unreferenced, malformed, or mapped to a different participant ID, the loader raises a `DatasetMappingError` instead of guessing.

Label mapping uses the documented `gt` encoding:

- `gt=False` maps to label `CNT`.
- `gt=True` maps to label `PD`.

Session/recording identity is preserved from the filename. `recording_id` is the raw filename stem. `session_id` is currently the filename repetition ID because the dataset documentation and loader expose repetitions through `repID`; MIT-CS1PD uses repeated sessions through `file_1` and `file_2`, while MIT-CS2PD has one recording per participant.

The validation layer is implemented in `ml/preprocessing/validate_dataset.py`. It reports suspicious records without removing or rewriting them. Validation rules currently report:

- missing key values
- missing timing values
- negative hold durations
- `release_time < press_time`
- non-positive press or release timestamps
- non-monotonic press times within each recording
- suspiciously long hold durations using the official loader's `>= 5` second sanity-check threshold
- duplicate raw event rows within a recording

Suspicious records are preserved because downstream preprocessing and feature extraction decisions have not been defined yet. Keeping the original rows visible prevents silent data loss, makes later filtering decisions auditable, and avoids accidentally hiding dataset quirks that could affect model evaluation or participant-level splitting.
