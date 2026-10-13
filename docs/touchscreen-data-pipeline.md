# Touchscreen Data Pipeline & Preprocessing

This document describes the read-only touchscreen dataset loader, data structure, schema normalization, validation framework, and structural findings for the **i-PROGNOSIS DS2.5 Sub1 Keystroke Timing & Pressure Dataset** (`ml/data/raw/touch_datasets/touchscreen_typing_pd/`).

---

## 1. Dataset Overview & Structure

The dataset consists of touchscreen typing recordings collected from Parkinson's Disease (PD) patients and healthy Control (CNT) participants typing text excerpts on smartphone touchscreens.

### File Tree Hierarchy
```text
ml/data/raw/touch_datasets/touchscreen_typing_pd/
└── i-PROGNOSIS_DS2.5_Sub1_KeystrokeTimingPressureData/
    └── i-PROGNOSIS_DS2.5_Sub1_KeystrokeTimingPressureData/
        ├── Demographics_Clinical_Characteristics.xlsx
        ├── ReadMe.txt
        └── Data/
            ├── S01/
            ├── S02/
            ├── ...
            └── S33/
```

### Dataset Key Facts
* **Subjects**: 33 participants (`S01` through `S33`).
* **Ground-Truth Cohort Breakdown**:
  * **PD**: 18 participants
  * **Control (CNT)**: 15 participants
* **Recordings**: 338 total `.txt` files across all subjects (e.g. `S01_TEX01.txt`, `S05_TEX010.txt`, `S06_TEX1.txt`).
* **TEX Identifiers**: Filenames contain variable-digit excerpt tags (`TEX01`, `TEX1`, `TEX010`), requiring flexible regular expression parsing.
* **Recording Counts per Subject**: Ranges from 5 recordings (Subject `S16`) to 11 recordings per subject.

---

## 2. Raw Event Format

Each `.txt` recording file contains headerless line-delimited records in the following exact format:

$$\texttt{Press,}\langle \text{press\_timestamp} \rangle\texttt{,Release }\langle \text{release\_timestamp} \rangle\texttt{,}\langle \text{pressure} \rangle$$

### Representative Raw Event Lines
```text
Press,16144290,Release 16144380,0.575
Press,16156363,Release 16156478,0.38750002
Press,16157517,Release 16157625,0.5625
```

### Data Fields & Units
* **Press Timestamp (`press_time_ms`)**: Integer/float millisecond timestamp marking touchscreen press-down.
* **Release Timestamp (`release_time_ms`)**: Integer/float millisecond timestamp marking touchscreen finger lift.
* **Hold Time (`hold_time_ms`)**: Derived duration $t_{\text{release}} - t_{\text{press}}$ in milliseconds.
* **Pressure (`pressure`)**: Normalized touch contact pressure value in $[0.0, 1.0]$.

---

## 3. Label Mapping & Metadata Integration

Ground-truth clinical labels are stored in `Demographics_Clinical_Characteristics.xlsx` under the `Demographics_Clinical` sheet.

### Mapping Rule
* **Subject ID** (e.g. `1`, `2`, ..., `33`) $\longrightarrow$ Folder Name `S01`, `S02`, ..., `S33`.
* **Group Column**:
  * `"Control"` $\longrightarrow$ Normalized label `"CNT"`
  * `"PD"` $\longrightarrow$ Normalized label `"PD"`

Every subject directory `S01`...`S33` in `Data/` is strictly validated against the Excel metadata. Any missing subject entry or unsupported group string raises a `DatasetMappingError`.

---

## 4. Normalized Schema

The data loader (`ml/preprocessing/touchscreen/data_loader.py`) produces two primary DataFrames and one malformed line tracking DataFrame.

### Event-Level DataFrame (`events_df`)

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `dataset` | `str` | `"i-PROGNOSIS"` |
| `participant_id` | `str` | Subject identifier (e.g., `"S01"`, `"S33"`) |
| `recording_id` | `str` | Unique recording identifier (e.g., `"S01_TEX01"`) |
| `text_id` | `str` | Text excerpt tag (e.g., `"TEX01"`, `"TEX1"`, `"TEX010"`) |
| `raw_file` | `str` | Baseline file basename (e.g., `"S01_TEX01.txt"`) |
| `raw_path` | `str` | Absolute path to source `.txt` file |
| `row_number` | `int` | 1-based physical line index in the `.txt` file |
| `press_time_ms` | `float/int` | Touch press timestamp in milliseconds |
| `release_time_ms`| `float/int` | Touch release timestamp in milliseconds |
| `hold_time_ms` | `float/int` | Calculated hold duration ($t_{\text{release}} - t_{\text{press}}$) in ms |
| `pressure` | `float` | Normalized touch pressure in $[0.0, 1.0]$ |
| `label` | `str` | Ground-truth class (`"PD"` or `"CNT"`) |

### Recording-Level Summary DataFrame (`recordings_df`)
Stores one summary row per `.txt` file containing `total_physical_lines`, `blank_lines`, `malformed_lines`, and `valid_events`.

---

## 5. Validation Rules & Data Quality Framework

The validation module (`ml/preprocessing/touchscreen/validate_dataset.py`) performs automated quality checks without modifying or dropping raw events:

1. **Blank Lines**: Identifies empty or whitespace-only lines (recorded in file line statistics).
2. **Malformed Non-Blank Lines**: Captures non-blank lines failing the `Press,T1,Release T2,P` regex.
3. **Missing Metadata / Unsupported Labels**: Flags any recording without Excel metadata or with unmapped class labels.
4. **Missing / Non-Numeric Values**: Flags NaN or non-numeric timestamps/pressures.
5. **Negative Hold Duration**: Flags instances where `hold_time_ms < 0`.
6. **Release Before Press**: Flags instances where `release_time_ms < press_time_ms`.
7. **Pressure Out of Range**: Flags touch pressure values $< 0.0$ or $> 1.0$.
8. **Non-Monotonic Press Timestamps**: Flags instances within a recording where press timestamp $t_{\text{press}, i} < t_{\text{press}, i-1}$.
9. **Empty & Low-Event Recordings**: Flags files with zero valid events or fewer than 10 valid events.

### Validation Flags vs Automatic Data Removal
> [!IMPORTANT]
> The validation framework strictly **reports** suspicious records and quality anomalies. It does **not** automatically filter, clean, or discard records during loading. Preserving raw anomalies ensures full auditability before feature engineering decisions are made.

---

## 6. Known Structural Findings (Audit Results)

Running `ml/preprocessing/touchscreen/audit_dataset.py` against the real dataset reveals the following empirical metrics:

* **Subject Count**: 33 (18 PD, 15 CNT)
* **Recording Count**: 338 (179 PD, 159 CNT)
* **Total Physical Lines**: 26,686
* **Blank Lines**: 343
* **Valid Event Records**: 26,343
* **Malformed Non-Blank Lines**: 0
* **Mean Physical Lines per File**: 78.95
* **Events per Recording**: Min: 6, Max: 233, Mean: 77.94, Median: 65.00
* **Recordings per Subject**: Min: 5, Max: 11, Mean: 10.24, Median: 11.00

### Discovered Data Quality Anomalies
1. **Negative Hold Durations**: **0** events.
2. **Release Before Press**: **0** events.
3. **Malformed Non-Blank Lines**: **0** lines.
4. **Pressure Out of Range ($> 1.0$)**: **7** events across 6 recordings:
   * `S09_TEX01.txt` Line 40: `pressure = 1.0125`
   * `S09_TEX02.txt` Line 39: `pressure = 1.0250`
   * `S22_TEX11.txt` Line 28: `pressure = 1.0375`
   * `S24_TEX02.txt` Line 6 : `pressure = 1.0250`
   * `S24_TEX03.txt` Line 53: `pressure = 1.0125`
   * `S24_TEX04.txt` Line 8 : `pressure = 1.0375`
   * `S24_TEX07.txt` Line 10: `pressure = 1.0125`
5. **Non-Monotonic Press Timestamps**: **1** event:
   * `S03_TEX02.txt` Line 78: `press_time_ms = 22049344` following previous line press `22296325` (timestamp reset / out-of-order event).

---

## 7. Key Engineering & Methodological Considerations

### Absence of Key Identity
The i-PROGNOSIS dataset logs touch timing (`press_time_ms`, `release_time_ms`) and contact pressure (`pressure`), but **does not log actual key characters** (e.g. letter typed, backspace, spacebar).
* **Consequence**: Feature extraction cannot calculate key-specific trigram latencies or character-specific hold metrics. Features must be derived strictly from timing dynamics, inter-tap flight times, hold durations, and touch pressure distributions.
* **Prohibition**: Synthesizing or fabricating artificial key characters is strictly prohibited.

### Requirement for Participant-Level Splitting
Multiple text recordings originate from each subject (up to 11 recordings per subject).
* **Data Leakage Risk**: Performing a random row-level or recording-level split would place recordings from the same subject into both training and test sets, leaking participant-specific touch habits.
* **Mandatory Rule**: All downstream train/test or cross-validation splits must perform **participant-aware partitioning** (GroupKFold by `participant_id`), keeping all recordings of a given subject strictly isolated in either training or testing.
