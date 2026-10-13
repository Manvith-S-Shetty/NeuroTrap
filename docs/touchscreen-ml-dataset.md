# Touchscreen ML Dataset & Participant-Aware Splitting

## Overview

This module (`ml/training/touchscreen_dataset.py`) defines the clean, ML-ready dataset representation and participant-aware splitting methodology for NeuroTap's touchscreen typing modality.

It builds upon the 27 numerical features extracted by `ml/features/touchscreen_feature_extraction.py` from raw touchscreen recordings.

---

## Dataset Schema

### Feature Matrix $X$ (27 Features)

The feature matrix $X$ contains exactly the 27 numerical features across four domain groups:

1. **Hold Time Features (8)**:
   - `hold_mean`, `hold_std`, `hold_variance`, `hold_median`, `hold_min`, `hold_max`, `hold_skewness`, `hold_kurtosis`
2. **Inter-Tap Interval Features (8)**:
   - `inter_tap_mean`, `inter_tap_std`, `inter_tap_variance`, `inter_tap_median`, `inter_tap_min`, `inter_tap_max`, `inter_tap_skewness`, `inter_tap_kurtosis`
3. **Touch Pressure Features (8)**:
   - `pressure_mean`, `pressure_std`, `pressure_variance`, `pressure_median`, `pressure_min`, `pressure_max`, `pressure_skewness`, `pressure_kurtosis`
4. **Recording-Level Activity Features (3)**:
   - `event_count`, `recording_duration`, `typing_rate`

### Target Vector $y$
- Target label column: `label`
- Allowed values: `"PD"` (Parkinson's Disease) and `"CNT"` (Control)

### Separate Metadata DataFrame
- `participant_id`: Unique identifier for each participant
- `recording_id`: Unique identifier for each touchscreen recording
- `text_id`: Text prompt identifier (e.g. `TXT_1`)
- `raw_file`: Filename of raw recording
- `raw_path`: Full filesystem path to raw file
- `label`: Duplicate copy kept in metadata for auditing

> [!IMPORTANT]
> The metadata DataFrame is strictly separated from the feature matrix $X$ to prevent identifier leakage into ML algorithms.

---

## Data Validation & Hygiene Rules

Before ML separation or splitting, the dataset must pass all data integrity checks:
1. **Feature Column Integrity**: All 27 numerical feature columns exist.
2. **Numeric Types**: Every feature column is numeric (`float64` or `int64`).
3. **No Missing Values**: 0 NaN values across feature matrix $X$ and target vector $y$.
4. **No Infinite Values**: 0 infinite values (`+inf` or `-inf`) in feature matrix $X$.
5. **Class Label Whitelist**: Labels are strictly `"PD"` or `"CNT"`.
6. **Unique Recording IDs**: No duplicate `recording_id` values.
7. **Participant Label Consistency**: Each participant belongs to exactly one label (`PD` or `CNT`) across all their recordings.
8. **Participant Single Ownership**: Each recording belongs to exactly one participant.

> [!NOTE]
> All 27 features are retained for ML readiness. No features were dropped based on previous correlation analysis.

---

## Participant-Aware Splitting Methodology

To evaluate model performance without data leakage across multiple recordings from the same subject, splitting is performed **strictly by participant**, never by recording.

### Strategy
- Target split ratio: ~80% development set, ~20% final-test set.
- Method: Stratified participant sampling (`random_state=42`, `stratify=True`).
- Stratification balances PD vs. CNT subject counts between development and final-test sets.

### Empirical Split Results (Real Touchscreen Dataset)

| Metric | Total Dataset | Development Set (80%) | Final-Test Set (20%) |
| :--- | :---: | :---: | :---: |
| **Total Recordings** | 338 | 269 | 69 |
| **Total Participants** | 33 | 26 | 7 |
| **PD Participants** | 18 | 14 | 4 |
| **CNT Participants** | 15 | 12 | 3 |
| **PD Recordings** | 179 | 142 | 37 |
| **CNT Recordings** | 159 | 127 | 32 |
| **Participant Overlap** | — | **0** | **0** |
| **Feature Matrix Shape ($X$)** | `(338, 27)` | `(269, 27)` | `(69, 27)` |

> [!WARNING]
> **Locked Final-Test Set Policy**:
> The 7 final-test participants (69 recordings) are **LOCKED**. They must NOT be used for feature selection, scaling parameter estimation, outlier removal, cross-validation, hyperparameter tuning, or model selection.

---

## Strict Data-Leakage Rule

The dataset module `ml/training/touchscreen_dataset.py` does **NOT**:
- Scale features (e.g., StandardScaler, MinMaxScaler)
- Normalize or transform distributions
- Impute missing values
- Select features or drop correlated features
- Remove outliers
- Tune hyperparameters
- Train models

All data transformations, feature scaling, and feature selection must be learned exclusively within training folds during cross-validation.

---

## Python API Usage

```python
from ml.training.touchscreen_dataset import (
    build_touchscreen_ml_dataset,
    split_touchscreen_by_participant,
)

# 1. Build ML dataset (separates X, y, metadata)
dataset = build_touchscreen_ml_dataset()

print("X shape:", dataset.X.shape)        # (338, 27)
print("y shape:", dataset.y.shape)        # (338,)
print("Metadata shape:", dataset.metadata.shape)  # (338, 6)

# 2. Perform reproducible participant-aware split (random_state=42)
split = split_touchscreen_by_participant(dataset, test_size=0.2, random_state=42)

print("Dev recordings:", len(split.X_dev))         # 269
print("Dev participants:", len(split.dev_participants)) # 26
print("Test recordings:", len(split.X_test))       # 69
print("Test participants:", len(split.final_test_participants)) # 7

# Assert zero participant overlap
assert split.dev_participants.isdisjoint(split.final_test_participants)
```
