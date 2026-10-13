# Touchscreen Recording-Level Feature Extraction

This document describes the recording-level feature extraction pipeline for the **i-PROGNOSIS touchscreen typing dataset** (`ml/features/touchscreen_feature_extraction.py`).

---

## 1. Unit of Analysis

The unit of analysis is **one touchscreen typing recording** (corresponding to one `.txt` excerpt file).

### Rationale
1. **Clinical Context**: Individual touch events (a single tap or hold) carry high variance and noise. Converting a multi-event typing session into a unified recording-level feature vector captures a subject's aggregate motor control dynamics (timing consistency, touch force distribution, typing rate).
2. **Tabular Machine Learning**: Standard classifiers (`LogisticRegression`, `RandomForestClassifier`, etc.) require fixed-length feature vectors per observation.
3. **Immutability & Safety**: Feature extraction operates strictly on DataFrame copies without modifying raw source files or altering raw line ordering in place.

---

## 2. Feature Schema (27 Numerical Features)

Each recording is converted into exactly **27 numerical features** organized into four functional groups:

```text
       +-----------------------------------------------------------------------+
       |               Touchscreen Feature Vector (27 Features)                 |
       +-----------------------------------------------------------------------+
           |                 |                     |                    |
           v                 v                     v                    v
    +--------------+  +-----------------+  +------------------+  +--------------------+
    | Hold-Time    |  | Inter-Tap Time  |  | Touch Pressure   |  | Recording Activity |
    | (8 Features) |  | (8 Features)    |  | (8 Features)     |  | (3 Features)       |
    +--------------+  +-----------------+  +------------------+  +--------------------+
```

### Exact Feature Names & Order

| Index | Feature Name | Description | Units |
| :---: | :--- | :--- | :--- |
| **1** | `hold_mean` | Mean touch hold/dwell duration | Seconds |
| **2** | `hold_std` | Standard deviation of hold durations | Seconds |
| **3** | `hold_variance` | Variance of hold durations | $\text{Seconds}^2$ |
| **4** | `hold_median` | Median touch hold duration | Seconds |
| **5** | `hold_min` | Minimum touch hold duration | Seconds |
| **6** | `hold_max` | Maximum touch hold duration | Seconds |
| **7** | `hold_skewness` | Adjusted Fisher-Pearson skewness of hold times | Unitless |
| **8** | `hold_kurtosis` | Fisher excess kurtosis of hold times | Unitless |
| **9** | `inter_tap_mean` | Mean inter-tap latency ($\text{press}_{i+1} - \text{press}_i$) | Seconds |
| **10** | `inter_tap_std` | Standard deviation of inter-tap latencies | Seconds |
| **11** | `inter_tap_variance` | Variance of inter-tap latencies | $\text{Seconds}^2$ |
| **12** | `inter_tap_median` | Median inter-tap latency | Seconds |
| **13** | `inter_tap_min` | Minimum inter-tap latency | Seconds |
| **14** | `inter_tap_max` | Maximum inter-tap latency | Seconds |
| **15** | `inter_tap_skewness` | Adjusted Fisher-Pearson skewness of inter-tap times | Unitless |
| **16** | `inter_tap_kurtosis` | Fisher excess kurtosis of inter-tap times | Unitless |
| **17** | `pressure_mean` | Mean normalized touch pressure | $[0.0, 1.0]$ |
| **18** | `pressure_std` | Standard deviation of touch pressure | Unitless |
| **19** | `pressure_variance` | Variance of touch pressure | Unitless |
| **20** | `pressure_median` | Median touch pressure | $[0.0, 1.0]$ |
| **21** | `pressure_min` | Minimum touch pressure | $[0.0, 1.0]$ |
| **22** | `pressure_max` | Maximum touch pressure | $[0.0, 1.0]$ |
| **23** | `pressure_skewness` | Adjusted Fisher-Pearson skewness of touch pressure | Unitless |
| **24** | `pressure_kurtosis` | Fisher excess kurtosis of touch pressure | Unitless |
| **25** | `event_count` | Count of valid chronological events in recording | Count |
| **26** | `recording_duration` | $\frac{\max(t_{\text{release}}) - \min(t_{\text{press}})}{1000.0}$ | Seconds |
| **27** | `typing_rate` | $\frac{\text{event\_count}}{\text{recording\_duration}}$ | Events / Sec |

---

## 3. Detailed Feature Group Specifications

### A. Hold-Time Features (`hold_*`)
* Derived from `hold_time_ms` converted to seconds ($t_{\text{hold, sec}} = t_{\text{hold, ms}} / 1000.0$).
* Computed on valid hold observations: $t_{\text{hold, ms}} \ge 0$ and $t_{\text{release}} \ge t_{\text{press}}$.

### B. Inter-Tap Timing Features (`inter_tap_*`)
* Calculated from a copied view sorted by $t_{\text{press}}$:
  $$\text{inter\_tap\_sec}_i = \frac{t_{\text{press}, i+1} - t_{\text{press}, i}}{1000.0}$$
* Named **`inter_tap`** rather than `flight_time` or `inter_key_time` because key identity is absent.
* Only strictly positive, finite time intervals ($\text{inter\_tap\_sec} > 0$) are retained.

### C. Touch Pressure Features (`pressure_*`)
* Computed on raw numeric pressure values.
* Values slightly exceeding $1.0$ (e.g. 7 recorded events with pressure up to $1.0375$) are **preserved as recorded without artificial clipping**.

### D. Recording-Level Features
* **`event_count`**: Total valid event count in recording.
* **`recording_duration`**: Time elapsed between earliest press-down and latest finger-lift ($\frac{\max(t_{\text{release}}) - \min(t_{\text{press}})}{1000.0}$).
* **`typing_rate`**: $\frac{\text{event\_count}}{\text{recording\_duration}}$. If duration is zero or invalid, returns `NaN` rather than infinity.

---

## 4. Handling of Edge Cases & Invalid Data

1. **Insufficient Observations for Higher-Order Statistics**:
   * Skewness (`pd.Series.skew()`) requires $N \ge 3$ observations. For $N < 3$, returns `NaN`.
   * Kurtosis (`pd.Series.kurt()`) requires $N \ge 4$ observations. For $N < 4$, returns `NaN`.
   * Standard deviation and variance require $N \ge 2$ observations. For $N < 2$, returns `NaN`.
   * Statistics are never invented or filled with zero fallbacks.
2. **Non-Monotonic Raw Ordering**:
   * Inter-tap calculation sorts a copied view by `press_time_ms` and `row_number` without altering the original DataFrame.
3. **Zero Duration Recordings**:
   * Returns `recording_duration = 0.0` and `typing_rate = NaN`.

---

## 5. Key Engineering & Methodological Distinctions

### Absence of Key Identity
The i-PROGNOSIS dataset logs touchscreen timing and contact force, but **does not capture key characters**.
* **Design Rule**: Key values are not fabricated. Features are named `inter_tap_*` rather than `flight_*` or `inter_key_*`.

### Separation of Metadata and Numerical Feature Matrix
The feature extraction API (`extract_all_touchscreen_features`) strictly separates identifiers from model inputs:
* `X`: $(N \times 27)$ DataFrame containing only numerical predictive features.
* `metadata`: $(N \times 6)$ DataFrame containing `participant_id`, `recording_id`, `text_id`, `label`, `raw_file`, `raw_path`.
* `y`: Series containing ground-truth labels (`PD` vs `CNT`).

### Mandatory Participant-Level Partitioning (Future Modeling)
Multiple recordings originate from each subject (5 to 11 recordings per participant).
* **Data Leakage Prohibition**: Downstream train/test splits must use **GroupKFold** grouped on `participant_id`. No participant may appear in both training and evaluation sets.
