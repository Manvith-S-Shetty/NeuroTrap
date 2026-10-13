# Touchscreen Feature Analysis

This document presents the exploratory feature analysis for the 27 numerical touchscreen features extracted from the **i-PROGNOSIS touchscreen dataset** (`ml/features/analyze_touchscreen_features.py`).

> [!IMPORTANT]
> **Exploratory Analysis Safeguard**: This document and analysis are strictly descriptive. No features were removed, selected, scaled, or transformed based on these findings. Downstream model development will perform participant-level splitting before any feature selection or hyperparameter tuning.

---

## 1. Data Integrity & Dataset Verification

Running `check_data_integrity` confirms the empirical structure of the extracted feature table:

* **Feature Matrix ($X$) Shape**: `(338, 27)` (Exactly 338 recordings, 27 numerical features)
* **Metadata Table Shape**: `(338, 6)` (`participant_id`, `recording_id`, `text_id`, `label`, `raw_file`, `raw_path`)
* **Target Labels ($y$)**: `PD`: 179 recordings, `CNT`: 159 recordings
* **Participants Total**: **33** (`PD`: 18 participants, `CNT`: 15 participants)
* **Feature Column Safety**: Zero metadata or string columns present in numerical matrix $X$. Participant IDs stored separately.

---

## 2. Missing & Infinite Values (Analysis 2)

| Statistic | Result |
| :--- | :--- |
| **NaN Count** | **0** across all 27 features |
| **+Infinity Count** | **0** across all 27 features |
| **-Infinity Count** | **0** across all 27 features |
| **Total Invalid Values** | **0** |

All 338 recordings contain complete, finite numerical data across all 27 extracted features.

---

## 3. High Correlation Analysis (Analysis 4)

Pearson correlation coefficients ($r$) were computed across all 27 numerical features ($27 \times 27$ correlation matrix).

### Correlation Pair Counts
* **$|r| \ge 0.90$**: **9 pairs**
* **$|r| \ge 0.95$**: **5 pairs**
* **$|r| \ge 0.99$**: **2 pairs**

### High Correlation Pairs Table

| Feature 1 | Feature 2 | Pearson $r$ | $|r|$ |
| :--- | :--- | :---: | :---: |
| `hold_mean` | `hold_median` | `+0.9905` | `0.9905` |
| `pressure_std` | `pressure_variance` | `+0.9901` | `0.9901` |
| `hold_std` | `hold_max` | `+0.9796` | `0.9796` |
| `pressure_mean` | `pressure_median` | `+0.9765` | `0.9765` |
| `inter_tap_skewness` | `inter_tap_kurtosis` | `+0.9524` | `0.9524` |
| `inter_tap_mean` | `inter_tap_median` | `+0.9478` | `0.9478` |
| `inter_tap_std` | `inter_tap_max` | `+0.9254` | `0.9254` |
| `inter_tap_std` | `inter_tap_variance` | `+0.9031` | `0.9031` |
| `hold_skewness` | `hold_kurtosis` | `+0.9007` | `0.9007` |

---

## 4. PD vs Control Exploratory Comparison (Analysis 5)

Standardized mean differences (Cohen's $d$) were computed between PD ($n=179$) and Control ($n=159$) groups:

$$d = \frac{m_{\text{pd}} - m_{\text{cnt}}}{s_{\text{pooled}}}$$

### Top Feature Differences by Effect Size ($|d|$)

| Feature | Control Mean | PD Mean | Mean Diff ($m_{\text{pd}} - m_{\text{cnt}}$) | Cohen's $d$ | Signal Direction |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `hold_mean` | $0.0945\text{s}$ | $0.1644\text{s}$ | $+0.0699\text{s}$ | **$+1.4394$** | Longer hold time in PD |
| `hold_median` | $0.0938\text{s}$ | $0.1593\text{s}$ | $+0.0655\text{s}$ | **$+1.4304$** | Longer median hold in PD |
| `typing_rate` | $1.4859\text{ev/s}$ | $0.8422\text{ev/s}$ | $-0.6437\text{ev/s}$ | **$-1.3423$** | Slower typing rate in PD |
| `inter_tap_median` | $0.5962\text{s}$ | $1.0653\text{s}$ | $+0.4691\text{s}$ | **$+1.2535$** | Longer inter-tap pause in PD |
| `pressure_median` | $0.6195$ | $0.5172$ | $-0.1023$ | **$-1.2333$** | Lower touch pressure in PD |
| `pressure_mean` | $0.5995$ | $0.5163$ | $-0.0833$ | **$-1.1978$** | Lower mean pressure in PD |
| `inter_tap_mean` | $0.8192\text{s}$ | $1.4578\text{s}$ | $+0.6385\text{s}$ | **$+1.1129$** | Longer inter-tap mean in PD |
| `hold_min` | $0.0464\text{s}$ | $0.0796\text{s}$ | $+0.0331\text{s}$ | **$+0.8676$** | Higher minimum hold in PD |
| `pressure_max` | $0.8016$ | $0.7253$ | $-0.0762$ | **$-0.8253$** | Lower maximum pressure in PD |
| `hold_max` | $0.1527\text{s}$ | $0.3686\text{s}$ | $+0.2159\text{s}$ | **$+0.6912$** | Higher maximum hold in PD |

---

## 5. Outlier Analysis (Analysis 6)

Using the $1.5 \times \text{IQR}$ rule ($\text{lower} = Q_1 - 1.5 \times \text{IQR}$, $\text{upper} = Q_3 + 1.5 \times \text{IQR}$):

### Features with Highest Outlier Counts

| Feature | Outlier Count | Outlier % | Lower Bound | Upper Bound |
| :--- | :---: | :---: | :---: | :---: |
| `hold_kurtosis` | 45 | 13.31% | $-2.0528$ | $4.2952$ |
| `hold_variance` | 38 | 11.24% | $-0.0028$ | $0.0061$ |
| `hold_skewness` | 38 | 11.24% | $-1.1578$ | $2.4410$ |
| `inter_tap_variance` | 33 | 9.76% | $-0.3242$ | $0.6277$ |
| `recording_duration` | 27 | 7.99% | $11.0370\text{s}$ | $114.7730\text{s}$ |
| `inter_tap_kurtosis` | 27 | 7.99% | $-1.6219$ | $3.5786$ |
| `hold_std` | 25 | 7.40% | $-0.0322\text{s}$ | $0.0782\text{s}$ |
| `inter_tap_max` | 24 | 7.10% | $-0.7712\text{s}$ | $3.5907\text{s}$ |
| `event_count` | 24 | 7.10% | $-15.5000$ | $132.5000$ |

> [!NOTE]
> Outliers reflect natural variation in typing behavior (e.g., occasional long pauses or hesitations). Outlier records are **preserved as recorded** and will not be removed or clipped.

---

## 6. Generated Analysis Plot Artifacts (Analysis 7)

Plots generated and saved under `ml/features/touchscreen_analysis_outputs/`:

1. `hold_features_distributions.png`: Histograms with KDE for all 8 hold-time features.
2. `inter_tap_features_distributions.png`: Histograms with KDE for all 8 inter-tap features.
3. `pressure_features_distributions.png`: Histograms with KDE for all 8 touch pressure features.
4. `activity_features_distributions.png`: Histograms with KDE for `event_count`, `recording_duration`, and `typing_rate`.
5. `correlation_heatmap.png`: Full $27 \times 27$ Pearson correlation matrix heatmap.
6. `pd_vs_cnt_boxplots.png`: Standalone boxplots comparing PD vs Control across representative feature groups (`hold_mean`, `inter_tap_mean`, `pressure_mean`, `typing_rate`).

---

## 7. Methodological Safeguards

* **No Feature Removal**: All 27 features remain in the feature set.
* **No Preprocessing / Scaling**: `StandardScaler` or normalization will be fitted exclusively on development training folds.
* **Participant Isolation**: Downstream splits will perform participant-aware partitioning (`GroupKFold` on `participant_id`) to ensure zero test-set leakage.
