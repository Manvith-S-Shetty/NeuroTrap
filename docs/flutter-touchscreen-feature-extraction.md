# Flutter Touchscreen Event Collection & 27-Feature Extraction Architecture

## Overview

This document details the architecture and mathematical specification for raw touchscreen event collection and 27-feature vector extraction in Flutter (`mobile/lib/src/ml/`).

> [!IMPORTANT]
> **No Key Identity Contract**:
> Touchscreen typing events record **press timestamp (`pressTimeMs`)**, **release timestamp (`releaseTimeMs`)**, and **normalized touch pressure (`pressure`)**.
> **Key identities (character names) are NOT recorded or fabricated.** Keyboard-specific features (flight time, key-pair digraphs) do not exist in the touchscreen modality.

---

## Clean Layered Architecture

```
  [Raw Device Touch Events] (Listener / GestureRecognizer)
              ↓
  [TouchscreenEventCollector] (Streams & Buffers valid TouchscreenTouchEvent objects)
              ↓
  [TouchscreenFeatureExtractor] (Computes exact 27-numerical-feature vector)
              ↓
  [TouchscreenFeatureVector] (Immutable 27-feature object in canonical order)
              ↓
  [TouchscreenInferenceEngine] (Calculates z-score, logit score, & sigmoid P(PD))
              ↓
  [TouchscreenPrediction] ("PD" or "CNT")
```

---

## 27-Feature Contract & Mathematical Formulas

The feature extractor strictly reproduces the reference Python implementation ([ml/features/touchscreen_feature_extraction.py](file:///c:/huMANvith/NeuroTrap/ml/features/touchscreen_feature_extraction.py)):

### 1. Hold Time Features (`1–8`)
For each valid event \( e \):
\[
\text{hold}_e = \frac{\text{releaseTimeMs}_e - \text{pressTimeMs}_e}{1000.0} \quad (\text{seconds})
\]

The 8 statistical aggregates computed over all valid hold durations are:
- `hold_mean`, `hold_std`, `hold_variance`, `hold_median`, `hold_min`, `hold_max`, `hold_skewness`, `hold_kurtosis`.

### 2. Inter-Tap Interval Features (`9–16`)
Events are sorted chronologically by `pressTimeMs`. For consecutive events \( e_i \) and \( e_{i+1} \):
\[
\text{inter\_tap}_i = \frac{\text{pressTimeMs}_{i+1} - \text{pressTimeMs}_i}{1000.0} \quad (\text{seconds for } \text{diff} > 0)
\]

The 8 statistical aggregates computed over positive inter-tap intervals are:
- `inter_tap_mean`, `inter_tap_std`, `inter_tap_variance`, `inter_tap_median`, `inter_tap_min`, `inter_tap_max`, `inter_tap_skewness`, `inter_tap_kurtosis`.

### 3. Touch Pressure Features (`17–24`)
Raw recorded pressure values are processed directly without artificial clipping:
- `pressure_mean`, `pressure_std`, `pressure_variance`, `pressure_median`, `pressure_min`, `pressure_max`, `pressure_skewness`, `pressure_kurtosis`.

### 4. Activity Features (`25–27`)
- `event_count`: Number of valid chronological events \( N \).
- `recording_duration`: \( \frac{\max(\text{releaseTimeMs}) - \min(\text{pressTimeMs})}{1000.0} \) (seconds).
- `typing_rate`: \( \frac{\text{event\_count}}{\text{recording\_duration}} \) (taps/second).

---

## Statistical Aggregation & Small-Sample Behavior

Statistical calculations reproduce `pandas` and `scipy.stats` definitions:
- **Mean & Median**: Standard arithmetic mean and sample median.
- **Sample Variance & Std**: Unbiased sample variance with degrees of freedom \( ddof = 1 \) (requires \( N \ge 2 \); returns `NaN` if \( N < 2 \)).
- **Adjusted Fisher-Pearson Skewness**:
  \[
  g_1 = \frac{n}{(n-1)(n-2)} \sum_{i=1}^n \left(\frac{x_i - \bar{x}}{s}\right)^3 \quad (\text{requires } N \ge 3)
  \]
- **Adjusted Fisher Excess Kurtosis**:
  \[
  g_2 = \frac{n(n+1)}{(n-1)(n-2)(n-3)} \sum_{i=1}^n \left(\frac{x_i - \bar{x}}{s}\right)^4 - \frac{3(n-1)^2}{(n-2)(n-3)} \quad (\text{requires } N \ge 4)
  \]

---

## Device Pressure Availability & Rejection Policy

> [!CAUTION]
> **Strict Non-Fabrication Rule**:
> Pressure must come from the device's actual touch-pressure measurement (`double? pressure`). If pressure is unavailable on a device, NeuroTrap **does NOT fabricate, default, or silently substitute** any constant value (`0`, `1`, `0.5`, or mean values).
> 
> **Rationale**:
> The trained touchscreen ML model relies on 8 pressure-derived features (`pressure_mean`, `pressure_std`, `pressure_median`, etc.) with significant model coefficients. Substituting artificial constant values would create a severe distribution mismatch between training data and mobile inference.

When real touch pressure is unavailable (`pressure == null`):
- `TouchscreenTouchEvent.hasPressure` evaluates to `false`.
- `TouchscreenFeatureExtractor.extract()` explicitly fails by throwing a typed `TouchPressureUnavailableException`.
- The system safely halts touchscreen ML screening rather than generating uncalibrated ML predictions.

---

## Verification & Test Summary

1. **`flutter analyze`**: **No issues found!**
2. **`flutter test`**: **23 / 23 passed** (Includes full pipeline integration tests: `EventCollector -> Extractor -> FeatureVector -> InferenceEngine`).
3. **`pytest -q tests/test_touchscreen_feature_extraction.py`**: **11 / 11 passed**.
