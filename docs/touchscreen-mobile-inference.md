# Touchscreen Mobile Inference Architecture & Integration Design

## Overview

This document describes the portable, lightweight mobile inference architecture for the NeuroTap touchscreen modality.

The selected touchscreen model (**Logistic Regression**, `C=0.001`, `class_weight=None`, `random_state=42`) is mathematically linear. It requires **zero third-party ML runtime frameworks** (such as TFLite or ONNX) on mobile devices. Inference can be computed natively in Dart with 100% precision equivalence.

---

## Model Parameters & Portable Specification

The complete model is represented as a single portable JSON file at [ml/models/touchscreen_mobile_model/model.json](file:///c:/huMANvith/NeuroTrap/ml/models/touchscreen_mobile_model/model.json).

- **Feature Count**: Exactly 27 numerical features.
- **Class Labels**: `["CNT", "PD"]` (`Index 0 = CNT`, `Index 1 = PD`).
- **Exported Parameters**:
  - `scaler.mean`: 27 floating-point means
  - `scaler.scale`: 27 floating-point scale factors
  - `logistic_regression.coefficients`: 27 floating-point feature weights
  - `logistic_regression.intercept`: Single scalar intercept (`-0.09886...`)

---

## Inference Calculation Mathematics

For a raw feature vector \( X = [x_1, x_2, \dots, x_{27}] \):

1. **Standardization**:
   \[
   z_i = \frac{x_i - \text{mean}_i}{\text{scale}_i} \quad \text{for } i \in \{1, \dots, 27\}
   \]

2. **Logit Decision Score Calculation**:
   \[
   \text{score} = \text{intercept} + \sum_{i=1}^{27} (\text{coefficient}_i \times z_i)
   \]

3. **Sigmoid Probability Transformation**:
   \[
   P(\text{PD}) = \frac{1}{1 + \exp(-\text{score})}
   \]
   \[
   P(\text{CNT}) = 1.0 - P(\text{PD})
   \]

4. **Classification Threshold**:
   - Class = `"PD"` if \( P(\text{PD}) \ge 0.50 \), else `"CNT"`.

---

## Proposed Flutter / Dart Implementation Blueprint

Below is the clean Dart implementation structure to be embedded in `mobile/lib/`:

```dart
import 'dart:math' as math;

class TouchscreenMobileInference {
  final List<double> means;
  final List<double> scales;
  final List<double> coefficients;
  final double intercept;

  TouchscreenMobileInference({
    required this.means,
    required this.scales,
    required this.coefficients,
    required this.intercept,
  });

  /// Predicts probability of PD given 27 extracted numerical features in exact order.
  double predictProbabilityPD(List<double> rawFeatures) {
    if (rawFeatures.length != 27) {
      throw ArgumentError('Expected 27 features, got ${rawFeatures.length}');
    }

    double score = intercept;
    for (int i = 0; i < 27; i++) {
      final double z = (rawFeatures[i] - means[i]) / scales[i];
      score += coefficients[i] * z;
    }

    return 1.0 / (1.0 + math.exp(-score));
  }

  /// Predicts class label ('CNT' or 'PD').
  String predictClass(List<double> rawFeatures, {double threshold = 0.50}) {
    final double pPD = predictProbabilityPD(rawFeatures);
    return pPD >= threshold ? 'PD' : 'CNT';
  }
}
```

---

## Mobile Execution Pipeline Workflow

```
[Touchscreen Gesture Events]
             ↓
  [Feature Extractor] → 27 Numerical Features
             ↓
  [Standardization]   → z_i = (x_i - mean_i) / scale_i
             ↓
  [Linear Dot Product] → score = intercept + Σ(coef_i * z_i)
             ↓
  [Sigmoid Function]  → P(PD) = 1 / (1 + exp(-score))
             ↓
  [Screening Result]  → "PD" (P >= 0.50) or "CNT" (P < 0.50)
```

---

## Data Leakage & Integrity Verification

- **Development Set Only**: Model training and parameter export used strictly the **269 development recordings**.
- **Final Test Locked**: The held-out 69-recording test set was never loaded or used during export or mobile parameter extraction.
- **Keyboard Artifact Separation**: Touchscreen joblib artifacts are stored with distinct filenames (`neurotap_touchscreen_*`) and do not overwrite keyboard artifacts.
