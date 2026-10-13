# Inference Pipeline

This stage demonstrates the Python-side inference path for a new raw typing session.

It does not retrain the model, refit the scaler, change feature definitions, change feature ordering, modify saved artifacts, modify raw data, connect Flutter, or use the final held-out test set.

## Training-Time Pipeline

The current deployment artifact was created with:

```text
canonical raw dataset
-> recording-level feature extraction
-> existing participant-aware development/final-test split
-> fit StandardScaler on development X only
-> train LogisticRegression(C=100.0, random_state=42) on scaled development X
-> save model, scaler, feature order, labels, and metadata
```

The final test split remains locked and untouched.

## Inference-Time Pipeline

The inference pipeline is:

```text
new raw typing events
-> in-memory normalized session
-> existing recording-level feature extraction
-> exact 27-feature ordered vector
-> saved StandardScaler transform
-> saved Logistic Regression prediction
-> predicted class and class probabilities
```

No `StandardScaler`, `LogisticRegression`, or other model is fitted during inference.

## Raw Event Format

A new typing event must provide:

```text
key
hold_time
press_time
release_time
```

These match the normalized timing semantics used by the existing preprocessing and feature extraction code. Timing values are expected in seconds.

The inference module adds temporary metadata in memory only so it can reuse the existing recording-level feature extractor.

## Feature Conversion

The function `extract_features_from_events` converts one new session into exactly the current 27 numerical features:

```text
hold_mean
hold_std
hold_variance
hold_median
hold_min
hold_max
hold_skewness
hold_kurtosis
flight_mean
flight_std
flight_variance
flight_median
flight_min
flight_max
flight_skewness
flight_kurtosis
inter_key_mean
inter_key_std
inter_key_variance
inter_key_median
inter_key_min
inter_key_max
inter_key_skewness
inter_key_kurtosis
key_count
recording_duration
typing_rate
```

The formulas are reused from `ml/features/feature_extraction.py`.

## Why Exact Feature Order Matters

The saved model receives a numeric array. It does not know column meanings at prediction time unless the pipeline enforces them.

The saved feature-name artifact defines the trained order. Inference validates that the extracted feature vector has exactly the same names and order before scaling and prediction.

## Validation

The inference pipeline rejects:

- empty sessions
- missing required event fields
- insufficient valid timing events for flight/inter-key timing
- missing, infinite, or invalid final feature values
- feature vectors with missing, extra, or reordered columns

It reports, without silently deleting source events:

- missing keys
- missing timing values
- non-positive timestamps
- negative hold durations
- release-before-press rows
- valid hold-event count
- valid chronological-event count
- valid press-event count

Existing feature-engineering rules are preserved. For example, missing keys do not block timing-only feature calculation when timing fields are valid.

## Flutter Integration Boundary

Flutter integration is not implemented in this stage.

Eventually, Flutter/Dart would collect typing or touch events, construct records with `key`, `hold_time`, `press_time`, and `release_time`, and pass those events into an equivalent feature-extraction and inference path. This Python pipeline defines the expected behavior before mobile integration.

## Synthetic Example

Running:

```powershell
python ml\inference\pipeline.py
```

uses a deterministic synthetic typing session. It does not use training recordings or final-test recordings.

## Safety Boundary

NeuroTap is a non-diagnostic research/prototype screening system. The inference result is a prototype risk-indicator output, not a medical diagnosis or clinical decision.
