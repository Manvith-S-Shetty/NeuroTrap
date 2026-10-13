# ML Dataset Construction

This stage builds the NeuroTap machine-learning table from recording-level features. It does not train a model, calculate accuracy, scale features, select features, export a model, or modify raw data.

## What X Represents

`X` is the numerical feature matrix. Each row is one recording-level sample, and each column is one numerical predictive feature derived by `ml/features/feature_extraction.py`.

`X` intentionally excludes identifiers, source filenames, dataset names, timestamps, ground-truth fields, and labels. Those values are metadata or targets, not model inputs.

Feature columns used by `X`:

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

## What y Represents

`y` is the target label column:

- `CNT`: control participant in the source ground truth
- `PD`: Parkinson's disease class in the source ground truth

The original boolean `gt` field is retained in metadata but is not included in `X`.

## One Recording, One Sample

One recording is one ML sample because feature extraction aggregates raw key events at the recording level. The raw dataset contains many rows per recording, but those rows are not independent samples once timing features are aggregated.

Using one row per recording preserves the intended modeling unit while keeping participant and recording metadata available for leakage checks.

## Metadata

Metadata is separated from `X`:

```text
dataset
participant_id
recording_id
session_id
repetition_id
experiment_id
raw_file
gt
```

`participant_id` is metadata rather than a feature because it identifies the person. Including it in `X` would allow a model to learn identity-specific patterns instead of typing-dynamics risk signals.

## Participant-Level Splitting

Splitting is implemented in `ml/training/dataset.py` with participant isolation:

- unique participants are assigned to train or test
- all recordings from a participant stay on the same side
- no participant may appear in both train and test
- the split is reproducible with `random_state`
- optional stratification is performed at participant level, not recording-row level

## Data Leakage

Data leakage means information from the test set is allowed to influence training or model selection. In NeuroTap, leakage can happen if recordings from the same participant appear in both train and test, because participant-specific typing behavior may be repeated across sessions.

Random row-level splitting is unsafe because:

- raw event rows from the same recording are highly dependent
- recording-level features are derived from many raw rows
- MIT-CS1PD has two recordings for each participant
- splitting recordings without participant grouping can put one participant's first session in train and second session in test

The minimum safe split unit is therefore `participant_id`.

## Validation

The dataset module raises clear errors when:

- required feature columns are missing
- required metadata columns are missing
- labels are missing
- unsupported labels are present
- non-numeric columns enter `X`
- participants have inconsistent labels
- a participant appears in both train and test

## Current Real-Data Verification

Running:

```powershell
python ml\training\dataset.py
```

produced:

```text
Total recordings: 116
Total participants: 85
X shape: (116, 27)
y shape: (116,)
Class counts: CNT 56, PD 60
Train recordings: 93
Test recordings: 23
Train participants: 68
Test participants: 17
Participant overlap: 0
Train class counts: CNT 45, PD 48
Test class counts: CNT 11, PD 12
```
