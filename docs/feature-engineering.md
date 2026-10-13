# Feature Engineering

This document describes the first NeuroTap recording-level feature-extraction pipeline. It does not train a model and does not modify raw dataset files.

## Scope

The first version converts each normalized raw recording into one feature vector.

```text
raw recording
-> validated feature-specific view
-> sort by press_time where chronological ordering is required
-> calculate timing measurements
-> aggregate measurements
-> one feature vector per recording
```

The implementation is in `ml/features/feature_extraction.py`.

## Input

The extractor expects the normalized DataFrame returned by `ml/preprocessing/data_loader.py`.

Required metadata columns:

```text
dataset
participant_id
recording_id
session_id
repetition_id
experiment_id
raw_file
gt
label
```

Required raw timing columns:

```text
row_number
key
hold_time
release_time
press_time
```

## Output

The extractor returns one row per recording.

Output metadata:

```text
dataset
participant_id
recording_id
session_id
repetition_id
experiment_id
raw_file
gt
label
```

Numerical feature columns:

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

Timing features are kept in seconds.

## Distribution Shape

Skewness is calculated with `pandas.Series.skew()`. This is the adjusted Fisher-Pearson standardized moment coefficient, normalized by `N - 1`.

Kurtosis is calculated with `pandas.Series.kurt()`. This is Fisher/excess kurtosis, so a normal distribution has kurtosis near `0`; it is not Pearson kurtosis, where a normal distribution is near `3`.

Hold shape features use the same valid hold-time observations as the other hold statistics. Flight shape features use the same valid flight-time observations as the other flight statistics.

If there are insufficient valid observations for a statistic, pandas returns `NaN`; the extractor preserves that `NaN` rather than inventing a value.

## Hold/Dwell Features

Hold features are calculated from `hold_time`.

Rows are excluded from the hold-specific view when:

- `hold_time` is missing
- `hold_time < 0`
- `release_time < press_time`

Rows are not excluded merely because they have a missing `key`, because hold-time features are timing-only in this first version.

Positive-timestamp long holds are retained in this first version. Long holds are known to require later sensitivity analysis, but they are not automatically removed here.

## Flight-Time Features

Flight time is calculated from a chronological valid view:

```text
flight_time[i] = press_time[i + 1] - release_time[i]
```

Rows are excluded from the chronological view when:

- `press_time` is missing
- `release_time` is missing
- `press_time <= 0`
- `release_time <= 0`
- `release_time < press_time`

The chronological view is sorted by `press_time` before calculating flight times. This is required because the raw files are generally stored in `release_time` order rather than `press_time` order.

Rows are not excluded merely because they were flagged as `non_monotonic_press_time` in the original file order.

If a recording has fewer than two valid chronological rows, flight statistics are returned as `NaN`.

## Inter-Key Timing

Inter-key interval is calculated from the press times of chronologically sorted valid key events:

```text
inter_key_interval[i] = press_time[i + 1] - press_time[i]
```

Rows are excluded from the inter-key view when:

- `press_time` is missing
- `press_time <= 0`

The inter-key view is sorted by `press_time` before interval calculation. Only positive intervals are retained. Missing `key` values do not block inter-key timing features because this feature group only uses timing.

Inter-key interval differs from flight time:

- Flight time measures the gap or overlap between one key release and the next key press: `next.press_time - current.release_time`.
- Inter-key interval measures press-to-press spacing: `next.press_time - current.press_time`.

These features are included because they capture rhythm and press-to-press cadence separately from dwell time and release-to-press flight timing.

## Activity Features

`key_count` is the number of raw rows in the recording. It includes rows with missing keys and rows with suspicious timing because it is a raw activity count, not a filtered count.

`recording_duration` is calculated from the chronological valid view:

```text
max(release_time) - min(press_time)
```

`typing_rate` is:

```text
key_count / recording_duration
```

If recording duration is missing, zero, or negative, `typing_rate` is returned as `NaN`.

## Data Preservation

The extractor works on copies and feature-specific views. It does not mutate the input DataFrame and does not modify raw dataset files.

Suspicious records are handled only by feature-specific inclusion rules. The original normalized rows remain available for later auditing, alternative feature calculations, and sensitivity analysis.

## Current Real-Data Run

Running:

```powershell
python ml\features\feature_extraction.py
```

on the canonical MIT-CS1PD and MIT-CS2PD data produced:

- input rows: 177,513
- feature rows: 116
- output columns: 36
- NaN feature values: none detected in the current full-dataset run

Dataset-level feature rows:

```text
MIT-CS1PD: 62 feature rows, 31 participants, 62 recordings
MIT-CS2PD: 54 feature rows, 54 participants, 54 recordings
```

Some hold features are strongly affected by retained long holds and non-positive press-time anomalies because the first-version hold rules only exclude missing holds, negative holds, and release-before-press rows. This is intentional for this version and should be revisited with sensitivity analysis before modeling.
