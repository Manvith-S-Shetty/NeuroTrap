# Feature Analysis

This stage performs read-only exploratory analysis of the current NeuroTap 27-feature recording-level dataset.

It does not train models, tune hyperparameters, remove features, perform feature selection, change feature definitions, create a production dataset, modify raw data, or make diagnostic claims.

## Scope

Feature analysis intended for model development uses only the existing training/development side of the participant-aware split:

- 93 development recordings
- 68 development participants

The final held-out test set is excluded:

- 23 held-out recordings
- 17 held-out participants

The analysis validates that final-test participants are not present in the development analysis set.

## Feature Set

The analysis uses exactly the 27 numerical feature columns defined by `ml/training/dataset.py`:

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

Metadata fields such as `participant_id`, `recording_id`, `dataset`, `raw_file`, `gt`, and `label` are not analyzed as input features.

## Analyses

The analysis reports:

- feature names and dtypes
- descriptive statistics: count, mean, std, min, 25%, median, 75%, max
- missing and infinite value counts
- unique value counts
- Pearson correlation matrix between numerical features
- highly correlated feature pairs
- CNT vs PD descriptive distribution comparisons
- IQR-based potential outlier counts

## Correlation Threshold

Highly correlated feature pairs are identified using:

```text
absolute Pearson correlation >= 0.90
```

This is a descriptive flag only. No features are removed automatically.

## CNT vs PD Comparison

For every feature, the analysis reports:

- CNT mean
- PD mean
- CNT median
- PD median
- CNT standard deviation
- PD standard deviation

These are descriptive comparisons only. They do not establish statistical significance and must not be used as medical evidence.

## Outlier Rule

Potential outliers are identified with the standard IQR rule:

```text
IQR = Q3 - Q1
lower bound = Q1 - 1.5 * IQR
upper bound = Q3 + 1.5 * IQR
```

Values outside those bounds are counted as potential outliers. They are not removed or corrected.

## Exploratory Outputs

Running:

```powershell
python ml\features\analyze_features.py
```

saves exploratory CSV tables and plots under:

```text
ml/features/analysis_outputs/
```

Generated plots:

- `feature_distributions.png`
- `cnt_vs_pd_feature_boxplots.png`
- `correlation_heatmap.png`

These are exploratory artifacts, not production datasets.

## Final Test Protection

The final held-out test set must not be used to choose, remove, transform, or tune features. It remains reserved for final model evaluation.

This matters because using final-test labels or distributions during development would leak evaluation information into model design.

## Current Development-Set Results

Running:

```powershell
python ml\features\analyze_features.py
```

on the existing participant-aware development set produced:

```text
Development recordings analyzed: 93
Development participants analyzed: 68
Final held-out test recordings excluded: 23
Final held-out test participants excluded: 17
Final-test participant overlap with analysis set: 0
Feature count: 27
Missing feature values: 0
Infinite feature values: 0
```

Feature dtypes:

```text
26 features: float64
key_count: int64
```

Highly correlated pairs using `abs(Pearson correlation) >= 0.90`:

```text
hold_mean / hold_variance: 0.999894
flight_max / inter_key_max: 0.998427
flight_median / inter_key_median: 0.993567
inter_key_mean / inter_key_median: 0.972192
inter_key_skewness / inter_key_kurtosis: 0.969216
flight_median / inter_key_mean: 0.965326
hold_skewness / hold_kurtosis: 0.954568
hold_mean / hold_std: 0.950395
hold_std / hold_variance: 0.949969
inter_key_std / inter_key_variance: 0.946184
key_count / typing_rate: 0.942462
flight_std / flight_min: -0.936299
flight_std / flight_variance: 0.934146
```

Largest descriptive CNT vs PD mean differences were observed in `hold_variance`, `hold_max`, `hold_std`, `hold_mean`, and `key_count`. These are descriptive differences only and do not establish statistical significance.

The IQR rule flagged potential outliers in multiple features. The highest counts were:

```text
inter_key_min: 20
hold_min: 20
flight_variance: 14
inter_key_kurtosis: 12
flight_max: 11
flight_mean: 10
inter_key_max: 10
inter_key_mean: 9
```

No features were removed or changed based on these findings.
