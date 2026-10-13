# Random Forest Experiment

This stage trains and evaluates the first NeuroTap Random Forest experiment. It uses the same recording-level ML table and the same participant-aware split settings as the Logistic Regression baseline.

The experiment does not tune hyperparameters, perform cross-validation, select features, export a model, scale features, train a diagnostic system, or modify raw data.

## What Random Forest Is

Random Forest is an ensemble classifier made from many decision trees. Each tree learns decision rules from a bootstrapped sample of the training data, and the forest combines the tree predictions by voting.

This can capture nonlinear relationships and feature interactions that a linear Logistic Regression model may miss.

## Why No Scaling Is Used

Random Forest splits features by threshold comparisons inside decision trees. Those threshold comparisons are not affected by whether a feature is measured in seconds, counts, or standardized units.

For this first experiment, the validated numerical feature matrix is passed directly to `RandomForestClassifier` without `StandardScaler`.

## Why Compare With Logistic Regression

Logistic Regression is the first simple baseline. Random Forest is a more flexible baseline. Comparing them helps separate pipeline issues from model-family behavior while keeping the dataset, features, split, and metrics fixed.

## Fair Comparison Rules

The Random Forest experiment uses:

- the same 116 recording-level samples
- the same 27 numerical feature columns
- the same participant-aware train/test split settings
- the same held-out test participants
- the same `CNT` and `PD` class labels
- the same evaluation metrics

Using the same participant-aware split is essential. If the Random Forest used a different test set, its metrics would not be directly comparable to the Logistic Regression baseline.

## Model Settings

Initial settings:

```text
RandomForestClassifier
n_estimators = 200
max_depth = None
min_samples_split = 2
min_samples_leaf = 1
random_state = 42
```

`max_depth = None` means each tree can expand until the stopping rules in scikit-learn are reached. This is documented explicitly, but it has not been tuned.

## Validation

Before training, the experiment validates that:

- expected feature columns exist
- no unexpected columns enter `X`
- feature column order matches the NeuroTap feature list
- features are numeric
- feature matrices contain no NaN values
- feature matrices contain no infinite values
- labels are limited to `CNT` and `PD`
- `participant_id` is present in train/test metadata
- train/test participant overlap is zero

Metadata fields such as `participant_id`, `dataset`, `raw_file`, `gt`, and `label` are not used as model input features.

## Metrics

Accuracy is the fraction of held-out recordings classified correctly.

Precision is calculated for `PD` as the positive class. It is the fraction of predicted `PD` recordings that are actually `PD` in the source ground truth.

Recall is calculated for `PD` as the positive class. It is the fraction of source-ground-truth `PD` recordings predicted as `PD`.

F1 is the harmonic mean of precision and recall for the `PD` class.

The confusion matrix is reported with labels ordered as `CNT`, `PD`. Rows are true labels and columns are predicted labels.

## Limitations

The held-out test set contains only 23 recordings from 17 participants. Metrics from this first experiment can move substantially with a different participant split, and no hyperparameter tuning or cross-validation has been performed yet.

This result is an engineering comparison between baseline classifiers. It is not a medical diagnosis and must not be interpreted as clinical evidence.
