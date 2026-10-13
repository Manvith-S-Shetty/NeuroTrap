# Cross-Validation

This stage adds participant-aware cross-validation for NeuroTap model development. It compares Logistic Regression and Random Forest on the existing training/development participants only.

The final held-out test set remains untouched by cross-validation.

## What Cross-Validation Is

Cross-validation is a development-time evaluation method that repeatedly splits available development data into training and validation portions. A model is trained on one portion and evaluated on the held-out validation portion.

A fold is one train/validation split inside the cross-validation process.

This project uses 5 folds where feasible.

## Why Ordinary KFold Is Unsafe

Ordinary KFold splits rows. That is unsafe for NeuroTap because one participant can contribute more than one recording. If recordings from the same participant appear in both training and validation, the validation result can be inflated by participant-specific typing patterns.

That is data leakage.

## GroupKFold

`GroupKFold` keeps all rows with the same group value together. NeuroTap uses:

```text
group = participant_id
```

This means a participant never appears in both the training and validation portion of the same fold.

## Final Test Set Protection

The existing participant-aware train/test split creates:

- 68 training/development participants
- 17 final held-out test participants

Cross-validation runs only on the 68 development participants. The 17 final test participants are reserved for final evaluation and must not influence model comparison, scaling, feature selection, tuning, or development decisions.

The implementation validates that supplied development data does not contain explicitly supplied final-test participant IDs.

## Logistic Regression Fold Pipeline

For every fold:

```text
fold training recordings
-> fit a fresh StandardScaler on fold training recordings only
-> transform fold training recordings
-> transform fold validation recordings with the same scaler
-> train a fresh LogisticRegression model
-> evaluate on fold validation recordings
```

The scaler is fitted separately inside every fold. A scaler fitted outside the fold would leak validation-fold information into training.

## Random Forest Fold Pipeline

For every fold:

```text
fold training recordings
-> train a fresh RandomForestClassifier
-> evaluate on fold validation recordings
```

Random Forest does not require feature scaling because tree splits use threshold comparisons on individual features.

## Fixed Model Settings

Logistic Regression uses the existing baseline configuration:

```text
LogisticRegression(random_state=42, max_iter=1000)
```

Random Forest uses the existing baseline configuration:

```text
RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
```

No hyperparameter tuning, `GridSearchCV`, or `RandomizedSearchCV` is used.

## Metrics

Each fold reports:

- accuracy
- precision
- recall
- F1
- training recording count
- validation recording count
- training participant count
- validation participant count
- participant overlap

Precision, recall, and F1 use `PD` as the positive class, with explicit class labels ordered as `CNT`, `PD`.

Aggregate results report the mean and sample standard deviation across folds for each metric. The mean summarizes typical validation performance across folds. The standard deviation shows how much the result varies across participant splits.

Predictions are not pooled across folds as the primary result.

## Limitations

The development set has 68 participants. That is enough for 5-fold GroupKFold, but the participant count is still small. Fold-level class balance can vary, and validation metrics can move noticeably from fold to fold.

Cross-validation is used for development and model comparison only. The final held-out test set remains reserved for final evaluation.

NeuroTap remains a non-diagnostic screening/risk-indicator prototype. These experiments must not be interpreted as medical diagnosis.
