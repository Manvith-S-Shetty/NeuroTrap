# Baseline Model

This stage trains and evaluates the first NeuroTap machine-learning baseline. It uses the existing recording-level ML table and the participant-aware train/test split from `ml/training/dataset.py`.

The baseline does not tune hyperparameters, perform cross-validation, select features, export a model, train a Random Forest, or modify raw data.

## Baseline Purpose

A baseline model is a simple first model used to verify that the training and evaluation pipeline works end to end. It gives later models a reference point, but it is not treated as a final clinical or diagnostic system.

Logistic Regression is used first because it is a well-understood linear classifier with explicit behavior, fast training, and reproducible settings. It is a pragmatic first check before trying more flexible models such as Random Forest.

## Pipeline

The implemented pipeline is:

```text
canonical raw data
-> recording-level feature extraction
-> ML table construction
-> participant-aware train/test split
-> fit StandardScaler on X_train only
-> transform X_train and X_test with the fitted scaler
-> train LogisticRegression on X_train
-> evaluate on held-out X_test
```

The model input `X` contains only numerical predictive features. It excludes participant identifiers, recording identifiers, filenames, dataset names, ground-truth metadata, and labels.

The target `y` is `label`:

- `CNT`: control participant in the source ground truth
- `PD`: Parkinson's disease class in the source ground truth

## Scaling

`StandardScaler` standardizes each numerical feature by subtracting the training-set mean and dividing by the training-set standard deviation.

Scaling is needed because Logistic Regression is sensitive to feature scale. NeuroTap timing features and count/rate features can have different numerical ranges, so scaling keeps the optimization problem better conditioned.

The scaler must be fitted only on `X_train`. Fitting it on `X_test` or on the combined dataset would leak information from the held-out evaluation set into training.

## Participant-Aware Split

Participant-aware splitting is required because recordings from the same participant can share identity-specific typing patterns. A random row-level split could put one participant's recording in train and another recording from the same participant in test, which would overstate model performance.

In this project, data leakage means any path where held-out participant information influences training, scaling, model selection, or feature selection. The baseline prevents participant leakage by reusing the existing split logic and validating that participant overlap is zero.

## Metrics

Accuracy is the fraction of held-out recordings classified correctly.

Precision is calculated for `PD` as the positive class. It answers: among recordings predicted as `PD`, what fraction are actually `PD` in the source ground truth?

Recall is calculated for `PD` as the positive class. It answers: among recordings labeled `PD` in the source ground truth, what fraction did the model predict as `PD`?

F1 is the harmonic mean of precision and recall for the `PD` class.

The confusion matrix is reported with labels ordered as `CNT`, `PD`. Rows are true labels and columns are predicted labels.

## Safety Boundary

This evaluation is not a medical diagnosis. NeuroTap is a non-diagnostic screening/risk-indicator prototype, and this baseline is only an engineering step for verifying the ML pipeline.

## Validation

Before scaling or training, the baseline validates that:

- train and test participants do not overlap
- feature matrices contain only numeric columns
- feature matrices contain no NaN values
- feature matrices contain no infinite values
- labels are limited to `CNT` and `PD`

Missing or infinite feature values raise an error rather than being silently filled.
