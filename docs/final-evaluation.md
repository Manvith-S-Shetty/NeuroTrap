# Final Evaluation

This stage performs the first held-out evaluation of the current tuned NeuroTap models.

It trains the selected configurations on all development recordings and evaluates once on the final held-out test set. It does not tune hyperparameters, perform feature selection, change thresholds, change the split, export a model, modify raw data, or make diagnostic claims.

## Development CV vs Final Test Evaluation

Development cross-validation is used for model development and hyperparameter selection. It repeatedly splits the 93-recording development set into training and validation folds using participant-aware grouping.

Final test evaluation is different. The final held-out test set was kept untouched during feature analysis, cross-validation, and tuning. It is used once here to estimate performance on participants not used for development decisions.

## Data Boundary

The existing participant-aware split is preserved:

- development set: 93 recordings from 68 participants
- final held-out test set: 23 recordings from 17 participants

Development and final-test participant overlap must be zero.

Final-test labels are used only to calculate the final metrics after model configurations have already been selected. They are not used for hyperparameter selection, feature selection, threshold selection, preprocessing fitting, or model selection.

## Selected Models

Selected Logistic Regression configuration:

```text
C = 100.0
random_state = 42
max_iter = 1000
StandardScaler required
```

Selected Random Forest configuration:

```text
n_estimators = 200
max_depth = 5
min_samples_split = 2
min_samples_leaf = 1
random_state = 42
no scaling
```

## Training Procedure

For Logistic Regression:

```text
fit StandardScaler on all 93 development recordings only
transform development recordings
transform final test recordings using the development-fitted scaler
train LogisticRegression on scaled development recordings
evaluate once on scaled final test recordings
```

For Random Forest:

```text
train RandomForestClassifier on all 93 unscaled development recordings
evaluate once on unscaled final test recordings
```

The selected models are retrained on all development data because tuning is complete. Using all development recordings gives the final selected configuration the full non-test training signal before held-out evaluation.

## Metrics

Accuracy is the fraction of final test recordings classified correctly.

Precision is calculated for `PD` as the positive class. It is the fraction of predicted `PD` recordings that are actually `PD` in the source ground truth.

Recall is calculated for `PD` as the positive class. It is the fraction of source-ground-truth `PD` recordings predicted as `PD`.

F1 is the harmonic mean of precision and recall for the `PD` class.

The confusion matrix is reported with labels ordered as `CNT`, `PD`. Rows are true labels and columns are predicted labels.

## Audit Predictions

The final evaluation reports each test recording with metadata, true label, and model predictions. This is for auditability only.

Those predictions must not be used to tune models, change features, select thresholds, or make additional model-selection decisions.

## Limitations

The final test set contains only 23 recordings from 17 participants. Results can be sensitive to a small number of recordings and should be interpreted as a limited engineering evaluation, not as clinical evidence.

NeuroTap remains a non-diagnostic screening/risk-indicator prototype. These results do not establish medical diagnostic performance.

## Current Final Held-Out Results

Running:

```powershell
python ml\training\final_evaluation.py
```

produced:

```text
Development recordings: 93
Development participants: 68
Final test recordings: 23
Final test participants: 17
Participant overlap: 0
Final test class counts: CNT 11, PD 12
```

Tuned Logistic Regression:

```text
C: 100.0
random_state: 42
scaler: StandardScaler fit on development data only
Accuracy: 0.7826
Precision: 0.7059
Recall: 1.0000
F1: 0.8276
Confusion matrix labels: CNT, PD
[[ 6  5]
 [ 0 12]]
```

Tuned Random Forest:

```text
n_estimators: 200
max_depth: 5
min_samples_split: 2
min_samples_leaf: 1
random_state: 42
scaler: none
Accuracy: 0.4783
Precision: 0.5000
Recall: 0.5833
F1: 0.5385
Confusion matrix labels: CNT, PD
[[4 7]
 [5 7]]
```

Final test comparison:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Tuned Logistic Regression | 0.7826 | 0.7059 | 1.0000 | 0.8276 |
| Tuned Random Forest | 0.4783 | 0.5000 | 0.5833 | 0.5385 |

Development CV and final test are separate evaluation stages:

| Model | CV Mean F1 | CV Std F1 | Final Test F1 |
|---|---:|---:|---:|
| Tuned Logistic Regression | 0.6413 | 0.0872 | 0.8276 |
| Tuned Random Forest | 0.5975 | 0.1539 | 0.5385 |

No additional tuning, threshold changes, feature changes, or model selection should be performed based on these final test results.
