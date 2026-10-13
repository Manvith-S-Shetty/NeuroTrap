# Model Artifact

This stage prepares the current NeuroTap deployment candidate as Python artifacts for later inference work.

It does not implement Flutter/mobile integration, change features, change hyperparameters, change the final test split, modify raw data, or perform additional model selection.

## Deployment Candidate

The current deployment candidate is:

```text
Model: LogisticRegression
C: 100.0
random_state: 42
max_iter: 1000
Scaler: StandardScaler
Features: 27 numerical recording-level features
Labels: CNT, PD
```

The scaler and model are fitted only on the development portion of the existing participant-aware split.

## Artifact Contents

Running:

```powershell
python ml\training\model_artifact.py
```

saves artifacts under:

```text
ml/models/
```

Artifact files:

```text
neurotap_logistic_regression.joblib
neurotap_standard_scaler.joblib
neurotap_feature_names.joblib
neurotap_model_metadata.joblib
```

The metadata contains:

- model type
- selected `C`
- `random_state`
- `max_iter`
- scaler type
- exact ordered feature names
- feature count
- class labels
- label mapping
- development training sample count
- development training participant count
- final-test sample and participant counts excluded from fitting
- final-test participant overlap

## Why The Scaler Is Saved

Logistic Regression uses scaled features. The `StandardScaler` learns training-set means and standard deviations during fitting.

Inference must reuse that fitted scaler. Refitting a scaler during inference would change the feature transformation and break consistency with model training.

## Why Feature Order Matters

The model receives a numeric matrix, not semantic column names. If inference supplies the same values in a different order, the model will interpret each value as the wrong feature.

The artifact stores the exact ordered list of 27 feature names. Inference rejects missing features, extra features, and incorrect feature ordering.

## Training vs Inference

Training:

```text
load ML dataset
recover existing participant-aware split
use only development X and y
fit StandardScaler on development X
train LogisticRegression(C=100.0, random_state=42)
save model, scaler, feature order, labels, and metadata
```

Inference:

```text
load saved model artifact
receive one recording-level feature vector
validate exact feature names and order
transform with the saved scaler
predict class with the saved Logistic Regression model
return predicted class and class probabilities
```

The final held-out test set is not used for fitting artifacts.

## New Recordings

A new recording must first pass through the same recording-level feature extraction logic so that it produces the same 27 numerical features in the same order.

Only then can the artifact inference function transform the feature vector with the saved scaler and pass it to the model.

## Safety Boundary

NeuroTap is a non-diagnostic research/prototype screening system. The model artifact can produce a class prediction and probabilities from the current research feature set, but those outputs are not medical diagnoses and do not establish clinical performance.
