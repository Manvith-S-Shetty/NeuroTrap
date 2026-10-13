# Model Tuning

This stage performs controlled hyperparameter tuning for NeuroTap using participant-aware cross-validation on the existing development set.

It does not evaluate tuned models on the final held-out test set, perform feature selection, remove correlated features, change feature definitions, export a model, create a production model, modify raw data, or make diagnostic claims.

## Why Tuning Is Needed

Hyperparameter tuning checks a small set of fixed model settings to understand which configuration works best during development. It is separate from final test evaluation.

A hyperparameter is a model setting chosen before training. It is not learned directly from the training data in the same way as model coefficients or tree splits.

## Data Boundary

The existing participant-aware split is preserved:

- development set: 93 recordings from 68 participants
- final held-out test set: 23 recordings from 17 participants

Tuning uses only the development set. The final held-out test participants must not influence configuration selection.

## GroupKFold

Tuning uses `GroupKFold` with:

```text
group = participant_id
```

This keeps each participant entirely in either the training portion or validation portion of a fold. Ordinary row-level KFold is unsafe because multiple recordings from one participant could otherwise leak identity-specific typing patterns across fold boundaries.

## Logistic Regression Grid

The only tuned Logistic Regression hyperparameter is `C`:

```text
C = 0.01, 0.1, 1.0, 10.0, 100.0
```

Conceptually, `C` controls regularization strength. Smaller `C` means stronger regularization; larger `C` means weaker regularization.

For every fold and every `C`, the pipeline creates a fresh `StandardScaler`, fits it only on that fold's training recordings, transforms fold training and validation recordings, creates a fresh `LogisticRegression`, trains on fold training data, and evaluates on fold validation data.

Preprocessing must happen inside each fold. A scaler fitted outside the fold would leak validation data into training.

## Random Forest Grid

Random Forest tuning uses this small predefined grid:

```text
n_estimators = 100, 200, 500
max_depth = None, 3, 5, 10
min_samples_leaf = 1, 2, 4
```

Unspecified settings keep the baseline values, including:

```text
min_samples_split = 2
random_state = 42
```

`n_estimators` controls the number of trees. `max_depth` limits tree depth when set to an integer; `None` lets trees grow until scikit-learn stopping rules apply. `min_samples_leaf` controls the minimum number of samples required in a leaf node.

Only a small grid is tested because the dataset has 68 development participants. A broad search would increase the risk of overfitting development folds.

Random Forest does not use `StandardScaler` because decision-tree splits are based on feature thresholds and do not require standardized units.

## Selection Metric

Mean F1 across GroupKFold validation folds is the primary model-selection metric.

For every configuration, tuning reports:

- mean F1
- standard deviation of F1
- mean accuracy
- mean precision
- mean recall

The best configuration is selected by:

1. highest mean F1
2. lower F1 standard deviation if mean F1 ties
3. simpler configuration if both still tie

For Logistic Regression, lower `C` is treated as simpler because it applies stronger regularization.

For Random Forest, simpler means shallower finite `max_depth`, then fewer trees, then larger `min_samples_leaf`. `max_depth=None` is treated as less simple than finite depths.

## Limitations

Tuning with 68 development participants is still small. Fold metrics can vary because each validation fold contains a limited number of participants. The selected configuration is a development result, not a final generalization estimate.

The final held-out test set remains reserved for a later final evaluation.

NeuroTap is a non-diagnostic screening/risk-indicator prototype. These tuning results must not be interpreted as medical diagnosis or clinical evidence.
