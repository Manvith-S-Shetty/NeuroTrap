# Touchscreen Baseline Models

## Purpose

This document describes the baseline ML training phase for NeuroTap's touchscreen typing modality. The goal is to establish honest baseline performance figures using Logistic Regression and Random Forest before any hyperparameter tuning, feature selection, or architectural decisions.

Baseline models operate on the 27 numerical touchscreen features defined in [`touchscreen_feature_extraction.py`](../ml/features/touchscreen_feature_extraction.py) and the ML-ready dataset from [`touchscreen_dataset.py`](../ml/training/touchscreen_dataset.py).

---

## Development / Final-Test Separation

All model training and cross-validation uses **ONLY the development set** (269 recordings, 26 participants) from Step 6.

| Set | Recordings | Participants | PD | CNT |
| :--- | :---: | :---: | :---: | :---: |
| **Development** | 269 | 26 | 142 | 127 |
| **Final-Test (LOCKED)** | 69 | 7 | 37 | 32 |

> [!CAUTION]
> **The 7 final-test participants (69 recordings) are PERMANENTLY LOCKED.**
> They are never loaded, inspected, or referenced during baseline training or CV.
> They are reserved exclusively for final evaluation after all design decisions are frozen.

The `validate_development_dataset()` function in `touchscreen_baseline.py` actively rejects any development set that contains a locked final-test participant ID.

---

## Participant-Aware Cross-Validation

### Strategy

| Parameter | Value |
| :--- | :--- |
| Strategy | `GroupKFold` (scikit-learn) |
| Groups | `participant_id` |
| Folds | 5 |
| `random_state` | 42 |

`GroupKFold` guarantees that **no participant's recordings appear in both the training and validation splits of any fold**. This prevents within-subject leakage — the primary leakage risk in small clinical datasets.

### Why GroupKFold over StratifiedGroupKFold

With only 26 development participants and 5 folds, label stratification across folds cannot be perfectly balanced at the participant level. `GroupKFold` is used for strict participant isolation; class balance per fold is reported but not enforced.

### Per-Fold Details (Real Data, `random_state=42`)

| Fold | Train Recordings | Val Recordings | Train Participants | Val Participants | Participant Overlap |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 216 | 53 | 21 | 5 | **0** |
| 2 | 209 | 60 | 20 | 6 | **0** |
| 3 | 217 | 52 | 21 | 5 | **0** |
| 4 | 217 | 52 | 21 | 5 | **0** |
| 5 | 217 | 52 | 21 | 5 | **0** |

---

## Models and Preprocessing

### Model A: Logistic Regression

| Parameter | Value |
| :--- | :--- |
| Scaler | `StandardScaler` (fitted **inside** each CV fold on train split only) |
| `C` (regularisation) | 1.0 (baseline — not tuned) |
| `max_iter` | 1000 |
| `random_state` | 42 |

> [!IMPORTANT]
> The StandardScaler is fitted **per-fold on the training portion only**. It is never fitted on the global development set or on any validation/test data. This prevents data leakage from scale estimation.

### Model B: Random Forest

| Parameter | Value |
| :--- | :--- |
| `n_estimators` | 200 |
| `max_depth` | None (unlimited) |
| `min_samples_split` | 2 |
| `min_samples_leaf` | 1 |
| `random_state` | 42 |

No scaler is applied to Random Forest (tree-based models are scale-invariant).

---

## Metrics

All metrics are computed with **PD as the positive class** (i.e., the class we care most about detecting).

| Metric | Definition |
| :--- | :--- |
| **Accuracy** | Fraction of all predictions that are correct |
| **Precision** | TP / (TP + FP) — of those predicted PD, how many are truly PD |
| **Recall** | TP / (TP + FN) — of all true PD, how many are correctly identified |
| **F1** | Harmonic mean of Precision and Recall |

Results are reported as **mean ± sample standard deviation across 5 folds**.

---

## Results (Development Set, 5-Fold GroupKFold CV)

### Logistic Regression

| Metric | Mean | Std |
| :--- | :---: | :---: |
| Accuracy | **0.8252** | 0.0876 |
| Precision | **0.8267** | 0.2310 |
| Recall | **0.8598** | 0.1739 |
| F1 | **0.8075** | 0.1351 |

### Random Forest

| Metric | Mean | Std |
| :--- | :---: | :---: |
| Accuracy | **0.7752** | 0.0685 |
| Precision | **0.7604** | 0.2012 |
| Recall | **0.8479** | 0.1522 |
| F1 | **0.7741** | 0.1007 |

> [!NOTE]
> Logistic Regression outperforms Random Forest on all four metrics at the baseline configuration. The high precision/recall standard deviation across folds reflects the small fold sizes (5–6 participants per fold), where a single participant with many recordings can strongly affect per-fold statistics.

---

## Limitations

1. **Small participant count per fold**: With 26 development participants and 5 folds, each validation fold contains 5–6 participants (52–60 recordings). Fold-level metrics have inherently high variance.
2. **Untuned baselines**: Neither `C` (LR) nor `n_estimators` / `max_depth` (RF) have been optimised. These are entry-point figures only.
3. **No feature selection**: All 27 features are used as-is. Some features are highly correlated (identified in the Step 4 feature analysis). Feature selection may improve generalisation.
4. **No class-reweighting**: PD/CNT class imbalance (142 PD vs 127 CNT in development) is modest but not explicitly addressed. `class_weight='balanced'` will be investigated in Step 8.

---

## Files

| File | Purpose |
| :--- | :--- |
| [`ml/training/touchscreen_baseline.py`](../ml/training/touchscreen_baseline.py) | Baseline training pipeline and CLI |
| [`tests/test_touchscreen_baseline.py`](../tests/test_touchscreen_baseline.py) | Unit tests |
| [`ml/training/touchscreen_dataset.py`](../ml/training/touchscreen_dataset.py) | ML-ready dataset and participant-aware split |

---

## Reproducing Results

```bash
# Run baseline and print results
python -m ml.training.touchscreen_baseline

# Run full test suite
pytest -q
```
