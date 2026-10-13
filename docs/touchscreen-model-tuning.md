# Touchscreen Model Tuning and Robustness Analysis

## Purpose

This document details the hyperparameter search, cross-validation strategy comparison, per-fold performance stability, and robustness analysis for the NeuroTap touchscreen modality baseline models (**Logistic Regression** and **Random Forest**).

> [!IMPORTANT]
> **Data Leakage Control**: All tuning, grid search evaluations, fold variance analyses, and model comparisons in Step 8 were conducted **exclusively on the 269-recording / 26-participant development set**.
> The final test set (**69 recordings / 7 participants**: 37 PD, 32 CNT) was **never loaded, inspected, or evaluated** during any part of this process.

---

## Executive Summary

1. **Logistic Regression Improvement**:
   - Baseline Step 7 F1 (GroupKFold, C=1.0, class_weight=None): `0.8075 ± 0.1351`
   - Best Tuned GroupKFold (C=0.1, class_weight='balanced'): **`0.8205 ± 0.1349`** (`+0.0130` F1 gain)
   - Best Tuned StratifiedGroupKFold (C=0.001, class_weight=None): **`0.8319 ± 0.1438`** (`+0.0244` F1 gain)

2. **Random Forest Improvement**:
   - Baseline Step 7 F1 (GroupKFold, defaults): `0.7741 ± 0.0889`
   - Best Tuned GroupKFold (n_estimators=500, max_depth=10, min_samples_leaf=2): **`0.7779 ± 0.0897`** (`+0.0038` F1 gain)
   - Best Tuned StratifiedGroupKFold (n_estimators=500, max_depth=5, min_samples_leaf=2): **`0.7920 ± 0.1699`** (`+0.0179` F1 gain)

---

## Grid Search Space & Methodology

- **Validation Strategy**: 5-Fold Participant-Aware Cross-Validation (`GroupKFold` & `StratifiedGroupKFold` on `participant_id`). Zero participant overlap between train and validation folds in all iterations.
- **Preprocessing Pipeline**: `StandardScaler` was strictly fitted **inside each CV training fold** for Logistic Regression. No feature scaling was used for Random Forest.

### Logistic Regression Search Space (10 combinations)
- `C`: `[0.001, 0.01, 0.1, 1.0, 100.0]`
- `class_weight`: `[None, 'balanced']`

### Random Forest Search Space (18 combinations)
- `n_estimators`: `[100, 300, 500]`
- `max_depth`: `[None, 5, 10]`
- `min_samples_leaf`: `[1, 2, 5]`
- `class_weight`: `[None, 'balanced']`

---

## Detailed Results & Comparison

### GroupKFold CV Summary (Primary Strategy)

| Model | Configuration | Mean F1 | Std F1 | Mean Acc | Mean Pre | Mean Rec |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Best LR** | `C=0.1, class_weight='balanced'` | **0.8205** | 0.1349 | 0.8416 | 0.8356 | 0.8680 |
| LR Baseline | `C=1.0, class_weight=None` | 0.8075 | 0.1351 | 0.8252 | 0.8267 | 0.8598 |
| **Best RF** | `n_est=500, max_depth=10, min_leaf=2` | **0.7779** | 0.0897 | 0.7785 | 0.7656 | 0.8483 |
| RF Baseline | `n_est=100, max_depth=None, min_leaf=1` | 0.7741 | 0.0889 | 0.7747 | 0.7621 | 0.8483 |

### StratifiedGroupKFold CV Summary

| Model | Configuration | Mean F1 | Std F1 | Mean Acc | Mean Pre | Mean Rec |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Best LR** | `C=0.001, class_weight=None` | **0.8319** | 0.1438 | 0.8090 | 0.8190 | 0.8533 |
| **Best RF** | `n_est=500, max_depth=5, min_leaf=2` | **0.7920** | 0.1699 | 0.7822 | 0.8021 | 0.8085 |

---

## Per-Fold Robustness (Best Models - GroupKFold)

### Best Logistic Regression (`C=0.1, class_weight='balanced'`)
- **Fold 1**: F1=0.9130 | Pre=0.8400 | Rec=1.0000 | (val: 53 rec / 5 p: PD=21, CNT=32)
- **Fold 2**: F1=0.8267 | Pre=0.8378 | Rec=0.8158 | (val: 60 rec / 6 p: PD=38, CNT=22)
- **Fold 3**: F1=0.6667 | Pre=0.5000 | Rec=1.0000 | (val: 52 rec / 5 p: PD=11, CNT=41)
- **Fold 4**: F1=0.9877 | Pre=1.0000 | Rec=0.9756 | (val: 52 rec / 5 p: PD=41, CNT=11)
- **Fold 5**: F1=0.7083 | Pre=1.0000 | Rec=0.5484 | (val: 52 rec / 5 p: PD=31, CNT=21)

---

## Verification & Artifacts

All outputs persist to [ml/training/touchscreen_tuning_outputs](file:///c:/huMANvith/NeuroTrap/ml/training/touchscreen_tuning_outputs):
- `best_configs_groupkfold.json` & `best_configs_stratifiedgroupkfold.json`
- `config_results_groupkfold.csv` & `config_results_stratifiedgroupkfold.csv`
- `fold_results_groupkfold.csv` & `fold_results_stratifiedgroupkfold.csv`
