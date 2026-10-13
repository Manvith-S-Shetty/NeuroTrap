# Touchscreen Final Locked-Test Evaluation

## Purpose

This document presents the final, held-out evaluation of NeuroTrap's primary touchscreen classification model (**Logistic Regression**, `C=0.001`, `class_weight=None`, `random_state=42`).

> [!IMPORTANT]
> **Experimental Prototype & Disclaimer**:
> - This is an **experimental machine-learning screening prototype**.
> - Results are based on a small dataset of 33 participants (338 recordings).
> - The held-out test set contains only **7 participants (69 recordings)**.
> - These results should **NOT be interpreted as clinical validation**.
> - The model is **NOT a medical diagnosis system**.
> - Performance on this dataset does **NOT establish real-world clinical effectiveness**.

---

## Data Leakage Control & Test Set Integrity

- **Locked Test Set Boundaries**:
  - Development dataset: **269 recordings / 26 participants** (142 PD, 127 CNT)
  - Final test dataset: **69 recordings / 7 participants** (37 PD, 32 CNT)
  - Participant Overlap: **0**
- **Strict Single-Shot Protocol**:
  - Model configuration (`LogisticRegression(C=0.001, class_weight=None)`) was fixed based on Step 8 development CV results before opening or inspecting the final test set.
  - `StandardScaler` was fitted **exclusively on the 269 development recordings**.
  - The model was trained **exclusively on the 269 development recordings**.
  - Final evaluation was performed **exactly once**. No hyperparameter tuning, model re-selection, feature adjustment, or retraining was performed post-evaluation.

---

## Final Held-Out Test Performance

| Metric | Score | Detail |
| :--- | :---: | :--- |
| **Accuracy** | **0.7101** | (49 / 69 correctly classified recordings) |
| **Precision** | **0.7297** | (27 true PD / 37 predicted PD) |
| **Recall** | **0.7297** | (27 true PD / 37 actual PD) |
| **F1 Score** | **0.7297** | Harmonic mean of precision and recall |

### Confusion Matrix

| | Predicted CNT | Predicted PD | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual CNT** | **22** | 10 | 32 |
| **Actual PD** | 10 | **27** | 37 |
| **Total Predicted** | 32 | 37 | 69 |

- **Predicted Class Distribution**: 32 CNT / 37 PD
- **Actual Class Distribution**: 32 CNT / 37 PD

---

## Participant-Level Prediction Breakdown

Because multiple typing recordings originate from each individual, participant-level predictions were analyzed via majority vote across their respective recordings:

| Participant ID | Actual Label | Recording Count | Predicted CNT | Predicted PD | Majority Prediction | Majority Correct? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S09** | PD | 11 | 10 | 1 | CNT | No |
| **S10** | PD | 11 | 0 | 11 | PD | **Yes** |
| **S16** | PD | 5 | 0 | 5 | PD | **Yes** |
| **S18** | PD | 10 | 0 | 10 | PD | **Yes** |
| **S22** | CNT | 11 | 11 | 0 | CNT | **Yes** |
| **S27** | CNT | 11 | 10 | 1 | CNT | **Yes** |
| **S32** | CNT | 10 | 1 | 9 | PD | No |

**Participant Majority Vote Summary**:
- **5 out of 7 participants** (71.4%) were correctly classified by majority voting (3/4 PD participants, 2/3 CNT participants).
- Individual S09 (PD) consistently triggered CNT predictions (10/11 recordings), indicating potential atypical or subtle motor expression.
- Individual S32 (CNT) consistently triggered PD predictions (9/10 recordings), indicating feature overlap with PD tapping patterns.

---

## Development CV vs. Final Held-Out Test Comparison

| Metric | Development CV (StratifiedGroupKFold) | Held-Out Final Test | Difference |
| :--- | :---: | :---: | :---: |
| **F1 Score** | `0.8319 ± 0.1438` | `0.7297` | `-0.1022` |
| **Accuracy** | `0.8090` | `0.7101` | `-0.0989` |
| **Precision** | `0.8190` | `0.7297` | `-0.0893` |
| **Recall** | `0.8533` | `0.7297` | `-0.1236` |

### Interpretation
- The final-test F1 of `0.7297` falls within less than 1 standard deviation (`-0.71 σ`) of the development CV estimate (`0.8319 ± 0.1438`), reflecting expected variance when evaluating on a small held-out set of 7 participants.
- Without a formal statistical hypothesis test, this difference should not be assumed to be statistically significant.

---

## Machine-Readable Outputs

All evaluation artifacts persist to [ml/training/touchscreen_final_evaluation_outputs/](file:///c:/huMANvith/NeuroTrap/ml/training/touchscreen_final_evaluation_outputs/):
- `final_evaluation_report.json`: Full configuration, metrics, confusion matrix, and participant summaries.
- `final_predictions.csv`: Recording-level predictions aligned with metadata.
- `participant_summary.csv`: Participant-level counts and majority vote outputs.
