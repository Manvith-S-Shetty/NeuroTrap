# Touchscreen Final-Test Error Analysis

## Purpose

> [!IMPORTANT]
> **Data Integrity & Final-Test Locked Status**:
> - The final test set was used **exclusively for evaluation and post-hoc error analysis**.
> - **No model, hyperparameter, feature, threshold, or preprocessing changes** were made based on these results.
> - This analysis is **purely descriptive** to understand error modes in held-out predictions.

---

## Recording-Level Error Analysis

Out of **69 held-out final-test recordings** (7 participants):

- **Correct Predictions**: `49` (71.01%)
  - True Positives (Actual PD -> Predicted PD): **27**
  - True Negatives (Actual CNT -> Predicted CNT): **22**
- **Incorrect Predictions**: `20` (28.99%)
  - False Positives (Actual CNT -> Predicted PD): **10** (14.49%)
  - False Negatives (Actual PD -> Predicted CNT): **10** (14.49%)

---

## Participant-Level Error Analysis

Majority voting across recordings per participant classified **5 of 7 participants correctly** (71.43%).

### Summary Table

| Participant ID | Actual Label | Recording Count | Pred CNT | Pred PD | Majority Prediction | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S09** | PD | 11 | 10 | 1 | CNT | **Misclassified (FN)** |
| **S10** | PD | 11 | 0 | 11 | PD | Correct |
| **S16** | PD | 5 | 0 | 5 | PD | Correct |
| **S18** | PD | 10 | 0 | 10 | PD | Correct |
| **S22** | CNT | 11 | 11 | 0 | CNT | Correct |
| **S27** | CNT | 11 | 10 | 1 | CNT | Correct |
| **S32** | CNT | 10 | 1 | 9 | PD | **Misclassified (FP)** |

### Misclassified Participants Detail
1. **Participant S09 (Actual PD)**:
   - 10 of 11 recordings were predicted as CNT.
   - S09 exhibits typing duration and press patterns that align closely with control group averages.
2. **Participant S32 (Actual CNT)**:
   - 9 of 10 recordings were predicted as PD.
   - S32 exhibits higher hold time variance and lower pressure averages, driving the model toward PD.

---

## Prediction Confidence Analysis

Model output probabilities were evaluated across prediction accuracy states:

| Metric | Correct Predictions (n=49) | Incorrect Predictions (n=20) |
| :--- | :---: | :---: |
| **Mean Confidence** | `0.6671` | `0.5630` |
| **Median Confidence** | `0.6280` | `0.5693` |
| **Min Confidence** | `0.5050` | `0.5012` |
| **Max Confidence** | `0.9702` | `0.7208` |

### Confidence Range Binning

| Confidence Range | Correct Predictions | Incorrect Predictions |
| :--- | :---: | :---: |
| **0.50 – 0.60** | 20 | **14** (70.0% of errors) |
| **0.60 – 0.70** | 12 | 5 |
| **0.70 – 0.80** | 5 | 1 |
| **0.80 – 0.90** | 4 | 0 |
| **0.90 – 1.00** | 8 | 0 |

> [!NOTE]
> 70% of misclassifications occurred near the decision boundary (`0.50 – 0.60` confidence), showing that the model expresses lower certainty when it makes mistakes.

---

## Feature-Level Descriptive Differences

Comparing mean feature values between correctly and incorrectly classified recordings:

| Feature | Mean Correct | Mean Incorrect | Absolute Difference |
| :--- | :---: | :---: | :---: |
| `recording_duration` | 100.2849 | 66.5161 | **33.7687** |
| `event_count` | 68.8980 | 78.9500 | **10.0520** |
| `inter_tap_kurtosis` | 13.8647 | 22.8673 | **9.0026** |
| `hold_kurtosis` | 7.3554 | 1.1073 | **6.2481** |
| `inter_tap_max` | 6.9358 | 8.7528 | **1.8170** |

---

## Logistic Regression Coefficients Interpretation

The trained model coefficients mathematically weight the 27 standardized features:

### Strongest Positive Coefficients (Pushes prediction toward PD class)
1. `hold_mean` (`+0.0552`): Higher mean hold duration increases log-odds of PD.
2. `hold_median` (`+0.0549`): Higher median hold duration increases log-odds of PD.
3. `inter_tap_median` (`+0.0471`): Longer pauses between taps increase log-odds of PD.
4. `hold_std` (`+0.0469`): Greater variance in hold time increases log-odds of PD.
5. `hold_max` (`+0.0431`): Longer maximum hold duration increases log-odds of PD.

### Strongest Negative Coefficients (Pushes prediction toward CNT class)
1. `pressure_median` (`-0.0604`): Higher median touch pressure decreases log-odds of PD (increases CNT likelihood).
2. `pressure_mean` (`-0.0602`): Higher mean touch pressure decreases log-odds of PD.
3. `pressure_max` (`-0.0493`): Higher peak touch pressure decreases log-odds of PD.
4. `typing_rate` (`-0.0476`): Faster typing speed decreases log-odds of PD.
5. `pressure_kurtosis` (`-0.0252`): Higher pressure distribution peakedness decreases log-odds of PD.

> [!CAUTION]
> **Interpretation Warning**: These coefficients represent mathematical weights fitted on standardized numerical features. They are **not medical biomarkers** and do **not establish causal medical mechanisms**.

---

## Visualizations Saved

Visualizations generated in [ml/training/touchscreen_error_analysis_outputs/](file:///c:/huMANvith/NeuroTrap/ml/training/touchscreen_error_analysis_outputs/):
- `plot_confusion_matrix.png`: Held-out test confusion matrix visualization.
- `plot_top_coefficients.png`: Bar chart of top 10 feature weights.
- `plot_participant_predictions.png`: Recording-level breakdown per participant.
