# Touchscreen Motor Pattern Screening Session Architecture

## Overview

The Touchscreen Motor Pattern Screening Session is a prototype feature in the NeuroTrap mobile application designed to evaluate touchscreen typing dynamics. It captures touch timing and pressure characteristics during a controlled text typing task using a custom Flutter-owned virtual keyboard (`CustomTouchscreenKeyboard`), extracts a 27-numerical-feature vector, executes pure Dart machine learning inference using a trained Logistic Regression model ($C=0.001$), and displays experimental screening indicators.

## User Flow & Session State Machine

The screening lifecycle is managed by `TouchscreenSessionController` and `TouchscreenScreen`, transitioning through six defined states:

```
[ idle ] ---> (START SESSION) ---> [ countdown (3..2..1) ]
                                             |
                                    (countdown finishes)
                                             v
[ completed ] <--- (Inference) <--- [ processing ] <--- (FINISH SCREENING) <--- [ collecting ]
      |                                      ^                                (Custom Keyboard active)
   (RETRY)                              (validation)                                  |
      |                                      |                                        v
   [ idle ]                              [ error ]                              [ collecting ]
                                                                             (if < 30 events: notice)
```

### Session States
1. **`idle`**: The user reviews the screening instructions, passage text, and non-diagnostic disclaimers. Displays the `START SESSION` button.
2. **`countdown`**: Displays a 3-second visual countdown (3... 2... 1...). Touch input received during countdown is strictly ignored.
3. **`collecting`**: Custom Flutter virtual keyboard (`CustomTouchscreenKeyboard`) automatically becomes visible and active. The Android soft keyboard (Gboard) is completely eliminated from the screening input flow. Live tap counts update with each completed key press/release pair.
4. **`processing`**: Halts collection, validates data sufficiency and real touch pressure availability, computes the 27-feature statistical vector, and executes Logistic Regression inference.
5. **`completed`**: Displays the experimental screening indicator, model-estimated pattern class, probability scores $P(\text{PD})$ and $P(\text{CNT})$, non-diagnostic explanation, disclaimer, and `TRY AGAIN` button.
6. **`error`**: Handles missing touch pressure hardware, invalid event timestamps, or feature extraction exceptions with clear user guidance, disclaimers, and `TRY AGAIN` option.

---

## Custom Flutter Keyboard Architecture (Option A)

### Why Custom Virtual Keyboard?
On physical mobile hardware (e.g. Android V2055), focusing a standard Flutter `TextField` delegates touch events to the system keyboard (Gboard), causing touches to occur outside Flutter's gesture pipeline. `CustomTouchscreenKeyboard` builds key buttons directly in Flutter:
- Eliminates System Soft Keyboard / Gboard invocation during screening.
- Each key button is an opaque Flutter `Listener` capturing `PointerDownEvent` and `PointerUpEvent`.
- Captures key-level `pressTimeMs`, `releaseTimeMs`, hold duration, and real `PointerEvent.pressure`.
- Manages an internal input text buffer displayed in a non-editable text container.
- Restricts typing event capture strictly to virtual key buttons. Non-key UI buttons (Start, Finish, Retry, Tabs) do not generate typing events.

---

## Controlled Passage & Minimum Event Threshold

### Controlled Text Passage
The model was trained and evaluated on controlled typing excerpts. To match the expected distribution, the application provides a fixed text passage stored in `kTouchscreenScreeningPassage`:

> `"The quick brown fox jumps over the lazy dog. Continuous typing dynamics provide subtle timing and pressure patterns for motor feature extraction."`

### Prototype Minimum Event Threshold (30 Taps)
- Completion requires a minimum of **30 touch events**.
- Tapping `FINISH SCREENING` with fewer than 30 events keeps the session in `collecting` state and displays: `"Please type more of the passage before finishing."`
- **Note**: The 30-event minimum is a prototype data-quality/usability threshold for completing the passage typing task. It is not a clinically validated or statistically established medical threshold.

---

## Feature Extraction & Model Inference

1. **Feature Vector**: `TouchscreenFeatureExtractor.extract` extracts 27 statistical aggregates:
   - 8 hold time features (mean, std, variance, median, min, max, skewness, kurtosis)
   - 8 inter-tap interval features (mean, std, variance, median, min, max, skewness, kurtosis)
   - 8 pressure features (mean, std, variance, median, min, max, skewness, kurtosis)
   - 3 activity features (event count, recording duration, typing rate)
2. **Inference Engine**: `TouchscreenInferenceEngine` standardizes raw features using frozen scaler means/scales and applies logistic regression coefficients and intercept:
   $$\text{score} = \text{intercept} + \sum_{i=1}^{27} w_i \cdot \frac{x_i - \mu_i}{\sigma_i}$$
   $$P(\text{PD}) = \frac{1}{1 + e^{-\text{score}}}, \quad P(\text{CNT}) = 1 - P(\text{PD})$$
3. Model parameters ($C=0.001$, threshold=0.50) are frozen and loaded from `assets/models/touchscreen_model.json`.

---

## Non-Diagnostic Wording & Disclaimers

All screens and results strictly comply with non-diagnostic rules:
- **Allowed Terms**: "Touchscreen Motor Pattern Screening", "Motor-pattern screening indicator", "Model-estimated pattern: PD-associated", "Model-estimated pattern: control-associated", "This experimental model produced a PD-associated pattern."
- **Prohibited Terms**: "You have Parkinson's", "Diagnosis", "Clinical validation", "Medical detection".
- **Mandatory Disclaimer**:
  > *"This is an experimental research prototype and is not a medical diagnosis. Results can be affected by typing style, device characteristics, and limited training data."*

---

## Data Privacy & Lifecycle

- **Memory-Only Scope**: Raw touch events and feature vectors are held strictly in temporary RAM during the active session.
- **Immediate Erasure**: Invoking `TRY AGAIN` or `reset()` completely purges raw events and typed text from memory.
- **No Persistence or Transmission**: No disk storage, local databases, cloud uploads, analytics, or backend network calls are implemented.
