# NeuroTap

NeuroTap is a non-diagnostic Parkinson's motor-risk screening prototype based on typing and touch dynamics. It is intended to explore whether interaction patterns from keyboard or touch input can support a screening-style motor-risk indicator.

## High-Level Architecture

The planned flow is:

```text
User typing
-> raw keyboard/touch events
-> feature extraction
-> ML model
-> on-device inference
-> non-diagnostic risk indicator
```

## Planned Stack

The planned ML stack is Python with Pandas, NumPy, scikit-learn, and a Random Forest model.

The planned mobile stack is Flutter and Dart.

Inference is intended to run on-device so that the mobile application can produce the screening/risk indicator locally.

## Medical Disclaimer

NeuroTap is a screening prototype only. Its output is a risk indicator and is not a medical diagnosis, clinical assessment, or replacement for professional medical advice.
