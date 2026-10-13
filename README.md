<div align="center">

# NeuroTrap

### Exploring motor-pattern differences through smartphone interaction dynamics

A research prototype that studies typing and touchscreen timing patterns with machine learning and brings experimental inference to a Flutter mobile app.

**Research prototype · On-device inference · Non-diagnostic**

</div>

---

## Overview

NeuroTrap explores whether timing characteristics captured during everyday-style typing or touchscreen tasks can reveal measurable differences in motor interaction patterns. It combines a Python-based machine-learning workflow with a Flutter application that collects interaction events, derives numerical features, and runs a model locally.

The project is intended for experimentation and learning about data collection, feature engineering, participant-aware evaluation, and mobile ML deployment. It is **not a medical device** and must not be used to diagnose Parkinson's disease or make healthcare decisions.

## How it works

```text
Typing / touchscreen task
          │
          ▼
Capture interaction timestamps
          │
          ▼
Validate and preprocess events
          │
          ▼
Extract timing features
          │
          ▼
Trained machine-learning model
          │
          ▼
Local inference in Flutter
          │
          ▼
Experimental, non-diagnostic result
```

## Key components

- **Interaction capture:** a Flutter interface collects input-event timing during a task. The touchscreen workflow uses a custom in-app keyboard so events can be recorded by the application.
- **Feature engineering:** Python extracts statistical summaries of key-hold or tap durations, intervals between events, and interaction rate. The touchscreen model has a separate 19-feature path that excludes pressure measurements.
- **Model development:** scikit-learn pipelines support preprocessing, model training, tuning, and evaluation with participant-aware splits.
- **On-device inference:** the mobile app uses a Dart implementation of the touchscreen no-pressure model so inference can run locally without sending a session to a prediction server.
- **Testing:** Python tests cover data/feature/model workflows, while Flutter tests cover mobile inference and session behaviour.

## Technology stack

| Area | Technologies |
|---|---|
| Mobile application | Flutter, Dart |
| Data processing | Python, Pandas, NumPy |
| Machine learning | scikit-learn, Logistic Regression, Random Forest experiments |
| Model evaluation | Participant-aware cross-validation, held-out participant test set |
| Mobile inference | Pure-Dart feature extraction and logistic-regression inference |
| Testing | pytest, Flutter test, flutter analyze |

## Dataset and evaluation

The project experiments with two distinct data sources and modelling paths:

### Keyboard dynamics

- Dataset: **neuroQWERTY MIT-CSXPD**
- 116 recordings from 85 participants
- 27 keyboard-derived features
- A participant-separated development/test setup was used to reduce participant leakage.

### Touchscreen dynamics

- Dataset: **smartphone touchscreen typing recordings** from the research dataset hosted on Zenodo (DOI: [10.5281/zenodo.2571623](https://doi.org/10.5281/zenodo.2571623))
- 338 recordings from 33 participants: 179 labelled PD and 159 labelled healthy controls in the project’s processed dataset.
- The locked final test set contains 69 recordings from 7 participants, with participants kept separate from development.
- Two touchscreen pipelines are retained: an original 27-feature version with pressure-derived features, and a separate 19-feature no-pressure version for devices where pressure readings are not usefully variable.

### Reported no-pressure touchscreen result

The project’s current evaluation report records the following metrics for the no-pressure model on its locked 69-recording test set:

| Metric | Reported result |
|---|---:|
| Accuracy | 85.51% |
| Precision (PD class) | 96.55% |
| Recall / sensitivity (PD class) | 75.68% |
| Specificity (control class) | 96.88% |
| F1 score (PD class) | 84.85% |
| ROC-AUC | 0.9544 |

These are **experimental dataset results, not clinical-performance estimates**. The test set contains only seven participants, so recordings should not be treated as seven-independent-person-equivalent evidence or as proof of generalisation to new populations or devices. Results can vary with the data, phone, recording conditions, and evaluation design.

## Repository layout

```text
NeuroTrap/
├── ml/
│   ├── data/              # Raw, intermediate and processed data
│   ├── preprocessing/    # Dataset loading and validation
│   ├── features/         # Keyboard and touchscreen feature extraction
│   ├── training/         # Training, tuning and evaluation workflows
│   ├── models/            # Model artifacts and export utilities
│   ├── inference/         # Python inference helpers
│   └── notebooks/         # Exploratory analysis
├── mobile/                # Flutter app and Dart inference
├── tests/                 # Python test suite
├── docs/                  # Project documentation
├── README.md
└── .gitignore
```

Some directories may be empty or evolve as development continues.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/Manvith-S-Shetty/NeuroTrap.git
cd NeuroTrap
```

### 2. Python environment

Use Python 3.10+ where compatible with the pinned dependencies.

**Windows PowerShell**

```powershell
python -m venv ml/venv
.\ml\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the packages listed in the project's dependency file if present. If a `requirements.txt` is added or maintained in a subdirectory, install from that file, for example:

```powershell
python -m pip install -r requirements.txt
```

Run the Python test suite from the repository root when the project's test dependencies are installed:

```bash
pytest -q
```

### 3. Flutter application

Install the Flutter SDK and configure an Android emulator or a physical Android device. Then:

```bash
cd mobile
flutter pub get
flutter analyze
flutter test
flutter run
```

Commands and dependency requirements may change as the project develops; consult the current files in `ml/` and `mobile/` if a setup command does not match your checkout.

## Data, privacy, and responsible use

- Do not commit passwords, API keys, signing files, personal data, or local environment files.
- Check the dataset's licence and redistribution conditions before publishing raw data. Dataset files should generally be obtained from their original source rather than copied into this repository.
- Treat model outputs as experimental predictions only. Do not use them to diagnose, rule out, or monitor a medical condition.
- Performance on a small research dataset does not establish clinical validity, demographic fairness, robustness across devices, or usefulness in real-world settings.

## Current limitations

- Research dataset sizes and participant counts are limited.
- Touchscreen measurements may vary with device hardware, operating system, screen, posture, and recording conditions.
- Pressure readings are not consistently useful across devices; the separate no-pressure path avoids using pressure as a model input.
- The prototype is not clinically validated and has not been established as a diagnostic tool.
- On-device inference and automated tests do not by themselves demonstrate real-world reliability.

## Contributing

Issues, reproducible bug reports, and improvements to documentation, evaluation methodology, and tests are welcome. Please avoid uploading private data or redistributing research datasets without permission.

## Disclaimer

**NeuroTrap is an educational and research prototype, not a medical device.** It does not provide medical advice, diagnosis, treatment, or a validated assessment of Parkinson's disease. Do not make healthcare decisions based on its output; consult a qualified healthcare professional about health concerns.

---

<div align="center">

Built to explore the intersection of mobile computing, behavioural signals, and machine learning.

</div>
