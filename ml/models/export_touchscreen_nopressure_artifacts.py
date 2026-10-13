"""Export script for NeuroTap pressure-independent touchscreen mobile inference model artifacts.

Trains the selected Logistic Regression configuration (C=0.001, class_weight=None, solver='liblinear',
random_state=42) on the complete 269-recording development dataset ONLY.

Generates:
1. Touchscreen joblib artifacts:
   - ml/models/neurotap_touchscreen_nopressure_logistic_regression.joblib
   - ml/models/neurotap_touchscreen_nopressure_standard_scaler.joblib
   - ml/models/neurotap_touchscreen_nopressure_feature_names.joblib
   - ml/models/neurotap_touchscreen_nopressure_model_metadata.joblib

2. Mobile-portable JSON representation:
   - ml/models/touchscreen_nopressure_mobile_model/model.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from ml.features.touchscreen_nopressure_feature_extraction import (  # type: ignore
        TOUCHSCREEN_NOPRESSURE_NUMERICAL_FEATURE_COLUMNS,
    )
    from ml.training.touchscreen_nopressure_baseline import DEFAULT_RANDOM_STATE  # type: ignore
    from ml.training.touchscreen_nopressure_dataset import (  # type: ignore
        EXACT_NOPRESSURE_FEATURE_COUNT,
        build_touchscreen_nopressure_ml_dataset,
        split_touchscreen_nopressure_by_participant,
    )
    from ml.training.touchscreen_nopressure_final_evaluation import (  # type: ignore
        EXPECTED_TEST_PARTICIPANTS_LIST,
        SELECTED_C,
        SELECTED_CLASS_WEIGHT,
        SELECTED_SOLVER,
        SELECTED_THRESHOLD,
    )
else:
    from ml.features.touchscreen_nopressure_feature_extraction import (
        TOUCHSCREEN_NOPRESSURE_NUMERICAL_FEATURE_COLUMNS,
    )
    from ml.training.touchscreen_nopressure_baseline import DEFAULT_RANDOM_STATE
    from ml.training.touchscreen_nopressure_dataset import (
        EXACT_NOPRESSURE_FEATURE_COUNT,
        build_touchscreen_nopressure_ml_dataset,
        split_touchscreen_nopressure_by_participant,
    )
    from ml.training.touchscreen_nopressure_final_evaluation import (
        EXPECTED_TEST_PARTICIPANTS_LIST,
        SELECTED_C,
        SELECTED_CLASS_WEIGHT,
        SELECTED_SOLVER,
        SELECTED_THRESHOLD,
    )

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MOBILE_MODEL_DIR = MODELS_DIR / "touchscreen_nopressure_mobile_model"


def export_touchscreen_nopressure_artifacts() -> dict[str, Path]:
    """Train selected 19-feature model on dev set only and export joblib & JSON artifacts."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MOBILE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset & split (ensures zero test leakage)
    dataset = build_touchscreen_nopressure_ml_dataset()
    split = split_touchscreen_nopressure_by_participant(dataset, test_size=0.2, random_state=DEFAULT_RANDOM_STATE, stratify=True)

    X_dev = split.X_dev
    y_dev = split.y_dev

    # 2. Fit StandardScaler on dev set ONLY
    scaler = StandardScaler()
    X_dev_scaled = scaler.fit_transform(X_dev)

    # 3. Fit LogisticRegression on dev set ONLY
    model = LogisticRegression(
        C=SELECTED_C,
        class_weight=SELECTED_CLASS_WEIGHT,
        solver=SELECTED_SOLVER,
        max_iter=1000,
        random_state=DEFAULT_RANDOM_STATE,
    )
    model.fit(X_dev_scaled, y_dev)

    feature_names = list(TOUCHSCREEN_NOPRESSURE_NUMERICAL_FEATURE_COLUMNS)
    class_labels = list(model.classes_)  # Expected ["CNT", "PD"]

    metadata = {
        "model_name": "NeuroTap Touchscreen No-Pressure Logistic Regression",
        "modality": "touchscreen_nopressure",
        "version": "1.0.0",
        "random_state": DEFAULT_RANDOM_STATE,
        "selected_C": SELECTED_C,
        "class_weight": SELECTED_CLASS_WEIGHT,
        "solver": SELECTED_SOLVER,
        "selected_threshold": SELECTED_THRESHOLD,
        "development_recordings": len(X_dev),
        "development_participants": int(split.dev_metadata["participant_id"].nunique()),
        "final_test_recordings": len(split.X_test),
        "final_test_participants": int(split.test_metadata["participant_id"].nunique()),
        "final_test_participant_ids": EXPECTED_TEST_PARTICIPANTS_LIST,
        "feature_count": EXACT_NOPRESSURE_FEATURE_COUNT,
        "classes": class_labels,
        "pressure_features_excluded": True,
        "notes": "Pressure-independent statistical screening classification model based on typing dynamics.",
    }

    # Output artifact paths
    paths = {
        "model": MODELS_DIR / "neurotap_touchscreen_nopressure_logistic_regression.joblib",
        "scaler": MODELS_DIR / "neurotap_touchscreen_nopressure_standard_scaler.joblib",
        "feature_names": MODELS_DIR / "neurotap_touchscreen_nopressure_feature_names.joblib",
        "metadata": MODELS_DIR / "neurotap_touchscreen_nopressure_model_metadata.joblib",
        "mobile_json": MOBILE_MODEL_DIR / "model.json",
    }

    joblib.dump(model, paths["model"])
    joblib.dump(scaler, paths["scaler"])
    joblib.dump(feature_names, paths["feature_names"])
    joblib.dump(metadata, paths["metadata"])

    # Build mobile JSON representation
    mobile_json = {
        "model_name": metadata["model_name"],
        "modality": metadata["modality"],
        "version": metadata["version"],
        "feature_count": EXACT_NOPRESSURE_FEATURE_COUNT,
        "feature_names": feature_names,
        "classes": class_labels,
        "selected_threshold": SELECTED_THRESHOLD,
        "pressure_features_excluded": True,
        "development_recordings": len(X_dev),
        "development_participants": metadata["development_participants"],
        "final_test_recordings": metadata["final_test_recordings"],
        "final_test_participants": metadata["final_test_participants"],
        "final_test_participant_ids": EXPECTED_TEST_PARTICIPANTS_LIST,
        "scaler": {
            "means": scaler.mean_.tolist(),
            "scales": scaler.scale_.tolist(),
            "var": scaler.var_.tolist(),
        },
        "logistic_regression": {
            "coefficients": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
            "C": SELECTED_C,
            "class_weight": SELECTED_CLASS_WEIGHT,
            "solver": SELECTED_SOLVER,
        },
    }

    with open(paths["mobile_json"], "w") as f:
        json.dump(mobile_json, f, indent=2)

    return paths


def main() -> int:
    print("# Exporting Pressure-Independent Touchscreen Artifacts\n")
    paths = export_touchscreen_nopressure_artifacts()

    for name, path in paths.items():
        print(f"- Exported {name}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
