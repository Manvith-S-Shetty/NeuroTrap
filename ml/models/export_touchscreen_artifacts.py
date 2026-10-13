"""Export script for NeuroTrap touchscreen mobile inference model artifacts.

Trains the selected Logistic Regression configuration (C=0.001, class_weight=None,
random_state=42) on the complete 269-recording development dataset ONLY.

Generates:
1. Touchscreen joblib artifacts:
   - ml/models/neurotap_touchscreen_logistic_regression.joblib
   - ml/models/neurotap_touchscreen_standard_scaler.joblib
   - ml/models/neurotap_touchscreen_feature_names.joblib
   - ml/models/neurotap_touchscreen_model_metadata.joblib

2. Mobile-portable JSON representation:
   - ml/models/touchscreen_mobile_model/model.json
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
    from ml.training.touchscreen_baseline import DEFAULT_RANDOM_STATE, validate_finite_features  # type: ignore
    from ml.training.touchscreen_dataset import (  # type: ignore
        EXACT_FEATURE_COUNT,
        TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS,
        build_touchscreen_ml_dataset,
        split_touchscreen_by_participant,
    )
    from ml.training.touchscreen_final_evaluation import (  # type: ignore
        SELECTED_C,
        SELECTED_CLASS_WEIGHT,
    )
else:
    from ml.training.touchscreen_baseline import DEFAULT_RANDOM_STATE, validate_finite_features
    from ml.training.touchscreen_dataset import (
        EXACT_FEATURE_COUNT,
        TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS,
        build_touchscreen_ml_dataset,
        split_touchscreen_by_participant,
    )
    from ml.training.touchscreen_final_evaluation import (
        SELECTED_C,
        SELECTED_CLASS_WEIGHT,
    )

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MOBILE_MODEL_DIR = MODELS_DIR / "touchscreen_mobile_model"


def export_touchscreen_model_artifacts() -> dict[str, Path]:
    """Train selected model on dev set only and export joblib & JSON artifacts."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MOBILE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset & split (ensures zero test leakage)
    dataset = build_touchscreen_ml_dataset()
    split = split_touchscreen_by_participant(dataset, test_size=0.2, random_state=DEFAULT_RANDOM_STATE, stratify=True)

    X_dev = split.X_dev
    y_dev = split.y_dev

    # 2. Fit StandardScaler on dev set ONLY
    scaler = StandardScaler()
    X_dev_scaled = scaler.fit_transform(X_dev)

    # 3. Fit LogisticRegression on dev set ONLY
    model = LogisticRegression(
        C=SELECTED_C,
        class_weight=SELECTED_CLASS_WEIGHT,
        max_iter=1000,
        random_state=DEFAULT_RANDOM_STATE,
    )
    model.fit(X_dev_scaled, y_dev)

    feature_names = list(TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS)
    class_labels = list(model.classes_)  # Expected ["CNT", "PD"]

    metadata = {
        "model_name": "NeuroTap Touchscreen Logistic Regression",
        "modality": "touchscreen",
        "version": "1.0.0",
        "random_state": DEFAULT_RANDOM_STATE,
        "selected_C": SELECTED_C,
        "class_weight": SELECTED_CLASS_WEIGHT,
        "development_recordings": len(X_dev),
        "development_participants": int(split.dev_metadata["participant_id"].nunique()),
        "feature_count": EXACT_FEATURE_COUNT,
        "classes": class_labels,
    }

    # 4. Save joblib artifacts
    lr_path = MODELS_DIR / "neurotap_touchscreen_logistic_regression.joblib"
    scaler_path = MODELS_DIR / "neurotap_touchscreen_standard_scaler.joblib"
    features_path = MODELS_DIR / "neurotap_touchscreen_feature_names.joblib"
    meta_path = MODELS_DIR / "neurotap_touchscreen_model_metadata.joblib"

    joblib.dump(model, lr_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_names, features_path)
    joblib.dump(metadata, meta_path)

    # 5. Export portable JSON representation with high floating-point precision
    coefficients = model.coef_[0].tolist()
    intercept = float(model.intercept_[0])
    means = scaler.mean_.tolist()
    scales = scaler.scale_.tolist()

    mobile_json = {
        "metadata": metadata,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "classes": class_labels,
        "scaler": {
            "mean": means,
            "scale": scales,
        },
        "logistic_regression": {
            "coefficients": coefficients,
            "intercept": intercept,
        },
        "inference_formula": {
            "standardization": "z_i = (x_i - mean_i) / scale_i",
            "decision_score": "score = intercept + sum(coefficient_i * z_i)",
            "sigmoid_probability": "p_PD = 1 / (1 + exp(-score))",
            "class_mapping": {
                "p_PD": "Probability of Parkinson's Disease (PD)",
                "p_CNT": "1.0 - p_PD (Probability of Control)",
                "decision_rule": "PD if p_PD >= 0.50 else CNT",
            },
        },
    }

    json_path = MOBILE_MODEL_DIR / "model.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(mobile_json, f, indent=2)

    return {
        "logistic_regression": lr_path,
        "scaler": scaler_path,
        "feature_names": features_path,
        "metadata": meta_path,
        "mobile_json": json_path,
    }


def main() -> int:
    exported = export_touchscreen_model_artifacts()
    print("============================================================")
    print("NeuroTap Touchscreen Model Artifact Export Complete")
    print("============================================================")
    for name, path in exported.items():
        print(f"  {name:<20}: {path}")
    print("============================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
