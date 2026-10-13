"""Portable Python inference engine for NeuroTap touchscreen model.

Reproduces exact scikit-learn Logistic Regression predictions and probabilities
using exported JSON parameters (`ml/models/touchscreen_mobile_model/model.json`).

Provides:
- load_mobile_model()
- validate_feature_vector()
- predict_touchscreen_probability()
- predict_touchscreen()

Does NOT require scikit-learn for inference.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

MODEL_JSON_PATH = Path(__file__).resolve().parents[1] / "models" / "touchscreen_mobile_model" / "model.json"


class TouchscreenMobileModel:
    """Portable lightweight inference model for touchscreen data."""

    def __init__(self, model_data: dict[str, Any]):
        self.metadata = model_data["metadata"]
        self.feature_names = list(model_data["feature_names"])
        self.feature_count = int(model_data["feature_count"])
        self.classes = list(model_data["classes"])

        self.means = np.array(model_data["scaler"]["mean"], dtype=np.float64)
        self.scales = np.array(model_data["scaler"]["scale"], dtype=np.float64)
        self.coefficients = np.array(model_data["logistic_regression"]["coefficients"], dtype=np.float64)
        self.intercept = float(model_data["logistic_regression"]["intercept"])

        if len(self.feature_names) != self.feature_count:
            raise ValueError(f"Feature names count ({len(self.feature_names)}) != feature_count ({self.feature_count})")
        if len(self.means) != self.feature_count or len(self.scales) != self.feature_count:
            raise ValueError("Scaler parameters length mismatch with feature_count")
        if len(self.coefficients) != self.feature_count:
            raise ValueError("Coefficients length mismatch with feature_count")

    def validate_feature_input(self, features: Sequence[float] | dict[str, float] | pd.Series | pd.DataFrame | np.ndarray) -> np.ndarray:
        """Validate input feature vector/matrix format, length, ordering, and values."""
        if isinstance(features, pd.DataFrame):
            if list(features.columns) != self.feature_names:
                missing = set(self.feature_names) - set(features.columns)
                extra = set(features.columns) - set(self.feature_names)
                if missing or extra:
                    raise ValueError(f"DataFrame columns mismatch. Missing: {missing}, Extra: {extra}")
                # Reorder columns to exact feature order
                features = features[self.feature_names]
            vec = features.to_numpy(dtype=np.float64)

        elif isinstance(features, pd.Series):
            if list(features.index) != self.feature_names:
                vec = np.array([features[col] for col in self.feature_names], dtype=np.float64)
            else:
                vec = features.to_numpy(dtype=np.float64)
            vec = vec.reshape(1, -1)

        elif isinstance(features, dict):
            if set(features.keys()) != set(self.feature_names):
                raise ValueError(f"Dictionary keys mismatch feature names. Missing: {set(self.feature_names) - set(features.keys())}")
            vec = np.array([[features[col] for col in self.feature_names]], dtype=np.float64)

        elif isinstance(features, (list, tuple, np.ndarray)):
            arr = np.asarray(features, dtype=np.float64)
            if arr.ndim == 1:
                if len(arr) != self.feature_count:
                    raise ValueError(f"Expected feature vector length {self.feature_count}, got {len(arr)}")
                vec = arr.reshape(1, -1)
            elif arr.ndim == 2:
                if arr.shape[1] != self.feature_count:
                    raise ValueError(f"Expected feature matrix shape (N, {self.feature_count}), got {arr.shape}")
                vec = arr
            else:
                raise ValueError(f"Unsupported array dimensions: {arr.ndim}")
        else:
            raise TypeError(f"Unsupported feature input type: {type(features)}")

        if np.isnan(vec).any():
            raise ValueError("Input feature vector contains NaN values.")
        if np.isinf(vec).any():
            raise ValueError("Input feature vector contains infinite (+inf or -inf) values.")

        return vec

    def predict_touchscreen_probability(
        self,
        features: Sequence[float] | dict[str, float] | pd.Series | pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """Calculate class probabilities using pure portable math.

        Returns array of shape (N, 2) where col 0 = P(CNT), col 1 = P(PD).
        """
        vec = self.validate_feature_input(features)

        # 1. Standardization: z_i = (x_i - mean_i) / scale_i
        z = (vec - self.means) / self.scales

        # 2. Decision score: score = intercept + sum(coef_i * z_i)
        scores = np.dot(z, self.coefficients) + self.intercept

        # 3. Sigmoid probability for PD class
        p_pd = 1.0 / (1.0 + np.exp(-scores))
        p_cnt = 1.0 - p_pd

        return np.column_stack([p_cnt, p_pd])

    def predict_touchscreen(
        self,
        features: Sequence[float] | dict[str, float] | pd.Series | pd.DataFrame | np.ndarray,
        threshold: float = 0.50,
    ) -> list[str]:
        """Predict class labels ('CNT' or 'PD')."""
        probs = self.predict_touchscreen_probability(features)
        return [self.classes[1] if p_pd >= threshold else self.classes[0] for p_pd in probs[:, 1]]


def load_mobile_model(model_path: Path = MODEL_JSON_PATH) -> TouchscreenMobileModel:
    """Load mobile model instance from exported JSON."""
    if not model_path.exists():
        raise FileNotFoundError(f"Mobile model JSON not found at {model_path}. Run export script first.")

    with open(model_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return TouchscreenMobileModel(data)


def validate_feature_vector(features: Any) -> np.ndarray:
    """Helper function to validate feature vector using default model."""
    model = load_mobile_model()
    return model.validate_feature_input(features)


def predict_touchscreen_probability(features: Any) -> np.ndarray:
    """Helper function for probability prediction."""
    model = load_mobile_model()
    return model.predict_touchscreen_probability(features)


def predict_touchscreen(features: Any) -> list[str]:
    """Helper function for class prediction."""
    model = load_mobile_model()
    return model.predict_touchscreen(features)
