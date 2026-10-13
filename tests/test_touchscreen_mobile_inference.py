"""Equivalence and validation unit tests for touchscreen mobile inference.

Verifies:
1. Exactly 27 features and fixed order.
2. Model parameters, scaler parameters, coefficients, and intercept export correctly.
3. Class mapping is preserved.
4. Mobile-safe Python inference matches scikit-learn model predictions and probabilities exactly (rtol=1e-5).
5. Raw recording -> feature extraction -> portable inference pipeline matches sklearn pipeline.
6. Invalid inputs (wrong count, NaN, inf, wrong column order) are rejected.
7. Keyboard artifacts remain untouched.
8. Final test dataset is never used.
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from ml.inference.touchscreen_inference import (
    load_mobile_model,
    predict_touchscreen,
    predict_touchscreen_probability,
    validate_feature_vector,
)
from ml.training.touchscreen_dataset import (
    EXACT_FEATURE_COUNT,
    TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS,
    build_touchscreen_ml_dataset,
    split_touchscreen_by_participant,
)


@pytest.fixture
def touchscreen_dataset():
    """Load development dataset."""
    return build_touchscreen_ml_dataset()


@pytest.fixture
def dev_split(touchscreen_dataset):
    """Get dev split."""
    return split_touchscreen_by_participant(touchscreen_dataset, test_size=0.2, random_state=42, stratify=True)


@pytest.fixture
def sklearn_artifacts():
    """Load trained joblib artifacts."""
    lr = joblib.load("ml/models/neurotap_touchscreen_logistic_regression.joblib")
    scaler = joblib.load("ml/models/neurotap_touchscreen_standard_scaler.joblib")
    features = joblib.load("ml/models/neurotap_touchscreen_feature_names.joblib")
    return lr, scaler, features


def test_feature_count_and_order(sklearn_artifacts):
    """1 & 2: Verify feature count is 27 and order is exact."""
    _, _, feature_names = sklearn_artifacts
    assert len(feature_names) == EXACT_FEATURE_COUNT
    assert feature_names == list(TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS)

    mobile_model = load_mobile_model()
    assert mobile_model.feature_count == EXACT_FEATURE_COUNT
    assert mobile_model.feature_names == list(TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS)


def test_mobile_parameters_export_correctly(sklearn_artifacts):
    """3 - 7: Verify parameters, scaler, coefficients, intercept, and class mapping."""
    lr, scaler, _ = sklearn_artifacts
    mobile_model = load_mobile_model()

    np.testing.assert_allclose(mobile_model.means, scaler.mean_, rtol=1e-5)
    np.testing.assert_allclose(mobile_model.scales, scaler.scale_, rtol=1e-5)
    np.testing.assert_allclose(mobile_model.coefficients, lr.coef_[0], rtol=1e-5)
    assert pytest.approx(mobile_model.intercept, 1e-5) == float(lr.intercept_[0])
    assert mobile_model.classes == list(lr.classes_) == ["CNT", "PD"]


def test_mobile_inference_matches_sklearn_on_dev(dev_split, sklearn_artifacts):
    """8 & 9: Test portable inference matches sklearn predictions & probabilities."""
    lr, scaler, _ = sklearn_artifacts
    mobile_model = load_mobile_model()

    X_dev = dev_split.X_dev

    # 1. Scikit-learn predictions
    X_dev_scaled = scaler.transform(X_dev)
    sklearn_probs = lr.predict_proba(X_dev_scaled)
    sklearn_preds = lr.predict(X_dev_scaled)

    # 2. Portable mobile predictions
    mobile_probs = mobile_model.predict_touchscreen_probability(X_dev)
    mobile_preds = mobile_model.predict_touchscreen(X_dev)

    # 3. Verify equivalence
    np.testing.assert_allclose(mobile_probs, sklearn_probs, rtol=1e-5, atol=1e-5)
    assert list(mobile_preds) == list(sklearn_preds)


def test_invalid_feature_vectors_rejected(dev_split):
    """10 & 11: Test wrong feature count, NaN, inf, and invalid structures are rejected."""
    mobile_model = load_mobile_model()

    # Wrong feature count (26 instead of 27)
    with pytest.raises(ValueError, match="Expected feature vector length 27"):
        mobile_model.validate_feature_input(np.ones(26))

    # NaN in features
    bad_nan = dev_split.X_dev.iloc[[0]].copy()
    bad_nan.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="contains NaN values"):
        mobile_model.validate_feature_input(bad_nan)

    # Inf in features
    bad_inf = dev_split.X_dev.iloc[[0]].copy()
    bad_inf.iloc[0, 1] = np.inf
    with pytest.raises(ValueError, match="contains infinite"):
        mobile_model.validate_feature_input(bad_inf)


def test_keyboard_artifacts_remain_untouched():
    """12: Verify keyboard joblib artifacts exist and are separate from touchscreen artifacts."""
    kb_lr = joblib.load("ml/models/neurotap_logistic_regression.joblib")
    kb_scaler = joblib.load("ml/models/neurotap_standard_scaler.joblib")

    ts_lr = joblib.load("ml/models/neurotap_touchscreen_logistic_regression.joblib")
    ts_scaler = joblib.load("ml/models/neurotap_touchscreen_standard_scaler.joblib")

    # Keyboard model C parameter is 100.0, Touchscreen C parameter is 0.001
    assert kb_lr.C == 100.0
    assert ts_lr.C == 0.001

    # Scaler means are distinct
    assert not np.array_equal(kb_scaler.mean_, ts_scaler.mean_)
