"""Unit tests for touchscreen feature analysis module."""

import numpy as np
import pandas as pd
import pytest

from ml.features.analyze_touchscreen_features import (
    calculate_correlation_matrix,
    calculate_descriptive_stats,
    calculate_iqr_outliers,
    calculate_missing_infinite_values,
    calculate_pd_vs_control_comparison,
    calculate_unique_value_counts,
    check_data_integrity,
    filter_high_correlations,
    run_touchscreen_feature_analysis,
    validate_touchscreen_analysis_inputs,
)
from ml.features.touchscreen_feature_extraction import (
    TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS,
    extract_all_touchscreen_features,
)
from ml.preprocessing.touchscreen.data_loader import (
    DEFAULT_TOUCHSCREEN_PACKAGE_ROOT,
    load_all_touchscreen_recordings,
)


@pytest.fixture
def dummy_feature_matrix_and_labels() -> tuple[pd.DataFrame, pd.Series]:
    """Fixture providing a mock 27-feature matrix and labels."""
    np.random.seed(42)
    n = 20
    data = {}
    for i, col in enumerate(TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS):
        data[col] = np.random.normal(loc=i * 0.1, scale=1.0, size=n)

    X = pd.DataFrame(data)
    y = pd.Series(["PD"] * 10 + ["CNT"] * 10, name="label")
    return X, y


def test_exact_27_feature_names(dummy_feature_matrix_and_labels: tuple[pd.DataFrame, pd.Series]):
    """Test validation of exact 27 touchscreen feature names."""
    X, y = dummy_feature_matrix_and_labels
    validate_touchscreen_analysis_inputs(X, y)

    # Test missing column error
    bad_X = X.drop(columns=["hold_mean"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_touchscreen_analysis_inputs(bad_X, y)


def test_basic_statistics_calculation(dummy_feature_matrix_and_labels: tuple[pd.DataFrame, pd.Series]):
    """Test calculation of basic descriptive statistics (count, mean, std, median, etc.)."""
    X, _ = dummy_feature_matrix_and_labels
    stats = calculate_descriptive_stats(X)

    assert len(stats) == 27
    assert list(stats.columns) == [
        "feature",
        "count",
        "mean",
        "std",
        "variance",
        "median",
        "min",
        "max",
        "q25",
        "q75",
        "iqr",
    ]
    assert (stats["count"] == 20).all()


def test_missing_infinite_detection(dummy_feature_matrix_and_labels: tuple[pd.DataFrame, pd.Series]):
    """Test detection of NaN, +inf, and -inf values."""
    X, _ = dummy_feature_matrix_and_labels
    X_modified = X.copy()
    X_modified.loc[0, "hold_mean"] = np.nan
    X_modified.loc[1, "hold_mean"] = np.inf
    X_modified.loc[2, "hold_mean"] = -np.inf

    report = calculate_missing_infinite_values(X_modified)
    hold_row = report[report["feature"] == "hold_mean"].iloc[0]

    assert hold_row["nan_count"] == 1
    assert hold_row["pos_inf_count"] == 1
    assert hold_row["neg_inf_count"] == 1
    assert hold_row["total_missing_or_infinite"] == 3


def test_unique_value_counts(dummy_feature_matrix_and_labels: tuple[pd.DataFrame, pd.Series]):
    """Test calculation of unique value counts."""
    X, _ = dummy_feature_matrix_and_labels
    counts = calculate_unique_value_counts(X)
    assert len(counts) == 27
    assert (counts == 20).all()


def test_correlation_matrix_and_high_correlation_filtering():
    """Test Pearson correlation matrix calculation and high correlation filtering."""
    np.random.seed(42)
    base = np.random.normal(size=20)
    data = {}
    for col in TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS:
        data[col] = base + np.random.normal(scale=0.01, size=20)

    # Make hold_std almost perfectly correlated with hold_mean
    data["hold_std"] = data["hold_mean"] * 1.0001

    X = pd.DataFrame(data)
    corr = calculate_correlation_matrix(X)
    assert corr.shape == (27, 27)

    high_corr_99 = filter_high_correlations(corr, threshold=0.99)
    assert not high_corr_99.empty
    pair = high_corr_99.iloc[0]
    assert set([pair["feature_1"], pair["feature_2"]]) == {"hold_mean", "hold_std"}
    assert pair["abs_correlation"] >= 0.99


def test_pd_vs_control_descriptive_stats(dummy_feature_matrix_and_labels: tuple[pd.DataFrame, pd.Series]):
    """Test PD vs Control mean, median, std, difference, and Cohen's d calculation."""
    X, y = dummy_feature_matrix_and_labels
    comp = calculate_pd_vs_control_comparison(X, y)

    assert len(comp) == 27
    assert "cohens_d" in comp.columns
    assert "mean_difference" in comp.columns
    assert "median_difference" in comp.columns


def test_iqr_outliers_calculation(dummy_feature_matrix_and_labels: tuple[pd.DataFrame, pd.Series]):
    """Test calculation of 1.5 * IQR outlier bounds and counts."""
    X, _ = dummy_feature_matrix_and_labels
    X_outlier = X.copy()
    # Insert extreme outlier
    X_outlier.loc[0, "hold_mean"] = 9999.0

    outliers = calculate_iqr_outliers(X_outlier)
    hold_row = outliers[outliers["feature"] == "hold_mean"].iloc[0]

    assert hold_row["outlier_count"] >= 1
    assert hold_row["outlier_percentage"] > 0.0


def test_no_input_mutation(dummy_feature_matrix_and_labels: tuple[pd.DataFrame, pd.Series]):
    """Test that input DataFrame X and y are not mutated by analysis functions."""
    X, y = dummy_feature_matrix_and_labels
    X_copy = X.copy(deep=True)
    y_copy = y.copy(deep=True)

    _ = calculate_descriptive_stats(X)
    _ = calculate_correlation_matrix(X)
    _ = calculate_pd_vs_control_comparison(X, y)
    _ = calculate_iqr_outliers(X)

    pd.testing.assert_frame_equal(X, X_copy)
    pd.testing.assert_series_equal(y, y_copy)


def test_real_dataset_analysis_integration():
    """Integration test verifying full touchscreen feature analysis on real dataset."""
    if not DEFAULT_TOUCHSCREEN_PACKAGE_ROOT.exists():
        pytest.skip("Touchscreen raw dataset package root not available.")

    res = run_touchscreen_feature_analysis(generate_plots=False)

    assert res.data_integrity["exactly_338_feature_rows"] is True
    assert res.data_integrity["exactly_33_participants"] is True
    assert res.data_integrity["pd_participants"] == 18
    assert res.data_integrity["cnt_participants"] == 15
    assert res.data_integrity["pd_recordings"] == 179
    assert res.data_integrity["cnt_recordings"] == 159
    assert res.data_integrity["exactly_27_numerical_features"] is True
    assert res.data_integrity["no_metadata_in_X"] is True

    assert len(res.descriptive_stats) == 27
    assert res.missing_infinite["total_missing_or_infinite"].sum() == 0
