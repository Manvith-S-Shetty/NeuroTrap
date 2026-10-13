import numpy as np
import pandas as pd
import pytest

from ml.features.analyze_features import (
    analyze_features,
    compare_by_label,
    correlation_matrix,
    highly_correlated_pairs,
    iqr_outlier_counts,
    missing_infinite_values,
    validate_feature_analysis_inputs,
)
from ml.training.dataset import FEATURE_COLUMNS, MLDataset


def make_feature_frame(rows: int = 6) -> pd.DataFrame:
    data = []
    for row_idx in range(rows):
        row = {}
        for column_idx, column in enumerate(FEATURE_COLUMNS):
            row[column] = float(row_idx + column_idx / 10)
        data.append(row)
    return pd.DataFrame(data, columns=FEATURE_COLUMNS)


def make_dataset(rows: int = 6) -> MLDataset:
    return MLDataset(
        X=make_feature_frame(rows),
        y=pd.Series(["CNT", "PD"] * (rows // 2), name="label"),
        metadata=pd.DataFrame(
            {
                "participant_id": [f"p{idx}" for idx in range(rows)],
                "recording_id": [f"r{idx}" for idx in range(rows)],
            }
        ),
    )


def test_feature_list_is_exactly_existing_27_features() -> None:
    dataset = make_dataset()

    validate_feature_analysis_inputs(dataset.X, dataset.y)

    assert list(dataset.X.columns) == FEATURE_COLUMNS
    assert len(dataset.X.columns) == 27


def test_numeric_validation_rejects_non_numeric_feature() -> None:
    dataset = make_dataset()
    dataset.X["hold_mean"] = "not numeric"

    with pytest.raises(ValueError, match="Non-numeric"):
        validate_feature_analysis_inputs(dataset.X, dataset.y)


def test_missing_and_infinite_detection() -> None:
    X = make_feature_frame()
    X.loc[0, "hold_mean"] = np.nan
    X.loc[1, "flight_mean"] = np.inf

    result = missing_infinite_values(X)

    hold_row = result.loc[result["feature"] == "hold_mean"].iloc[0]
    flight_row = result.loc[result["feature"] == "flight_mean"].iloc[0]
    assert hold_row["missing_count"] == 1
    assert hold_row["infinite_count"] == 0
    assert flight_row["missing_count"] == 0
    assert flight_row["infinite_count"] == 1


def test_correlation_calculation_and_high_correlation_detection() -> None:
    X = make_feature_frame()
    X["hold_mean"] = [1, 2, 3, 4, 5, 6]
    X["hold_std"] = [2, 4, 6, 8, 10, 12]
    X["hold_variance"] = [6, 5, 4, 3, 2, 1]

    correlations = correlation_matrix(X)
    pairs = highly_correlated_pairs(correlations, threshold=0.90)

    assert correlations.loc["hold_mean", "hold_std"] == pytest.approx(1.0)
    assert not pairs[
        (pairs["feature_1"] == "hold_mean") & (pairs["feature_2"] == "hold_std")
    ].empty


def test_cnt_pd_grouping() -> None:
    X = make_feature_frame(4)
    X["hold_mean"] = [1.0, 10.0, 3.0, 30.0]
    y = pd.Series(["CNT", "PD", "CNT", "PD"], name="label")

    comparison = compare_by_label(X, y)

    hold = comparison.loc[comparison["feature"] == "hold_mean"].iloc[0]
    assert hold["CNT_mean"] == pytest.approx(2.0)
    assert hold["PD_mean"] == pytest.approx(20.0)
    assert hold["CNT_median"] == pytest.approx(2.0)
    assert hold["PD_median"] == pytest.approx(20.0)


def test_iqr_outlier_counts() -> None:
    X = make_feature_frame(8)
    X["hold_mean"] = [1, 1, 1, 1, 1, 1, 1, 100]

    outliers = iqr_outlier_counts(X)

    hold = outliers.loc[outliers["feature"] == "hold_mean"].iloc[0]
    assert hold["outlier_count"] == 1


def test_analysis_does_not_mutate_raw_inputs() -> None:
    dataset = make_dataset()
    final_test_metadata = pd.DataFrame({"participant_id": ["heldout"]})
    original_X = dataset.X.copy(deep=True)
    original_y = dataset.y.copy(deep=True)
    original_metadata = dataset.metadata.copy(deep=True)

    analyze_features(dataset, final_test_metadata, create_plots=False)

    pd.testing.assert_frame_equal(dataset.X, original_X)
    pd.testing.assert_series_equal(dataset.y, original_y)
    pd.testing.assert_frame_equal(dataset.metadata, original_metadata)


def test_final_test_participants_are_excluded_from_development_analysis() -> None:
    dataset = make_dataset()
    final_test_metadata = pd.DataFrame({"participant_id": ["p0"]})

    with pytest.raises(ValueError, match="final held-out test participants"):
        analyze_features(dataset, final_test_metadata, create_plots=False)


def test_analysis_reports_final_test_exclusion_when_no_overlap() -> None:
    dataset = make_dataset()
    final_test_metadata = pd.DataFrame({"participant_id": ["heldout_1", "heldout_2"]})

    result = analyze_features(dataset, final_test_metadata, create_plots=False)

    assert result.development_recordings == 6
    assert result.development_participants == 6
    assert result.final_test_participants == 2
    assert result.final_test_overlap == 0
