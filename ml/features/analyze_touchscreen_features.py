"""Exploratory feature analysis for touchscreen 27-numerical-feature matrix.

Calculates basic statistics, missing/infinite detection, unique value counts,
Pearson correlations, PD vs Control descriptive comparisons, Cohen's d effect sizes,
IQR outlier analysis, and data integrity checks. Generates standalone plots saved under
ml/features/touchscreen_analysis_outputs/.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from touchscreen_feature_extraction import (  # type: ignore
        TOUCHSCREEN_METADATA_COLUMNS,
        TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS,
    )
    from preprocessing.touchscreen.data_loader import (  # type: ignore
        DEFAULT_TOUCHSCREEN_PACKAGE_ROOT,
        load_all_touchscreen_recordings,
    )
else:
    from .touchscreen_feature_extraction import (
        TOUCHSCREEN_METADATA_COLUMNS,
        TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS,
    )
    from ..preprocessing.touchscreen.data_loader import (
        DEFAULT_TOUCHSCREEN_PACKAGE_ROOT,
        load_all_touchscreen_recordings,
    )

DEFAULT_TOUCHSCREEN_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "touchscreen_analysis_outputs"
)


@dataclass(frozen=True)
class TouchscreenFeatureAnalysisResult:
    descriptive_stats: pd.DataFrame
    missing_infinite: pd.DataFrame
    unique_counts: pd.Series
    correlation_matrix: pd.DataFrame
    high_correlations_90: pd.DataFrame
    high_correlations_95: pd.DataFrame
    high_correlations_99: pd.DataFrame
    label_comparison: pd.DataFrame
    outliers: pd.DataFrame
    data_integrity: dict
    plot_files: list[Path]


def validate_touchscreen_analysis_inputs(X: pd.DataFrame, y: pd.Series) -> None:
    """Validate that X has exactly the 27 touchscreen features and y has valid labels."""
    missing_cols = set(TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS) - set(X.columns)
    if missing_cols:
        raise ValueError(f"Feature matrix X missing required columns: {sorted(missing_cols)}")

    if len(X.columns) != 27 or list(X.columns) != TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS:
        raise ValueError("Feature matrix X must contain exactly the 27 touchscreen numerical features in order.")

    non_numeric = [col for col in TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS if not pd.api.types.is_numeric_dtype(X[col])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns found in X: {non_numeric}")

    if y.isna().any():
        raise ValueError("Labels y contain missing values.")

    unsupported = sorted(set(y) - {"PD", "CNT"})
    if unsupported:
        raise ValueError(f"Unsupported class labels found in y: {unsupported}")


def calculate_descriptive_stats(X: pd.DataFrame) -> pd.DataFrame:
    """Analysis 1: Calculate basic descriptive statistics for every feature."""
    validate_touchscreen_analysis_inputs(X, pd.Series(["PD"] * len(X)))

    rows = []
    for col in TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS:
        series = X[col].dropna()
        q25 = float(series.quantile(0.25)) if not series.empty else float("nan")
        q75 = float(series.quantile(0.75)) if not series.empty else float("nan")
        iqr = q75 - q25 if not (np.isnan(q25) or np.isnan(q75)) else float("nan")

        rows.append(
            {
                "feature": col,
                "count": int(series.count()),
                "mean": float(series.mean()) if not series.empty else float("nan"),
                "std": float(series.std()) if len(series) >= 2 else float("nan"),
                "variance": float(series.var()) if len(series) >= 2 else float("nan"),
                "median": float(series.median()) if not series.empty else float("nan"),
                "min": float(series.min()) if not series.empty else float("nan"),
                "max": float(series.max()) if not series.empty else float("nan"),
                "q25": q25,
                "q75": q75,
                "iqr": iqr,
            }
        )

    return pd.DataFrame(rows)


def calculate_missing_infinite_values(X: pd.DataFrame) -> pd.DataFrame:
    """Analysis 2: Check NaN, +inf, -inf counts per feature."""
    validate_touchscreen_analysis_inputs(X, pd.Series(["PD"] * len(X)))

    rows = []
    for col in TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS:
        series = X[col]
        nan_cnt = int(series.isna().sum())
        arr = series.dropna().to_numpy(dtype=float)
        pos_inf_cnt = int(np.isposinf(arr).sum())
        neg_inf_cnt = int(np.isneginf(arr).sum())
        tot_cnt = nan_cnt + pos_inf_cnt + neg_inf_cnt

        rows.append(
            {
                "feature": col,
                "nan_count": nan_cnt,
                "pos_inf_count": pos_inf_cnt,
                "neg_inf_count": neg_inf_cnt,
                "total_missing_or_infinite": tot_cnt,
            }
        )

    return pd.DataFrame(rows)


def calculate_unique_value_counts(X: pd.DataFrame) -> pd.Series:
    """Analysis 3: Calculate unique value counts per feature."""
    validate_touchscreen_analysis_inputs(X, pd.Series(["PD"] * len(X)))
    return X[TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS].nunique(dropna=False)


def calculate_correlation_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Analysis 4: Calculate Pearson correlation matrix across all 27 numerical features."""
    validate_touchscreen_analysis_inputs(X, pd.Series(["PD"] * len(X)))
    return X[TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS].corr(method="pearson")


def filter_high_correlations(corr_matrix: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    """Filter distinct pair correlations where |r| >= threshold."""
    rows = []
    columns = list(corr_matrix.columns)
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1, col2 = columns[i], columns[j]
            val = float(corr_matrix.loc[col1, col2])
            if not np.isnan(val) and abs(val) >= threshold:
                rows.append(
                    {
                        "feature_1": col1,
                        "feature_2": col2,
                        "correlation": val,
                        "abs_correlation": abs(val),
                    }
                )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("abs_correlation", ascending=False).reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["feature_1", "feature_2", "correlation", "abs_correlation"])
    return df


def calculate_pd_vs_control_comparison(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Analysis 5: Calculate descriptive stats comparison between PD and Control groups."""
    validate_touchscreen_analysis_inputs(X, y)

    rows = []
    for col in TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS:
        cnt_series = X.loc[y == "CNT", col].dropna()
        pd_series = X.loc[y == "PD", col].dropna()

        cnt_mean = float(cnt_series.mean()) if not cnt_series.empty else float("nan")
        pd_mean = float(pd_series.mean()) if not pd_series.empty else float("nan")
        cnt_median = float(cnt_series.median()) if not cnt_series.empty else float("nan")
        pd_median = float(pd_series.median()) if not pd_series.empty else float("nan")
        cnt_std = float(cnt_series.std()) if len(cnt_series) >= 2 else float("nan")
        pd_std = float(pd_series.std()) if len(pd_series) >= 2 else float("nan")

        mean_diff = pd_mean - cnt_mean if not (np.isnan(pd_mean) or np.isnan(cnt_mean)) else float("nan")
        median_diff = pd_median - cnt_median if not (np.isnan(pd_median) or np.isnan(cnt_median)) else float("nan")

        # Cohen's d calculation
        n_cnt, n_pd = len(cnt_series), len(pd_series)
        if n_cnt >= 2 and n_pd >= 2 and not (np.isnan(cnt_std) or np.isnan(pd_std)):
            pooled_var = (((n_pd - 1) * (pd_std ** 2)) + ((n_cnt - 1) * (cnt_std ** 2))) / (n_pd + n_cnt - 2)
            pooled_std = np.sqrt(pooled_var)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else float("nan")
        else:
            cohens_d = float("nan")

        rows.append(
            {
                "feature": col,
                "control_mean": cnt_mean,
                "pd_mean": pd_mean,
                "control_median": cnt_median,
                "pd_median": pd_median,
                "control_std": cnt_std,
                "pd_std": pd_std,
                "mean_difference": mean_diff,
                "median_difference": median_diff,
                "cohens_d": float(cohens_d),
            }
        )

    return pd.DataFrame(rows)


def calculate_iqr_outliers(X: pd.DataFrame) -> pd.DataFrame:
    """Analysis 6: Calculate IQR rule outlier counts and bounds per feature."""
    validate_touchscreen_analysis_inputs(X, pd.Series(["PD"] * len(X)))

    total_rows = len(X)
    rows = []

    for col in TOUCHSCREEN_NUMERICAL_FEATURE_COLUMNS:
        series = X[col].dropna()
        if series.empty:
            rows.append(
                {
                    "feature": col,
                    "q25": float("nan"),
                    "q75": float("nan"),
                    "iqr": float("nan"),
                    "lower_bound": float("nan"),
                    "upper_bound": float("nan"),
                    "outlier_count": 0,
                    "outlier_percentage": 0.0,
                }
            )
            continue

        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr

        outliers_mask = (series < lower_bound) | (series > upper_bound)
        outlier_cnt = int(outliers_mask.sum())
        outlier_pct = float((outlier_cnt / total_rows) * 100.0) if total_rows > 0 else 0.0

        rows.append(
            {
                "feature": col,
                "q25": q25,
                "q75": q75,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": outlier_cnt,
                "outlier_percentage": outlier_pct,
            }
        )

    return pd.DataFrame(rows)


def check_data_integrity(
    X: pd.DataFrame,
    metadata: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """Analysis 8: Verify strict sample sizes, participant distributions, and feature schema."""
    validate_touchscreen_analysis_inputs(X, y)

    total_rows = len(X)
    total_meta_rows = len(metadata)
    total_y_rows = len(y)

    total_participants = metadata["participant_id"].nunique()
    pd_participants = metadata.loc[metadata["label"] == "PD", "participant_id"].nunique()
    cnt_participants = metadata.loc[metadata["label"] == "CNT", "participant_id"].nunique()

    pd_recordings = int((y == "PD").sum())
    cnt_recordings = int((y == "CNT").sum())

    metadata_in_X = set(TOUCHSCREEN_METADATA_COLUMNS).intersection(set(X.columns))

    integrity = {
        "exactly_338_feature_rows": total_rows == 338 and total_meta_rows == 338 and total_y_rows == 338,
        "exactly_33_participants": total_participants == 33,
        "pd_participants": pd_participants,
        "cnt_participants": cnt_participants,
        "pd_recordings": pd_recordings,
        "cnt_recordings": cnt_recordings,
        "exactly_27_numerical_features": len(X.columns) == 27,
        "no_metadata_in_X": len(metadata_in_X) == 0,
        "participant_ids_separate": "participant_id" in metadata.columns and "participant_id" not in X.columns,
    }

    return integrity


def generate_analysis_plots(
    X: pd.DataFrame,
    y: pd.Series,
    corr_matrix: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Analysis 7: Generate and save clean standalone plot images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files: list[Path] = []

    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Hold Features Distributions
    hold_cols = [col for col in X.columns if col.startswith("hold_")]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for idx, col in enumerate(hold_cols):
        sns.histplot(X[col], kde=True, ax=axes[idx], color="skyblue")
        axes[idx].set_title(col, fontsize=10)
        axes[idx].set_xlabel("")
    plt.tight_layout()
    hold_file = output_dir / "hold_features_distributions.png"
    plt.savefig(hold_file, dpi=150)
    plt.close()
    generated_files.append(hold_file)

    # 2. Inter-tap Features Distributions
    inter_cols = [col for col in X.columns if col.startswith("inter_tap_")]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for idx, col in enumerate(inter_cols):
        sns.histplot(X[col], kde=True, ax=axes[idx], color="salmon")
        axes[idx].set_title(col, fontsize=10)
        axes[idx].set_xlabel("")
    plt.tight_layout()
    inter_file = output_dir / "inter_tap_features_distributions.png"
    plt.savefig(inter_file, dpi=150)
    plt.close()
    generated_files.append(inter_file)

    # 3. Pressure Features Distributions
    press_cols = [col for col in X.columns if col.startswith("pressure_")]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for idx, col in enumerate(press_cols):
        sns.histplot(X[col], kde=True, ax=axes[idx], color="lightgreen")
        axes[idx].set_title(col, fontsize=10)
        axes[idx].set_xlabel("")
    plt.tight_layout()
    press_file = output_dir / "pressure_features_distributions.png"
    plt.savefig(press_file, dpi=150)
    plt.close()
    generated_files.append(press_file)

    # 4. Activity Features Distributions
    act_cols = ["event_count", "recording_duration", "typing_rate"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, col in enumerate(act_cols):
        sns.histplot(X[col], kde=True, ax=axes[idx], color="orchid")
        axes[idx].set_title(col, fontsize=11)
        axes[idx].set_xlabel("")
    plt.tight_layout()
    act_file = output_dir / "activity_features_distributions.png"
    plt.savefig(act_file, dpi=150)
    plt.close()
    generated_files.append(act_file)

    # 5. Correlation Heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, annot=False, linewidths=0.5)
    plt.title("Touchscreen Features Correlation Heatmap (27x27)", fontsize=14)
    plt.tight_layout()
    corr_file = output_dir / "correlation_heatmap.png"
    plt.savefig(corr_file, dpi=150)
    plt.close()
    generated_files.append(corr_file)

    # 6. PD vs Control Boxplots for Main Feature Groups
    group_representatives = ["hold_mean", "inter_tap_mean", "pressure_mean", "typing_rate"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    plot_df = X[group_representatives].copy()
    plot_df["label"] = y.values

    for idx, col in enumerate(group_representatives):
        sns.boxplot(data=plot_df, x="label", y=col, ax=axes[idx], hue="label", palette="Set2", legend=False)
        axes[idx].set_title(f"PD vs CNT: {col}", fontsize=11)
        axes[idx].set_xlabel("")

    plt.tight_layout()
    boxplot_file = output_dir / "pd_vs_cnt_boxplots.png"
    plt.savefig(boxplot_file, dpi=150)
    plt.close()
    generated_files.append(boxplot_file)

    return generated_files


def run_touchscreen_feature_analysis(
    package_root: Path | None = None,
    output_dir: Path | None = None,
    generate_plots: bool = True,
) -> TouchscreenFeatureAnalysisResult:
    """Execute complete touchscreen exploratory feature analysis."""
    root = package_root or DEFAULT_TOUCHSCREEN_PACKAGE_ROOT
    out_dir = output_dir or DEFAULT_TOUCHSCREEN_OUTPUT_DIR

    # Load dataset & extract features
    events_df, recs_df, malformed_df = load_all_touchscreen_recordings(root)

    # Import feature extractor function
    if __package__ in {None, ""}:
        from touchscreen_feature_extraction import extract_all_touchscreen_features
    else:
        from .touchscreen_feature_extraction import extract_all_touchscreen_features

    X, metadata, y = extract_all_touchscreen_features(events_df)

    descriptive_stats = calculate_descriptive_stats(X)
    missing_infinite = calculate_missing_infinite_values(X)
    unique_counts = calculate_unique_value_counts(X)
    corr_matrix = calculate_correlation_matrix(X)

    high_corr_90 = filter_high_correlations(corr_matrix, threshold=0.90)
    high_corr_95 = filter_high_correlations(corr_matrix, threshold=0.95)
    high_corr_99 = filter_high_correlations(corr_matrix, threshold=0.99)

    label_comparison = calculate_pd_vs_control_comparison(X, y)
    outliers = calculate_iqr_outliers(X)
    integrity = check_data_integrity(X, metadata, y)

    plot_files: list[Path] = []
    if generate_plots:
        plot_files = generate_analysis_plots(X, y, corr_matrix, out_dir)

    return TouchscreenFeatureAnalysisResult(
        descriptive_stats=descriptive_stats,
        missing_infinite=missing_infinite,
        unique_counts=unique_counts,
        correlation_matrix=corr_matrix,
        high_correlations_90=high_corr_90,
        high_correlations_95=high_corr_95,
        high_correlations_99=high_corr_99,
        label_comparison=label_comparison,
        outliers=outliers,
        data_integrity=integrity,
        plot_files=plot_files,
    )
