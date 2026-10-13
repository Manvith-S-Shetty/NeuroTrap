"""Read-only exploratory analysis for NeuroTap recording-level features.

The default analysis uses only the existing 93-recording development set. The
final held-out test participants are excluded from exploratory feature analysis.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from ml.training.baseline import CLASS_LABELS, validate_finite_features  # type: ignore
    from ml.training.dataset import (  # type: ignore
        FEATURE_COLUMNS,
        MLDataset,
        build_feature_table,
        build_ml_dataset,
        split_by_participant,
    )
    from ml.training.random_forest import validate_expected_features  # type: ignore
else:
    from ml.training.baseline import CLASS_LABELS, validate_finite_features
    from ml.training.dataset import FEATURE_COLUMNS, MLDataset, build_feature_table, build_ml_dataset, split_by_participant
    from ml.training.random_forest import validate_expected_features


DEFAULT_CORRELATION_THRESHOLD = 0.90
DEFAULT_OUTPUT_DIR = Path("ml") / "features" / "analysis_outputs"


@dataclass(frozen=True)
class FeatureAnalysisResult:
    dtypes: pd.Series
    descriptive_stats: pd.DataFrame
    missing_infinite: pd.DataFrame
    unique_counts: pd.Series
    correlation_matrix: pd.DataFrame
    high_correlations: pd.DataFrame
    label_comparison: pd.DataFrame
    outliers: pd.DataFrame
    development_recordings: int
    development_participants: int
    final_test_recordings: int
    final_test_participants: int
    final_test_overlap: int
    plot_files: list[Path]


def validate_feature_analysis_inputs(X: pd.DataFrame, y: pd.Series) -> None:
    """Validate the 27-feature matrix and CNT/PD labels without mutating input."""
    validate_expected_features(X, "feature analysis X")
    non_numeric = [column for column in FEATURE_COLUMNS if not pd.api.types.is_numeric_dtype(X[column])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns found: {non_numeric}")
    if y.isna().any():
        raise ValueError("Feature analysis labels contain missing values.")
    unsupported = sorted(set(y) - set(CLASS_LABELS))
    if unsupported:
        raise ValueError(f"Unsupported class labels found: {unsupported}")


def feature_dtypes(X: pd.DataFrame) -> pd.Series:
    validate_expected_features(X, "feature analysis X")
    return X.dtypes.copy()


def descriptive_statistics(X: pd.DataFrame) -> pd.DataFrame:
    validate_expected_features(X, "feature analysis X")
    stats = X.describe().T
    return stats.rename(columns={"50%": "median"})[
        ["count", "mean", "std", "min", "25%", "median", "75%", "max"]
    ]


def missing_infinite_values(X: pd.DataFrame) -> pd.DataFrame:
    validate_expected_features(X, "feature analysis X")
    numeric = X.to_numpy(dtype=float)
    inf_counts = np.isinf(numeric).sum(axis=0)
    return pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "missing_count": X[FEATURE_COLUMNS].isna().sum().to_numpy(),
            "infinite_count": inf_counts,
        }
    )


def unique_value_counts(X: pd.DataFrame) -> pd.Series:
    validate_expected_features(X, "feature analysis X")
    return X[FEATURE_COLUMNS].nunique(dropna=False)


def correlation_matrix(X: pd.DataFrame) -> pd.DataFrame:
    validate_expected_features(X, "feature analysis X")
    return X[FEATURE_COLUMNS].corr()


def highly_correlated_pairs(
    correlations: pd.DataFrame,
    threshold: float = DEFAULT_CORRELATION_THRESHOLD,
) -> pd.DataFrame:
    """Return feature pairs with absolute Pearson correlation above threshold."""
    if not 0 <= threshold <= 1:
        raise ValueError("correlation threshold must be between 0 and 1.")

    rows: list[dict[str, object]] = []
    columns = list(correlations.columns)
    for left_idx, left in enumerate(columns):
        for right in columns[left_idx + 1 :]:
            value = correlations.loc[left, right]
            if pd.notna(value) and abs(float(value)) >= threshold:
                rows.append(
                    {
                        "feature_1": left,
                        "feature_2": right,
                        "correlation": float(value),
                        "abs_correlation": abs(float(value)),
                    }
                )
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False).reset_index(drop=True)


def compare_by_label(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Compare descriptive CNT/PD distributions without significance claims."""
    validate_feature_analysis_inputs(X, y)
    labeled = X[FEATURE_COLUMNS].copy()
    labeled["label"] = y.reset_index(drop=True).to_numpy()

    rows: list[dict[str, object]] = []
    for feature in FEATURE_COLUMNS:
        cnt = labeled.loc[labeled["label"] == "CNT", feature]
        pd_values = labeled.loc[labeled["label"] == "PD", feature]
        rows.append(
            {
                "feature": feature,
                "CNT_mean": float(cnt.mean()),
                "PD_mean": float(pd_values.mean()),
                "CNT_median": float(cnt.median()),
                "PD_median": float(pd_values.median()),
                "CNT_std": float(cnt.std()),
                "PD_std": float(pd_values.std()),
                "mean_difference_PD_minus_CNT": float(pd_values.mean() - cnt.mean()),
            }
        )
    return pd.DataFrame(rows)


def iqr_outlier_counts(X: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
    """Count descriptive outliers using Q1 - 1.5*IQR and Q3 + 1.5*IQR."""
    validate_expected_features(X, "feature analysis X")
    rows: list[dict[str, object]] = []
    for feature in FEATURE_COLUMNS:
        values = X[feature]
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        mask = values.notna() & ((values < lower) | (values > upper))
        rows.append(
            {
                "feature": feature,
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "outlier_count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def build_development_dataset(
    ml_dataset: MLDataset | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[MLDataset, pd.DataFrame]:
    """Return development-only data plus final-test metadata using the existing split."""
    dataset = build_ml_dataset() if ml_dataset is None else ml_dataset
    split = split_by_participant(
        dataset,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    development = MLDataset(
        X=split.X_train.copy(),
        y=split.y_train.copy(),
        metadata=split.train_metadata.copy(),
    )
    return development, split.test_metadata.copy()


def generate_exploratory_plots(
    X: pd.DataFrame,
    y: pd.Series,
    correlations: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    """Generate exploratory plots under the analysis output directory."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_files: list[Path] = []

    distribution_path = output_dir / "feature_distributions.png"
    axes = X[FEATURE_COLUMNS].hist(bins=15, figsize=(18, 16))
    for axis in np.ravel(axes):
        axis.tick_params(labelsize=6)
        axis.title.set_size(8)
    plt.tight_layout()
    plt.savefig(distribution_path, dpi=150)
    plt.close()
    plot_files.append(distribution_path)

    comparison_path = output_dir / "cnt_vs_pd_feature_boxplots.png"
    labeled = X[FEATURE_COLUMNS].copy()
    labeled["label"] = y.reset_index(drop=True).to_numpy()
    fig, axes = plt.subplots(7, 4, figsize=(18, 24))
    for axis, feature in zip(np.ravel(axes), FEATURE_COLUMNS):
        labeled.boxplot(column=feature, by="label", ax=axis)
        axis.set_title(feature, fontsize=8)
        axis.set_xlabel("")
        axis.tick_params(labelsize=6)
    for axis in np.ravel(axes)[len(FEATURE_COLUMNS) :]:
        axis.axis("off")
    fig.suptitle("")
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=150)
    plt.close()
    plot_files.append(comparison_path)

    heatmap_path = output_dir / "correlation_heatmap.png"
    fig, axis = plt.subplots(figsize=(14, 12))
    image = axis.imshow(correlations, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(FEATURE_COLUMNS)))
    axis.set_xticklabels(FEATURE_COLUMNS, rotation=90, fontsize=6)
    axis.set_yticks(range(len(FEATURE_COLUMNS)))
    axis.set_yticklabels(FEATURE_COLUMNS, fontsize=6)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    plot_files.append(heatmap_path)

    return plot_files


def analyze_features(
    development: MLDataset,
    final_test_metadata: pd.DataFrame,
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    create_plots: bool = True,
) -> FeatureAnalysisResult:
    """Analyze development-set features without using final test labels."""
    X = development.X.copy(deep=True)
    y = development.y.copy(deep=True)
    metadata = development.metadata.copy(deep=True)
    validate_feature_analysis_inputs(X, y)
    validate_finite_features(X, "feature analysis X")

    dev_participants = set(metadata["participant_id"].astype(str))
    final_test_participants = set(final_test_metadata["participant_id"].astype(str))
    final_test_overlap = len(dev_participants & final_test_participants)
    if final_test_overlap:
        raise ValueError("Development analysis contains final held-out test participants.")

    correlations = correlation_matrix(X)
    plot_files = generate_exploratory_plots(X, y, correlations, output_dir=output_dir) if create_plots else []
    return FeatureAnalysisResult(
        dtypes=feature_dtypes(X),
        descriptive_stats=descriptive_statistics(X),
        missing_infinite=missing_infinite_values(X),
        unique_counts=unique_value_counts(X),
        correlation_matrix=correlations,
        high_correlations=highly_correlated_pairs(correlations, threshold=correlation_threshold),
        label_comparison=compare_by_label(X, y),
        outliers=iqr_outlier_counts(X),
        development_recordings=len(X),
        development_participants=len(dev_participants),
        final_test_recordings=len(final_test_metadata),
        final_test_participants=len(final_test_participants),
        final_test_overlap=final_test_overlap,
        plot_files=plot_files,
    )


def save_analysis_tables(result: FeatureAnalysisResult, output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    """Save analysis tables as CSV files under the analysis output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "feature_dtypes.csv": result.dtypes.astype(str).rename("dtype").to_frame(),
        "descriptive_statistics.csv": result.descriptive_stats,
        "missing_infinite_values.csv": result.missing_infinite,
        "unique_value_counts.csv": result.unique_counts.rename("unique_count").to_frame(),
        "correlation_matrix.csv": result.correlation_matrix,
        "highly_correlated_pairs.csv": result.high_correlations,
        "cnt_vs_pd_comparison.csv": result.label_comparison,
        "iqr_outlier_counts.csv": result.outliers,
    }
    paths: list[Path] = []
    for filename, table in outputs.items():
        path = output_dir / filename
        table.to_csv(path)
        paths.append(path)
    return paths


def main() -> int:
    feature_table = build_feature_table()
    ml_dataset = build_ml_dataset(feature_table)
    development, final_test_metadata = build_development_dataset(ml_dataset)
    result = analyze_features(development, final_test_metadata, create_plots=True)
    table_files = save_analysis_tables(result)

    missing_or_inf = result.missing_infinite[
        (result.missing_infinite["missing_count"] > 0) | (result.missing_infinite["infinite_count"] > 0)
    ]
    outliers = result.outliers[result.outliers["outlier_count"] > 0].sort_values(
        "outlier_count",
        ascending=False,
    )
    notable_differences = result.label_comparison.assign(
        abs_mean_difference=lambda frame: frame["mean_difference_PD_minus_CNT"].abs()
    ).sort_values("abs_mean_difference", ascending=False)

    print("# NeuroTap Feature Analysis")
    print("\n## Scope")
    print(f"Development recordings analyzed: {result.development_recordings}")
    print(f"Development participants analyzed: {result.development_participants}")
    print(f"Final held-out test recordings excluded: {result.final_test_recordings}")
    print(f"Final held-out test participants excluded: {result.final_test_participants}")
    print(f"Final-test participant overlap with analysis set: {result.final_test_overlap}")
    print("\n## Feature Columns")
    print(f"Feature count: {len(FEATURE_COLUMNS)}")
    print(FEATURE_COLUMNS)
    print("\n## Missing or Infinite Values")
    if missing_or_inf.empty:
        print("No missing or infinite feature values detected in the development set.")
    else:
        print(missing_or_inf.to_string(index=False))
    print(f"\n## Highly Correlated Feature Pairs (abs(correlation) >= {DEFAULT_CORRELATION_THRESHOLD:.2f})")
    if result.high_correlations.empty:
        print("No highly correlated feature pairs detected.")
    else:
        print(result.high_correlations.to_string(index=False))
    print("\n## Largest CNT vs PD Mean Differences")
    print(
        notable_differences[
            ["feature", "CNT_mean", "PD_mean", "CNT_median", "PD_median", "mean_difference_PD_minus_CNT"]
        ]
        .head(10)
        .to_string(index=False)
    )
    print("\n## IQR Outlier Counts")
    if outliers.empty:
        print("No IQR outliers detected.")
    else:
        print(outliers[["feature", "outlier_count", "lower_bound", "upper_bound"]].to_string(index=False))
    print("\n## Output Files")
    for path in table_files + result.plot_files:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
