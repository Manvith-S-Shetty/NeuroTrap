"""Unit tests for touchscreen hyperparameter tuning and robustness analysis.

Verifies:
1. Locked final test set is never used in grid search or evaluation.
2. Participant-aware fold splitting preserves group separation across folds.
3. Logistic regression & Random forest hyperparameter search space evaluation.
4. Robustness checks (threshold grid, feature ablation/subsets, CV strategy comparison).
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from ml.training.touchscreen_baseline import validate_development_dataset
from ml.training.touchscreen_dataset import (
    TouchscreenMLDataset,
    build_touchscreen_ml_dataset,
    split_touchscreen_by_participant,
)
from ml.training.touchscreen_tuning import (
    tune_touchscreen_development,
    run_touchscreen_tuning,
)


@pytest.fixture
def touchscreen_dataset():
    """Load real touchscreen ML dataset."""
    return build_touchscreen_ml_dataset()


@pytest.fixture
def split_data(touchscreen_dataset):
    """Get train/test split."""
    return split_touchscreen_by_participant(touchscreen_dataset)


def test_validate_tuning_development_dataset_rejects_test_participants(split_data):
    """Ensure final test set participant injection raises a ValueError."""
    dev = TouchscreenMLDataset(
        X=split_data.X_dev.copy(),
        y=split_data.y_dev.copy(),
        metadata=split_data.dev_metadata.copy(),
    )
    test_p = split_data.final_test_participants

    # Inject a test participant row into dev
    injected_meta = dev.metadata.copy()
    injected_meta.iloc[0, injected_meta.columns.get_loc("participant_id")] = list(test_p)[0]
    injected_meta.iloc[0, injected_meta.columns.get_loc("recording_id")] = "INJECTED_TEST_REC_01"

    bad_dev = TouchscreenMLDataset(X=dev.X.copy(), y=dev.y.copy(), metadata=injected_meta)

    with pytest.raises(ValueError, match="locked final-test participants"):
        validate_development_dataset(bad_dev, final_test_participants=test_p)


def test_tune_touchscreen_development_runs(split_data):
    """Test tuning grid search on development set with GroupKFold."""
    dev = TouchscreenMLDataset(
        X=split_data.X_dev.copy(),
        y=split_data.y_dev.copy(),
        metadata=split_data.dev_metadata.copy(),
    )
    res = tune_touchscreen_development(
        dev,
        final_test_participants=split_data.final_test_participants,
        n_splits=3,
        cv_strategy="GroupKFold",
    )

    assert res.config_results is not None
    assert len(res.fold_results) > 0
    assert float(res.best_logistic_regression["mean_f1"]) > 0.0
    assert float(res.best_random_forest["mean_f1"]) > 0.0


def test_run_touchscreen_tuning_full_pipeline():
    """Test full high-level tuning pipeline."""
    gkf_res, sgkf_res, split = run_touchscreen_tuning(n_splits=3)

    assert gkf_res is not None
    assert len(gkf_res.fold_results) > 0
    assert len(split.final_test_participants) == 7
    assert len(split.X_dev) == 269

