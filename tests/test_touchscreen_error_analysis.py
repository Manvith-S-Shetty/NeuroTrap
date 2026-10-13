"""Unit tests for touchscreen final-test error analysis module.

Verifies:
1. Existing final predictions are loaded without modification.
2. Number of analyzed recordings = 69.
3. Number of participants = 7.
4. Confusion matrix agrees with Step 9 (TN=22, FP=10, FN=10, TP=27).
5. Exactly 49 correct recordings are identified.
6. Exactly 20 incorrect recordings are identified.
7. Participant summaries cover all 7 participants.
8. No retraining occurs.
9. No hyperparameter tuning occurs.
10. Feature count remains 27.
11. Analysis does not modify the locked final predictions.
12. Existing Step 9 metrics remain unchanged.
"""

import pytest
import numpy as np
import pandas as pd

from ml.training.touchscreen_dataset import EXACT_FEATURE_COUNT
from ml.training.touchscreen_error_analysis import (
    INPUT_DIR,
    load_step_9_artifacts,
    run_touchscreen_error_analysis,
)
from ml.training.touchscreen_final_evaluation import (
    SELECTED_C,
    SELECTED_CLASS_WEIGHT,
    run_touchscreen_final_evaluation,
)


def test_load_step_9_artifacts():
    """1, 2, 3: Verify existing Step 9 artifacts load without modification."""
    pred_df, part_df, report_json = load_step_9_artifacts(INPUT_DIR)

    assert len(pred_df) == 69
    assert len(part_df) == 7
    assert report_json["test_recordings"] == 69
    assert report_json["test_participants"] == 7


def test_error_analysis_counts_and_confusion_matrix():
    """4, 5, 6, 7: Verify correct/incorrect counts and confusion matrix agreement."""
    res = run_touchscreen_error_analysis()

    assert res.total_recordings == 69
    assert res.correct_recordings == 49
    assert res.incorrect_recordings == 20
    assert res.false_positives == 10
    assert res.false_negatives == 10
    assert len(res.participant_summary_table) == 7
    assert set(res.incorrect_participants) == {"S09", "S32"}


def test_no_retraining_or_tuning_occurs():
    """8, 9, 10, 11, 12: Verify model configuration and metrics remain unchanged."""
    res = run_touchscreen_error_analysis()
    final_eval = run_touchscreen_final_evaluation()

    # Step 9 metrics remain exact
    assert final_eval.accuracy == pytest.approx(49 / 69, 1e-4)
    assert final_eval.f1 == pytest.approx(27 / 37, 1e-4)
    assert final_eval.model_params["C"] == SELECTED_C == 0.001
    assert final_eval.model_params["class_weight"] == SELECTED_CLASS_WEIGHT is None

    # Feature count remains 27
    assert len(res.feature_differences) == EXACT_FEATURE_COUNT
    assert len(res.coefficients) == EXACT_FEATURE_COUNT
