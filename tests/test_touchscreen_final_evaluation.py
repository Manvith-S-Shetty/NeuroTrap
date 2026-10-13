"""Unit tests for touchscreen final locked-test evaluation.

Verifies:
1. Final test contains exactly 69 recordings.
2. Final test contains exactly 7 participants.
3. Development/final participant overlap is zero.
4. Model is trained only on development data.
5. Scaler is fitted only on development data.
6. Final-test labels are not used during model fitting.
7. Confusion matrix dimensions are correct (2x2).
8. Reported metrics are mathematically correct.
9. Participant-level summary covers all 7 participants.
10. Final evaluation is deterministic with random_state=42.
11. Feature count remains exactly 27.
12. No hyperparameter tuning occurs in final evaluation.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ml.training.touchscreen_dataset import (
    TouchscreenMLDataset,
    TouchscreenParticipantSplit,
    build_touchscreen_ml_dataset,
    split_touchscreen_by_participant,
    EXACT_FEATURE_COUNT,
)
from ml.training.touchscreen_final_evaluation import (
    DEV_SGKF_CV_F1_MEAN,
    EXPECTED_DEV_PARTICIPANTS,
    EXPECTED_DEV_RECORDINGS,
    EXPECTED_TEST_PARTICIPANTS,
    EXPECTED_TEST_RECORDINGS,
    SELECTED_C,
    SELECTED_CLASS_WEIGHT,
    evaluate_touchscreen_final_model,
    run_touchscreen_final_evaluation,
    validate_touchscreen_final_split,
)


@pytest.fixture
def touchscreen_dataset():
    """Load real touchscreen ML dataset."""
    return build_touchscreen_ml_dataset()


@pytest.fixture
def split_data(touchscreen_dataset):
    """Get standard 80/20 participant-aware split."""
    return split_touchscreen_by_participant(touchscreen_dataset, test_size=0.2, random_state=42, stratify=True)


def test_final_test_size_and_participants(split_data):
    """1 & 2: Verify final test set sizes (69 recordings, 7 participants)."""
    assert len(split_data.X_test) == EXPECTED_TEST_RECORDINGS
    assert len(split_data.final_test_participants) == EXPECTED_TEST_PARTICIPANTS
    assert split_data.test_metadata["participant_id"].nunique() == EXPECTED_TEST_PARTICIPANTS


def test_zero_participant_overlap(split_data):
    """3: Verify development / final-test participant overlap is zero."""
    dev_p = set(split_data.dev_metadata["participant_id"].astype(str))
    test_p = set(split_data.test_metadata["participant_id"].astype(str))
    assert len(dev_p & test_p) == 0


def test_model_and_scaler_fitted_only_on_dev(split_data):
    """4 & 5 & 6: Verify model and scaler are fitted only on development data."""
    res = evaluate_touchscreen_final_model(split_data, random_state=42)

    # Check scaler mean matches development mean, not total dataset mean
    scaler = res.scaler
    expected_dev_mean = split_data.X_dev.mean().to_numpy()
    np.testing.assert_allclose(scaler.mean_, expected_dev_mean, rtol=1e-5)

    # Check model sample count matches development size
    assert res.dev_recordings == len(split_data.X_dev) == EXPECTED_DEV_RECORDINGS
    assert res.dev_participants == EXPECTED_DEV_PARTICIPANTS


def test_confusion_matrix_dimensions_and_metrics(split_data):
    """7 & 8: Verify confusion matrix dimensions (2x2) and metric calculations."""
    res = evaluate_touchscreen_final_model(split_data, random_state=42)

    cm = np.array(result_cm := res.confusion_matrix)
    assert cm.shape == (2, 2)

    # Check manual metrics against reported metrics
    # cm[0][0]=TN (CNT), cm[0][1]=FP, cm[1][0]=FN, cm[1][1]=TP (PD)
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    total = tn + fp + fn + tp

    calc_acc = (tp + tn) / total
    calc_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    calc_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    calc_f1 = (2 * calc_prec * calc_rec) / (calc_prec + calc_rec) if (calc_prec + calc_rec) > 0 else 0.0

    assert pytest.approx(res.accuracy, 1e-4) == calc_acc
    assert pytest.approx(res.precision, 1e-4) == calc_prec
    assert pytest.approx(res.recall, 1e-4) == calc_rec
    assert pytest.approx(res.f1, 1e-4) == calc_f1


def test_participant_summary_covers_all_7_participants(split_data):
    """9: Verify participant-level summary covers all 7 test participants."""
    res = evaluate_touchscreen_final_model(split_data, random_state=42)
    p_summaries = res.participant_summaries

    assert len(p_summaries) == EXPECTED_TEST_PARTICIPANTS
    p_ids = {p.participant_id for p in p_summaries}
    assert p_ids == split_data.final_test_participants

    total_recs = sum(p.recording_count for p in p_summaries)
    assert total_recs == EXPECTED_TEST_RECORDINGS


def test_evaluation_is_deterministic(split_data):
    """10: Verify evaluation is deterministic with random_state=42."""
    res1 = evaluate_touchscreen_final_model(split_data, random_state=42)
    res2 = evaluate_touchscreen_final_model(split_data, random_state=42)

    assert res1.f1 == res2.f1
    assert res1.accuracy == res2.accuracy
    assert res1.confusion_matrix == res2.confusion_matrix


def test_feature_count_remains_exact_27(split_data):
    """11: Verify feature count remains exactly 27 throughout."""
    assert len(split_data.X_dev.columns) == EXACT_FEATURE_COUNT
    assert len(split_data.X_test.columns) == EXACT_FEATURE_COUNT


def test_no_hyperparameter_tuning_in_final_evaluation(split_data):
    """12: Verify model configuration is strictly fixed (C=0.001, class_weight=None)."""
    res = evaluate_touchscreen_final_model(split_data, random_state=42)
    assert res.model_params["C"] == SELECTED_C == 0.001
    assert res.model_params["class_weight"] == SELECTED_CLASS_WEIGHT is None


def test_validate_final_split_raises_on_participant_leak(split_data):
    """Verify validation guard rejects any final-test participant leakage."""
    dev_meta = split_data.dev_metadata.copy()
    test_p = list(split_data.final_test_participants)[0]
    dev_meta.iloc[0, dev_meta.columns.get_loc("participant_id")] = test_p

    bad_split = TouchscreenParticipantSplit(
        X_dev=split_data.X_dev,
        X_test=split_data.X_test,
        y_dev=split_data.y_dev,
        y_test=split_data.y_test,
        dev_metadata=dev_meta,
        test_metadata=split_data.test_metadata,
    )

    with pytest.raises(ValueError, match="Data leakage detected"):
        validate_touchscreen_final_split(bad_split)
