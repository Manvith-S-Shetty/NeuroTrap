import numpy as np
import pandas as pd
import pytest

from ml.training.dataset import FEATURE_COLUMNS, ParticipantSplit
from ml.training.final_evaluation import SELECTED_LOGISTIC_C
from ml.training.model_artifact import (
    load_model_artifact,
    predict_single_recording,
    train_artifact_from_split,
)
from ml.training.random_forest import DEFAULT_RANDOM_STATE


def make_feature_frame(rows: int, offset: float = 0.0) -> pd.DataFrame:
    data = []
    for row_idx in range(rows):
        row = {}
        base = offset + row_idx
        for column_idx, column in enumerate(FEATURE_COLUMNS):
            row[column] = float(base + column_idx / 100)
        data.append(row)
    return pd.DataFrame(data, columns=FEATURE_COLUMNS)


def make_split() -> ParticipantSplit:
    X_train = pd.concat(
        [
            make_feature_frame(4, offset=0.0),
            make_feature_frame(4, offset=10.0),
        ],
        ignore_index=True,
    )
    X_test = pd.concat(
        [
            make_feature_frame(2, offset=20.0),
            make_feature_frame(2, offset=30.0),
        ],
        ignore_index=True,
    )
    return ParticipantSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=pd.Series(["CNT"] * 4 + ["PD"] * 4, name="label"),
        y_test=pd.Series(["CNT", "CNT", "PD", "PD"], name="label"),
        train_metadata=pd.DataFrame(
            {
                "participant_id": [f"train_p{idx}" for idx in range(8)],
                "recording_id": [f"train_r{idx}" for idx in range(8)],
            }
        ),
        test_metadata=pd.DataFrame(
            {
                "participant_id": [f"test_p{idx}" for idx in range(4)],
                "recording_id": [f"test_r{idx}" for idx in range(4)],
            }
        ),
    )


def test_artifacts_can_be_saved_and_loaded(tmp_path) -> None:
    artifact = train_artifact_from_split(make_split(), output_dir=tmp_path)
    loaded = load_model_artifact(tmp_path)

    assert artifact.paths is not None
    assert artifact.paths.model_path.exists()
    assert artifact.paths.scaler_path.exists()
    assert artifact.paths.feature_names_path.exists()
    assert artifact.paths.metadata_path.exists()
    assert loaded.metadata["model_type"] == "LogisticRegression"


def test_feature_order_is_preserved(tmp_path) -> None:
    train_artifact_from_split(make_split(), output_dir=tmp_path)
    loaded = load_model_artifact(tmp_path)

    assert loaded.feature_names == FEATURE_COLUMNS
    assert loaded.metadata["feature_names"] == FEATURE_COLUMNS


def test_scaler_is_actually_fitted(tmp_path) -> None:
    split = make_split()

    artifact = train_artifact_from_split(split, output_dir=tmp_path)

    assert hasattr(artifact.scaler, "mean_")
    assert artifact.scaler.mean_ == pytest.approx(split.X_train.mean().to_numpy())


def test_inference_uses_loaded_scaler(tmp_path) -> None:
    split = make_split()
    train_artifact_from_split(split, output_dir=tmp_path)
    loaded = load_model_artifact(tmp_path)

    prediction = predict_single_recording(split.X_train.iloc[0], loaded)

    manual_scaled = loaded.scaler.transform(split.X_train.iloc[[0]])
    manual_prediction = loaded.model.predict(manual_scaled)[0]
    assert prediction["predicted_class"] == manual_prediction


def test_inference_produces_same_result_before_and_after_save_load(tmp_path) -> None:
    split = make_split()
    artifact = train_artifact_from_split(split, output_dir=tmp_path)
    loaded = load_model_artifact(tmp_path)

    before = predict_single_recording(split.X_test.iloc[0], artifact)
    after = predict_single_recording(split.X_test.iloc[0], loaded)

    assert before == after


def test_incorrect_feature_order_is_rejected(tmp_path) -> None:
    split = make_split()
    artifact = train_artifact_from_split(split, output_dir=tmp_path)
    reversed_features = list(reversed(FEATURE_COLUMNS))
    feature_row = split.X_test.iloc[[0]][reversed_features]

    with pytest.raises(ValueError, match="trained feature order"):
        predict_single_recording(feature_row, artifact)


def test_missing_features_are_rejected(tmp_path) -> None:
    split = make_split()
    artifact = train_artifact_from_split(split, output_dir=tmp_path)
    feature_row = split.X_test.iloc[0].drop(labels=["hold_mean"])

    with pytest.raises(ValueError, match="missing required features"):
        predict_single_recording(feature_row, artifact)


def test_extra_features_are_rejected(tmp_path) -> None:
    split = make_split()
    artifact = train_artifact_from_split(split, output_dir=tmp_path)
    feature_row = split.X_test.iloc[0].copy()
    feature_row["participant_id"] = "test_p0"

    with pytest.raises(ValueError, match="extra features"):
        predict_single_recording(feature_row, artifact)


def test_inference_does_not_mutate_input(tmp_path) -> None:
    split = make_split()
    artifact = train_artifact_from_split(split, output_dir=tmp_path)
    feature_row = split.X_test.iloc[[0]].copy(deep=True)
    original = feature_row.copy(deep=True)

    predict_single_recording(feature_row, artifact)

    pd.testing.assert_frame_equal(feature_row, original)


def test_final_test_participants_are_never_used_for_fitting(tmp_path) -> None:
    split = make_split()
    changed_split = ParticipantSplit(
        X_train=split.X_train,
        X_test=split.X_test + 10000,
        y_train=split.y_train,
        y_test=split.y_test,
        train_metadata=split.train_metadata,
        test_metadata=split.test_metadata,
    )

    original = train_artifact_from_split(split, output_dir=tmp_path / "original")
    changed = train_artifact_from_split(changed_split, output_dir=tmp_path / "changed")

    assert original.scaler.mean_ == pytest.approx(changed.scaler.mean_)
    assert original.model.coef_ == pytest.approx(changed.model.coef_)


def test_artifact_metadata_records_selected_configuration_and_training_counts(tmp_path) -> None:
    split = make_split()

    artifact = train_artifact_from_split(split, output_dir=tmp_path)

    assert artifact.metadata["C"] == SELECTED_LOGISTIC_C
    assert artifact.metadata["random_state"] == DEFAULT_RANDOM_STATE
    assert artifact.metadata["feature_count"] == len(FEATURE_COLUMNS)
    assert artifact.metadata["training_sample_count"] == len(split.X_train)
    assert artifact.metadata["training_participant_count"] == split.train_metadata["participant_id"].nunique()
    assert artifact.metadata["final_test_participant_overlap"] == 0
    assert artifact.label_mapping == {"CNT": 0, "PD": 1}


def test_prediction_returns_probability_when_available(tmp_path) -> None:
    split = make_split()
    artifact = train_artifact_from_split(split, output_dir=tmp_path)

    prediction = predict_single_recording(split.X_test.iloc[0], artifact)

    assert set(prediction["probabilities"]) == {"CNT", "PD"}
    assert sum(prediction["probabilities"].values()) == pytest.approx(1.0)
