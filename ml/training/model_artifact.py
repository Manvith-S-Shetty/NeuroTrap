"""Deployment artifact preparation for the selected NeuroTap model.

The artifact is trained only on the existing development split. It serializes
the fitted Logistic Regression model, fitted StandardScaler, ordered feature
names, label information, and training metadata for later inference.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from ml.training.baseline import CLASS_LABELS, validate_finite_features  # type: ignore
    from ml.training.dataset import FEATURE_COLUMNS, MLDataset, build_ml_dataset, split_by_participant  # type: ignore
    from ml.training.final_evaluation import (  # type: ignore
        SELECTED_LOGISTIC_C,
        validate_final_split,
    )
    from ml.training.random_forest import DEFAULT_RANDOM_STATE  # type: ignore
else:
    from ml.training.baseline import CLASS_LABELS, validate_finite_features
    from ml.training.dataset import FEATURE_COLUMNS, MLDataset, build_ml_dataset, split_by_participant
    from ml.training.final_evaluation import SELECTED_LOGISTIC_C, validate_final_split
    from ml.training.random_forest import DEFAULT_RANDOM_STATE


DEFAULT_ARTIFACT_DIR = Path("ml") / "models"
MODEL_FILENAME = "neurotap_logistic_regression.joblib"
SCALER_FILENAME = "neurotap_standard_scaler.joblib"
FEATURES_FILENAME = "neurotap_feature_names.joblib"
METADATA_FILENAME = "neurotap_model_metadata.joblib"


@dataclass(frozen=True)
class ArtifactPaths:
    model_path: Path
    scaler_path: Path
    feature_names_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class ModelArtifact:
    model: LogisticRegression
    scaler: StandardScaler
    feature_names: list[str]
    label_mapping: dict[str, int]
    metadata: dict[str, Any]
    paths: ArtifactPaths | None = None


def artifact_paths(output_dir: Path = DEFAULT_ARTIFACT_DIR) -> ArtifactPaths:
    """Return canonical artifact paths under the selected output directory."""
    return ArtifactPaths(
        model_path=output_dir / MODEL_FILENAME,
        scaler_path=output_dir / SCALER_FILENAME,
        feature_names_path=output_dir / FEATURES_FILENAME,
        metadata_path=output_dir / METADATA_FILENAME,
    )


def _participant_overlap(train_metadata: pd.DataFrame, test_metadata: pd.DataFrame) -> int:
    train = set(train_metadata["participant_id"].astype(str))
    test = set(test_metadata["participant_id"].astype(str))
    return len(train & test)


def build_model_metadata(
    model: LogisticRegression,
    scaler: StandardScaler,
    split,
) -> dict[str, Any]:
    """Build serializable metadata describing the trained artifact."""
    return {
        "model_type": "LogisticRegression",
        "C": SELECTED_LOGISTIC_C,
        "random_state": DEFAULT_RANDOM_STATE,
        "max_iter": 1000,
        "scaler_type": "StandardScaler",
        "feature_count": len(FEATURE_COLUMNS),
        "feature_names": list(FEATURE_COLUMNS),
        "class_labels": list(model.classes_),
        "label_mapping": {label: int(index) for index, label in enumerate(model.classes_)},
        "training_sample_count": int(len(split.X_train)),
        "training_participant_count": int(split.train_metadata["participant_id"].nunique()),
        "final_test_sample_count": int(len(split.X_test)),
        "final_test_participant_count": int(split.test_metadata["participant_id"].nunique()),
        "final_test_participant_overlap": _participant_overlap(split.train_metadata, split.test_metadata),
        "fitted_on": "development_split_only",
        "inference_requires_exact_feature_order": True,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }


def train_artifact_from_split(split, output_dir: Path = DEFAULT_ARTIFACT_DIR) -> ModelArtifact:
    """Train selected Logistic Regression on development data and save artifacts."""
    validate_final_split(split)
    output_dir.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(split.X_train)
    model = LogisticRegression(C=SELECTED_LOGISTIC_C, random_state=DEFAULT_RANDOM_STATE, max_iter=1000)
    model.fit(X_train_scaled, split.y_train)

    metadata = build_model_metadata(model, scaler, split)
    paths = artifact_paths(output_dir)
    joblib.dump(model, paths.model_path)
    joblib.dump(scaler, paths.scaler_path)
    joblib.dump(list(FEATURE_COLUMNS), paths.feature_names_path)
    joblib.dump(metadata, paths.metadata_path)

    return ModelArtifact(
        model=model,
        scaler=scaler,
        feature_names=list(FEATURE_COLUMNS),
        label_mapping=metadata["label_mapping"],
        metadata=metadata,
        paths=paths,
    )


def train_and_save_artifact(
    ml_dataset: MLDataset | None = None,
    output_dir: Path = DEFAULT_ARTIFACT_DIR,
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratify: bool = True,
) -> ModelArtifact:
    """Load/build the ML dataset, recover the existing split, and save the artifact."""
    dataset = build_ml_dataset() if ml_dataset is None else ml_dataset
    split = split_by_participant(dataset, test_size=test_size, random_state=random_state, stratify=stratify)
    return train_artifact_from_split(split, output_dir=output_dir)


def load_model_artifact(output_dir: Path = DEFAULT_ARTIFACT_DIR) -> ModelArtifact:
    """Load the serialized model artifact from disk."""
    paths = artifact_paths(output_dir)
    model = joblib.load(paths.model_path)
    scaler = joblib.load(paths.scaler_path)
    feature_names = joblib.load(paths.feature_names_path)
    metadata = joblib.load(paths.metadata_path)
    return ModelArtifact(
        model=model,
        scaler=scaler,
        feature_names=list(feature_names),
        label_mapping=dict(metadata["label_mapping"]),
        metadata=dict(metadata),
        paths=paths,
    )


def _feature_frame(feature_vector: pd.Series | pd.DataFrame | dict[str, float], feature_names: list[str]) -> pd.DataFrame:
    if isinstance(feature_vector, pd.DataFrame):
        frame = feature_vector.copy(deep=True)
        if len(frame) != 1:
            raise ValueError("Inference expects exactly one recording feature row.")
    elif isinstance(feature_vector, pd.Series):
        frame = feature_vector.to_frame().T
    else:
        frame = pd.DataFrame([feature_vector])

    columns = list(frame.columns)
    missing = sorted(set(feature_names) - set(columns))
    extra = sorted(set(columns) - set(feature_names))
    if missing:
        raise ValueError(f"Inference feature vector is missing required features: {missing}")
    if extra:
        raise ValueError(f"Inference feature vector contains extra features: {extra}")
    if columns != feature_names:
        raise ValueError("Inference feature vector columns are not in the trained feature order.")

    validate_finite_features(frame, "inference feature vector")
    return frame


def predict_single_recording(
    feature_vector: pd.Series | pd.DataFrame | dict[str, float],
    artifact: ModelArtifact,
) -> dict[str, Any]:
    """Predict one recording from loaded feature values without refitting preprocessing."""
    frame = _feature_frame(feature_vector, artifact.feature_names)
    scaled = artifact.scaler.transform(frame)
    prediction = artifact.model.predict(scaled)[0]
    result: dict[str, Any] = {"predicted_class": prediction}
    if hasattr(artifact.model, "predict_proba"):
        probabilities = artifact.model.predict_proba(scaled)[0]
        result["probabilities"] = {
            str(label): float(probability)
            for label, probability in zip(artifact.model.classes_, probabilities)
        }
    return result


def main() -> int:
    ml_dataset = build_ml_dataset()
    split = split_by_participant(ml_dataset, test_size=0.2, random_state=DEFAULT_RANDOM_STATE, stratify=True)
    artifact = train_artifact_from_split(split, output_dir=DEFAULT_ARTIFACT_DIR)
    loaded = load_model_artifact(DEFAULT_ARTIFACT_DIR)
    sample_prediction = predict_single_recording(split.X_train.iloc[0], loaded)

    print("# NeuroTap Model Artifact")
    print("\n## Artifact Paths")
    print(f"Model: {artifact.paths.model_path}")
    print(f"Scaler: {artifact.paths.scaler_path}")
    print(f"Feature names: {artifact.paths.feature_names_path}")
    print(f"Metadata: {artifact.paths.metadata_path}")
    print("\n## Artifact Contents")
    print(f"Model type: {artifact.metadata['model_type']}")
    print(f"C: {artifact.metadata['C']}")
    print(f"random_state: {artifact.metadata['random_state']}")
    print(f"Feature count: {artifact.metadata['feature_count']}")
    print(f"Class labels: {artifact.metadata['class_labels']}")
    print(f"Label mapping: {artifact.metadata['label_mapping']}")
    print(f"Training samples: {artifact.metadata['training_sample_count']}")
    print(f"Training participants: {artifact.metadata['training_participant_count']}")
    print(f"Final test samples excluded from fitting: {artifact.metadata['final_test_sample_count']}")
    print(f"Final test participants excluded from fitting: {artifact.metadata['final_test_participant_count']}")
    print(f"Final-test participant overlap: {artifact.metadata['final_test_participant_overlap']}")
    print("\n## Inference Smoke Test")
    print(f"Predicted class: {sample_prediction['predicted_class']}")
    print(f"Probabilities: {sample_prediction.get('probabilities')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
