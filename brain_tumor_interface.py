from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import joblib
import numpy as np
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

CLASS_NAMES: List[str] = ["glioma", "meningioma", "notumor", "pituitary"]
DATASET_DIR = Path("data/preprocessed_data/preprocessedForMLP_numpy")
MLP_MODEL_PATH = Path("model/MLP_IT24104191/brain_tumor_mlp_best.h5")
CLASSICAL_MODEL_DIR = Path("model/classical_models")


@dataclass
class Prediction:
    label: str
    confidence: float


@dataclass
class PredictionBundle:
    model_name: str
    predictions: Sequence[Prediction]


def preprocess_image(image_path: Path, target_size: int = 64) -> np.ndarray:
    """
    Convert an input image to a flattened grayscale vector that matches the
    training data used for the MLP and classical baselines.
    """
    image = Image.open(image_path).convert("L").resize((target_size, target_size))
    normalized = np.asarray(image, dtype=np.float32) / 255.0
    return normalized.flatten().reshape(1, -1)


def load_numpy_split(dataset_dir: Path = DATASET_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Load the flattened training tensors and labels that back the classical models."""
    X_path = dataset_dir / "X_train.npy"
    y_path = dataset_dir / "y_train.npy"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            "Could not locate preprocessed numpy arrays. Expected to find "
            f"`X_train.npy` and `y_train.npy` in {dataset_dir}."
        )

    X_train = np.load(X_path)
    y_train = np.load(y_path)
    return X_train, y_train


def ensure_classical_models(
    dataset_dir: Path = DATASET_DIR, force_retrain: bool = False
) -> Dict[str, CalibratedClassifierCV]:
    """
    Load the classical ML models if they exist on disk, or fit them from the
    preprocessed numpy data on-demand.
    """
    CLASSICAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_specs = {
        "logistic_regression": (
            CLASSICAL_MODEL_DIR / "brain_tumor_logreg.joblib",
            CalibratedClassifierCV(
                base_estimator=LogisticRegression(
                    max_iter=1_000, solver="lbfgs", multi_class="multinomial"
                ),
                cv=3,
            ),
        ),
        "random_forest": (
            CLASSICAL_MODEL_DIR / "brain_tumor_random_forest.joblib",
            RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
        ),
        "linear_svm": (
            CLASSICAL_MODEL_DIR / "brain_tumor_linear_svm.joblib",
            CalibratedClassifierCV(
                base_estimator=make_pipeline(
                    StandardScaler(with_mean=False), LogisticRegression(max_iter=1_000)
                ),
                cv=3,
            ),
        ),
    }

    trained: Dict[str, CalibratedClassifierCV] = {}
    X_train: np.ndarray | None = None
    y_train: np.ndarray | None = None

    for model_name, (path, estimator) in model_specs.items():
        if path.exists() and not force_retrain:
            trained[model_name] = joblib.load(path)
            continue

        if X_train is None or y_train is None:
            X_train, y_train = load_numpy_split(dataset_dir)

        fitted = estimator.fit(X_train, y_train)
        joblib.dump(fitted, path)
        trained[model_name] = fitted

    return trained


def load_mlp_model(model_path: Path = MLP_MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Expected the trained MLP weights at {model_path}, but the file was not found."
        )
    return load_model(model_path)


def top_predictions(probabilities: np.ndarray, limit: int = 3) -> List[Prediction]:
    sorted_indices = np.argsort(probabilities)[::-1][:limit]
    return [
        Prediction(label=CLASS_NAMES[idx], confidence=float(probabilities[idx]))
        for idx in sorted_indices
    ]


def predict_with_classical(
    models: Dict[str, CalibratedClassifierCV], image_vector: np.ndarray
) -> Iterable[PredictionBundle]:
    for model_name, estimator in models.items():
        class_probs = estimator.predict_proba(image_vector)[0]
        yield PredictionBundle(model_name=model_name, predictions=top_predictions(class_probs))


def predict_with_mlp(image_vector: np.ndarray) -> PredictionBundle:
    mlp_model = load_mlp_model()
    class_probs = mlp_model.predict(image_vector, verbose=0)[0]
    return PredictionBundle(model_name="mlp", predictions=top_predictions(class_probs))


def format_prediction(prediction: Prediction) -> str:
    percentage = round(prediction.confidence * 100, 2)
    return f"{prediction.label} ({percentage}%)"


def display_results(bundles: Iterable[PredictionBundle]) -> None:
    for bundle in bundles:
        formatted = ", ".join(format_prediction(p) for p in bundle.predictions)
        print(f"[{bundle.model_name}] {formatted}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect brain tumor types using multiple trained models. "
            "The script will train classical ML baselines on-demand if they are missing."
        )
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the MRI image to classify.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mlp", "logistic_regression", "random_forest", "linear_svm"],
        choices=["mlp", "logistic_regression", "random_forest", "linear_svm"],
        help="Subset of models to run. Defaults to all available models.",
    )
    parser.add_argument(
        "--retrain-classical",
        action="store_true",
        help="Force retraining of the classical ML models (logistic, RF, SVM).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Directory containing X_train.npy and y_train.npy for classical model training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Image path does not exist: {args.image}")

    image_vector = preprocess_image(args.image)

    bundles: List[PredictionBundle] = []
    if "mlp" in args.models:
        bundles.append(predict_with_mlp(image_vector))

    classical_requested = {
        name for name in args.models if name in {"logistic_regression", "random_forest", "linear_svm"}
    }
    if classical_requested:
        classical_models = ensure_classical_models(
            dataset_dir=args.dataset_dir, force_retrain=args.retrain_classical
        )
        filtered_models = {name: classical_models[name] for name in classical_requested}
        bundles.extend(predict_with_classical(filtered_models, image_vector))

    display_results(bundles)


if __name__ == "__main__":
    main()
