"""Model loading and single-image prediction."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image

from waste_ml.config import CLASSES, IMAGE_SIZE
from waste_ml.preprocessing import preprocess_image, preprocess_image_bytes


class WasteClassifier:
    """Load a trained Keras model once and serve predictions."""

    def __init__(self, model_path: Path, metadata_path: Path) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Model metadata not found: {self.metadata_path}")

        with self.metadata_path.open(encoding="utf-8") as handle:
            self.metadata: dict[str, Any] = json.load(handle)

        metadata_classes = tuple(self.metadata.get("classes", []))
        if metadata_classes != CLASSES:
            raise ValueError(
                f"Model metadata classes {metadata_classes} do not match expected {CLASSES}"
            )

        self.classes = list(metadata_classes)
        self.model_version = str(self.metadata.get("model_version", "unknown"))
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        self._lock = threading.Lock()

    def model_info(self) -> dict[str, Any]:
        return {
            "model_name": self.metadata.get("model_name", "waste-classifier"),
            "model_version": self.model_version,
            "architecture": self.metadata.get("architecture", "MobileNetV2"),
            "classes": self.classes,
            "input_size": list(self.metadata.get("input_shape", [*IMAGE_SIZE, 3])[:2]),
        }

    def predict(self, image: Image.Image) -> dict[str, Any]:
        array = preprocess_image(image, image_size=IMAGE_SIZE)
        return self._predict_array(array)

    def predict_bytes(self, data: bytes) -> dict[str, Any]:
        array = preprocess_image_bytes(data, image_size=IMAGE_SIZE)
        return self._predict_array(array)

    def _predict_array(self, array: np.ndarray) -> dict[str, Any]:
        batch = np.expand_dims(array, axis=0).astype(np.float32)
        with self._lock:
            probabilities = self.model.predict(batch, verbose=0)[0]

        probabilities = np.asarray(probabilities, dtype=np.float64)
        predicted_index = int(np.argmax(probabilities))
        predicted_class = self.classes[predicted_index]

        return {
            "predicted_class": predicted_class,
            "confidence": float(probabilities[predicted_index]),
            "probabilities": {
                class_name: float(probabilities[index])
                for index, class_name in enumerate(self.classes)
            },
            "model_version": self.model_version,
        }
