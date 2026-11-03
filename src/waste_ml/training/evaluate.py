"""Evaluate the trained classifier on the held-out test split."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from waste_ml.config import CLASSES, IMAGE_SIZE, SERVICE_ROOT
from waste_ml.preprocessing import preprocess_image

logger = logging.getLogger("waste_ml.evaluate")

ARTIFACTS_DIR = SERVICE_ROOT / "artifacts"
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "waste_classifier.keras"
DEFAULT_METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
DEFAULT_EVALUATION_DIR = ARTIFACTS_DIR / "evaluation"
DEFAULT_TEST_DIR = SERVICE_ROOT / "dataset" / "test"

AUTOTUNE = tf.data.AUTOTUNE


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_EVALUATION_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latency-samples", type=int, default=20)
    return parser.parse_args(argv)


def load_test_dataset(test_dir: Path, batch_size: int) -> tf.data.Dataset:
    dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        class_names=list(CLASSES),
        color_mode="rgb",
        batch_size=batch_size,
        image_size=IMAGE_SIZE,
        shuffle=False,
        verbose=False,
    )
    dataset = dataset.map(
        lambda images, labels: (preprocess_input(images), labels),
        num_parallel_calls=AUTOTUNE,
    )
    return dataset.prefetch(AUTOTUNE)


def save_confusion_matrix(
    matrix: np.ndarray, output_path: Path, labels: tuple[str, ...] = CLASSES
) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Waste Classifier Confusion Matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")

    threshold = matrix.max() / 2.0 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_training_history(history_path: Path, output_path: Path) -> None:
    if not history_path.is_file():
        logger.warning("Training history not found at %s; skipping plot", history_path)
        return

    with history_path.open(encoding="utf-8") as handle:
        history = json.load(handle)

    metrics = history.get("metrics", history)
    epochs = range(1, len(metrics.get("loss", [])) + 1)
    boundary = history.get("phase_one_epochs")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, metric in zip(axes, ("accuracy", "loss"), strict=True):
        axis.plot(epochs, metrics.get(metric, []), label=f"train {metric}")
        axis.plot(epochs, metrics.get(f"val_{metric}", []), label=f"validation {metric}")
        if boundary:
            axis.axvline(boundary, color="gray", linestyle="--", label="fine-tuning start")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric.capitalize())
        axis.set_title(f"{metric.capitalize()} over training")
        axis.grid(alpha=0.3)
        axis.legend()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def cpu_model() -> str:
    try:
        with Path("/proc/cpuinfo").open(encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def environment_info() -> dict[str, Any]:
    gpus = [device.name for device in tf.config.list_physical_devices("GPU")]
    return {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu_model(),
        "cpu_count": os.cpu_count(),
        "gpus": gpus,
        "inference_device": "GPU" if gpus else "CPU",
        "image_size": list(IMAGE_SIZE),
    }


def measure_latency(
    model: tf.keras.Model,
    test_dir: Path,
    sample_count: int,
) -> dict[str, float]:
    paths = sorted(
        path
        for class_name in CLASSES
        for path in (test_dir / class_name).glob("*")
        if path.is_file()
    )
    if not paths:
        raise FileNotFoundError(f"No test images found under {test_dir}")

    if len(paths) > sample_count:
        indexes = np.linspace(0, len(paths) - 1, num=sample_count, dtype=int)
        paths = [paths[index] for index in indexes]

    warmup = min(3, len(paths))
    for path in paths[:warmup]:
        with Image.open(path) as image:
            array = preprocess_image(image.convert("RGB"))
        model.predict(np.expand_dims(array, axis=0), verbose=0)

    preprocess_times: list[float] = []
    inference_times: list[float] = []
    total_times: list[float] = []

    for path in paths:
        started = time.perf_counter()
        with Image.open(path) as image:
            array = preprocess_image(image.convert("RGB"))
        preprocessed_at = time.perf_counter()
        model.predict(np.expand_dims(array, axis=0), verbose=0)
        finished = time.perf_counter()

        preprocess_times.append((preprocessed_at - started) * 1000)
        inference_times.append((finished - preprocessed_at) * 1000)
        total_times.append((finished - started) * 1000)

    def summarize(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": round(float(array.mean()), 3),
            "std": round(float(array.std(ddof=0)), 3),
            "min": round(float(array.min()), 3),
            "p95": round(float(np.percentile(array, 95)), 3),
            "max": round(float(array.max()), 3),
        }

    return {
        "samples": len(paths),
        "batch_size": 1,
        "preprocessing_ms": summarize(preprocess_times),
        "inference_ms": summarize(inference_times),
        "end_to_end_ms": summarize(total_times),
    }


def update_metadata(
    metadata_path: Path,
    metrics: dict[str, Any],
    test_dir: Path,
) -> None:
    if not metadata_path.is_file():
        logger.warning("Metadata not found at %s; skipping update", metadata_path)
        return

    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    try:
        test_dataset = str(test_dir.relative_to(SERVICE_ROOT))
    except ValueError:
        test_dataset = str(test_dir)

    metadata["evaluation"] = {
        "accuracy": metrics["accuracy"],
        "test_samples": metrics["test_samples"],
        "test_date": metrics["evaluation_date"],
        "test_dataset": test_dataset,
        "metrics_path": "artifacts/evaluation/metrics.json",
        "confusion_matrix_path": "artifacts/evaluation/confusion_matrix.png",
        "classification_report_path": "artifacts/evaluation/classification_report.txt",
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def markdown_table(matrix: np.ndarray, labels: tuple[str, ...] = CLASSES) -> str:
    header = "| True \\ Predicted | " + " | ".join(labels) + " |"
    separator = "|---|" + "---|" * len(labels)
    rows = [
        "| " + label + " | " + " | ".join(str(int(value)) for value in row) + " |"
        for label, row in zip(labels, matrix, strict=True)
    ]
    return "\n".join([header, separator, *rows])


def write_summary(
    output_path: Path,
    *,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    report = metrics["classification_report"]
    class_rows = [
        "| Class | Precision | Recall | F1-score | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for class_name in CLASSES:
        values = report[class_name]
        class_rows.append(
            f"| {class_name} | {values['precision']:.4f} | {values['recall']:.4f} | "
            f"{values['f1-score']:.4f} | {int(values['support'])} |"
        )

    matrix = np.asarray(metrics["confusion_matrix"])
    latency = metrics["latency_ms"]
    environment = metrics["environment"]

    lines = [
        "# Evaluation Summary",
        "",
        f"- **Model:** {metadata.get('model_name', 'waste-classifier')} "
        f"v{metadata.get('model_version', '1.0.0')}",
        f"- **Architecture:** {metadata.get('architecture', 'MobileNetV2')}",
        f"- **Test samples:** {metrics['test_samples']}",
        f"- **Test accuracy:** {metrics['accuracy'] * 100:.2f}%",
        f"- **Evaluation date:** {metrics['evaluation_date']}",
        "",
        "## Classification Report",
        "",
        *class_rows,
        "",
        "## Confusion Matrix",
        "",
        markdown_table(matrix),
        "",
        "## Inference Latency",
        "",
        f"Measured on {latency['samples']} test images with batch size 1 on "
        f"{environment['inference_device']} ({environment['cpu']}).",
        "",
        "| Stage | Mean (ms) | Std (ms) | Min (ms) | P95 (ms) | Max (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stage in ("preprocessing_ms", "inference_ms", "end_to_end_ms"):
        values = latency[stage]
        lines.append(
            f"| {stage.replace('_ms', '').replace('_', ' ').title()} | "
            f"{values['mean']:.3f} | {values['std']:.3f} | {values['min']:.3f} | "
            f"{values['p95']:.3f} | {values['max']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Environment",
            "",
            f"- Python: {environment['python']}",
            f"- TensorFlow: {environment['tensorflow']}",
            f"- Platform: {environment['platform']}",
            f"- CPU: {environment['cpu']}",
            f"- GPU: {', '.join(environment['gpus']) if environment['gpus'] else 'none'}",
            f"- Input size: {environment['image_size'][0]} x {environment['image_size'][1]}",
            "",
            "Accuracy is computed only on the held-out test split. Confidence returned by "
            "the API is the model probability for one prediction, not model accuracy.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    model_path = args.model_path.resolve()
    metadata_path = args.metadata_path.resolve()
    test_dir = args.test_dir.resolve()
    evaluation_dir = args.evaluation_dir.resolve()
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")

    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    model = tf.keras.models.load_model(model_path, compile=False)
    test_ds = load_test_dataset(test_dir, batch_size=args.batch_size)

    true_labels = np.concatenate([labels.numpy() for _, labels in test_ds], axis=0)
    probability_batches = model.predict(test_ds, verbose=0)
    predicted_labels = np.argmax(probability_batches, axis=1)

    matrix = confusion_matrix(true_labels, predicted_labels, labels=list(range(len(CLASSES))))
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(len(CLASSES))),
        target_names=list(CLASSES),
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(len(CLASSES))),
        target_names=list(CLASSES),
        zero_division=0,
    )
    accuracy = float(accuracy_score(true_labels, predicted_labels))
    latency = measure_latency(model, test_dir, sample_count=args.latency_samples)

    metrics: dict[str, Any] = {
        "accuracy": round(accuracy, 6),
        "test_samples": int(len(true_labels)),
        "classes": list(CLASSES),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "latency_ms": latency,
        "environment": environment_info(),
        "evaluation_date": datetime.now(UTC).isoformat(),
    }

    with (evaluation_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    (evaluation_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    save_confusion_matrix(matrix, evaluation_dir / "confusion_matrix.png")
    save_training_history(
        evaluation_dir / "training_history.json",
        evaluation_dir / "training_history.png",
    )
    write_summary(
        evaluation_dir / "evaluation_summary.md",
        metadata=metadata,
        metrics=metrics,
    )
    update_metadata(metadata_path, metrics, test_dir)

    logger.info("Test accuracy: %.4f (%d samples)", accuracy, len(true_labels))
    logger.info("Evaluation artifacts written to %s", evaluation_dir)
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        evaluate(args)
    except Exception:
        logger.exception("Evaluation failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
