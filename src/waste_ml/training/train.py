"""Train the MobileNetV2 waste classifier with transfer learning."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from waste_ml.config import (
    CLASSES,
    DEFAULT_MODEL_VERSION,
    DEFAULT_SEED,
    IMAGE_SIZE,
    INPUT_SHAPE,
    SERVICE_ROOT,
)
from waste_ml.model import build_model, unfreeze_backbone

logger = logging.getLogger("waste_ml.train")

DATASET_ROOT = SERVICE_ROOT / "dataset"
ARTIFACTS_DIR = SERVICE_ROOT / "artifacts"

AUTOTUNE = tf.data.AUTOTUNE


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--epochs", type=int, default=20, help="Phase 1 epochs.")
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--fine-tune-layers", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--no-fine-tune", action="store_true")
    return parser.parse_args(argv)


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def count_images(dataset_dir: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        split_dir = dataset_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing dataset split: {split_dir}")
        split_counts: dict[str, int] = {}
        for class_name in CLASSES:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            split_counts[class_name] = sum(
                1
                for path in class_dir.iterdir()
                if path.is_file() and not path.name.startswith(".")
            )
        counts[split] = split_counts
    return counts


def load_split(
    dataset_dir: Path,
    split: str,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> tf.data.Dataset:
    dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir / split,
        labels="inferred",
        label_mode="int",
        class_names=list(CLASSES),
        color_mode="rgb",
        batch_size=batch_size,
        image_size=IMAGE_SIZE,
        shuffle=shuffle,
        seed=seed if shuffle else None,
        verbose=False,
    )
    return dataset


def build_datasets(
    dataset_dir: Path,
    batch_size: int,
    seed: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.1, seed=seed),
            tf.keras.layers.RandomZoom(0.1, seed=seed),
            tf.keras.layers.RandomContrast(0.1, seed=seed),
        ],
        name="augmentation",
    )

    train_ds = load_split(dataset_dir, "train", batch_size=batch_size, seed=seed, shuffle=True)
    validation_ds = load_split(
        dataset_dir, "validation", batch_size=batch_size, seed=seed, shuffle=False
    )
    train_ds = train_ds.map(
        lambda images, labels: (augmentation(images, training=True), labels),
        num_parallel_calls=AUTOTUNE,
    )
    train_ds = train_ds.map(
        lambda images, labels: (preprocess_input(images), labels),
        num_parallel_calls=AUTOTUNE,
    )
    validation_ds = validation_ds.map(
        lambda images, labels: (preprocess_input(images), labels),
        num_parallel_calls=AUTOTUNE,
    )

    return train_ds.prefetch(AUTOTUNE), validation_ds.prefetch(AUTOTUNE)


def make_callbacks(
    *,
    checkpoint_path: Path,
    patience: int,
    monitor: str = "val_accuracy",
) -> list[tf.keras.callbacks.Callback]:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor,
            mode="max" if "accuracy" in monitor else "min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="max" if "accuracy" in monitor else "min",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, patience // 2),
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def history_to_dict(history: tf.keras.callbacks.History) -> dict[str, list[float]]:
    return {key: [float(value) for value in values] for key, values in history.history.items()}


def merge_histories(
    phase_one: dict[str, list[float]],
    phase_two: dict[str, list[float]] | None,
) -> dict[str, Any]:
    merged: dict[str, list[float]] = {}
    for key, values in phase_one.items():
        merged.setdefault(key, []).extend(values)
    for key, values in (phase_two or {}).items():
        merged.setdefault(key, []).extend(values)
    return {
        "metrics": merged,
        "phase_one_epochs": len(next(iter(phase_one.values()), [])),
        "phase_two_epochs": len(next(iter((phase_two or {}).values()), [])),
    }


def load_manifest(dataset_dir: Path) -> dict[str, Any] | None:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    with manifest_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def dataset_reference(manifest: dict[str, Any] | None) -> dict[str, Any]:
    """Describe the dataset sources for model metadata (single- or multi-source manifest)."""
    manifest = manifest or {}
    reference: dict[str, Any] = {
        "manifest": str((SERVICE_ROOT / "dataset" / "manifest.json").relative_to(SERVICE_ROOT)),
    }
    sources = manifest.get("sources")
    if sources:
        reference["sources"] = [
            {"name": name, "url": value.get("url"), "license": value.get("license")}
            for name, value in sources.items()
        ]
    else:
        reference["source"] = manifest.get("source")
        reference["license"] = manifest.get("license")
    return reference


def write_metadata(
    *,
    metadata_path: Path,
    model_version: str,
    counts: dict[str, dict[str, int]],
    manifest: dict[str, Any] | None,
    args: argparse.Namespace,
    training_seconds: float,
    history: dict[str, Any],
) -> None:
    best_val_accuracy = max(history["metrics"].get("val_accuracy", [0.0]))
    metadata = {
        "model_name": "waste-classifier",
        "model_version": model_version,
        "framework": f"TensorFlow {tf.__version__}/Keras",
        "architecture": "MobileNetV2",
        "input_shape": list(INPUT_SHAPE),
        "classes": list(CLASSES),
        "dataset_reference": dataset_reference(manifest),
        "dataset": counts,
        "training_date": datetime.now(UTC).isoformat(),
        "training_seconds": round(training_seconds, 2),
        "best_val_accuracy": round(float(best_val_accuracy), 6),
        "hyperparameters": {
            "batch_size": args.batch_size,
            "phase_one_epochs": args.epochs,
            "fine_tune_epochs": 0 if args.no_fine_tune else args.fine_tune_epochs,
            "fine_tune_layers": 0 if args.no_fine_tune else args.fine_tune_layers,
            "learning_rate": args.learning_rate,
            "fine_tune_learning_rate": args.fine_tune_learning_rate,
            "dropout": args.dropout,
            "optimizer": "Adam",
            "loss": "SparseCategoricalCrossentropy",
            "seed": args.seed,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "evaluation": None,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def train(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    set_reproducibility(args.seed)

    dataset_dir = args.dataset_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    checkpoint_path = artifacts_dir / "waste_classifier.keras"
    metadata_path = artifacts_dir / "model_metadata.json"
    evaluation_dir = artifacts_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    counts = count_images(dataset_dir)
    logger.info("Dataset counts: %s", counts)

    train_ds, validation_ds = build_datasets(
        dataset_dir, batch_size=args.batch_size, seed=args.seed
    )

    model = build_model(
        num_classes=len(CLASSES),
        input_shape=INPUT_SHAPE,
        dropout_rate=args.dropout,
        trainable_backbone=False,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.summary(print_fn=logger.info)

    started_at = datetime.now(UTC)
    phase_one = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=args.epochs,
        callbacks=make_callbacks(checkpoint_path=checkpoint_path, patience=args.patience),
        verbose=2,
    )
    phase_one_history = history_to_dict(phase_one)

    phase_two_history: dict[str, list[float]] | None = None
    if not args.no_fine_tune:
        unfreeze_backbone(model, num_layers=args.fine_tune_layers)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.fine_tune_learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        phase_two = model.fit(
            train_ds,
            validation_data=validation_ds,
            initial_epoch=len(phase_one_history.get("loss", [])),
            epochs=len(phase_one_history.get("loss", [])) + args.fine_tune_epochs,
            callbacks=make_callbacks(checkpoint_path=checkpoint_path, patience=args.patience),
            verbose=2,
        )
        phase_two_history = history_to_dict(phase_two)

    training_seconds = (datetime.now(UTC) - started_at).total_seconds()

    if checkpoint_path.is_file():
        logger.info("Loading best checkpoint from %s", checkpoint_path)
        best_model = tf.keras.models.load_model(checkpoint_path, compile=False)
    else:
        best_model = model
    best_model.save(checkpoint_path)

    history = merge_histories(phase_one_history, phase_two_history)
    with (evaluation_dir / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
        handle.write("\n")

    with (evaluation_dir / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, default=str)
        handle.write("\n")

    write_metadata(
        metadata_path=metadata_path,
        model_version=args.model_version,
        counts=counts,
        manifest=load_manifest(dataset_dir),
        args=args,
        training_seconds=training_seconds,
        history=history,
    )
    logger.info("Saved model to %s", checkpoint_path)
    logger.info("Saved metadata to %s", metadata_path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        train(args)
    except Exception:
        logger.exception("Training failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
