"""Build a reproducible organic/plastic/paper dataset from Hugging Face.

Source dataset: NeoAivara/Waste_Classification_data (Apache-2.0 per dataset card).
The source contains 12 garbage classes. This script selects `biological`,
`plastic`, and `paper`, deduplicates near-identical images, and creates a
stratified 70/15/15 train/validation/test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

SOURCE_REPO = "NeoAivara/Waste_Classification_data"
SOURCE_FILE = "data/train-00000-of-00001.parquet"
SOURCE_URL = f"https://huggingface.co/datasets/{SOURCE_REPO}"
SOURCE_LICENSE = "Apache-2.0 (as declared on the Hugging Face dataset card)"

SOURCE_CLASSES: dict[str, int] = {
    "battery": 0,
    "biological": 1,
    "brown-glass": 2,
    "cardboard": 3,
    "clothes": 4,
    "green-glass": 5,
    "metal": 6,
    "paper": 7,
    "plastic": 8,
    "shoes": 9,
    "trash": 10,
    "white-glass": 11,
}

CLASS_MAPPING: dict[str, str] = {
    "organic": "biological",
    "plastic": "plastic",
    "paper": "paper",
}

SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
HASH_SIZE = 8


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--per-class", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-duplicate-distance", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def average_hash(image: Image.Image, hash_size: int = HASH_SIZE) -> np.ndarray:
    grayscale = image.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.float64)
    return pixels >= pixels.mean()


def is_near_duplicate(candidate: np.ndarray, accepted: list[np.ndarray], distance: int) -> bool:
    return any(np.count_nonzero(candidate != other) <= distance for other in accepted)


def read_source_rows() -> list[dict[str, Any]]:
    parquet_path = hf_hub_download(
        repo_id=SOURCE_REPO,
        filename=SOURCE_FILE,
        repo_type="dataset",
    )
    table = pq.read_table(parquet_path, columns=["image", "label"])
    return table.to_pylist()


def split_counts(per_class: int, class_count: int) -> dict[str, int]:
    train = int(per_class * SPLIT_RATIOS["train"])
    validation = int(per_class * SPLIT_RATIOS["validation"])
    test = per_class - train - validation
    return {"train": train, "validation": validation, "test": test}


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    for split in SPLIT_RATIOS:
        split_dir = output_dir / split
        if split_dir.exists() and any(split_dir.iterdir()):
            if not args.force:
                raise FileExistsError(
                    f"{split_dir} already contains files. Use --force to replace the dataset."
                )
            shutil.rmtree(split_dir)

    rows = read_source_rows()
    random.seed(args.seed)
    random.shuffle(rows)

    selected: dict[str, list[tuple[Image.Image, str]]] = {
        class_name: [] for class_name in CLASS_MAPPING
    }
    hashes: dict[str, list[np.ndarray]] = {class_name: [] for class_name in CLASS_MAPPING}
    duplicate_stats: dict[str, dict[str, int]] = {}

    for class_name, source_name in CLASS_MAPPING.items():
        source_label = SOURCE_CLASSES[source_name]
        accepted = selected[class_name]
        accepted_hashes = hashes[class_name]
        skipped_duplicates = 0
        skipped_invalid = 0

        for row in rows:
            label = row.get("label")
            if isinstance(label, str):
                if label != source_name:
                    continue
            elif int(label) != source_label:
                continue

            if len(accepted) >= args.per_class:
                break

            image_payload = row.get("image") or {}
            image_bytes = image_payload.get("bytes")
            if not image_bytes:
                skipped_invalid += 1
                continue

            try:
                with Image.open(BytesIO(image_bytes)) as image:
                    image.load()
                    rgb_image = image.convert("RGB")
            except (OSError, ValueError):
                skipped_invalid += 1
                continue

            image_hash = average_hash(rgb_image)
            if is_near_duplicate(image_hash, accepted_hashes, args.near_duplicate_distance):
                skipped_duplicates += 1
                continue

            accepted.append((rgb_image, hashlib.sha256(image_bytes).hexdigest()))
            accepted_hashes.append(image_hash)

        if len(accepted) < args.per_class:
            raise RuntimeError(
                f"Only found {len(accepted)} unique usable images for {class_name}; "
                f"requested {args.per_class}"
            )

        duplicate_stats[class_name] = {
            "requested": args.per_class,
            "selected": len(accepted),
            "skipped_near_duplicates": skipped_duplicates,
            "skipped_invalid": skipped_invalid,
        }

    counts_per_class = split_counts(args.per_class, len(CLASS_MAPPING))
    manifest: dict[str, Any] = {
        "name": "smart-waste-organic-plastic-paper",
        "source": SOURCE_URL,
        "source_repo": SOURCE_REPO,
        "source_file": SOURCE_FILE,
        "license": SOURCE_LICENSE,
        "class_mapping": CLASS_MAPPING,
        "source_class_ids": {
            name: SOURCE_CLASSES[source] for name, source in CLASS_MAPPING.items()
        },
        "requested_per_class": args.per_class,
        "seed": args.seed,
        "near_duplicate_hamming_distance": args.near_duplicate_distance,
        "split_ratio": SPLIT_RATIOS,
        "split_counts_per_class": counts_per_class,
        "preprocessing": [
            "Decode source image bytes",
            "Convert to RGB",
            "Average-hash near-duplicate filtering per class",
            "Stratified 70/15/15 split by class",
            "Save re-encoded JPEG (quality=92)",
        ],
        "created_at": datetime.now(UTC).isoformat(),
        "duplicate_filtering": duplicate_stats,
        "counts": {},
    }

    for class_name, images in selected.items():
        random.Random(args.seed).shuffle(images)
        train_end = counts_per_class["train"]
        validation_end = train_end + counts_per_class["validation"]
        split_images = {
            "train": images[:train_end],
            "validation": images[train_end:validation_end],
            "test": images[validation_end:],
        }
        for split, split_items in split_images.items():
            split_dir = output_dir / split / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for index, (image, _) in enumerate(split_items):
                image.save(split_dir / f"{class_name}_{index:04d}.jpg", quality=92, optimize=True)
            manifest["counts"].setdefault(split, {})[class_name] = len(split_items)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = prepare(args)
    except Exception as exc:
        print(f"Dataset preparation failed: {exc}", file=sys.stderr)
        return 1

    print(f"Dataset created at {args.output_dir.resolve()}")
    for split, counts in manifest["counts"].items():
        print(f"  {split}: {counts} (total {sum(counts.values())})")
    print(f"  total: {sum(sum(counts.values()) for counts in manifest['counts'].values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
