"""Build a reproducible organic/plastic/paper dataset from two Hugging Face sources.

1. NeoAivara/Waste_Classification_data (Apache-2.0 per dataset card): clean single-object
   photos. `biological`, `plastic` and `paper` are selected exactly as in the first dataset
   version, so with the same seed these files are byte-identical to the previous release.
2. shahzaibvohra/realwaste (RealWaste, CC BY 4.0 per dataset card): photos taken at a
   landfill material recovery facility, closer to field photos from the mobile app.
   `Food Organics` + `Vegetation` map to organic, `Plastic` to plastic and `Paper` +
   `Cardboard` to paper. Files are prefixed `realwaste_` and downscaled to a 256 px short side.

Each source is deduplicated with an average hash and split 70/15/15 per class, so every
split contains both photo styles.
"""

from __future__ import annotations

import argparse
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
from huggingface_hub import hf_hub_download, snapshot_download
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

REALWASTE_REPO = "shahzaibvohra/realwaste"
REALWASTE_URL = f"https://huggingface.co/datasets/{REALWASTE_REPO}"
REALWASTE_LICENSE = "CC BY 4.0 (as declared on the Hugging Face dataset card)"
REALWASTE_CITATION = (
    "S. Single, S. Iranmanesh, R. Raad, 'RealWaste: A Novel Real-Life Data Set for "
    "Landfill Waste Classification Using Deep Learning', Information 14(12):633, 2023."
)
REALWASTE_MAPPING: dict[str, tuple[str, ...]] = {
    "organic": ("Food Organics", "Vegetation"),
    "plastic": ("Plastic",),
    "paper": ("Paper", "Cardboard"),
}
REALWASTE_SHORT_SIDE = 256

SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
HASH_SIZE = 8


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--per-class", type=int, default=150, help="NeoAivara images per class.")
    parser.add_argument(
        "--realwaste-per-class",
        type=int,
        default=250,
        help="RealWaste images per class, spread evenly over its mapped source classes.",
    )
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


def downscale(image: Image.Image, short_side: int) -> Image.Image:
    """Shrink so the short side equals `short_side`; never upscale."""
    width, height = image.size
    current = min(width, height)
    if current <= short_side:
        return image
    scale = short_side / current
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def even_quotas(total: int, parts: int) -> list[int]:
    """Split `total` into `parts` near-equal integers, larger ones first."""
    base, remainder = divmod(total, parts)
    return [base + (1 if index < remainder else 0) for index in range(parts)]


def realwaste_class(filename: str) -> str:
    """`Food Organics_12.jpg` -> `Food Organics`."""
    return Path(filename).stem.rsplit("_", 1)[0]


def read_source_rows() -> list[dict[str, Any]]:
    parquet_path = hf_hub_download(
        repo_id=SOURCE_REPO,
        filename=SOURCE_FILE,
        repo_type="dataset",
    )
    table = pq.read_table(parquet_path, columns=["image", "label"])
    return table.to_pylist()


def read_realwaste_files() -> dict[str, list[Path]]:
    wanted = sorted({name for names in REALWASTE_MAPPING.values() for name in names})
    root = Path(
        snapshot_download(
            repo_id=REALWASTE_REPO,
            repo_type="dataset",
            allow_patterns=[f"{name}_*.jpg" for name in wanted],
        )
    )
    files: dict[str, list[Path]] = {name: [] for name in wanted}
    for path in sorted(root.glob("*.jpg")):
        name = realwaste_class(path.name)
        if name in files:
            files[name].append(path)
    return files


def split_counts(per_class: int, class_count: int) -> dict[str, int]:
    train = int(per_class * SPLIT_RATIOS["train"])
    validation = int(per_class * SPLIT_RATIOS["validation"])
    test = per_class - train - validation
    return {"train": train, "validation": validation, "test": test}


def load_rgb(data: bytes) -> Image.Image | None:
    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
            return image.convert("RGB")
    except (OSError, ValueError):
        return None


def select_neoaivara(
    args: argparse.Namespace,
) -> tuple[dict[str, list[Image.Image]], dict[str, dict[str, int]]]:
    rows = read_source_rows()
    random.seed(args.seed)
    random.shuffle(rows)

    selected: dict[str, list[Image.Image]] = {class_name: [] for class_name in CLASS_MAPPING}
    stats: dict[str, dict[str, int]] = {}

    for class_name, source_name in CLASS_MAPPING.items():
        source_label = SOURCE_CLASSES[source_name]
        accepted = selected[class_name]
        accepted_hashes: list[np.ndarray] = []
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

            image_bytes = (row.get("image") or {}).get("bytes")
            rgb_image = load_rgb(image_bytes) if image_bytes else None
            if rgb_image is None:
                skipped_invalid += 1
                continue

            image_hash = average_hash(rgb_image)
            if is_near_duplicate(image_hash, accepted_hashes, args.near_duplicate_distance):
                skipped_duplicates += 1
                continue

            accepted.append(rgb_image)
            accepted_hashes.append(image_hash)

        if len(accepted) < args.per_class:
            raise RuntimeError(
                f"Only found {len(accepted)} unique usable NeoAivara images for "
                f"{class_name}; requested {args.per_class}"
            )
        stats[class_name] = {
            "requested": args.per_class,
            "selected": len(accepted),
            "skipped_near_duplicates": skipped_duplicates,
            "skipped_invalid": skipped_invalid,
        }

    return selected, stats


def select_realwaste(
    args: argparse.Namespace,
) -> tuple[dict[str, list[Image.Image]], dict[str, dict[str, Any]]]:
    files = read_realwaste_files()
    rng = random.Random(args.seed)
    selected: dict[str, list[Image.Image]] = {}
    stats: dict[str, dict[str, Any]] = {}

    for class_name, source_names in REALWASTE_MAPPING.items():
        accepted: list[Image.Image] = []
        accepted_hashes: list[np.ndarray] = []
        per_source: dict[str, int] = {}
        skipped_duplicates = 0
        skipped_invalid = 0

        for source_name, quota in zip(
            source_names, even_quotas(args.realwaste_per_class, len(source_names)), strict=True
        ):
            candidates = list(files[source_name])
            rng.shuffle(candidates)
            taken = 0
            for path in candidates:
                if taken >= quota:
                    break
                rgb_image = load_rgb(path.read_bytes())
                if rgb_image is None:
                    skipped_invalid += 1
                    continue
                image_hash = average_hash(rgb_image)
                if is_near_duplicate(image_hash, accepted_hashes, args.near_duplicate_distance):
                    skipped_duplicates += 1
                    continue
                accepted.append(downscale(rgb_image, REALWASTE_SHORT_SIDE))
                accepted_hashes.append(image_hash)
                taken += 1
            if taken < quota:
                raise RuntimeError(
                    f"Only found {taken} unique usable RealWaste '{source_name}' images; "
                    f"requested {quota}"
                )
            per_source[source_name] = taken

        rng.shuffle(accepted)
        selected[class_name] = accepted
        stats[class_name] = {
            "requested": args.realwaste_per_class,
            "selected": len(accepted),
            "per_source_class": per_source,
            "skipped_near_duplicates": skipped_duplicates,
            "skipped_invalid": skipped_invalid,
        }

    return selected, stats


def write_split(
    images: list[Image.Image],
    counts: dict[str, int],
    output_dir: Path,
    class_name: str,
    prefix: str,
) -> dict[str, int]:
    train_end = counts["train"]
    validation_end = train_end + counts["validation"]
    split_images = {
        "train": images[:train_end],
        "validation": images[train_end:validation_end],
        "test": images[validation_end:],
    }
    written: dict[str, int] = {}
    for split, split_items in split_images.items():
        split_dir = output_dir / split / class_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(split_items):
            image.save(
                split_dir / f"{prefix}{class_name}_{index:04d}.jpg", quality=92, optimize=True
            )
        written[split] = len(split_items)
    return written


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

    neo_images, neo_stats = select_neoaivara(args)
    real_images, real_stats = select_realwaste(args)

    neo_counts = split_counts(args.per_class, len(CLASS_MAPPING))
    real_counts = split_counts(args.realwaste_per_class, len(REALWASTE_MAPPING))

    manifest: dict[str, Any] = {
        "name": "smart-waste-organic-plastic-paper",
        "version": 2,
        "sources": {
            "neoaivara": {
                "url": SOURCE_URL,
                "repo": SOURCE_REPO,
                "file": SOURCE_FILE,
                "license": SOURCE_LICENSE,
                "class_mapping": CLASS_MAPPING,
                "source_class_ids": {
                    name: SOURCE_CLASSES[source] for name, source in CLASS_MAPPING.items()
                },
                "per_class": args.per_class,
                "split_counts_per_class": neo_counts,
                "file_prefix": "",
                "duplicate_filtering": neo_stats,
            },
            "realwaste": {
                "url": REALWASTE_URL,
                "repo": REALWASTE_REPO,
                "license": REALWASTE_LICENSE,
                "citation": REALWASTE_CITATION,
                "class_mapping": {k: list(v) for k, v in REALWASTE_MAPPING.items()},
                "per_class": args.realwaste_per_class,
                "split_counts_per_class": real_counts,
                "file_prefix": "realwaste_",
                "short_side_px": REALWASTE_SHORT_SIDE,
                "duplicate_filtering": real_stats,
            },
        },
        "seed": args.seed,
        "near_duplicate_hamming_distance": args.near_duplicate_distance,
        "split_ratio": SPLIT_RATIOS,
        "preprocessing": [
            "Decode source image bytes",
            "Convert to RGB",
            "Average-hash near-duplicate filtering per class and source",
            "Downscale RealWaste images to a 256 px short side (NeoAivara unchanged)",
            "Stratified 70/15/15 split per class and source",
            "Save re-encoded JPEG (quality=92)",
        ],
        "created_at": datetime.now(UTC).isoformat(),
        "counts": {},
        "counts_by_source": {"neoaivara": {}, "realwaste": {}},
    }

    for class_name, images in neo_images.items():
        random.Random(args.seed).shuffle(images)
        written = write_split(images, neo_counts, output_dir, class_name, prefix="")
        for split, count in written.items():
            manifest["counts_by_source"]["neoaivara"].setdefault(split, {})[class_name] = count
            totals = manifest["counts"].setdefault(split, {})
            totals[class_name] = totals.get(class_name, 0) + count

    for class_name, images in real_images.items():
        written = write_split(images, real_counts, output_dir, class_name, prefix="realwaste_")
        for split, count in written.items():
            manifest["counts_by_source"]["realwaste"].setdefault(split, {})[class_name] = count
            totals = manifest["counts"].setdefault(split, {})
            totals[class_name] = totals.get(class_name, 0) + count

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
