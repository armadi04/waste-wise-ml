"""Suggest a minimum-confidence threshold from waste and non-waste photos.

The classifier always returns one of its classes, so a photo of anything else is forced into
organic, plastic or paper. This script scores two folders with the real model and reports how
well a minimum top-1 probability separates them:

    uv run python scripts/calibrate_threshold.py --foreign-dir /path/to/non-waste-photos

``--waste-dir`` defaults to ``dataset/test``. Softmax confidence is a weak out-of-distribution
signal, so read the acceptance and rejection numbers before trusting a threshold.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from waste_ml.config import SERVICE_ROOT, get_settings
from waste_ml.inference.predictor import WasteClassifier

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
THRESHOLDS = tuple(round(0.5 + step * 0.05, 2) for step in range(10))
MIN_WASTE_ACCEPTED = 0.95
MIN_FOREIGN_SEPARATED = 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waste-dir", type=Path, default=SERVICE_ROOT / "dataset" / "test")
    parser.add_argument("--foreign-dir", type=Path, required=True)
    return parser.parse_args()


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def top1_and_margin(probabilities: dict[str, float]) -> tuple[float, float]:
    """Return the top probability and its gap to the runner-up."""
    ranked = sorted(probabilities.values(), reverse=True)
    return ranked[0], ranked[0] - ranked[1]


def share_at_or_above(scores: list[float], threshold: float) -> float:
    if not scores:
        return 0.0
    return float(np.mean([score >= threshold for score in scores]))


def evaluate(waste_scores: list[float], foreign_scores: list[float]) -> list[dict[str, float]]:
    """Per threshold: share of waste accepted and share of foreign photos rejected."""
    return [
        {
            "threshold": threshold,
            "waste_accepted": share_at_or_above(waste_scores, threshold),
            "foreign_rejected": 1.0 - share_at_or_above(foreign_scores, threshold),
        }
        for threshold in THRESHOLDS
    ]


def recommend(rows: list[dict[str, float]]) -> dict[str, float] | None:
    """Highest threshold that still accepts enough waste, or None when none does."""
    eligible = [row for row in rows if row["waste_accepted"] >= MIN_WASTE_ACCEPTED]
    return eligible[-1] if eligible else None


def score_folder(classifier: WasteClassifier, folder: Path) -> tuple[list[float], list[float]]:
    tops: list[float] = []
    margins: list[float] = []
    for path in list_images(folder):
        result = classifier.predict_bytes(path.read_bytes())
        top, margin = top1_and_margin(result["probabilities"])
        tops.append(top)
        margins.append(margin)
    return tops, margins


def describe(name: str, tops: list[float]) -> None:
    if not tops:
        print(f"{name}: no images")
        return
    p10, p50, p90 = np.percentile(tops, [10, 50, 90])
    print(
        f"{name}: n={len(tops)} min={min(tops):.3f} p10={p10:.3f} "
        f"median={p50:.3f} p90={p90:.3f} max={max(tops):.3f}"
    )


def main() -> int:
    args = parse_args()
    for folder in (args.waste_dir, args.foreign_dir):
        if not folder.is_dir():
            raise SystemExit(f"Not a directory: {folder}")

    settings = get_settings()
    classifier = WasteClassifier(
        settings.resolved_model_path, settings.resolved_model_metadata_path
    )

    waste_tops, _ = score_folder(classifier, args.waste_dir)
    foreign_tops, _ = score_folder(classifier, args.foreign_dir)
    if not waste_tops or not foreign_tops:
        raise SystemExit("Both folders need at least one image.")

    describe("waste  ", waste_tops)
    describe("foreign", foreign_tops)

    rows = evaluate(waste_tops, foreign_tops)
    print("\nthreshold  waste accepted  foreign rejected")
    for row in rows:
        print(
            f"{row['threshold']:>9.2f}  {row['waste_accepted']:>13.1%}  "
            f"{row['foreign_rejected']:>15.1%}"
        )

    best = recommend(rows)
    print()
    if best is None:
        print(f"No threshold keeps >= {MIN_WASTE_ACCEPTED:.0%} of waste photos accepted.")
        return 1
    passed = best["foreign_rejected"] >= MIN_FOREIGN_SEPARATED
    print(
        f"Suggested ML_MIN_CONFIDENCE={best['threshold']:.2f} "
        f"(waste accepted {best['waste_accepted']:.1%}, "
        f"foreign rejected {best['foreign_rejected']:.1%})."
    )
    print(
        f"MVP gate (>= {MIN_FOREIGN_SEPARATED:.0%} foreign rejected): "
        f"{'PASS' if passed else 'FAIL, consider adding an `other` class'}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
