# Dataset Documentation

## Source

| Field | Value |
|---|---|
| Dataset | `NeoAivara/Waste_Classification_data` |
| URL | https://huggingface.co/datasets/NeoAivara/Waste_Classification_data |
| Source file used | `data/train-00000-of-00001.parquet` |
| Publisher / uploader | `NeoAivara` on Hugging Face |
| License | Apache-2.0, as declared on the Hugging Face dataset card |
| Modality | Image classification |
| Original classes | 12 garbage categories |
| Images used in this project | 450 (150 organic, 150 plastic, 150 paper) |

The dataset card is the authority for the license declaration. Before redistributing this
dataset outside the technical-test submission, verify the original source and license
directly. The uploader may have aggregated images from other sources; no additional
ownership is claimed by this project.

## Class mapping

The source dataset does not have an `organic` class. This project maps:

| Project class | Source class | Source label ID | Description |
|---|---|---:|---|
| `organic` | `biological` | 1 | Biological / food-derived waste |
| `plastic` | `plastic` | 8 | Plastic bottles, packaging, and containers |
| `paper` | `paper` | 7 | Paper, cardboard, newspapers, and similar material |

Folder names and model labels use `organic`, `plastic`, and `paper` consistently. The
`biological → organic` mapping is a project decision; it is not an official label from the
source dataset.

## Preparation pipeline

Reproduce with:

```bash
uv run python scripts/prepare_dataset.py --per-class 150 --seed 42
```

The script performs:

1. Download the public parquet file from Hugging Face (cached locally).
2. Filter rows for source labels `biological`, `plastic`, and `paper`.
3. Decode each image and convert it to RGB.
4. Reject unreadable images.
5. Compute an 8×8 average hash per image and reject images with Hamming distance ≤ 2 from
   an already accepted image of the same class. This reduces exact and near duplicates
   within each class.
6. Select 150 accepted images per class in a deterministic seeded order.
7. Split each class deterministically into 70% training, 15% validation, and 15% test.
8. Save re-encoded JPEG files (quality 92) into the train/validation/test folders.
9. Write `dataset/manifest.json` with source, license, mapping, counts, deduplication
   statistics, split ratios, seed, and creation timestamp.

The test split is never used for training, hyperparameter tuning, or model selection.

## Deduplication statistics

From `dataset/manifest.json` for the recorded run:

| Class | Requested | Selected | Near-duplicates skipped | Invalid skipped |
|---|---:|---:|---:|---:|
| organic | 150 | 150 | 1 | 0 |
| plastic | 150 | 150 | 2 | 0 |
| paper | 150 | 150 | 16 | 0 |

## Final distribution

| Split | Organic | Plastic | Paper | Total |
|---|---:|---:|---:|---:|
| Training | 105 | 105 | 105 | 315 |
| Validation | 22 | 22 | 22 | 66 |
| Test | 23 | 23 | 23 | 69 |
| **Total** | **150** | **150** | **150** | **450** |

Per class:

- Training: `floor(150 × 0.70) = 105`
- Validation: `floor(150 × 0.15) = 22`
- Test: `150 - 105 - 22 = 23`

The split is stratified by construction because each class is split independently with the
same ratios.

## Preprocessing applied to stored images

- Decode source image bytes.
- Convert to RGB.
- Near-duplicate filter before splitting.
- Re-encode as JPEG quality 92.
- No manual relabeling, cropping, or color correction.

At training and inference time, images are resized to 224×224 with Pillow bilinear
resampling, converted to float32, and passed through
`tensorflow.keras.applications.mobilenet_v2.preprocess_input`.

## Data leakage controls

- Near-duplicate filtering happens before the split, so accepted duplicates cannot cross
  subsets.
- The test folder is written once and is not read by `waste_ml.training.train`.
- Random seeds are fixed (`--seed 42`) in both dataset preparation and training.
- The test split is evaluated only by `waste_ml.training.evaluate`.

## Limitations

- Only 450 images are used, from a single public source.
- The test set has 69 images, so accuracy has a wide uncertainty interval.
- The source is class-balanced by sampling, not by collecting new images; it may not
  represent all real-world waste conditions.
- `biological` is treated as a proxy for `organic`. Images containing non-organic
  contaminants in that class would not be detected by the source labels.
