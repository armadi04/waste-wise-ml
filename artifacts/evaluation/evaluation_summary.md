# Evaluation Summary

- **Model:** waste-classifier v1.1.0
- **Architecture:** MobileNetV2
- **Test samples:** 183
- **Test accuracy:** 95.63%
- **Evaluation date:** 2026-11-05T08:19:11.066393+00:00

## Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| organic | 0.9677 | 0.9836 | 0.9756 | 61 |
| plastic | 0.9828 | 0.9344 | 0.9580 | 61 |
| paper | 0.9206 | 0.9508 | 0.9355 | 61 |

## Accuracy by Photo Source

| Source | Samples | Correct | Accuracy |
|---|---:|---:|---:|
| neoaivara | 69 | 66 | 95.65% |
| realwaste | 114 | 109 | 95.61% |

## Confusion Matrix

| True \ Predicted | organic | plastic | paper |
|---|---|---|---|
| organic | 60 | 0 | 1 |
| plastic | 0 | 57 | 4 |
| paper | 2 | 1 | 58 |

## Inference Latency

Measured on 20 test images with batch size 1 on CPU (Intel(R) Core(TM) i7-8665U CPU @ 1.90GHz).

| Stage | Mean (ms) | Std (ms) | Min (ms) | P95 (ms) | Max (ms) |
|---|---:|---:|---:|---:|---:|
| Preprocessing | 1.515 | 0.430 | 1.037 | 2.395 | 2.452 |
| Inference | 55.870 | 1.762 | 53.491 | 58.284 | 61.476 |
| End To End | 57.386 | 1.953 | 54.630 | 59.816 | 63.007 |

## Environment

- Python: 3.12.16
- TensorFlow: 2.21.0
- Platform: Linux-7.0.0-42-generic-x86_64-with-glibc2.43
- CPU: Intel(R) Core(TM) i7-8665U CPU @ 1.90GHz
- GPU: none
- Input size: 224 x 224

Accuracy is computed only on the held-out test split. Confidence returned by the API is the model probability for one prediction, not model accuracy.
