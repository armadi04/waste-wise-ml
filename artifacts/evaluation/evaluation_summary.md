# Evaluation Summary

- **Model:** waste-classifier v1.0.0
- **Architecture:** MobileNetV2
- **Test samples:** 69
- **Test accuracy:** 91.30%
- **Evaluation date:** 2026-11-03T02:08:42.447067+00:00

## Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| organic | 1.0000 | 1.0000 | 1.0000 | 23 |
| plastic | 0.9474 | 0.7826 | 0.8571 | 23 |
| paper | 0.8148 | 0.9565 | 0.8800 | 23 |

## Confusion Matrix

| True \ Predicted | organic | plastic | paper |
|---|---|---|---|
| organic | 23 | 0 | 0 |
| plastic | 0 | 18 | 5 |
| paper | 0 | 1 | 22 |

## Inference Latency

Measured on 20 test images with batch size 1 on CPU (Intel(R) Core(TM) i7-8665U CPU @ 1.90GHz).

| Stage | Mean (ms) | Std (ms) | Min (ms) | P95 (ms) | Max (ms) |
|---|---:|---:|---:|---:|---:|
| Preprocessing | 2.447 | 1.005 | 1.447 | 3.940 | 4.780 |
| Inference | 80.436 | 15.679 | 70.475 | 98.485 | 144.365 |
| End To End | 82.883 | 15.973 | 71.922 | 101.823 | 147.278 |

## Environment

- Python: 3.12.16
- TensorFlow: 2.21.0
- Platform: Linux-7.0.0-42-generic-x86_64-with-glibc2.43
- CPU: Intel(R) Core(TM) i7-8665U CPU @ 1.90GHz
- GPU: none
- Input size: 224 x 224

Accuracy is computed only on the held-out test split. Confidence returned by the API is the model probability for one prediction, not model accuracy.
