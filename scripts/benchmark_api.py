"""Measure end-to-end HTTP latency of a running prediction API."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/evaluation/api_latency.json")
    )
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(array.mean()), 3),
        "std": round(float(array.std(ddof=0)), 3),
        "min": round(float(array.min()), 3),
        "p50": round(float(np.percentile(array, 50)), 3),
        "p95": round(float(np.percentile(array, 95)), 3),
        "max": round(float(array.max()), 3),
    }


def main() -> int:
    args = parse_args()
    image_bytes = args.image.read_bytes()
    endpoint = f"{args.base_url.rstrip('/')}/api/v1/predict"
    files = {"file": (args.image.name, image_bytes, "image/jpeg")}
    latencies: list[float] = []
    statuses: list[int] = []

    with httpx.Client(timeout=60.0) as client:
        for _ in range(args.warmup):
            client.post(endpoint, files=files)
        for _ in range(args.requests):
            started = time.perf_counter()
            response = client.post(endpoint, files=files)
            latencies.append((time.perf_counter() - started) * 1000)
            statuses.append(response.status_code)

    if any(status != 200 for status in statuses):
        raise RuntimeError(f"Non-200 responses from {endpoint}: {sorted(set(statuses))}")

    report = {
        "endpoint": endpoint,
        "image": str(args.image),
        "requests": args.requests,
        "warmup_requests": args.warmup,
        "status_codes": sorted(set(statuses)),
        "latency_ms": summarize(latencies),
        "median_ms": round(statistics.median(latencies), 3),
        "measured_at": datetime.now(UTC).isoformat(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
