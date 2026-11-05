import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_threshold.py"
spec = importlib.util.spec_from_file_location("calibrate_threshold", SCRIPT)
assert spec is not None and spec.loader is not None
calibrate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibrate)


def test_top1_and_margin_uses_runner_up() -> None:
    top, margin = calibrate.top1_and_margin({"organic": 0.2, "plastic": 0.5, "paper": 0.3})
    assert top == 0.5
    assert abs(margin - 0.2) < 1e-9


def test_share_at_or_above_is_inclusive_and_safe_when_empty() -> None:
    assert calibrate.share_at_or_above([0.4, 0.5, 0.9], 0.5) == 2 / 3
    assert calibrate.share_at_or_above([], 0.5) == 0.0


def test_evaluate_reports_acceptance_and_rejection() -> None:
    rows = calibrate.evaluate([0.99, 0.95, 0.6], [0.55, 0.52, 0.98])
    row = next(r for r in rows if r["threshold"] == 0.7)
    assert abs(row["waste_accepted"] - 2 / 3) < 1e-9
    assert abs(row["foreign_rejected"] - 2 / 3) < 1e-9


def test_recommend_picks_highest_threshold_that_keeps_enough_waste() -> None:
    rows = [
        {"threshold": 0.5, "waste_accepted": 1.0, "foreign_rejected": 0.2},
        {"threshold": 0.7, "waste_accepted": 0.96, "foreign_rejected": 0.7},
        {"threshold": 0.9, "waste_accepted": 0.80, "foreign_rejected": 0.95},
    ]
    best = calibrate.recommend(rows)
    assert best is not None and best["threshold"] == 0.7


def test_recommend_returns_none_when_waste_is_lost_everywhere() -> None:
    rows = [{"threshold": 0.5, "waste_accepted": 0.9, "foreign_rejected": 0.9}]
    assert calibrate.recommend(rows) is None
