import numpy as np

from waste_ml.training.evaluate import accuracy_by_source, source_of, source_rows


def test_source_of_uses_the_file_prefix() -> None:
    assert source_of("dataset/test/paper/realwaste_paper_0003.jpg") == "realwaste"
    assert source_of("dataset/test/paper/paper_0003.jpg") == "neoaivara"


def test_accuracy_by_source_reports_each_source_separately() -> None:
    paths = ["a/paper_0.jpg", "a/paper_1.jpg", "a/realwaste_paper_0.jpg", "a/realwaste_paper_1.jpg"]
    true = np.array([0, 1, 2, 2])
    predicted = np.array([0, 1, 2, 0])

    result = accuracy_by_source(paths, true, predicted)

    assert result["neoaivara"] == {"samples": 2, "correct": 2, "accuracy": 1.0}
    assert result["realwaste"] == {"samples": 2, "correct": 1, "accuracy": 0.5}


def test_source_rows_are_skipped_for_a_single_source() -> None:
    single = {"neoaivara": {"samples": 2, "correct": 2, "accuracy": 1.0}}
    assert source_rows(single) == []
    assert source_rows(None) == []


def test_source_rows_render_a_markdown_table() -> None:
    both = {
        "neoaivara": {"samples": 2, "correct": 2, "accuracy": 1.0},
        "realwaste": {"samples": 2, "correct": 1, "accuracy": 0.5},
    }
    text = "\n".join(source_rows(both))
    assert "| realwaste | 2 | 1 | 50.00% |" in text
