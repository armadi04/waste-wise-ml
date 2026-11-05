from waste_ml.training.train import dataset_reference


def test_dataset_reference_lists_every_source_of_a_multi_source_manifest() -> None:
    manifest = {
        "sources": {
            "neoaivara": {"url": "https://example/neo", "license": "Apache-2.0"},
            "realwaste": {"url": "https://example/real", "license": "CC BY 4.0"},
        }
    }

    reference = dataset_reference(manifest)

    assert [s["name"] for s in reference["sources"]] == ["neoaivara", "realwaste"]
    assert reference["sources"][1]["license"] == "CC BY 4.0"
    assert reference["manifest"] == "dataset/manifest.json"


def test_dataset_reference_still_reads_a_single_source_manifest() -> None:
    reference = dataset_reference({"source": "https://example/neo", "license": "Apache-2.0"})

    assert reference["source"] == "https://example/neo"
    assert "sources" not in reference


def test_dataset_reference_tolerates_a_missing_manifest() -> None:
    assert dataset_reference(None)["manifest"] == "dataset/manifest.json"
