
import json

import pytest

from ocr_engine.evaluation.dataset import load_dataset
from ocr_engine.evaluation.metrics import evaluate_corpus


def test_valid_dataset(tmp_path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"dummy")

    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps([{
            "id": "sample_001",
            "image": "sample.png",
            "language": "en",
            "ground_truth": "Hello world",
        }]),
        encoding="utf-8",
    )

    samples = load_dataset(dataset, tmp_path)
    assert len(samples) == 1
    assert samples[0].id == "sample_001"


def test_duplicate_ids(tmp_path):
    (tmp_path / "sample.png").write_bytes(b"dummy")

    record = {
        "id": "duplicate",
        "image": "sample.png",
        "language": "en",
        "ground_truth": "Hello",
    }

    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps([record, record]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate"):
        load_dataset(dataset, tmp_path)


def test_perfect_corpus():
    result = evaluate_corpus(
        ["Hello world", "Good morning"],
        ["Hello world", "Good morning"],
    )

    assert result["cer"] == 0
    assert result["wer"] == 0


def test_mismatched_lengths():
    with pytest.raises(ValueError):
        evaluate_corpus(["Hello"], [])
