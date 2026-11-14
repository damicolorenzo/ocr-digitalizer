
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from ocr_engine.evaluation.dataset import load_dataset
from ocr_engine.evaluation.metrics import (
    evaluate_corpus,
    evaluate_recognition,
)
from ocr_engine.pipeline.handwriting_pipeline import HandwritingPipeline


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmarks/handwriting_samples.json"
OUTPUT = ROOT / "output/benchmarks/handwriting_results.json"


def main():
    samples = load_dataset(DATASET, ROOT)
    print(f"Loaded {len(samples)} samples")

    start = time.perf_counter()
    pipeline = HandwritingPipeline()
    model_load_seconds = time.perf_counter() - start

    results = []
    references = []
    predictions = []

    for sample in samples:
        start = time.perf_counter()
        result = pipeline.process(sample.image)
        elapsed = time.perf_counter() - start

        prediction = result["text"]
        metrics = evaluate_recognition(
            sample.ground_truth,
            prediction,
        )

        references.append(sample.ground_truth)
        predictions.append(prediction)

        results.append({
            "id": sample.id,
            "language": sample.language,
            "ground_truth": sample.ground_truth,
            "prediction": prediction,
            "elapsed_seconds": round(elapsed, 3),
            "metrics": metrics,
        })

        print(f"\nSample: {sample.id}")
        print(f"Prediction:\n{prediction}")
        print(f"CER: {metrics['cer']:.4f}")
        print(f"WER: {metrics['wer']:.4f}")
        print(f"Time: {elapsed:.2f}s")

    corpus_metrics = evaluate_corpus(references, predictions)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": "microsoft/trocr-base-handwritten",
        "sample_count": len(samples),
        "model_load_seconds": round(model_load_seconds, 3),
        "corpus_metrics": corpus_metrics,
        "samples": results,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n--- CORPUS METRICS ---")
    print(json.dumps(corpus_metrics, indent=2))
    print(f"\nSaved to: {OUTPUT}")


if __name__ == "__main__":
    main()
