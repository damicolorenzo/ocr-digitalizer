
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from ocr_engine.evaluation.metrics import evaluate_recognition
from ocr_engine.pipeline.handwriting_pipeline import HandwritingPipeline


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmarks/handwriting_samples.json"
OUTPUT = ROOT / "output/benchmarks/handwriting_results.json"


def main():
    samples = json.loads(DATASET.read_text(encoding="utf-8"))

    if not samples:
        raise ValueError("Benchmark dataset is empty")

    pipeline = HandwritingPipeline()
    results = []

    for sample in samples:
        image_path = ROOT / sample["image"]

        start = time.perf_counter()
        result = pipeline.process(image_path)
        elapsed = time.perf_counter() - start

        prediction = result["text"]
        metrics = evaluate_recognition(
            sample["ground_truth"],
            prediction,
        )

        results.append({
            "id": sample["id"],
            "language": sample["language"],
            "ground_truth": sample["ground_truth"],
            "prediction": prediction,
            "elapsed_seconds": round(elapsed, 3),
            "metrics": metrics,
        })

        print(f"\nSample: {sample['id']}")
        print(f"Prediction:\n{prediction}")
        print(f"CER: {metrics['cer']:.4f}")
        print(f"WER: {metrics['wer']:.4f}")
        print(f"Time: {elapsed:.2f}s")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": "microsoft/trocr-base-handwritten",
        "samples": results,
        "mean_cer": sum(
            item["metrics"]["cer"] for item in results
        ) / len(results),
        "mean_wer": sum(
            item["metrics"]["wer"] for item in results
        ) / len(results),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nBenchmark saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
