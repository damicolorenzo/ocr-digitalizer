
import json

from pathlib import Path

from ocr_engine.pipeline.handwriting_pipeline import HandwritingPipeline


def main():
    pipeline = HandwritingPipeline()

    result = pipeline.process("samples/test.png")

    output = Path("output/handwriting")
    output.mkdir(parents=True, exist_ok=True)

    output_path = output / "result.json"

    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n--- RECOGNIZED TEXT ---")
    print(result["text"])

    print(f"\nJSON saved to: {output_path}")


if __name__ == "__main__":
    main()
