
import time
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from ocr_engine.preprocessing.line_segmenter import LineSegmenter

MODEL_ID = "microsoft/trocr-base-handwritten"
INPUT = Path("samples/test.png")

def main():
    image = cv2.imread(str(INPUT))

    if image is None:
        raise FileNotFoundError(INPUT)

    segmenter = LineSegmenter()
    lines = segmenter.segment(image)

    print(f"Detected lines: {len(lines)}")

    processor = TrOCRProcessor.from_pretrained(
        MODEL_ID,
        use_fast=False,
    )

    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
    model.eval()

    results = []

    for index, line in enumerate(lines):
        rgb = cv2.cvtColor(line.image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        pixel_values = processor(
            images=pil_image,
            return_tensors="pt",
        ).pixel_values

        start = time.perf_counter()

        with torch.inference_mode():
            ids = model.generate(
                pixel_values,
                max_new_tokens=128,
            )

        text = processor.batch_decode(
            ids,
            skip_special_tokens=True,
        )[0]

        elapsed = time.perf_counter() - start

        results.append({
            "line": index,
            "text": text,
            "bbox": line.bbox,
            "inference_seconds": round(elapsed, 2),
        })

        print(f"[{index}] {text} ({elapsed:.2f}s)")

    print("\n--- FULL TRANSCRIPTION ---")
    print("\n".join(item["text"] for item in results))

if __name__ == "__main__":
    main()