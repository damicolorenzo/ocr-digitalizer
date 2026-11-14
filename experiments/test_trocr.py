
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL_ID = "microsoft/trocr-base-handwritten"
IMAGE_PATH = Path("samples/testa.png")


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    device = torch.device("cpu")
    torch.set_num_threads(4)

    print(f"Device: {device}")
    print(f"Loading model: {MODEL_ID}")

    start = time.perf_counter()

    processor = TrOCRProcessor.from_pretrained(MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()

    print(f"Model loaded in {time.perf_counter() - start:.2f}s")

    image = Image.open(IMAGE_PATH).convert("RGB")

    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values.to(device)

    start = time.perf_counter()

    with torch.inference_mode():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=128
        )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    print("\n--- OCR RESULT ---")
    print(text)
    print(f"\nInference time: {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
