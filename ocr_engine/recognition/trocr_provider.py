
import torch
import numpy as np

from ocr_engine.recognition.base import OCRRecognizer

from PIL import Image
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)


class TrOCRProvider(OCRRecognizer):
    def __init__(
        self,
        model_id="microsoft/trocr-base-handwritten",
        device="cpu",
    ):
        self.device = torch.device(device)

        self.processor = TrOCRProcessor.from_pretrained(
            model_id,
            use_fast=False,
        )

        self.model = VisionEncoderDecoderModel.from_pretrained(
            model_id
        )

        self.model.to(self.device)
        self.model.eval()

    def recognize(self, image: np.ndarray) -> str:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")

        # The segmenter returns OpenCV BGR images.
        import cv2

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        pixel_values = self.processor(
            images=pil_image,
            return_tensors="pt",
        ).pixel_values.to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                pixel_values,
                max_new_tokens=128,
            )

        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
