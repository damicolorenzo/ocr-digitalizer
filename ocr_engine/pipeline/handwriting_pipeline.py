
from pathlib import Path

import cv2

from ocr_engine.preprocessing.line_segmenter import LineSegmenter
from ocr_engine.recognition.base import OCRRecognizer
from ocr_engine.recognition.trocr_provider import TrOCRProvider


class HandwritingPipeline:
    def __init__(
        self,
        recognizer: OCRRecognizer | None = None,
        segmenter: LineSegmenter | None = None,
    ):
        self.segmenter = segmenter or LineSegmenter()
        self.recognizer = recognizer or TrOCRProvider()

    def process(self, image_path: str | Path) -> dict:
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(image_path)

        lines = self.segmenter.segment(image)
        elements = []

        for index, line in enumerate(lines):
            text = self.recognizer.recognize(line.image)

            elements.append({
                "id": f"line_{index:03d}",
                "type": "text_line",
                "content": text,
                "bbox": list(line.bbox),
                "model": type(self.recognizer).__name__,
            })

        return {
            "source": str(image_path),
            "elements": elements,
            "text": "\n".join(
                element["content"] for element in elements
            ),
        }
