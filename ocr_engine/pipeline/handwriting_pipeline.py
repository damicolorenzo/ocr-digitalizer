
from pathlib import Path

import cv2

from ocr_engine.preprocessing.line_segmenter import LineSegmenter
from ocr_engine.recognition.trocr_provider import TrOCRProvider


class HandwritingPipeline:
    def __init__(self):
        self.segmenter = LineSegmenter()
        self.recognizer = TrOCRProvider()

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
                "model": "microsoft/trocr-base-handwritten",
            })

        return {
            "source": str(image_path),
            "elements": elements,
            "text": "\n".join(
                element["content"]
                for element in elements
            ),
        }
