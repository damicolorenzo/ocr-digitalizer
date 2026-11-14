
import cv2
import numpy as np

from ocr_engine.pipeline.handwriting_pipeline import HandwritingPipeline
from ocr_engine.recognition.base import OCRRecognizer


class MockRecognizer(OCRRecognizer):
    def recognize(self, image: np.ndarray) -> str:
        return "Hello world"


def test_pipeline_with_mock(tmp_path):
    image_path = tmp_path / "sample.png"

    image = np.full((100, 300, 3), 255, dtype=np.uint8)

    cv2.putText(
        image,
        "Hello",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
    )

    cv2.imwrite(str(image_path), image)

    pipeline = HandwritingPipeline(
        recognizer=MockRecognizer()
    )

    result = pipeline.process(image_path)

    assert len(result["elements"]) >= 1
    assert result["text"] == "Hello world"
