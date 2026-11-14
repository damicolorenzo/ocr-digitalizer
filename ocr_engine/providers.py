from typing import Protocol
from PIL import Image


class OCRProvider(Protocol):
    def recognize(self, image: Image.Image) -> str: ...


class NoOCRProvider:
    """Pipeline smoke-test provider; does not perform recognition."""
    def recognize(self, image: Image.Image) -> str:
        return ""


class TesseractProvider:
    """Printed-text baseline, NOT reliable handwritten/math recognition."""
    def __init__(self, language: str = "eng"):
        self.language = language

    def recognize(self, image: Image.Image) -> str:
        import pytesseract
        return pytesseract.image_to_string(image, lang=self.language).strip()
