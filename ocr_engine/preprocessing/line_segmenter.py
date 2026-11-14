
from dataclasses import dataclass

import cv2
import numpy as np

@dataclass
class TextLine:
    image: np.ndarray
    bbox: tuple[int, int, int, int]

class LineSegmenter:
    def __init__(
        self,
        threshold: int = 140,
        min_height: int = 8,
        padding: int = 5,
        min_ink_pixels: int = 3,
        max_row_gap: int = 3,
    ):
        self.threshold = threshold
        self.min_height = min_height
        self.padding = padding
        self.min_ink_pixels = min_ink_pixels
        self.max_row_gap = max_row_gap

    def segment(self, image: np.ndarray) -> list[TextLine]:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        _, binary = cv2.threshold(
            gray,
            self.threshold,
            255,
            cv2.THRESH_BINARY_INV,
        )

        projection = np.count_nonzero(binary, axis=1)
        active = projection >= self.min_ink_pixels

        # Bridge small vertical gaps within the same line.
        kernel = np.ones(
            2 * self.max_row_gap + 1,
            dtype=np.uint8,
        )
        active = np.convolve(
            active.astype(np.uint8),
            kernel,
            mode="same",
        ) > 0

        regions = []
        start = None

        for y, is_active in enumerate(active):
            if is_active and start is None:
                start = y
            elif not is_active and start is not None:
                if y - start >= self.min_height:
                    regions.append((start, y))
                start = None

        if start is not None:
            regions.append((start, len(active)))

        lines = []
        height, width = image.shape[:2]

        for y1, y2 in regions:
            top = max(0, y1 - self.padding)
            bottom = min(height, y2 + self.padding)

            # Crop horizontal whitespace too.
            region_binary = binary[top:bottom, :]
            ys, xs = np.where(region_binary > 0)

            if len(xs) == 0:
                continue

            left = max(0, int(xs.min()) - self.padding)
            right = min(width, int(xs.max()) + self.padding + 1)

            crop = image[top:bottom, left:right]

            lines.append(
                TextLine(
                    image=crop,
                    bbox=(left, top, right, bottom),
                )
            )

        return lines