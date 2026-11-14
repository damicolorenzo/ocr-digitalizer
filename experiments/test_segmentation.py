
from pathlib import Path

import cv2

from ocr_engine.preprocessing.line_segmenter import LineSegmenter

INPUT = Path("samples/test.png")
OUTPUT = Path("output/segmentation")

def main():
    image = cv2.imread(str(INPUT))

    if image is None:
        raise FileNotFoundError(INPUT)

    segmenter = LineSegmenter()
    lines = segmenter.segment(image)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    preview = image.copy()

    for index, line in enumerate(lines):
        path = OUTPUT / f"line_{index:03d}.png"

        cv2.imwrite(str(path), line.image)

        x1, y1, x2, y2 = line.bbox

        cv2.rectangle(
            preview,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        print(f"Line {index}: {line.bbox}")

    cv2.imwrite(str(OUTPUT / "preview.png"), preview)

    print(f"\nDetected {len(lines)} lines")

if __name__ == "__main__":
    main()