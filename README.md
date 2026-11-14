# OCR Digitalizer — Milestone 1 starter

Local-first Python CLI scaffold for converting image/PDF pages into structured JSON and a LaTeX document.

**Current limitations:** The default `none` provider does not perform OCR. The optional Tesseract provider is a *printed-text baseline* and is not a reliable handwritten/math recognizer. No automatic layout segmentation, diagram extraction, mathematical OCR, or cloud provider is implemented yet. LaTeX generation supports paragraphs, headings, equations and image references in the intermediate schema. This is a foundation, not a completed OCR system.

## Install

Python 3.11+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e '.[dev]'
```

## Smoke test (no OCR)

```bash
ocr-digitalizer --input samples/page.png --output output/example
```

This produces page images, `document.json`, and `latex/main.tex`, with no recognized text.

## Optional printed-text OCR baseline

Install the Tesseract executable and ensure it is on PATH. Then:

```bash
pip install -e '.[ocr,dev]'
ocr-digitalizer --input samples/page.png --output output/example --provider tesseract
```

For Italian, install Tesseract's `ita` language pack and add `--language ita`.

## Optional PDF compilation

Install Tectonic separately and add `--compile`. The output PDF is placed in the chosen output directory. Without it, the generated `.tex` can be compiled elsewhere.

## Run tests

```bash
pytest -q
```

## Next implementation milestone

Create a 20–30 page handwritten evaluation dataset, manually transcribe ground truth, implement layout-region extraction, and benchmark specialized handwriting and math recognition models against this baseline. Keep all model calls behind the `OCRProvider` interface.
