import pytest
from PIL import Image
from ocr_engine.ingestion import load_pages
from ocr_engine.latex import render_document


def test_image_ingestion(tmp_path):
    path = tmp_path / "test.png"
    Image.new("RGB", (120, 80), "white").save(path)
    pages = load_pages(path)
    assert len(pages) == 1
    assert pages[0].size == (120, 80)


def test_latex_escaping_and_equation():
    document = {"pages": [{"elements": [{"type": "paragraph", "content": "a_b & 20%"}, {"type": "equation", "content": r"\frac{1}{2}"}]}]}
    tex = render_document(document)
    assert r"a\_b \& 20\%" in tex
    assert r"\frac{1}{2}" in tex


def test_unsupported_input(tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("hi")
    with pytest.raises(ValueError):
        load_pages(path)
