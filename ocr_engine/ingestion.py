from pathlib import Path
from PIL import Image, ImageOps


def load_pages(path: Path, dpi: int = 200) -> list[Image.Image]:
    """Load raster images or render PDF pages to RGB PIL images."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import fitz
        pages = []
        with fitz.open(path) as doc:
            if doc.needs_pass:
                raise ValueError("Password-protected PDFs are not supported")
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
                pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        return pages
    if suffix not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        raise ValueError(f"Unsupported input type: {suffix}")
    with Image.open(path) as img:
        return [ImageOps.exif_transpose(img).convert("RGB")]
