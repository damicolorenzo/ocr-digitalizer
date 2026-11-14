import argparse
import json
import shutil
import subprocess
from pathlib import Path
from .ingestion import load_pages
from .providers import NoOCRProvider, TesseractProvider
from .latex import save_latex


def process(input_path: Path, output_dir: Path, provider_name: str, compile_pdf: bool, language: str) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(exist_ok=True)
    provider = TesseractProvider(language) if provider_name == "tesseract" else NoOCRProvider()
    document = {"schema_version": "0.1", "source": input_path.name, "pages": []}
    for i, image in enumerate(load_pages(input_path), start=1):
        image.save(pages_dir / f"page_{i:03}.png")
        text = provider.recognize(image)
        elements = ([{"id": f"p{i}_e1", "type": "paragraph", "content": text, "bbox": None, "status": "unverified", "provider": provider_name}] if text else [])
        document["pages"].append({"page_number": i, "width": image.width, "height": image.height, "elements": elements})
    (output_dir / "document.json").write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    tex_dir = output_dir / "latex"
    tex_path = tex_dir / "main.tex"
    save_latex(document, tex_path)
    if compile_pdf:
        if not shutil.which("tectonic"):
            raise RuntimeError("Tectonic not found on PATH; .tex and JSON were generated")
        subprocess.run(["tectonic", "main.tex", "--outdir", str(output_dir.resolve())], cwd=tex_dir, check=True)
    print(f"Processed {len(document['pages'])} page(s) into {output_dir}")
    print(f"Structured data: {output_dir / 'document.json'}")
    print(f"LaTeX: {tex_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR Digitalizer starter")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provider", choices=["none", "tesseract"], default="none")
    parser.add_argument("--language", default="eng")
    parser.add_argument("--compile", action="store_true", help="Compile with installed Tectonic")
    args = parser.parse_args()
    process(args.input, args.output, args.provider, args.compile, args.language)


if __name__ == "__main__":
    main()
