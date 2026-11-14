from pathlib import Path

ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def escape_text(value: str) -> str:
    return "".join(ESCAPES.get(ch, ch) for ch in value)


def render_document(document: dict) -> str:
    lines = [r"\documentclass{article}", r"\usepackage[T1]{fontenc}", r"\usepackage{amsmath,amssymb,graphicx}", r"\begin{document}"]
    for index, page in enumerate(document["pages"]):
        if index:
            lines.append(r"\clearpage")
        for element in page["elements"]:
            content = element.get("content", "")
            kind = element["type"]
            if kind == "paragraph":
                lines.append(escape_text(content) + "\n")
            elif kind == "heading":
                lines.append(r"\section{" + escape_text(content) + "}")
            elif kind == "equation":
                lines.extend([r"\begin{equation}", content, r"\end{equation}"])
            elif kind == "figure":
                asset = element.get("asset")
                if asset:
                    lines.append(r"\includegraphics[width=\linewidth]{\detokenize{" + asset + "}}")
            else:
                raise ValueError(f"Unsupported element type: {kind}")
    lines.append(r"\end{document}")
    return "\n".join(lines) + "\n"


def save_latex(document: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_document(document), encoding="utf-8")
