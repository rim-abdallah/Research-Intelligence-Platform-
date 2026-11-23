from argparse import ArgumentParser
from pathlib import Path

import pymupdf


def render_evidence_page(
    pdf_bytes: bytes, pdf_page: int, quote: str
) -> tuple[bytes, bool]:
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
        if not 1 <= pdf_page <= len(pdf):
            raise ValueError("PDF page number is out of range.")

        page = pdf[pdf_page - 1]
        quote = " ".join(quote.split())
        matches = page.search_for(quote, quads=True) if quote else []
        if quote and not matches:
            matches = match_extracted_words(page, quote)


        if matches:
            page.add_highlight_annot(matches)

        scale = min(2, 1500 / max(page.rect.width, page.rect.height))
        image = page.get_pixmap(
            matrix=pymupdf.Matrix(scale, scale), alpha=False
        )
        return image.tobytes("png"), bool(matches)
    
    
def match_extracted_words(page, quote):
    target = quote.split()
    if not target:
        return []

    words = page.get_text("words")
    tokens = [word[4] for word in words]
    size = len(target)
    quads = []

    for start in range(len(tokens) - size + 1):
        if tokens[start:start + size] != target:
            continue

        lines = {}
        for word in words[start:start + size]:
            key = (word[5], word[6])  # Block number, line number
            rect = pymupdf.Rect(*word[:4])
            lines[key] = lines[key] | rect if key in lines else rect

        quads.extend(rect.quad for rect in lines.values())

    return quads


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("pdf_page", type=int)
    parser.add_argument("quote")
    args = parser.parse_args()

    png, found = render_evidence_page(
        args.pdf.read_bytes(), args.pdf_page, args.quote
    )
    Path("evidence_preview.png").write_bytes(png)
    print("Highlight found:", found)
    print("Preview saved: evidence_preview.png")
