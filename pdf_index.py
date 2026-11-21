from argparse import ArgumentParser
from pathlib import Path
import re
import pymupdf

from rag_utils import create_embeddings, load_embedding_model


CHUNK_WORDS = 100
OVERLAP_WORDS = 25
MAX_EXTRA_WORDS = 40
REFERENCES_HEADING = re.compile(
    r"^[ \t]*(references|bibliography)[ \t]*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


def build_pdf_index(pdf_path: Path, model):
    chunks = []

    with pymupdf.open(pdf_path, filetype="pdf") as pdf:
        if pdf.needs_pass:
            raise ValueError("Password-protected PDFs are not supported yet.")

        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text")
            references = REFERENCES_HEADING.search(text)

            if references:
                text = text[:references.start()]

            words = text.split()
            start = 0

            while start < len(words):
                end = min(start + CHUNK_WORDS, len(words))
                limit = min(end + MAX_EXTRA_WORDS, len(words))

                while end < limit and not words[end - 1].endswith((".", "?", "!")):
                    end += 1

                chunks.append({
                    "source": pdf_path.name,
                    "page": page_number,
                    "text": " ".join(words[start:end]),
                })

                if end == len(words):
                    break

                start = end - OVERLAP_WORDS

    if not chunks:
        raise ValueError("No selectable text found. Scanned PDFs need OCR.")

    texts = [chunk["text"] for chunk in chunks]
    return chunks, create_embeddings(model, texts)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("pdf", type=Path)
    args = parser.parse_args()

    chunks, embeddings = build_pdf_index(args.pdf, load_embedding_model())
    print(f"Chunks: {len(chunks)}")
    print(f"Embeddings shape: {embeddings.shape}")
    print("First chunk:", chunks[0])
