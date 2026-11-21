from pathlib import Path
import json
import numpy as np

from pypdf import PdfReader
from rag_utils import load_embedding_model, create_embeddings


pdf_path = Path(
    r"C:\Users\User\Desktop\papers\light gcn\light gcn.pdf"
)

chunks_path = Path("chunks.json")
embeddings_path = Path("chunk_embeddings.npy")


# Read the PDF page by page
reader = PdfReader(pdf_path)

pages = []

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""

    pages.append({
        "source": pdf_path.name,
        "page": page_number,
        "text": text
    })

    print(f"Page {page_number}: {len(text)} characters")

print("\nPages stored:", len(pages))

# Split each page into overlapping word chunks
chunk_size = 100
overlap = 25
step = chunk_size - overlap

chunks = []

for page_record in pages:
    words = page_record["text"].split()

    for start in range(0, len(words), step):
        chunk_text = " ".join(words[start:start + chunk_size])

        chunks.append({
            "source": page_record["source"],
            "page": page_record["page"],
            "text": chunk_text
        })

        if start + chunk_size >= len(words):
            break

print("\nNumber of chunks:", len(chunks))

# Save the chunks to a JSON file
with open(chunks_path, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print("Chunks saved to:", chunks_path)

# Load the embedding model
print("\nLoading embedding model...")

model = load_embedding_model()

# Extract only the text from each chunk
chunk_texts = [chunk["text"] for chunk in chunks]

# Create embeddings
print("Creating embeddings...")

chunk_embeddings = create_embeddings(
    model,
    chunk_texts,
    show_progress=True
)

print("Embeddings shape:", chunk_embeddings.shape)

# Save embeddings
np.save(embeddings_path, chunk_embeddings)

print("Embeddings saved to:", embeddings_path)