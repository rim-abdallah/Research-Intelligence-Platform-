import json
from pathlib import Path


chunks_path = Path("chunks.json")

with open(chunks_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)


keyword = input("Search chunks for: ").strip().lower()

for index, chunk in enumerate(chunks):
    if keyword in chunk["text"].lower():

        print("\n" + "=" * 70)
        print("Chunk index:", index)
        print("Page:", chunk["page"])
        print("Text:")
        print(chunk["text"])