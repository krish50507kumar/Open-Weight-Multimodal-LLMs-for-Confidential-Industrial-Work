import sys
import os
sys.path.append('src')

from extract import extract_text_from_pdf
from chunker import chunk_text
from embed import embed_chunks
from vectorstore import add_chunks

pdf_folder = "data/raw_pdfs"
skip_pages_map = {
    "equipment-manual-pdf": 2,
}

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        path = os.path.join(pdf_folder, filename)
        source_name = os.path.splitext(filename)[0]
        skip = skip_pages_map.get(source_name, 0)

        print(f"\nProcessing: {filename} (skipping first {skip} pages)")

        text = extract_text_from_pdf(path, skip_pages=skip)
        chunks = chunk_text(text)
        print(f"  {len(chunks)} chunks")

        embeddings = embed_chunks(chunks)
        add_chunks(chunks, embeddings, source=source_name)
        print(f"  Stored.")

print("\nAll PDFs ingested.")
