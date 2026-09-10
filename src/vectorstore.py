import chromadb
import os
import sys

def get_db_path():
    if getattr(sys, 'frozen', False):
        base = sys._MEIPASS
        return os.path.join(base, "data", "chroma_db")
    else:
        base = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base, "..", "data", "chroma_db")

DB_PATH = get_db_path()

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name="industry_docs")

def add_chunks(chunks, embeddings, source="unknown"):
    ids = [f"{source}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source} for _ in chunks]
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=ids,
        metadatas=metadatas
    )

def query(question_embedding, n_results=3):
    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=n_results
    )
    return results['documents'][0]