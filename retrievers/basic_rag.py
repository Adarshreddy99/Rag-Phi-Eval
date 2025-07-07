# retrievers/basic_rag.py

import os
import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# Paths
# -----------------------------
EMBEDDING_PATH = "embeddings/chunks.pkl"
INDEX_PATH = "faiss/chunks.index"
DATA_FOLDER = "data"
FILES = ["health.json", "science.json"]  # Only two categories used

# -----------------------------
# Load FAISS index
# -----------------------------
index = faiss.read_index(INDEX_PATH)
print(f"📦 FAISS index loaded with {index.ntotal} vectors")
print(f"🧭 Index dimension: {index.d}")

# -----------------------------
# Load Chunk Embeddings
# -----------------------------
with open(EMBEDDING_PATH, "rb") as f:
    embeddings = pickle.load(f)
print(f"🔢 Chunk embeddings shape: {embeddings.shape}")

# -----------------------------
# Build Index-to-Text Mapping
# -----------------------------
index_to_chunk = {}

for file in FILES:
    path = os.path.join(DATA_FOLDER, file)
    with open(path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    for doc in docs:
        title = doc.get("title", "No Title")
        for chunk in doc.get("chunks", []):
            chunk_text = chunk["text"]
            chunk_id = chunk["index_id"]
            index_to_chunk[chunk_id] = {
                "title": title,
                "text": chunk_text
            }

print(f"📄 Total indexed chunks: {len(index_to_chunk)}")

# -----------------------------
# Load SentenceTransformer Model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
print("🧠 SentenceTransformer model loaded")

# -----------------------------
# Query Loop
# -----------------------------
while True:
    query = input("\n🔍 Enter your query (or type 'exit'): ").strip()
    if query.lower() in ["exit", "quit"]:
        break

    # Encode the query
    query_vec = model.encode([query])
    
    # Search top-k
    k = 3
    D, I = index.search(query_vec, k)

    print("\n🔎 Top Retrieved Chunks:\n")
    for rank, (score, chunk_id) in enumerate(zip(D[0], I[0])):
        chunk_data = index_to_chunk.get(int(chunk_id), {})
        chunk_text = chunk_data.get("text", "[Missing]")
        chunk_title = chunk_data.get("title", "[No Title]")

        print(f"#{rank + 1} | ID: {chunk_id} | Score: {score:.4f}")
        print(f"Title: {chunk_title}")
        print(f"Chunk:\n{chunk_text[:500]}...\n")  # Truncate for readability
