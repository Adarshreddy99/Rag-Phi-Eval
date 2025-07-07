import os
import pickle
import faiss
import numpy as np

EMBED_PATH = "embeddings"
INDEX_PATH = "faiss"

os.makedirs(INDEX_PATH, exist_ok=True)

def load_embeddings(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_index(index, path):
    faiss.write_index(index, path)

def build_flat_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # L2 = Euclidean (or you can use cosine with normalization)
    index.add(embeddings)
    return index

def build_and_save_all():
    print("🔍 Loading embeddings...")

    chunk_embeds = load_embeddings(os.path.join(EMBED_PATH, "chunks.pkl"))
    summary_embeds = load_embeddings(os.path.join(EMBED_PATH, "summaries.pkl"))

    print(f"📦 Chunk embeddings: {chunk_embeds.shape}")
    print(f"📦 Summary embeddings: {summary_embeds.shape}")

    print("\n⚙️ Building FAISS flat index for chunks...")
    chunk_index = build_flat_index(np.array(chunk_embeds))
    save_index(chunk_index, os.path.join(INDEX_PATH, "chunks.index"))
    print("✅ Chunks index saved.")

    print("\n⚙️ Building FAISS flat index for summaries...")
    summary_index = build_flat_index(np.array(summary_embeds))
    save_index(summary_index, os.path.join(INDEX_PATH, "summaries.index"))
    print("✅ Summaries index saved.")

if __name__ == "__main__":
    build_and_save_all()


# build_chunk_index.py
import os
import json
import pickle
import faiss
from sklearn.preprocessing import normalize

EMBED_PATH = "embeddings/chunks.pkl"
DATA_FILES = ["data/health.json", "data/science.json"]
INDEX_PATH = "faiss/chunks.index"
MAP_PATH = "faiss/chunks_index_map.json"

# Load chunk embeddings
with open(EMBED_PATH, "rb") as f:
    chunk_embeddings = pickle.load(f)
chunk_embeddings = normalize(chunk_embeddings, axis=1)

# Create FAISS index
dim = chunk_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(chunk_embeddings)
faiss.write_index(index, INDEX_PATH)
print(f"✅ FAISS index saved to {INDEX_PATH}")

# Create and save FAISS ID -> Chunk mapping
index_map = {}
counter = 0

for path in DATA_FILES:
    with open(path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    for doc in docs:
        for chunk in doc.get("chunks", []):
            index_map[counter] = {
                "title": doc["title"],
                "chunk": chunk
            }
            counter += 1

with open(MAP_PATH, "w", encoding="utf-8") as f:
    json.dump(index_map, f, indent=2)

print(f"✅ Mapping file saved to {MAP_PATH} with {len(index_map)} entries.")
