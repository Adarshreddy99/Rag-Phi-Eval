# process_all.py
import os
import json
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from embeddings.chunking import chunk_documents
from embeddings.summarizer import summarize_text

DATA_PATH = "data"
EMBED_PATH = "embeddings"
FILES = ["health.json", "science.json"]

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def save_embeddings(embeddings, path):
    with open(path, "wb") as f:
        pickle.dump(embeddings, f)

def process_all():
    all_chunks = []
    all_summaries = []

    print("🚀 Starting document processing...\n")

    for file in FILES:
        full_path = os.path.join(DATA_PATH, file)
        docs = load_json(full_path)['docs']
        print(f"📄 Processing {file} with {len(docs)} docs...")

        chunks = chunk_documents(docs)
        print(f"🔪 {len(chunks)} chunks created")

        chunk_map = {c["chunk_id"]: c["text"] for c in chunks}

        for doc in tqdm(docs, desc=f"📚 Enriching {file}"):
            doc["chunks"] = [c["text"] for c in chunks if c["title"] == doc["title"]]
            doc["summary"] = summarize_text(doc["text"])
            all_chunks.extend([c for c in chunks if c["title"] == doc["title"]])
            all_summaries.append({"title": doc["title"], "summary": doc["summary"]})

        save_json(docs, full_path)
        print(f"💾 Saved enriched {file}\n")

    print("🔍 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("🧬 Generating chunk embeddings...")
    chunk_embeddings = model.encode([c["text"] for c in all_chunks], show_progress_bar=True)

    print("🧬 Generating document summary embeddings...")
    summary_embeddings = model.encode([s["summary"] for s in all_summaries], show_progress_bar=True)

    save_embeddings(chunk_embeddings, os.path.join(EMBED_PATH, "chunks.pkl"))
    save_embeddings(summary_embeddings, os.path.join(EMBED_PATH, "summaries.pkl"))

    print("✅ All embeddings saved.")

if __name__ == "__main__":
    process_all()
