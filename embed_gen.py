# process_all.py
import os
import json
import pickle
import sys
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from chunking import chunk_documents
from summarizer import summarize_chunks

DATA_PATH = "data"
EMBED_PATH = "embeddings"
FILES = ["health.json", "science.json", "indian_law.json"]

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
        raw = load_json(full_path)
        docs = raw["docs"] if isinstance(raw, dict) and "docs" in raw else raw
        print(f"📄 Processing {file} with {len(docs)} docs...")

        # Chunking
        chunks = chunk_documents(docs)
        print(f"🔪 {len(chunks)} chunks created")

        # Summarization
        summaries = summarize_chunks(chunks)
        print(f"📚 {len(summaries)} summaries generated")

        # Enrich docs
        chunk_map = {c["chunk_id"]: c["text"] for c in chunks}
        summary_map = {s["chunk_id"]: s["summary"] for s in summaries}

        for doc in tqdm(docs, desc=f"🔗 Enriching {file}"):
            doc_chunks = [c["text"] for c in chunks if c["title"] == doc["title"]]
            doc_summaries = [summary_map[c["chunk_id"]] for c in chunks if c["title"] == doc["title"]]
            doc["chunks"] = doc_chunks
            doc["summaries"] = doc_summaries

        # Save enriched file back
        enriched = {"category": raw.get("category", file.replace(".json", "")), "docs": docs}
        save_json(enriched, full_path)
        print(f"💾 Saved enriched {file}\n")

        all_chunks.extend(chunks)
        all_summaries.extend(summaries)

    # Embedding section
    print("🔍 Loading SentenceTransformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    chunk_texts = [chunk["text"] for chunk in all_chunks]
    summary_texts = [s["summary"] for s in all_summaries]

    print("\n🧬 Generating chunk embeddings...")
    chunk_embeddings = model.encode(chunk_texts, show_progress_bar=True)

    print("🧬 Generating summary embeddings...")
    summary_embeddings = model.encode(summary_texts, show_progress_bar=True)

    save_embeddings(chunk_embeddings, os.path.join(EMBED_PATH, "chunks.pkl"))
    save_embeddings(summary_embeddings, os.path.join(EMBED_PATH, "summaries.pkl"))

    print("\n✅ All chunks, summaries, and embeddings processed and saved.")

if __name__ == "__main__":
    process_all()
