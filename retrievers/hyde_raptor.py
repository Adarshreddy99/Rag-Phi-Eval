# retrievers/hyde_raptor.py

import os
import json
import pickle
import faiss
import numpy as np
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

# -----------------------------
# Paths and Constants
# -----------------------------
SUMMARY_EMBED_PATH = "embeddings/summaries.pkl"
SUMMARY_INDEX_PATH = "faiss/summaries.index"
DATA_FOLDER = "data"
FILES = ["health.json", "science.json"]
MODEL_PATH = "models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf"
TOP_K = 3

# -----------------------------
# Load FAISS Summary Index
# -----------------------------
index = faiss.read_index(SUMMARY_INDEX_PATH)
print(f"📦 FAISS summary index loaded with {index.ntotal} vectors")
print(f"🧭 Index dimension: {index.d}")

# -----------------------------
# Load Summary Embeddings
# -----------------------------
with open(SUMMARY_EMBED_PATH, "rb") as f:
    embeddings = pickle.load(f)
print(f"🔢 Summary embeddings shape: {embeddings.shape}")

# -----------------------------
# Build Index-to-Document Mapping
# -----------------------------
index_to_doc = {}

for file in FILES:
    path = os.path.join(DATA_FOLDER, file)
    with open(path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    for i, doc in enumerate(docs):
        if "summary" in doc:
            index_to_doc[len(index_to_doc)] = {
                "title": doc.get("title", "No Title"),
                "summary": doc["summary"],
                "text": doc["text"]
            }

print(f"📄 Total indexed summaries: {len(index_to_doc)}")

# -----------------------------
# Load Local TinyLlama Model
# -----------------------------
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=20  # Adjust based on your GPU setup
)
print("🧠 TinyLlama model loaded")

# -----------------------------
# Load SentenceTransformer
# -----------------------------
encoder = SentenceTransformer("all-MiniLM-L6-v2")
print("🔤 SentenceTransformer loaded")

# -----------------------------
# HyDE Prompt Template
# -----------------------------
def generate_hypothetical_answer(query: str) -> str:
    prompt = f"""### Instruction:
Answer the question below briefly.

### Question:
{query}

### Answer:"""
    response = llm(prompt=prompt, temperature=0.5, max_tokens=128, stop=["###"])
    return response['choices'][0]['text'].strip()

# -----------------------------
# Query Loop
# -----------------------------
while True:
    query = input("\n🔍 Enter your query (or 'exit'): ").strip()
    if query.lower() in ["exit", "quit"]:
        break

    print("\n🤖 Generating hypothetical answer...")
    hypothetical = generate_hypothetical_answer(query)
    print(f"💡 Hypothetical Answer:\n{hypothetical}\n")

    query_vec = encoder.encode([hypothetical])
    D, I = index.search(query_vec, TOP_K)

    print("📄 Top Retrieved Documents:\n")
    for rank, (score, doc_id) in enumerate(zip(D[0], I[0])):
        doc = index_to_doc.get(int(doc_id), {})
        print(f"#{rank + 1} | ID: {doc_id} | Score: {score:.4f}")
        print(f"Title: {doc.get('title', '[No Title]')}")
        print(f"Summary: {doc.get('summary', '[No Summary]')[:500]}...\n")
