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
CHUNK_EMBED_PATH = "embeddings/chunks.pkl"
DATA_FOLDER = "data"
FILES = ["health.json", "science.json"]
MODEL_PATH = "models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf"
TOP_K = 3
TOP_CHUNKS_PER_DOC = 2

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
doc_lookup = {}

for file in FILES:
    path = os.path.join(DATA_FOLDER, file)
    with open(path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    for doc in docs:
        if "summary" in doc:
            doc_id = len(index_to_doc)
            index_to_doc[doc_id] = {
                "title": doc.get("title", "No Title"),
                "summary": doc["summary"],
                "text": doc["text"]
            }
            doc_lookup[doc_id] = doc  # full doc with chunks

print(f"📄 Total indexed summaries: {len(index_to_doc)}")

# -----------------------------
# Load Chunk Embeddings
# -----------------------------
with open(CHUNK_EMBED_PATH, "rb") as f:
    chunk_embeddings = pickle.load(f)

# -----------------------------
# Load Local TinyLlama Model
# -----------------------------
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=20  # adjust based on your setup
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
# RAPTOR: Refine Chunks from Matched Summaries
# -----------------------------
def refine_chunks_from_summary_match(hyde_embedding, doc_ids, top_k=2):
    refined_chunks = []
    for doc_id in doc_ids:
        doc = doc_lookup.get(int(doc_id), {})
        candidate_chunks = doc.get("chunks", [])
        scored = []
        for chunk in candidate_chunks:
            chunk_id = chunk.get("index_id")
            if chunk_id in chunk_embeddings:
                emb = chunk_embeddings[chunk_id]
                score = np.dot(hyde_embedding, emb)
                scored.append((score, chunk["text"]))
        scored.sort(reverse=True)
        refined_chunks.extend([text for _, text in scored[:top_k]])
    return refined_chunks

# -----------------------------
# Final Answer Prompt
# -----------------------------
def generate_final_answer(query: str, context_chunks: list) -> str:
    context = "\n\n".join(context_chunks)
    prompt = f"""### Instruction:
You are a helpful assistant. Use the following context to answer the user's question accurately and in a very detailed and structured manner.

### Context:
{context}

### Question:
{query}

### Answer:"""
    response = llm(prompt=prompt, temperature=0.5, max_tokens=256, stop=["###"])
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
    top_doc_ids = []
    for rank, (score, doc_id) in enumerate(zip(D[0], I[0])):
        doc = index_to_doc.get(int(doc_id), {})
        print(f"#{rank + 1} | ID: {doc_id} | Score: {score:.4f}")
        print(f"Title: {doc.get('title', '[No Title]')}")
        print(f"Summary: {doc.get('summary', '[No Summary]')[:500]}...\n")
        top_doc_ids.append(int(doc_id))

    print("🔎 Refining with RAPTOR to get best chunks...")
    best_chunks = refine_chunks_from_summary_match(query_vec[0], top_doc_ids, top_k=TOP_CHUNKS_PER_DOC)

    print("\n🧠 Final Answer Generation...\n")
    final_answer = generate_final_answer(query, best_chunks)
    print(f"✅ Answer:\n{final_answer}\n")

