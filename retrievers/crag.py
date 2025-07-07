# retrievers/crag.py

import os
import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# -----------------------------
# Paths
# -----------------------------
EMBEDDING_PATH = "embeddings/chunks.pkl"
INDEX_PATH = "faiss/chunks.index"
DATA_FOLDER = "data"
FILES = ["health.json", "science.json"]
MODEL_PATH = "models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf"

# -----------------------------
# Load FAISS index and embeddings
# -----------------------------
index = faiss.read_index(INDEX_PATH)
with open(EMBEDDING_PATH, "rb") as f:
    embeddings = pickle.load(f)

# -----------------------------
# Load index-to-text mapping
# -----------------------------
index_to_chunk = {}
for file in FILES:
    path = os.path.join(DATA_FOLDER, file)
    with open(path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    for doc in docs:
        title = doc.get("title", "No Title")
        for chunk in doc.get("chunks", []):
            chunk_id = chunk["index_id"]
            index_to_chunk[chunk_id] = {
                "title": title,
                "text": chunk["text"]
            }

# -----------------------------
# Load Embedding Model and LLM
# -----------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4, verbose=False)
print("✅ Models loaded")

# -----------------------------
# Helper: Generate prompt for TinyLlama
# -----------------------------
def ask_llama(prompt: str, max_tokens: int = 256) -> str:
    output = llm(prompt=prompt, max_tokens=max_tokens, stop=["</s>"])
    return output["choices"][0]["text"].strip()

# -----------------------------
# Step 1: Query Decomposition
# -----------------------------
def decompose_query(query: str) -> list:
    prompt = f"""
You are a helpful assistant. Break the following question into exactly 2 simpler sub-questions for better retrieval:

Original Question: "{query}"

Sub-questions:
1."""
    response = ask_llama(prompt, max_tokens=200)
    subquestions = response.split("\n")
    subquestions = [q.strip("0123456789. ").strip() for q in subquestions if q.strip()]
    return subquestions[:2]  # 🔧 Ensure only the first 2 sub-questions are returned


# -----------------------------
# Step 2: Retrieve Chunks for Subquestion
# -----------------------------
def retrieve_chunks(question, k=2):
    q_vec = embed_model.encode([question])
    D, I = index.search(q_vec, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        chunk_data = index_to_chunk.get(int(idx), {})
        text = chunk_data.get("text", "[Missing]")
        results.append((text, int(idx), float(score)))  # ✅ Return text, id, score
    return results


# -----------------------------
# Step 3: Answer Subquestion
# -----------------------------
def answer_subquestion(subq, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""
You are a helpful assistant. Use the provided context to answer the question.

Context:
{context}

Question: {subq}
Answer:"""
    return ask_llama(prompt, max_tokens=200)

# -----------------------------
# Step 4: Final Corrective Answer
# -----------------------------
def generate_final_answer(query, sub_qas):
    combined = "\n".join([f"Q: {q}\nA: {a}" for q, a in sub_qas])
    prompt = f"""
You are a helpful assistant. Use the following question-answer pairs to provide a final, comprehensive and corrected answer to the original query and keep the answer very detailed.

Original Query: {query}

Sub-answers:
{combined}

Final Answer:"""
    return ask_llama(prompt, max_tokens=300)

# -----------------------------
# Main Loop
# -----------------------------
while True:
    query = input("\n🔍 Enter your query (or 'exit'): ").strip()
    if query.lower() in ["exit", "quit"]:
        break

    print("\n🤖 Decomposing query...")
    subqs = decompose_query(query)
    print(f"🔹 Sub-questions:\n" + "\n".join([f"- {q}" for q in subqs]))

    sub_qas = []
    for sq in subqs:
        print(f"\n🔎 Retrieving for: {sq}")
        chunks = retrieve_chunks(sq)
        for i, (text, idx, score) in enumerate(chunks):
            print(f"#{i+1} | ID: {idx} | Score: {score:.4f}")
    
    # ✅ Fix: Extract just the text for answering
        retrieved_texts = [text for text, _, _ in chunks]
        answer = answer_subquestion(sq, retrieved_texts)
        sub_qas.append((sq, answer))
        print(f"💬 Answer: {answer}")

    print("\n🧠 Generating final answer...")
    final_answer = generate_final_answer(query, sub_qas)
    print(f"\n✅ Final Answer:\n{final_answer}")
