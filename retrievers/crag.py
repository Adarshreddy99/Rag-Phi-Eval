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
SCORE_THRESHOLD = 0.5

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
# Helper: Ask LLM
# -----------------------------
def ask_llama(prompt: str, max_tokens: int = 256) -> str:
    output = llm(prompt=prompt, max_tokens=max_tokens, stop=["</s>"])
    return output["choices"][0]["text"].strip()

# -----------------------------
# Decompose Query (Optionally Guided)
# -----------------------------
def decompose_query(query: str, prev_subqs=None, prev_chunks=None) -> list:
    guidance = ""
    if prev_subqs:
        joined_prev = "\n".join([f"- {q}" for q in prev_subqs])
        guidance += f"\nPreviously attempted sub-questions:\n{joined_prev}\n"
    if prev_chunks:
        flat_chunks = "\n".join(prev_chunks[:3])
        guidance += f"\nSome previous retrieved context:\n{flat_chunks}\n"

    prompt = f"""
You are a helpful assistant. Break the following complex question into exactly 2 simpler and clearer sub-questions for better retrieval.
{guidance}
Original Question: "{query}"

Sub-questions:
1."""
    response = ask_llama(prompt, max_tokens=200)
    subquestions = response.split("\n")
    subquestions = [q.strip("0123456789. ").strip() for q in subquestions if q.strip()]
    return subquestions[:2]

# -----------------------------
# Retrieve Chunks (Embed + Search)
# -----------------------------
def retrieve_chunks(question, k=3):
    q_vec = embed_model.encode([question])
    D, I = index.search(q_vec, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        chunk_data = index_to_chunk.get(int(idx), {})
        text = chunk_data.get("text", "[Missing]")
        results.append((text, int(idx), float(score)))
    return results

# -----------------------------
# Answer Subquestion
# -----------------------------
def answer_subquestion(subq, retrieved_texts):
    context = "\n\n".join(retrieved_texts)
    prompt = f"""
You are a helpful assistant. Use the provided context to answer the question.

Context:
{context}

Question: {subq}
Answer:"""
    return ask_llama(prompt, max_tokens=200)

# -----------------------------
# Final Answer
# -----------------------------
# -----------------------------
# Final Answer (Updated with Chunks)
# -----------------------------
def generate_final_answer(query, sub_qas, context_chunks):
    combined_qa = "\n".join([f"Q: {q}\nA: {a}" for q, a in sub_qas])
    combined_chunks = "\n\n".join(context_chunks[:5])  # limit to 5 chunks if too long

    prompt = f"""
You are a helpful assistant. Use the following evidence and question-answer pairs to provide a final, comprehensive, and corrected answer to the original query.

Original Query:
{query}

Sub-questions and answers:
{combined_qa}

Relevant retrieved context:
{combined_chunks}

Final Answer:"""
    return ask_llama(prompt, max_tokens=300)

# -----------------------------
# Main Loop
# -----------------------------
while True:
    query = input("\n🔍 Enter your query (or 'exit'): ").strip()
    if query.lower() in ["exit", "quit"]:
        break

    attempt = 0
    sub_qas = []
    good_enough = False
    previous_subqs = []
    previous_chunks = []

    while not good_enough and attempt < 2:
        print(f"\n🧠 Attempt #{attempt+1} at decomposing query...")
        subqs = decompose_query(query, previous_subqs, previous_chunks)
        print("🔹 Sub-questions:")
        for sq in subqs:
            print(f"- {sq}")

        current_chunks = []
        sub_qas = []

        total_good_chunks = 0

        for sq in subqs:
            print(f"\n🔍 Retrieving for: {sq}")
            results = retrieve_chunks(sq)
            for i, (text, idx, score) in enumerate(results):
                print(f"#{i+1} | ID: {idx} | Score: {score:.4f}")
            current_chunks.extend([text for text, _, _ in results])
            good_count = sum(1 for _, _, score in results if score > SCORE_THRESHOLD)
            total_good_chunks += good_count

            # Only use text for answer generation
            texts_only = [text for text, _, _ in results]
            answer = answer_subquestion(sq, texts_only)
            sub_qas.append((sq, answer))
            print(f"💬 Answer: {answer}")

        if total_good_chunks >= 2:
            good_enough = True
        else:
            print("\n⚠️ Retrieved chunks are weak. Re-decomposing with feedback...")
            previous_subqs = subqs
            previous_chunks = current_chunks
            attempt += 1

    print("\n🧠 Generating final answer...")
    final_answer = generate_final_answer(query, sub_qas, current_chunks)
    print(f"\n✅ Final Answer:\n{final_answer}")
