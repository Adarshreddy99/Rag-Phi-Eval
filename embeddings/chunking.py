# utils/chunking.py
import uuid
from typing import List

def chunk_documents(docs, chunk_size=300, overlap=50):
    all_chunks = []
    for doc_id, doc in enumerate(docs):
        text = doc.get("text", "")
        title = doc.get("title", f"doc_{doc_id}")
        if not text.strip():
            continue

        words = text.split()
        i = 0
        chunk_id = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)

            all_chunks.append({
                "chunk_id": f"{title}_{chunk_id}",
                "title": title,
                "text": chunk_text
            })

            chunk_id += 1
            i += chunk_size - overlap
    return all_chunks
