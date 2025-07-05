# utils/summarizer.py
from transformers import pipeline

def summarize_chunks(chunks, model_name="facebook/bart-large-cnn", max_length=120):
    summarizer = pipeline("summarization", model=model_name)
    summarized = []

    for chunk in chunks:
        try:
            result = summarizer(chunk["text"], max_length=max_length, min_length=30, do_sample=False)[0]
            summary = result["summary_text"]
        except Exception as e:
            print(f"⚠️ Error summarizing chunk {chunk['chunk_id']}: {e}")
            summary = ""
        
        summarized.append({
            "chunk_id": chunk["chunk_id"],
            "summary": summary
        })
    return summarized
