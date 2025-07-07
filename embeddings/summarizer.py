# summarizer.py
from transformers import pipeline

# Use a small summarizer for fast processing
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_text(text, max_tokens=512):
    if len(text) == 0:
        return ""

    text = text.strip().replace("\n", " ")
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    result = summarizer(chunks, max_length=128, min_length=30, do_sample=False)
    summary = " ".join([r["summary_text"] for r in result])
    return summary
