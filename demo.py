import os, json
from datasets import load_dataset

os.makedirs("data", exist_ok=True)

# Health
medquad = load_dataset("lavita/MedQuAD", split="train[:500]")
health_docs = [{"title": d["question"], "text": d["answer"]} for d in medquad]
with open("data/health.json", "w", encoding="utf8") as f:
    json.dump({"category": "health", "docs": health_docs}, f, ensure_ascii=False, indent=2)
print(f"✅ Saved {len(health_docs)} health entries")

# Indian Law
legal = load_dataset("MeeraR/legal-qa-dataset", split="train[:500]")
law_docs = [{"title": d["question"], "text": d["answer"]} for d in legal]
with open("data/indian_law.json", "w", encoding="utf8") as f:
    json.dump({"category": "indian_law", "docs": law_docs}, f, ensure_ascii=False, indent=2)
print(f"✅ Saved {len(law_docs)} law entries")

# Science Facts – using `sciq`
science = load_dataset("sciq", split="train[:500]")
sci_docs = [{"title": d["question"], "text": d["support"]} for d in science]
with open("data/science.json", "w", encoding="utf8") as f:
    json.dump({"category": "science", "docs": sci_docs}, f, ensure_ascii=False, indent=2)
print(f"✅ Saved {len(sci_docs)} science entries")
