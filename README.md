# Rag-Phi-Eval

# 🔍 Comparing Retrieval-Augmented Generation (RAG) Techniques

In this project, I implemented and compared **three different RAG pipelines** on a custom-built dataset spanning multiple knowledge domains. The goal was to explore different retrieval strategies, from standard dense lookups to more advanced query rewriting and generative search prompts.

---

## 📦 RAG Variants Explored

### 1️⃣ Basic RAG (Dense Retrieval)

- Standard dense embedding-based retrieval.
- Each query is embedded and compared using **local FAISS index**.
- Returned chunks are passed directly to the model for generation.
- Baseline RAG pipeline without any query enhancements.

---

### 2️⃣ HyDE + RAPTOR

- **HyDE (Hypothetical Document Embeddings):**
  - I first generated hypothetical answers using a local LLM (Phi 2.8B).
  - These answers were then embedded and used as search queries.
- **RAPTOR-style Summary-Aware Retrieval:**
  - I generated both **document-level summaries** and **chunk-level summaries**.
  - Queries retrieved from a FAISS index built on these summaries.
- This led to **more focused and abstracted context matching** than basic dense retrieval.

---

### 3️⃣ Query Decomposition + CRAG (Corrective RAG)

- The most sophisticated approach in this project.
- Used a **query decomposition technique** with Phi 2.8B:
  - Broke complex queries into simpler sub-questions.
  - Retrieved chunks for each sub-question using dense vector search.
- **Corrective mechanism**:
  - If the top chunks’ scores were below a threshold:
    - I generated **even more refined sub-questions**.
    - Retrieved their chunks and generated partial answers.
  - Only when the scores were good enough, the model generated the final answer through the retrieved chunks and answers generated for sub chunks.
- **No web search was used** — everything was performed locally.
- Full **reverse mapping** of chunk indices back to original documents and summaries was implemented.

---

## 🧠 Language Model Used

- All reasoning steps (HyDE, RAPTOR summaries, query decomposition, sub-question answering) were powered by the **Phi 2.8B** model running locally.

---

## 🗂️ Dataset

I created a domain-diverse dataset containing documents from **health**, **science**, and **Indian law**. All documents were collected from trusted open datasets and preprocessed into a standard format.

```python
# Health QA
lavita/MedQuAD → 500 Q&A pairs

# Science QA
sciq → 500 science facts (question + support)


Final Learnings
All three RAG pipelines performed similarly on this relatively simple, clean, and small dataset.

The limited document count and low query complexity allowed Basic RAG to work effectively in most cases.

However, as the query complexity increases or document structure becomes denser or noisier, basic dense retrieval alone begins to fail.

In such situations, advanced RAG methods like HyDE (for semantic abstraction) or CRAG (for decomposition and correction) become essential.

The nature of the task and documents should guide the RAG pipeline — not all tasks need complexity, but complex tasks absolutely demand it.
