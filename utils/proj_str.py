import os

# List of folders to create
folders = [
    "Rag-Phi-Eval/data",
    "Rag-Phi-Eval/embeddings",
    "Rag-Phi-Eval/chains",
    "Rag-Phi-Eval/retrievers",
    "Rag-Phi-Eval/chunkers",
    "Rag-Phi-Eval/utils",
    "Rag-Phi-Eval/ui/components"
]

# List of empty files to create
files = [
    "Rag-Phi-Eval/main.py",
    "Rag-Phi-Eval/requirements.txt",

    # Chains
    "Rag-Phi-Eval/chains/base_rag_chain.py",
    "Rag-Phi-Eval/chains/hyde_rag_chain.py",
    "Rag-Phi-Eval/chains/crag_chain.py",
    "Rag-Phi-Eval/chains/query_decomp_chain.py",

    # Retrievers
    "Rag-Phi-Eval/retrievers/rank_retriever.py",
    "Rag-Phi-Eval/retrievers/crag_retriever.py",
    "Rag-Phi-Eval/retrievers/hyde_retriever.py",

    # Chunkers
    "Rag-Phi-Eval/chunkers/semantic_chunker.py",
    "Rag-Phi-Eval/chunkers/raptor_chunker.py",
    "Rag-Phi-Eval/chunkers/dense_chunker.py",

    # Utils
    "Rag-Phi-Eval/utils/loader.py",
    "Rag-Phi-Eval/utils/evaluator.py",
    "Rag-Phi-Eval/utils/config.py",

    # UI
    "Rag-Phi-Eval/ui/app.py",
    "Rag-Phi-Eval/ui/memory.py"
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create empty files
for file_path in files:
    with open(file_path, "w") as f:
        pass  # Creates an empty file

print("✅ Full 'Rag-Phi-Eval' folder structure with empty files created successfully.")
