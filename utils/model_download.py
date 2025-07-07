import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

MODEL_REPO = "andrijdavid/TinyLlama-1.1B-Chat-v1.0-GGUF"
MODEL_FILENAME = "TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf"
MODEL_DIR = "models/tinyllama"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH):
        print("✅ Model already exists.")
        return

    print("⬇️ Downloading TinyLlama GGUF model via Hugging Face...")
    hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME, local_dir=MODEL_DIR, local_dir_use_symlinks=False)
    print("✅ Model downloaded successfully.")

def test_model():
    print("🚀 Loading model...")
    llm = Llama(model_path=MODEL_PATH, n_ctx=512, n_threads=4, verbose=False)

    prompt = "Explain gravity"
    output = llm(prompt, max_tokens=80, stop=["</s>"])
    print("\n🧠 TinyLlama Response:\n", output["choices"][0]["text"].strip())

    llm.__del__()  # prevent cleanup warning

if __name__ == "__main__":
    download_model()
    test_model()
