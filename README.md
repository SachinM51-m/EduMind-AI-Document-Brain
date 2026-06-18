# 🧠 EduMind AI - Intelligent Study Brain

EduMind AI is a premium, high-performance local study assistant and document Q&A brain. It allows you to upload textbooks, notes, and handouts (PDF, DOCX, DOC) to semantic search indices and query them using state-of-the-art AI.

The application supports **100% Offline Local Execution** via Ollama and a seamless **Hybrid Cloud Fallback** mode for serverless deployment on Streamlit Cloud.

---

## ✨ Key Features

1. **💬 Q&A Document Brain:** Semantic search-driven chat. Ask questions of your files, retrieve relevant snippets, and view direct source citations.
2. **📝 Auto-Notes Generator:** Analyzes text context to produce structured study guides, summaries, and key concept outlines.
3. **✍️ Interactive Quiz Generator:** Creates custom multiple-choice practices with automated scoring and comprehensive explanations.
4. **🌐 Hybrid Local-Cloud Engine:** Automatically connects to your local Ollama instance. If offline, it dynamically routes traffic to Groq or Hugging Face serverless APIs.
5. **🎨 Premium Dark Tech UI:** Beautiful user interface featuring Outfits typography, glassmorphism containers, and reactive status indicators.

---

## 🛠️ Getting Started

### 1. Installation

Clone the repository and install the required Python packages:
```bash
pip install -r requirements.txt
```

### 2. Setup Local LLM (Offline Mode)

1. Download and install **[Ollama](https://ollama.com/)**.
2. Start the Ollama application or run `ollama serve` in your terminal.
3. Pull the required Llama model:
   ```bash
   ollama pull llama3.2:1b
   ```

### 3. Setup Cloud Fallback (Optional)

If your local Ollama server is stopped, or when deploying the app online, you can use a cloud fallback. The app will look for these keys in your system environment variables or Streamlit secrets:
* **Option A (Groq API):** Set `GROQ_API_KEY` (runs `llama-3.2-1b-preview` model).
* **Option B (Hugging Face API):** Set `HF_TOKEN` (runs `meta-llama/Llama-3.2-1B-Instruct` model).

For Streamlit Cloud, add the secrets in your dashboard configuration:
```toml
GROQ_API_KEY = "your-groq-api-key"
# OR
HF_TOKEN = "your-huggingface-token"
```

---

## 🚀 Running the App

To launch the Streamlit dashboard, run the following command in your terminal:
```bash
streamlit run app/main.py
```

Open `http://localhost:8501` in your browser.

---

## 📂 Project Structure

```
├── app/
│   └── main.py              # Premium Streamlit UI and dashboard logic
├── src/
│   ├── document_processor.py # Text extraction (PDF/DOCX/DOC) and chunking
│   ├── llm_handler.py        # HybridLLMHandler client (Ollama/Groq/HF)
│   └── vector_store.py       # Local FAISS embedding indexer
├── requirements.txt         # Dependencies
└── README.md                # This instruction manual
```
