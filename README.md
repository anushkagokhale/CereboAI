# 📄 PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions **strictly from uploaded PDF documents**. Built with LangChain, FAISS, HuggingFace Sentence Transformers, Google Gemini, and Streamlit.

---

## 🗂️ Project Structure

```
rag_chatbot/
├── app.py            # Streamlit UI (upload, chat interface, session history)
├── ingest.py         # PDF loading, chunking, embedding, FAISS vector store
├── chatbot.py        # RetrievalQA chain with strict context-only prompt
├── requirements.txt  # All Python dependencies
├── README.md         # This file
└── data/             # (optional) Put your sample PDFs here
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get a Google Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click **Create API Key**
3. Copy the key — you'll paste it into the app's sidebar

> **Alternatively**, create a `.env` file in the project root:
> ```
> GOOGLE_API_KEY=your_key_here
> ```
> The app will automatically read it on startup.

---

## ▶️ Running the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 🛠️ How to Use

1. **Enter your Gemini API key** in the sidebar (or set it via `.env`)
2. **Upload one or more PDF files** using the file uploader
3. Click **⚡ Process Documents** — this loads, chunks, embeds, and indexes your PDFs
4. **Ask questions** in the chat box — answers come only from your documents

---

## 🧠 Technical Design

| Component | Choice | Reason |
|-----------|--------|--------|
| PDF Loader | `PyPDFLoader` | Fast, page-aware, preserves metadata |
| Chunking | `RecursiveCharacterTextSplitter` (1000 / 200) | Balances context richness and retrieval precision |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace) | Lightweight, fast, no API cost |
| Vector DB | `FAISS` (CPU) | Local, zero-latency, no external service needed |
| LLM | `gemini-1.5-flash` | Fast, generous free tier, strong instruction following |
| UI | Streamlit | Minimal setup, built-in session state |

### Chunking strategy (interview discussion point)

- **chunk_size = 1000** — large enough to preserve complete sentences and surrounding context, ideal for factual Q&A over research papers and manuals.
- **chunk_overlap = 200** — 20% overlap prevents answers from being severed at chunk boundaries.
- Smaller chunks (≤ 500) increase retrieval precision but lose context; larger chunks (≥ 2000) improve context but reduce retrieval precision and increase token usage.

### Strict context enforcement

The LLM prompt explicitly instructs the model to answer **only** from the retrieved chunks and to say *"I don't have enough information in the uploaded documents to answer this question."* if the answer is absent — preventing hallucination from prior training knowledge.

---

## 📦 Sample PDF

Place any PDF (research paper, manual, handbook) inside the `data/` folder and upload it via the Streamlit sidebar.

---

## 🐛 Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside your virtual env |
| Gemini API error 403 | Check your API key; ensure the Generative Language API is enabled |
| FAISS `allow_dangerous_deserialization` warning | Expected — the index is local and safe |
| Slow first run | Sentence-transformer model downloads on first use (~90 MB) |