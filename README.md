# RAG Chatbot (PDF + Audio Knowledge Base)

This project builds a **Retrieval-Augmented Generation (RAG)** chatbot that can:
- Ingest and process knowledge from **PDF files and audio lectures**.
- Transcribe audio using **Whisper (Hugging Face)**.
- Store embeddings in **ChromaDB**.
- Answer user questions using **Groq LLM** and retrieved context.

---

## 🧩 Project Workflow
1. Extract text from PDFs and audio.
2. Chunk and embed the text.
3. Store embeddings in a vector DB.
4. Query relevant chunks for user questions.
5. Generate answers using Groq.

---

## 🚀 How to Run
```bash
pip install -r requirements.txt
python src/app.py 
Run - python -m src.app