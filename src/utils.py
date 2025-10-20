import os
from PyPDF2 import PdfReader
import whisper
from sentence_transformers import SentenceTransformer

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def transcribe_audio(audio_path):
    """Transcribe audio file using Whisper (Hugging Face local model)."""
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

def chunk_text(text, chunk_size=500):
    """Split text into smaller meaningful chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

def embed_text(chunks):
    """Generate embeddings for text chunks."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(chunks)