import os
import chromadb
from chromadb.config import Settings
from src.utils import extract_text_from_pdf, transcribe_audio, chunk_text, embed_text

def create_vector_db(pdf_folder, audio_folder):
    """Process all PDFs and audio files, store embeddings in ChromaDB."""
    client = chromadb.Client(Settings(persist_directory="./outputs/results"))
    collection = client.get_or_create_collection("knowledge_base")

    # Process PDFs
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            text = extract_text_from_pdf(os.path.join(pdf_folder, file))
            chunks = chunk_text(text)
            embeddings = embed_text(chunks)
            collection.add(
                ids=[f"pdf_{i}" for i in range(len(chunks))],
                embeddings=embeddings.tolist(),
                documents=chunks
            )

    # Process Audio Files
    for file in os.listdir(audio_folder):
        if file.endswith((".mp3", ".wav", ".m4a")):
            text = transcribe_audio(os.path.join(audio_folder, file))
            chunks = chunk_text(text)
            embeddings = embed_text(chunks)
            collection.add(
                ids=[f"audio_{i}" for i in range(len(chunks))],
                embeddings=embeddings.tolist(),
                documents=chunks
            )

    return collection

def retrieve_context(collection, query):
    """Retrieve top matching text chunks for a query."""
    query_embedding = embed_text([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    context = " ".join(results["documents"][0])
    return context