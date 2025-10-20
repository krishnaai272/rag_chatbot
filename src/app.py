import os
from src.pipeline import create_vector_db, retrieve_context
from src.model import generate_answer
import datetime

PDF_PATH = "media/pdfs"
AUDIO_PATH = "media/videos"
LOG_PATH = "outputs/logs/chat_log.txt"

# Build database if not exists
collection = create_vector_db(PDF_PATH, AUDIO_PATH)

print("RAG Chatbot ready. Ask a question:")

while True:
    query = input("\n❓ You: ")
    if query.lower() in ["exit", "quit"]:
        print("Chat ended.")
        break

    context = retrieve_context(collection, query)
    answer = generate_answer(context, query)

    # Display and log
    print(f"🤖 Bot: {answer}\n")

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now()}]\nQ: {query}\nA: {answer}\n\n")