import os
import pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import time

load_dotenv()

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)

history_index = pinecone.Index(os.getenv("PINECONE_HISTORY_INDEX"))
model = SentenceTransformer("all-MiniLM-L6-v2")

def save_chat(username: str, question: str, answer: str):
    timestamp = int(time.time())
    text = question + " " + answer
    vector = model.encode(text).tolist()

    history_index.upsert([
        (
            f"{username}-{timestamp}",
            vector,
            {
                "username": username,
                "question": question,
                "answer": answer,
                "timestamp": timestamp
            }
        )
    ])
    print(f"💾 Riwayat chat user {username} tersimpan.")

def get_user_history(username: str):
    # (Optional): hanya mock pengambilan semua data user karena Pinecone bukan untuk query kompleks
    print("Note: Pinecone tidak ideal untuk full retrieval tanpa filter eksternal.")
