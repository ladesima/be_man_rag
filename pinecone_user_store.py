import os
import pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Init Pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)

user_index = pinecone.Index(os.getenv("PINECONE_USERS_INDEX"))

model = SentenceTransformer("all-MiniLM-L6-v2")  # ringan & cepat

def register_user(username: str, password: str):
    user_vector = model.encode(username + password).tolist()
    user_index.upsert([
        (f"user-{username}", user_vector, {"username": username, "password": password})
    ])
    print(f"✅ User {username} berhasil disimpan.")

def login_user(username: str, password: str) -> bool:
    user_vector = model.encode(username + password).tolist()
    result = user_index.query(vector=user_vector, top_k=1, include_metadata=True)
    
    if result.matches and result.matches[0].metadata['username'] == username:
        return result.matches[0].metadata['password'] == password
    return False
