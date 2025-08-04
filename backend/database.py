# database.py

import hashlib
import uuid
from datetime import datetime
from typing import Optional, List, Dict
from pinecone import Pinecone, ServerlessSpec

# Inisialisasi koneksi Pinecone
pc = Pinecone(api_key="pcsk_4SGf9f_NTpLgzUwjTKe9FjBhMguHC9uATQox8LLqyv3U44KuVJPCWnMnA6jbWwBC8Q62tU")
index_name = "studiva-users"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=128,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# ===== Helper: Hash password =====
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# ===== Signup user =====
def signup_user(username: str, email: str, password: str) -> bool:
    if get_user_by_email(email):
        return False  # Email sudah terdaftar

    user_id = str(uuid.uuid4())
    user_data = {
        "user_id": user_id,
        "username": username,
        "email": email,
        "password_hash": hash_password(password),
        "created_at": datetime.utcnow().isoformat(),
        "type": "user"
    }

    index.upsert(vectors=[
        {"id": f"user:{user_id}", "values": [0.0] * 128, "metadata": user_data}
    ])
    return True

# ===== Login user =====
def login_user(email: str, password: str) -> Optional[Dict]:
    user = get_user_by_email(email)
    if not user:
        return None

    if user["password_hash"] != hash_password(password):
        return None

    return user

# ===== Get user by email =====
def get_user_by_email(email: str) -> Optional[Dict]:
    query = index.query(
        vector=[0.0] * 128,
        filter={"email": {"$eq": email}, "type": {"$eq": "user"}},
        top_k=1,
        include_metadata=True
    )

    if query["matches"]:
        return query["matches"][0]["metadata"]
    return None

# ===== Simpan chat =====
def save_chat(user_id: str, question: str, answer: str) -> None:
    chat_id = str(uuid.uuid4())
    chat_data = {
        "chat_id": chat_id,
        "user_id": user_id,
        "question": question,
        "answer": answer,
        "timestamp": datetime.utcnow().isoformat(),
        "type": "chat"
    }

    index.upsert(vectors=[
        {"id": f"chat:{chat_id}", "values": [0.0] * 128, "metadata": chat_data}
    ])

# ===== Ambil riwayat chat user =====
def get_chat_history(user_id: str, limit: int = 20) -> List[Dict]:
    query = index.query(
        vector=[0.0] * 128,
        filter={"user_id": {"$eq": user_id}, "type": {"$eq": "chat"}},
        top_k=limit,
        include_metadata=True
    )

    return [match["metadata"] for match in query["matches"]]
