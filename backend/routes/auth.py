from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from hashlib import sha256
from pinecone import Pinecone
import os

# Setup Pinecone
pc = Pinecone(api_key=os.getenv("pcsk_4SGf9f_NTpLgzUwjTKe9FjBhMguHC9uATQox8LLqyv3U44KuVJPCWnMnA6jbWwBC8Q62tU"))
index = pc.Index(os.getenv("studiva-users"))  # contoh: "users-index"

router = APIRouter()

# === User schema ===
class User(BaseModel):
    username: str
    email: str
    password: str

def hash_password(password: str) -> str:
    return sha256(password.encode()).hexdigest()

@router.post("/signup")
def signup(user: User):
    user_id = user.email
    hashed = hash_password(user.password)

    # Check if user already exists
    existing = index.fetch(ids=[user_id])
    if existing and user_id in existing.get("vectors", {}):
        raise HTTPException(status_code=409, detail="Email sudah terdaftar.")

    index.upsert([
        {
            "id": user_id,
            "values": [0.0] * 10,  # placeholder embedding
            "metadata": {
                "username": user.username,
                "email": user.email,
                "password": hashed
            }
        }
    ])
    return {"message": "Signup berhasil"}

@router.post("/login")
def login(user: User):
    user_id = user.email
    hashed = hash_password(user.password)
    result = index.fetch(ids=[user_id])

    data = result.get("vectors", {}).get(user_id)
    if not data:
        raise HTTPException(status_code=404, detail="Akun tidak ditemukan.")

    if data["metadata"]["password"] != hashed:
        raise HTTPException(status_code=401, detail="Password salah.")

    return {"message": "Login berhasil", "username": data["metadata"]["username"]}
