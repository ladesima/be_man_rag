# src/backend/chat.py
from fastapi import APIRouter, Request
from pydantic import BaseModel
from pinecone import Pinecone
import os
import uuid
import time

router = APIRouter()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

class ChatRequest(BaseModel):
    username: str
    question: str
    answer: str

@router.post("/chat-store")
def store_chat(data: ChatRequest):
    vector_id = f"{data.username}_{uuid.uuid4().hex}"
    metadata = {
        "username": data.username,
        "question": data.question,
        "answer": data.answer,
        "timestamp": time.time()
    }
    index.upsert(vectors=[{
        "id": vector_id,
        "values": [0.0] * 1536,  # dummy vector (tidak digunakan)
        "metadata": metadata
    }])
    return {"status": "stored", "id": vector_id}


@router.get("/chat-history/{username}")
def get_chat_history(username: str):
    res = index.query(
        vector=[0.0] * 1536,
        top_k=100,
        include_metadata=True,
        filter={"username": {"$eq": username}}
    )
    history = sorted(
        [match["metadata"] for match in res["matches"]],
        key=lambda x: x["timestamp"]
    )
    return {"history": history}
