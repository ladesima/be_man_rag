from pinecone_config import pinecone_index
import uuid

def store_chat_history(username: str, question: str, answer: str):
    vector_id = f"{username}-{uuid.uuid4()}"
    metadata = {
        "username": username,
        "question": question,
        "answer": answer
    }
    # Dummy vector, replace with real embedding
    pinecone_index.upsert([(vector_id, [0.0]*1536, metadata)])

def fetch_chat_history(username: str):
    # Dummy implementation - adjust if storing vectors with filters
    return pinecone_index.query(
        vector=[0.0]*1536, top_k=100, include_metadata=True, filter={"username": {"$eq": username}}
    )
