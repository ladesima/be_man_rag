import os
import time
import uuid
import bcrypt
import json
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable not set.")
pc = Pinecone(api_key=PINECONE_API_KEY)

DUMMY_VECTOR_DIM = 384
DUMMY_VECTOR = [0.01] + [0.0] * (DUMMY_VECTOR_DIM - 1)
USER_INDEX_NAME = os.getenv("PINECONE_USER_INDEX", "studiva-users")
CHAT_INDEX_NAME = os.getenv("PINECONE_CHAT_INDEX", "studiva-history")

def create_pinecone_index_if_not_exists(name, dimension):
    if name not in pc.list_indexes().names():
        print(f"Index '{name}' tidak ditemukan, membuat index baru...")
        pc.create_index(
            name=name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Index '{name}' berhasil dibuat.")

try:
    create_pinecone_index_if_not_exists(USER_INDEX_NAME, DUMMY_VECTOR_DIM)
    create_pinecone_index_if_not_exists(CHAT_INDEX_NAME, DUMMY_VECTOR_DIM)
    user_index = pc.Index(USER_INDEX_NAME)
    chat_index = pc.Index(CHAT_INDEX_NAME)
except Exception as e:
    print(f"Gagal inisialisasi Pinecone: {e}")
    raise e

def init_db():
    print("Database dan index Pinecone sudah diinisialisasi.")
    pass

# --- PERBAIKAN UTAMA ADA DI SINI ---
def get_user_details(user_id: str):
    """Mengambil metadata lengkap seorang pengguna berdasarkan user_id."""
    try:
        fetch_response = user_index.fetch(ids=[user_id])
        # Gunakan atribut .vectors untuk mengakses data, bukan .get()
        vector_data = fetch_response.vectors.get(user_id)
        if vector_data:
            return vector_data.metadata
        return None
    except Exception as e:
        print(f"Gagal mengambil detail user {user_id}: {e}")
        return None
# --- AKHIR DARI PERBAIKAN ---

def create_user(name, nisn, grade, password):
    user_id = str(uuid.uuid4())
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    response = user_index.query(vector=DUMMY_VECTOR, filter={"name": {"$eq": name}}, top_k=1)
    if response.get("matches", []):
        return None 

    metadata = {
        "name": name,
        "nisn": nisn,
        "grade": str(grade),
        "password": hashed_pw
    }
    user_index.upsert(vectors=[(user_id, DUMMY_VECTOR, metadata)])
    return user_id

def login_user(name, password):
    response = user_index.query(
        vector=DUMMY_VECTOR,
        filter={"name": {"$eq": name}},
        top_k=1,
        include_metadata=True
    )
    if response.get("matches", []):
        match = response["matches"][0]
        stored_hash = match["metadata"]["password"]
        if bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
            return match["id"]
    return None

def save_chat_turn(user_id: str, session_id: str, turn: dict, session_name: str = None):
    message_id = f"{session_id}-{int(time.time() * 1000)}"
    metadata = {
        "user_id": user_id,
        "session_id": session_id,
        "sender": turn.get("sender"),
        "text": turn.get("text"),
        "timestamp": int(time.time())
    }
    if session_name:
        metadata["session_name"] = session_name
    if "sources" in turn and turn["sources"]:
        metadata["sources_json"] = json.dumps(turn["sources"])
    chat_index.upsert(vectors=[(message_id, DUMMY_VECTOR, metadata)])
    print(f"Giliran chat untuk sesi {session_id} berhasil disimpan.")

def get_messages_for_session(session_id: str) -> list:
    if not session_id: return []
    response = chat_index.query(
        vector=DUMMY_VECTOR,
        filter={"session_id": {"$eq": session_id}},
        top_k=1000,
        include_metadata=True
    )
    matches = response.get("matches", [])
    if not matches: return []
    sorted_matches = sorted(matches, key=lambda m: m["metadata"].get("timestamp", 0))
    messages = []
    for m in sorted_matches:
        msg_data = {
            "id": m["id"],
            "sender": m["metadata"].get("sender"),
            "text": m["metadata"].get("text"),
        }
        if "sources_json" in m["metadata"]:
            msg_data["sources"] = json.loads(m["metadata"]["sources_json"])
        messages.append(msg_data)
    return messages

def get_sessions_for_user(user_id: str) -> list:
    response = chat_index.query(
        vector=DUMMY_VECTOR,
        filter={"user_id": {"$eq": user_id}},
        top_k=10000,
        include_metadata=True
    )
    matches = response.get("matches", [])
    if not matches: return []
    sessions = {}
    for m in matches:
        metadata = m["metadata"]
        session_id = metadata.get("session_id")
        if not session_id: continue
        timestamp = metadata.get("timestamp", 0)
        if session_id not in sessions:
            sessions[session_id] = {
                "id": session_id,
                "name": metadata.get("session_name", "Sesi Tanpa Nama"),
                "timestamp": timestamp
            }
        else:
            if "session_name" in metadata:
                sessions[session_id]["name"] = metadata["session_name"]
            if timestamp > sessions[session_id]["timestamp"]:
                sessions[session_id]["timestamp"] = timestamp
    return sorted(list(sessions.values()), key=lambda s: s["timestamp"], reverse=True)

def delete_session_for_user(user_id: str, session_id: str):
    try:
        chat_index.delete(filter={
            "user_id": {"$eq": user_id},
            "session_id": {"$eq": session_id}
        })
        print(f"Berhasil menghapus sesi {session_id} untuk user {user_id}.")
        return True
    except Exception as e:
        print(f"Gagal menghapus sesi {session_id}: {e}")
        return False