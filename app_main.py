"""
Backend Utama Studiva.AI (Versi Final dengan Database per Kelas & Pencarian Hibrida)
======================================================================================
- Menginisialisasi database ChromaDB terpisah untuk setiap kelas (10, 11, 12).
- Menggunakan database yang sesuai berdasarkan kelas pengguna saat chat.
- Mengimplementasikan Pencarian Hibrida (Vektor + Kata Kunci BM25) dengan RRF.
- Menyempurnakan semua fitur yang ada (sesi chat, kuis, kalkulator, otentikasi).
"""
import os
import re
import logging
import uuid
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import chromadb
import numexpr as ne
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from config import config
from database.pinecone_db import (
    init_db,
    create_user,
    login_user,
    save_chat_turn,
    get_sessions_for_user,
    get_messages_for_session,
    delete_session_for_user,
    get_user_details
)
from groq import AsyncGroq, APIError

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Inisialisasi Global ---
embedding_model = None
reranker_model = None
groq_client = None
# Gunakan dictionary untuk menyimpan resources per kelas
chroma_collections = {}
bm25_indexes = {}
bm25_documents_map = {}

def initialize_system():
    global embedding_model, reranker_model, groq_client, chroma_collections, bm25_indexes, bm25_documents_map
    logger.info("=== MEMULAI INISIALISASI SISTEM AI ===")
    try:
        if config.GROQ_API_KEY:
            groq_client = AsyncGroq(api_key=config.GROQ_API_KEY)
            logger.info("   [OK] Groq client berhasil diinisialisasi.")
        else:
            logger.warning("   [WARNING] GROQ_API_KEY tidak ditemukan.")

        logger.info(f"1. Memuat model embedding: {config.EMBEDDING_MODEL_NAME}")
        embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, cache_folder=config.MODEL_CACHE_DIR, device=config.EMBEDDING_DEVICE)
        logger.info("   [OK] Model embedding berhasil dimuat.")

        logger.info(f"2. Memuat model Re-ranker: {config.RERANKER_MODEL_NAME}")
        reranker_model = CrossEncoder(config.RERANKER_MODEL_NAME)
        logger.info("   [OK] Model Re-ranker berhasil dimuat.")

        # Muat database untuk setiap kelas
        for grade in ["10", "11", "12"]:
            db_path = f"./data/chroma_{grade}"
            collection_name = f"studiva_grade_{grade}"
            if not os.path.exists(db_path):
                logger.warning(f"Database untuk Kelas {grade} di '{db_path}' tidak ditemukan. Fitur RAG untuk kelas ini tidak akan berfungsi.")
                continue
            
            logger.info(f"Menghubungkan ke ChromaDB & membangun BM25 untuk Kelas {grade}...")
            chroma_client = chromadb.PersistentClient(path=db_path)
            chroma_collections[grade] = chroma_client.get_collection(name=collection_name)
            
            all_docs = chroma_collections[grade].get(include=["documents", "metadatas"])
            bm25_documents_map[grade] = [{"content": doc, "metadata": meta} for doc, meta in zip(all_docs["documents"], all_docs["metadatas"])]
            
            if bm25_documents_map.get(grade):
                tokenized_corpus = [doc["content"].split(" ") for doc in bm25_documents_map[grade]]
                bm25_indexes[grade] = BM25Okapi(tokenized_corpus)
                logger.info(f"   [OK] Indeks BM25 untuk Kelas {grade} berhasil dibangun.")
            else:
                logger.warning(f"   [WARNING] Tidak ada dokumen untuk membangun BM25 untuk Kelas {grade}.")

        logger.info("4. Menginisialisasi basis data pengguna...")
        init_db()
        logger.info("   [OK] Basis data pengguna berhasil diinisialisasi.")
    except Exception as e:
        logger.error(f"   [FATAL] Gagal saat inisialisasi sistem: {e}", exc_info=True)
        raise
    
    logger.info("=== INISIALISASI SELESAI ===")

app = FastAPI(title="Studiva API", version=config.VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
initialize_system()

# --- Pydantic Models ---
class ChatTurn(BaseModel): role: str; content: str
class SignupRequest(BaseModel): name: str; nisn: str; grade: str; password: str
class QuizRequest(BaseModel): topic: str; request_detail: str; user_id: str
class MathRequest(BaseModel): problem: str
class NewChatRequest(BaseModel): user_id: str; question: str
class ContinueChatRequest(BaseModel): user_id: str; question: str

# --- Helper & Core RAG Functions ---
def sigmoid(x): return 1 / (1 + np.exp(-x))

async def call_groq_api(prompt_context: str, question: str, system_prompt: str, history: List[ChatTurn] = []) -> str:
    if not groq_client: raise ValueError("Groq client tidak diinisialisasi.")
    messages = [{"role": "system", "content": system_prompt}] + [{"role": turn.role, "content": turn.content} for turn in history]
    user_prompt = f"KONTEKS:\n---\n{prompt_context}\n---\n\nPERMINTAAN ANDA:\n{question}"
    messages.append({"role": "user", "content": user_prompt})
    try:
        chat_completion = await groq_client.chat.completions.create(messages=messages, model=config.GENERATIVE_MODEL_NAME)
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error saat memanggil Groq API: {e}", exc_info=True)
        return "Maaf, terjadi kesalahan pada sistem AI."

async def generate_hypothetical_answer(question: str) -> str:
    system_prompt = "Anda adalah asisten yang sangat membantu. Tuliskan satu paragraf jawaban yang ideal dan faktual untuk pertanyaan berikut. Jawaban ini akan digunakan untuk mencari dokumen yang relevan."
    return await call_groq_api("", question, system_prompt)

def reciprocal_rank_fusion(results: List[List[dict]], k: int = 60) -> List[dict]:
    fused_scores = {}
    def get_doc_id(doc):
        path = doc['metadata'].get('source_path', '')
        pages_str = doc['metadata'].get('pages', '')
        return f"{path}_{pages_str}"
    for doc_list in results:
        for rank, doc in enumerate(doc_list):
            doc_id = get_doc_id(doc)
            if doc_id not in fused_scores: fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1 / (k + rank)
    unique_docs = {get_doc_id(doc): doc for doc_list in results for doc in doc_list}
    sorted_doc_ids = sorted(fused_scores.keys(), key=lambda id: fused_scores[id], reverse=True)
    return [unique_docs[doc_id] for doc_id in sorted_doc_ids]

async def retrieve_and_rerank(query: str, grade: str) -> List:
    if grade not in chroma_collections:
        logger.warning(f"Pencarian dibatalkan: tidak ada database untuk kelas {grade}.")
        return []

    collection = chroma_collections[grade]
    bm25_idx = bm25_indexes.get(grade)
    bm25_docs = bm25_documents_map.get(grade, [])
    
    logger.info(f"Menjalankan pencarian hibrida untuk: '{query}' pada database kelas {grade}...")
    
    hypothetical_document = await generate_hypothetical_answer(query)
    embedding = embedding_model.encode(hypothetical_document).tolist()
    vector_results_raw = collection.query(query_embeddings=[embedding], n_results=config.TOP_K_RETRIEVAL, include=['metadatas', 'documents'])
    vector_results = [{"content": doc, "metadata": meta} for doc, meta in zip(vector_results_raw["documents"][0], vector_results_raw["metadatas"][0])]

    bm25_results = []
    if bm25_idx:
        tokenized_query = query.lower().split(" ")
        bm25_scores = bm25_idx.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:config.TOP_K_RETRIEVAL]
        bm25_results = [bm25_docs[i] for i in top_bm25_indices if bm25_scores[i] > 0]
    
    fused_documents = reciprocal_rank_fusion([vector_results, bm25_results])
    
    if not fused_documents: return []

    docs_to_rerank = [doc["content"] for doc in fused_documents]
    metadatas_to_rerank = [doc["metadata"] for doc in fused_documents]
    pairs = [[query, doc] for doc in docs_to_rerank]
    scores = sigmoid(reranker_model.predict(pairs))
    
    return sorted(zip(scores, docs_to_rerank, metadatas_to_rerank), key=lambda x: x[0], reverse=True)

async def generate_real_answer(question: str, history: List[ChatTurn], grade: str) -> Dict[str, Any]:
    logger.info(f"Menjalankan pipeline RAG untuk: '{question}'")
    scored_docs = await retrieve_and_rerank(question, grade=grade)
    
    context_docs, context, retrieved_contents = [], "", []
    final_sources = [] # Inisialisasi daftar sumber di awal

    if scored_docs:
        RELEVANCE_THRESHOLD = 0.3
        relevant_docs = [doc for doc in scored_docs if doc[0] > RELEVANCE_THRESHOLD]
        if relevant_docs:
            context_docs = relevant_docs[:config.TOP_K_RERANKED]
            context = "\n\n---\n\n".join([doc[1] for doc in context_docs])
            retrieved_contents = [doc[1] for doc in context_docs]

    # --- PERBAIKAN UTAMA: Logika Pemilihan Prompt dan Sumber ---
    if context:
        # Mode 1: Konteks Ditemukan (Jawaban berbasis RAG)
        logger.info("Konteks ditemukan. Menjawab berdasarkan dokumen.")
        system_prompt = (
            "Anda adalah asisten AI bernama Studiva. Anda cerdas, membantu, dan dapat mengingat percakapan sebelumnya.\n"
            "JAWABLAH PERTANYAAN PENGGUNA HANYA BERDASARKAN KONTEKS DOKUMEN YANG DISEDIAKAN.\n"
            "Di akhir jawaban Anda, WAJIB sebutkan sumber yang Anda gunakan dengan format '[SUMBER 1]'."
        )
        
        # Ambil HANYA DOKUMEN PERTAMA (skor tertinggi) untuk dijadikan sumber
        top_doc_score, _, top_doc_meta = context_docs[0]
        
        if top_doc_meta and (path_str := top_doc_meta.get('source_path')):
            pages_str = top_doc_meta.get('pages', '') 
            relevant_pages = set()
            if pages_str:
                try:
                    page_nums = [int(p) for p in pages_str.split(',')]
                    for page_num in page_nums:
                        relevant_pages.add(page_num + 1)
                except (ValueError, TypeError):
                    logger.warning(f"Gagal mengolah nomor halaman dari string: '{pages_str}'")
            
            final_sources = [{
                "path": path_str,
                "score": float(top_doc_score),
                "relevant_pages": sorted(list(relevant_pages))
            }]
    else:
        # Mode 2: Konteks Tidak Ditemukan (Jawaban berbasis Pengetahuan Umum)
        logger.info("Tidak ada konteks relevan. Menjawab dari pengetahuan umum.")
        system_prompt = (
            "Anda adalah asisten AI bernama Studiva. Anda cerdas, membantu, dan dapat mengingat percakapan sebelumnya.\n"
            "JAWABLAH PERTANYAAN PENGGUNA BERDASARKAN PENGETAHUAN UMUM ANDA.\n"
            "PENTING: Di akhir jawaban Anda, tambahkan baris baru yang berisi disclaimer berikut: "
            "'(Jawaban ini dihasilkan berdasarkan pengetahuan umum AI dan bukan dari buku pelajaran Anda.)'"
        )
        final_sources = [] # Pastikan tidak ada sumber yang dikirim
    # --- AKHIR DARI PERBAIKAN ---
    
    final_answer = await call_groq_api(context, question, system_prompt, history)
    
    return {
        "answer": final_answer, 
        "sources": final_sources,
        "retrieved_contexts": retrieved_contents
    }
    
# --- ENDPOINTS API ---
@app.post("/signup", tags=["Authentication"])
async def signup(request: SignupRequest):
    user_id = create_user(request.name, request.nisn, request.grade, request.password)
    if not user_id: raise HTTPException(status_code=409, detail=f"Nama '{request.name}' sudah terdaftar.")
    return {"message": "Signup berhasil", "user_id": user_id}

@app.post("/login", tags=["Authentication"])
async def login(name: str = Form(...), password: str = Form(...)):
    user_id = login_user(name, password)
    if not user_id: raise HTTPException(status_code=401, detail="Nama atau password salah.")
    return {"message": "Login sukses", "user_id": user_id}

@app.get("/pdf/{file_path:path}", tags=["Utilities"])
async def serve_pdf(file_path: str):
    full_path = Path(config.PDF_DIR) / file_path
    if not full_path.is_file(): raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(full_path, media_type='application/pdf', headers={"Content-Disposition": "inline"})

@app.post("/calculate", tags=["Utilities"])
async def calculate(request: MathRequest):
    problem = request.problem.strip()
    safe_problem_expr = problem.replace('^', '**')
    try:
        result = ne.evaluate(safe_problem_expr).item()
        return {"problem": problem, "expression": safe_problem_expr, "result": result}
    except Exception:
        system_prompt = "Anda adalah ahli penerjemah matematika..."
        try:
            translated_expression = await call_groq_api("", request.problem, system_prompt)
            safe_translated_expr = translated_expression.replace('^', '**')
            result = ne.evaluate(safe_translated_expr).item()
            return {"problem": request.problem, "expression": safe_translated_expr, "result": result}
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Tidak dapat menyelesaikan soal: {e}")

@app.post("/generate-quiz", tags=["Core RAG"])
async def generate_quiz(request: QuizRequest):
    try:
        user_details = get_user_details(request.user_id)
        if not user_details: raise HTTPException(status_code=404, detail="User tidak ditemukan.")
        grade = user_details.get("grade")

        logger.info(f"Menerima permintaan kuis untuk topik: '{request.topic}' untuk kelas {grade}")
        top_docs_with_metadata = await retrieve_and_rerank(request.topic, grade=grade)
        context = ""
        if top_docs_with_metadata:
            context_docs = top_docs_with_metadata[:config.TOP_K_RERANKED]
            context = "\n\n---\n\n".join([doc[1] for doc in context_docs])
            system_prompt_quiz = "Anda adalah guru. Berdasarkan KONTEKS DOKUMEN..."
        else:
            system_prompt_quiz = "Anda adalah seorang guru yang berpengetahuan luas..."
        quiz_result = await call_groq_api(context, request.request_detail, system_prompt_quiz)
        return {"quiz": quiz_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Gagal membuat soal latihan.")

@app.get("/sessions/{user_id}", tags=["Chat Session"])
async def get_user_sessions(user_id: str):
    try:
        return {"sessions": get_sessions_for_user(user_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Gagal mengambil riwayat sesi.")

@app.get("/chat/{session_id}", tags=["Chat Session"])
async def get_session_messages(session_id: str):
    try:
        return {"messages": get_messages_for_session(session_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Gagal memuat percakapan.")

async def handle_chat(request: Any, session_id: str = None):
    user_details = get_user_details(request.user_id)
    if not user_details: raise HTTPException(status_code=404, detail="User tidak ditemukan.")
    grade = user_details.get("grade")

    history = get_messages_for_session(session_id) if session_id else []
    formatted_history = [ChatTurn(role=("assistant" if msg.get('sender') == 'bot' else 'user'), content=msg.get('text', '')) for msg in history]

    response_data = await generate_real_answer(request.question, formatted_history, grade=grade)

    active_session_id = session_id or str(uuid.uuid4())
    if not session_id:
        session_name = request.question[:50]
        save_chat_turn(user_id=request.user_id, session_id=active_session_id, session_name=session_name, turn={"sender": "user", "text": request.question})
        response_data["session_id"] = active_session_id
        response_data["session_name"] = session_name
    else:
        save_chat_turn(user_id=request.user_id, session_id=active_session_id, turn={"sender": "user", "text": request.question})
    
    save_chat_turn(user_id=request.user_id, session_id=active_session_id, turn={"sender": "bot", "text": response_data["answer"], "sources": response_data["sources"]})
    
    return response_data

@app.post("/chat/new", tags=["Chat Session"])
async def start_new_chat(request: NewChatRequest):
    return await handle_chat(request)

@app.post("/chat/{session_id}", tags=["Chat Session"])
async def continue_chat(session_id: str, request: ContinueChatRequest):
    return await handle_chat(request, session_id=session_id)

@app.delete("/chat/{session_id}", tags=["Chat Session"])
async def delete_chat_session(session_id: str, user_id: str):
    try:
        if delete_session_for_user(user_id=user_id, session_id=session_id):
            return {"message": "Sesi berhasil dihapus."}
        else:
            raise HTTPException(status_code=500, detail="Gagal menghapus sesi di database.")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Terjadi kesalahan internal saat menghapus sesi.")
    
@app.post("/generate/direct", tags=["LLM Direct"])
async def direct_generation(request: NewChatRequest):
    try:
        logger.info(f"Menerima direct generation request: '{request.question}'")
        system_prompt = "Anda adalah asisten AI bernama Studiva..."
        answer = await call_groq_api(prompt_context="", question=request.question, system_prompt=system_prompt, history=[])
        return {"answer": answer, "sources": [], "retrieved_contexts": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Terjadi kesalahan pada sistem AI.")