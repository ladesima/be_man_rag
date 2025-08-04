#!/usr/bin/env python3
"""
config.py - Konfigurasi Terpusat untuk Studiva.AI
Mengelola semua pengaturan penting dari satu tempat.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Memuat variabel dari file .env
load_dotenv()

logger = logging.getLogger(__name__)

class GlobalConfig:
    """Kelas konfigurasi terpusat untuk seluruh aplikasi."""
    
    # --- Informasi Proyek ---
    PROJECT_NAME = "Studiva.AI RAG System"
    PROJECT_VERSION = "4.0.0-best"
    
    # --- Konfigurasi Kunci API & Model ---
    # Diambil dari file .env Anda
    HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Model untuk mengubah teks menjadi vektor (untuk pencarian awal)
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")
    
    # **PERBAIKAN**: Menambahkan variabel yang hilang untuk model Re-ranker
    # Model untuk menilai ulang relevansi (langkah penting untuk akurasi)
    RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Model untuk menghasilkan jawaban akhir
    GENERATIVE_MODEL_NAME = os.getenv("GENERATIVE_MODEL_NAME", "gemini-1.5-pro-latest")

    # --- Konfigurasi Database ---
    # Path untuk menyimpan database vektor ChromaDB
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma")
    # Nama koleksi di dalam database vektor
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "islamic_curriculum_docs")

    # --- Konfigurasi Pemrosesan Dokumen ---
    # Direktori utama tempat semua file PDF Anda disimpan
    PDF_DIR = Path(os.getenv("PDF_DIR", "./app/pdf"))
    # Direktori cache untuk menyimpan model yang diunduh dari Hugging Face
    HUGGINGFACE_CACHE_DIR = os.getenv("HUGGINGFACE_CACHE_DIR", "./data/models")
    
    # Pengaturan untuk memecah teks dari PDF menjadi potongan-potongan
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    # --- Konfigurasi Logika RAG ---
    # Jumlah dokumen awal yang diambil dari ChromaDB sebelum di-filter
    INITIAL_K_RETRIEVAL = int(os.getenv("INITIAL_K_RETRIEVAL", "25"))
    # Jumlah dokumen teratas yang dikirim ke Gemini setelah di-filter oleh re-ranker
    TOP_K_RERANKED = int(os.getenv("TOP_K_RERANKED", "5"))

# Membuat instance dari kelas di atas.
# Objek 'config' inilah yang akan diimpor dan digunakan oleh file lain.
config = GlobalConfig()

# Validasi sederhana saat file diimpor untuk memastikan konfigurasi penting ada.
if not config.GEMINI_API_KEY:
    logger.warning("PERINGATAN: GEMINI_API_KEY tidak ditemukan di file .env. Fungsi chat tidak akan bekerja.")
if not config.HUGGINGFACE_API_TOKEN:
    logger.warning("PERINGATAN: HUGGINGFACE_API_TOKEN tidak ditemukan. Beberapa model mungkin gagal diunduh.")