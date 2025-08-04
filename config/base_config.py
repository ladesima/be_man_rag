# config/base_config.py - Konfigurasi dasar sistem

import os
from pathlib import Path
from dotenv import load_dotenv

# Muat environment variables
load_dotenv()

class BaseConfig:
    """Konfigurasi dasar untuk sistem RAG Islami."""
    
    def __init__(self):
        # --- Pengaturan Path Proyek ---
        self.PROJECT_ROOT = Path(__file__).parent.parent
        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.LOGS_DIR = self.PROJECT_ROOT / "logs" 
        self.PDF_DIR = self.PROJECT_ROOT / "app" / "pdf" 
        
        # Path ke file pickle diatur untuk menggunakan direktori PDF
        # PASTIKAN nama file ini sudah sesuai dengan nama file Anda yang sebenarnya
        self.PICKLE_FILE_PATH = self.PDF_DIR / "merge_data.pkl"

        # --- Pengaturan Lingkungan ---
        self.ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
        self.DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
        self.VERSION = "5.0.0-tuned"

        # --- Pengaturan Pemrosesan Dokumen (Chunking) ---
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.2"))

        # Membuat direktori yang diperlukan saat aplikasi dimulai
        self._create_directories()
    
    def _create_directories(self):
        """Membuat direktori yang diperlukan jika belum ada."""
        directories = [
            self.DATA_DIR,
            self.LOGS_DIR,
            self.PDF_DIR,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def validate_base_config(self):
        """Memvalidasi konfigurasi dasar."""
        required_dirs = [self.DATA_DIR, self.LOGS_DIR, self.PDF_DIR]
        
        for directory in required_dirs:
            if not directory.exists():
                raise FileNotFoundError(f"Direktori yang diperlukan tidak ditemukan: {directory}")
        
        if self.CHUNK_SIZE <= 0 or self.CHUNK_OVERLAP < 0:
            raise ValueError("CHUNK_SIZE dan CHUNK_OVERLAP harus bernilai positif.")
        
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP harus lebih kecil dari CHUNK_SIZE.")
            
        print("✅ Base configuration validated")
        return True