# config/model_config.py - AI model configuration

import os
from typing import Dict

class ModelConfig:
    """Configuration for AI models"""
    
    def __init__(self):
        # --- KUNCI API ---
        self.HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self.GROQ_API_KEY = os.getenv("groq_api_key")  # Optional, for Groq models

        # --- NAMA MODEL ---
        self.GENERATIVE_MODEL_NAME = os.getenv("GENERATIVE_MODEL_NAME", "meta-llama/llama-4-scout-17b-16e-instruct")
        self.EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        self.RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")

        # --- PENGATURAN CACHE & PEMUATAN MODEL ---
        self.MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./models_cache")
        # PERBAIKAN: Menambahkan variabel yang hilang untuk device embedding
        self.EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")  # Ganti ke "cuda" jika punya GPU
        self.PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "true").lower() == "true"
        
        # Pengaturan lain yang mungkin Anda perlukan
        self.MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "1024"))
        self.TEMPERATURE = float(os.getenv("TEMPERATURE", "0.5"))
        self.TOP_P = float(os.getenv("TOP_P", "0.9"))
        self.DO_SAMPLE = os.getenv("DO_SAMPLE", "true").lower() == "true"


    def validate_model_config(self):
        """Validate all model-related configurations"""
        if not self.HUGGINGFACE_API_TOKEN:
            raise ValueError("❌ HUGGINGFACE_API_TOKEN is required in your .env file!")
            
        if not self.GEMINI_API_KEY:
            raise ValueError("❌ GEMINI_API_KEY is required in your .env file!")

        if not self.GENERATIVE_MODEL_NAME:
            raise ValueError("❌ GENERATIVE_MODEL_NAME is not defined!")

        if not self.EMBEDDING_MODEL_NAME:
            raise ValueError("❌ EMBEDDING_MODEL_NAME is not defined!")
        
        if not self.RERANKER_MODEL_NAME:
            raise ValueError("❌ RERANKER_MODEL_NAME is not defined!")

        if not self.MODEL_CACHE_DIR:
            raise ValueError("❌ MODEL_CACHE_DIR is not defined!")
            
        # PERBAIKAN: Menambahkan validasi untuk device embedding
        if not self.EMBEDDING_DEVICE:
            raise ValueError("❌ EMBEDDING_DEVICE is not defined!")

        print("✅ Model configuration validated")
        return True