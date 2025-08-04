# backend/__init__.py - Backend package initializer

"""
Islamic RAG Backend Package
===========================

This package contains all backend components for the Islamic curriculum RAG system.

Main Components:
- AI & LLM: Model management and inference
- Document Processing: PDF extraction and chunking
- Vector Store: ChromaDB and search functionality
- Islamic Components: Arabic text and curriculum-specific features
- Pipeline: Processing orchestration
- Analytics: Monitoring and metrics
- Utilities: Helper functions and services
"""

# Core imports
from .ultra_fast_rag import IslamicFastRAGEngine, islamic_ultra_fast_get_answer
from .llm_service import LLMService, HuggingFaceFalconLLM
from .document_processor import DocumentProcessor
from .vector_store import ChromaVectorStore
from .subject_detector import SubjectDetector
# from .arabic_processor import ArabicTextProcessor

# Service imports
# from .api_service import APIService
from .cache_manager import CacheManager
# from .analytics_service import AnalyticsService
# from .health_checker import HealthChecker

# Configuration
from config import __all__ as config_all
from config import Config

# Version info
__version__ = "3.0.0-islamic"
__author__ = "Islamic RAG Team"
__description__ = "Backend components for Islamic curriculum RAG system"

# Initialize core services
def initialize_backend():
    """Initialize backend services"""
    try:
        # Initialize core components
        rag_engine = IslamicFastRAGEngine()
        subject_detector = SubjectDetector()
        # arabic_processor = ArabicTextProcessor()
        
        print("✅ Backend services initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Backend initialization failed: {e}")
        return False

# Export main components
__all__ = [
    'IslamicFastRAGEngine',
    'islamic_ultra_fast_get_answer',
    'LLMService',
    'DocumentProcessor',
    'VectorStoreManager',
    'SubjectDetector',
    'ArabicTextProcessor',
    'APIService',
    'CacheManager',
    'AnalyticsService',
    'HealthChecker',
    'initialize_backend'
]