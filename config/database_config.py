# config/database_config.py - Vector database configuration

import os
from pathlib import Path

class DatabaseConfig:
    """Configuration for vector database (ChromaDB)"""
    
    def __init__(self):
        # ChromaDB Settings
        self.CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma")
        self.COLLECTION_NAME = os.getenv("COLLECTION_NAME", "islamic_curriculum_docs")
        self.PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "./data/chroma")
        
        # Vector Store Settings
        self.VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma")
        self.SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
        self.TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "3"))
        
        # Document Processing
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.MAX_CHUNK_SIZE = 1000
        self.MIN_CHUNK_SIZE = 200
        
        # Text Splitter Settings
        self.TEXT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
        self.LENGTH_FUNCTION = "token_count"  # or "character_count"
        
        # Batch Processing
        self.BATCH_SIZE = int(os.getenv("BATCH_PROCESSING_SIZE", "12"))
        self.MAX_BATCH_SIZE = 50
        self.PROCESSING_TIMEOUT = 300  # 5 minutes per batch
        
        # Database Optimization
        self.ENABLE_INDEXING = True
        self.INDEX_TYPE = "hnsw"  # Hierarchical Navigable Small World
        self.INDEX_PARAMETERS = {
            "M": 16,  # Number of connections
            "efConstruction": 200,  # Size of dynamic candidate list
            "ef": 50  # Size of candidate list for search
        }
        
        # Persistence Settings
        self.AUTO_PERSIST = True
        self.PERSIST_INTERVAL = 100  # Persist every 100 operations
        self.BACKUP_ENABLED = True
        self.BACKUP_INTERVAL_HOURS = 24
        
        # Memory Management
        self.MAX_MEMORY_USAGE_MB = 2000
        self.CLEANUP_INTERVAL = 1000  # Cleanup every 1000 operations
        self.ENABLE_COMPRESSION = True
        
        # Search Settings
        self.SEARCH_TYPE = "similarity"  # or "similarity_score_threshold", "mmr"
        self.MMR_DIVERSITY_THRESHOLD = 0.5  # For MMR search
        self.FETCH_K = 20  # Initial fetch size for MMR
        
        # Create directories
        self._create_database_directories()
    
    def _create_database_directories(self):
        """Create database directories"""
        directories = [
            Path(self.CHROMA_DB_PATH),
            Path(self.PERSIST_DIRECTORY),
            Path(self.CHROMA_DB_PATH).parent / "backups"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_chroma_config(self) -> dict:
        """Get ChromaDB configuration"""
        return {
            "persist_directory": self.PERSIST_DIRECTORY,
            "collection_name": self.COLLECTION_NAME,
        }
    
    def get_text_splitter_config(self) -> dict:
        """Get text splitter configuration"""
        return {
            "chunk_size": self.CHUNK_SIZE,
            "chunk_overlap": self.CHUNK_OVERLAP,
            "separators": self.TEXT_SEPARATORS,
            "length_function": self.LENGTH_FUNCTION
        }
    
    def get_retrieval_config(self) -> dict:
        """Get document retrieval configuration"""
        return {
            "search_type": self.SEARCH_TYPE,
            "search_kwargs": {
                "k": self.TOP_K_RETRIEVAL,
                "score_threshold": self.SIMILARITY_THRESHOLD,
                "fetch_k": self.FETCH_K
            }
        }
    
    def validate_database_config(self):
        """Validate database configuration"""
        # Check paths exist
        if not Path(self.CHROMA_DB_PATH).parent.exists():
            raise ValueError(f"Database parent directory does not exist: {Path(self.CHROMA_DB_PATH).parent}")
        
        # Validate chunk sizes
        if not (self.MIN_CHUNK_SIZE <= self.CHUNK_SIZE <= self.MAX_CHUNK_SIZE):
            raise ValueError(f"CHUNK_SIZE must be between {self.MIN_CHUNK_SIZE} and {self.MAX_CHUNK_SIZE}")
        
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        
        # Validate similarity threshold
        if not (0.0 <= self.SIMILARITY_THRESHOLD <= 1.0):
            raise ValueError("SIMILARITY_THRESHOLD must be between 0.0 and 1.0")
        
        # Validate retrieval settings
        if self.TOP_K_RETRIEVAL <= 0:
            raise ValueError("TOP_K_RETRIEVAL must be positive")
        
        print("✅ Database configuration validated")
        return True