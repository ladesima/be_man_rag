# config/performance_config.py - Performance optimization configuration

import os
import psutil

class PerformanceConfig:
    """Configuration for performance optimizations"""
    
    def __init__(self):
        # System Resources
        self.TOTAL_RAM_GB = psutil.virtual_memory().total / (1024**3)
        self.CPU_CORES = psutil.cpu_count()

        # --- PERBAIKAN ---
        # Menambahkan variabel yang hilang untuk logika RAG
        # RAG Pipeline Performance
        self.TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "25"))
        self.TOP_K_RERANKED = int(os.getenv("TOP_K_RERANKED", "5"))
        
        # Memory Management
        self.MAX_MEMORY_USAGE_GB = float(os.getenv("MAX_MEMORY_USAGE_GB", "6"))
        self.MEMORY_WARNING_THRESHOLD = 0.8  # 80% of max memory
        self.MEMORY_CRITICAL_THRESHOLD = 0.9  # 90% of max memory
        
        # Processing Performance
        self.BATCH_PROCESSING_SIZE = int(os.getenv("BATCH_PROCESSING_SIZE", "12"))
        self.MAX_WORKERS = min(self.CPU_CORES, int(os.getenv("MAX_WORKERS", str(self.CPU_CORES))))
        self.PROCESSING_TIMEOUT = int(os.getenv("PROCESSING_TIMEOUT", "300"))  # 5 minutes
        
        # Caching Settings
        self.CACHE_SIZE = int(os.getenv("CACHE_SIZE", "500"))
        self.CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
        self.ENABLE_DISK_CACHE = os.getenv("ENABLE_DISK_CACHE", "true").lower() == "true"
        self.DISK_CACHE_SIZE_MB = int(os.getenv("DISK_CACHE_SIZE_MB", "1000"))  # 1GB
        
        # Model Loading Optimization
        self.PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "true").lower() == "true"
        self.MODEL_CACHE_SIZE_GB = float(os.getenv("MODEL_CACHE_SIZE_GB", "2"))
        self.LAZY_LOADING = os.getenv("LAZY_LOADING", "false").lower() == "true"
        
        # Garbage Collection
        self.ENABLE_GC_OPTIMIZATION = os.getenv("ENABLE_GC_OPTIMIZATION", "true").lower() == "true"
        self.GC_THRESHOLD = (700, 10, 10)  # More aggressive collection
        self.CLEANUP_INTERVAL_MINUTES = int(os.getenv("CLEANUP_INTERVAL_MINUTES", "30"))
        
        # Concurrent Processing
        self.MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
        self.THREAD_POOL_SIZE = int(os.getenv("THREAD_POOL_SIZE", "4"))
        self.ASYNC_PROCESSING = os.getenv("ASYNC_PROCESSING", "true").lower() == "true"
        
        # Database Performance
        self.DB_CONNECTION_POOL_SIZE = int(os.getenv("DB_CONNECTION_POOL_SIZE", "10"))
        self.DB_QUERY_TIMEOUT = int(os.getenv("DB_QUERY_TIMEOUT", "30"))
        self.ENABLE_DB_INDEXING = os.getenv("ENABLE_DB_INDEXING", "true").lower() == "true"
        
        # Vector Search Optimization
        self.VECTOR_SEARCH_BATCH_SIZE = int(os.getenv("VECTOR_SEARCH_BATCH_SIZE", "100"))
        self.ENABLE_VECTOR_CACHE = os.getenv("ENABLE_VECTOR_CACHE", "true").lower() == "true"
        self.VECTOR_INDEX_TYPE = os.getenv("VECTOR_INDEX_TYPE", "hnsw")
        
        # Response Optimization
        self.ENABLE_RESPONSE_COMPRESSION = os.getenv("ENABLE_RESPONSE_COMPRESSION", "true").lower() == "true"
        self.RESPONSE_CACHE_SIZE = int(os.getenv("RESPONSE_CACHE_SIZE", "1000"))
        self.ENABLE_STREAMING = os.getenv("ENABLE_STREAMING", "false").lower() == "true"
        
        # Monitoring and Alerting
        self.ENABLE_PERFORMANCE_MONITORING = os.getenv("ENABLE_PERFORMANCE_MONITORING", "true").lower() == "true"
        self.PERFORMANCE_LOG_INTERVAL = int(os.getenv("PERFORMANCE_LOG_INTERVAL", "60"))  # 1 minute
        self.ALERT_SLOW_QUERIES = os.getenv("ALERT_SLOW_QUERIES", "true").lower() == "true"
        self.SLOW_QUERY_THRESHOLD = float(os.getenv("SLOW_QUERY_THRESHOLD", "5.0"))  # 5 seconds
        
        # Auto-optimization based on system resources
        self._auto_optimize_settings()
    
    def _auto_optimize_settings(self):
        """Auto-optimize settings based on available resources"""
        
        # Adjust batch size based on RAM
        if self.TOTAL_RAM_GB >= 16:
            self.BATCH_PROCESSING_SIZE = min(15, self.BATCH_PROCESSING_SIZE)
            self.CACHE_SIZE = min(1000, self.CACHE_SIZE)
        elif self.TOTAL_RAM_GB >= 8:
            self.BATCH_PROCESSING_SIZE = min(10, self.BATCH_PROCESSING_SIZE)
            self.CACHE_SIZE = min(500, self.CACHE_SIZE)
        else:
            self.BATCH_PROCESSING_SIZE = min(6, self.BATCH_PROCESSING_SIZE)
            self.CACHE_SIZE = min(200, self.CACHE_SIZE)
        
        # Adjust worker count based on CPU cores
        if self.CPU_CORES >= 8:
            self.MAX_WORKERS = min(8, self.MAX_WORKERS)
            self.THREAD_POOL_SIZE = min(6, self.THREAD_POOL_SIZE)
        elif self.CPU_CORES >= 4:
            self.MAX_WORKERS = min(4, self.MAX_WORKERS)
            self.THREAD_POOL_SIZE = min(4, self.THREAD_POOL_SIZE)
        else:
            self.MAX_WORKERS = min(2, self.MAX_WORKERS)
            self.THREAD_POOL_SIZE = min(2, self.THREAD_POOL_SIZE)
        
        # Adjust memory limits
        available_memory = psutil.virtual_memory().available / (1024**3)
        self.MAX_MEMORY_USAGE_GB = min(self.MAX_MEMORY_USAGE_GB, available_memory * 0.8)
    
    def get_optimization_config(self) -> dict:
        """Get optimization configuration dictionary"""
        return {
            "memory": {
                "max_usage_gb": self.MAX_MEMORY_USAGE_GB,
                "warning_threshold": self.MEMORY_WARNING_THRESHOLD,
                "critical_threshold": self.MEMORY_CRITICAL_THRESHOLD
            },
            "processing": {
                "batch_size": self.BATCH_PROCESSING_SIZE,
                "max_workers": self.MAX_WORKERS,
                "timeout": self.PROCESSING_TIMEOUT
            },
            "caching": {
                "cache_size": self.CACHE_SIZE,
                "cache_ttl": self.CACHE_TTL,
                "enable_disk_cache": self.ENABLE_DISK_CACHE
            },
            "concurrent": {
                "max_requests": self.MAX_CONCURRENT_REQUESTS,
                "thread_pool_size": self.THREAD_POOL_SIZE,
                "async_processing": self.ASYNC_PROCESSING
            }
        }
    
    def get_system_info(self) -> dict:
        """Get current system information"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            "total_ram_gb": round(self.TOTAL_RAM_GB, 2),
            "available_ram_gb": round(memory.available / (1024**3), 2),
            "ram_usage_percent": memory.percent,
            "cpu_cores": self.CPU_CORES,
            "cpu_usage_percent": cpu_percent,
            "recommended_batch_size": self.BATCH_PROCESSING_SIZE,
            "recommended_cache_size": self.CACHE_SIZE
        }
    
    def validate_performance_config(self):
        """Validate performance configuration"""
        # Check memory limits
        if self.MAX_MEMORY_USAGE_GB > self.TOTAL_RAM_GB:
            raise ValueError(f"MAX_MEMORY_USAGE_GB ({self.MAX_MEMORY_USAGE_GB}) cannot exceed total RAM ({self.TOTAL_RAM_GB:.1f})")
        
        # Check batch processing size
        if self.BATCH_PROCESSING_SIZE <= 0:
            raise ValueError("BATCH_PROCESSING_SIZE must be positive")
        
        if self.BATCH_PROCESSING_SIZE > 50:
            raise ValueError("BATCH_PROCESSING_SIZE too large, may cause memory issues")
        
        # Check worker limits
        if self.MAX_WORKERS <= 0:
            raise ValueError("MAX_WORKERS must be positive")
        
        if self.MAX_WORKERS > self.CPU_CORES * 2:
            raise ValueError(f"MAX_WORKERS ({self.MAX_WORKERS}) should not exceed 2x CPU cores ({self.CPU_CORES * 2})")
        
        # Check cache settings
        if self.CACHE_SIZE <= 0:
            raise ValueError("CACHE_SIZE must be positive")
        
        if self.CACHE_TTL <= 0:
            raise ValueError("CACHE_TTL must be positive")

        # --- PERBAIKAN ---
        # Menambahkan validasi untuk variabel RAG
        if self.TOP_K_RETRIEVAL <= 0:
            raise ValueError("TOP_K_RETRIEVAL must be positive")
        if self.TOP_K_RERANKED <= 0:
            raise ValueError("TOP_K_RERANKED must be positive")
        if self.TOP_K_RERANKED > self.TOP_K_RETRIEVAL:
            raise ValueError(f"TOP_K_RERANKED ({self.TOP_K_RERANKED}) cannot be greater than TOP_K_RETRIEVAL ({self.TOP_K_RETRIEVAL})")
        
        print("✅ Performance configuration validated")
        return True