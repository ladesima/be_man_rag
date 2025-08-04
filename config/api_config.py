### **8. `config/api_config.py` - FastAPI Server Configuration**
# config/api_config.py - FastAPI server configuration

import os
from typing import List

class APIConfig:
    """Configuration for FastAPI server"""
    
    def __init__(self):
        # Server Settings
        self.API_HOST = os.getenv("API_HOST", "127.0.0.1")
        self.API_PORT = int(os.getenv("API_PORT", "8000"))
        self.API_RELOAD = os.getenv("API_RELOAD", "false").lower() == "true"
        self.WORKERS = int(os.getenv("WORKERS", "1"))
        
        # API Metadata
        self.API_TITLE = "Islamic Curriculum RAG API"
        self.API_DESCRIPTION = "Ultra-fast RAG untuk 12 mata pelajaran: 5 Islamic Studies + 7 General Education"
        self.API_VERSION = "3.0.0-islamic"
        
        # CORS Settings
        self.CORS_ORIGINS = self._parse_cors_origins(os.getenv("CORS_ORIGINS", '["*"]'))
        self.CORS_CREDENTIALS = os.getenv("CORS_CREDENTIALS", "true").lower() == "true"
        self.CORS_METHODS = self._parse_list(os.getenv("CORS_METHODS", '["GET", "POST"]'))
        self.CORS_HEADERS = self._parse_list(os.getenv("CORS_HEADERS", '["*"]'))
        self.CORS_MAX_AGE = int(os.getenv("CORS_MAX_AGE", "7200"))
        
        # Request/Response Settings
        self.MAX_REQUEST_SIZE = int(os.getenv("MAX_REQUEST_SIZE", "10485760"))  # 10MB
        self.REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
        self.RESPONSE_TIMEOUT = int(os.getenv("RESPONSE_TIMEOUT", "30"))
        
        # Rate Limiting
        self.ENABLE_RATE_LIMITING = os.getenv("ENABLE_RATE_LIMITING", "false").lower() == "true"
        self.RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
        self.RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))  # 1 hour
        
        # Security Settings
        self.ENABLE_HTTPS = os.getenv("ENABLE_HTTPS", "false").lower() == "true"
        self.SSL_CERT_FILE = os.getenv("SSL_CERT_FILE")
        self.SSL_KEY_FILE = os.getenv("SSL_KEY_FILE")
        
        # Middleware Settings
        self.ENABLE_GZIP = os.getenv("ENABLE_GZIP", "true").lower() == "true"
        self.GZIP_MINIMUM_SIZE = int(os.getenv("GZIP_MINIMUM_SIZE", "500"))
        
        # Logging Settings
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
        self.ACCESS_LOG = os.getenv("ACCESS_LOG", "true").lower() == "true"
        self.LOG_CONFIG = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                }
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout"
                }
            },
            "root": {
                "level": self.LOG_LEVEL.upper(),
                "handlers": ["default"]
            }
        }
        
        # Health Check Settings
        self.HEALTH_CHECK_PATH = "/health"
        self.METRICS_PATH = "/metrics"
        self.DOCS_PATH = "/docs"
        self.OPENAPI_PATH = "/openapi.json"
        
        # Islamic Curriculum Specific
        self.ISLAMIC_ENDPOINTS = {
            "ask": "/ask",
            "subjects": "/subjects", 
            "health": "/health",
            "performance": "/admin/islamic/performance",
            "cache_clear": "/admin/islamic/cache/clear"
        }
    
    def _parse_cors_origins(self, origins_str: str) -> List[str]:
        """Parse CORS origins from string"""
        try:
            import json
            return json.loads(origins_str)
        except:
            return ["*"]
    
    def _parse_list(self, list_str: str) -> List[str]:
        """Parse list from string"""
        try:
            import json
            return json.loads(list_str)
        except:
            return ["*"]
    
    def get_uvicorn_config(self) -> dict:
        """Get Uvicorn server configuration"""
        config = {
            "host": self.API_HOST,
            "port": self.API_PORT,
            "reload": self.API_RELOAD,
            "log_level": self.LOG_LEVEL,
            "access_log": self.ACCESS_LOG,
            "workers": self.WORKERS,
            "timeout_keep_alive": 30,
            "timeout_graceful_shutdown": 10
        }
        
        if self.ENABLE_HTTPS:
            config.update({
                "ssl_certfile": self.SSL_CERT_FILE,
                "ssl_keyfile": self.SSL_KEY_FILE
            })
        
        return config
    
    def get_cors_config(self) -> dict:
        """Get CORS configuration"""
        return {
            "allow_origins": self.CORS_ORIGINS,
            "allow_credentials": self.CORS_CREDENTIALS,
            "allow_methods": self.CORS_METHODS,
            "allow_headers": self.CORS_HEADERS,
            "max_age": self.CORS_MAX_AGE
        }
    
    def validate_api_config(self):
        """Validate API configuration"""
        # Validate host and port
        if not self.API_HOST:
            raise ValueError("API_HOST cannot be empty")
        
        if not (1 <= self.API_PORT <= 65535):
            raise ValueError("API_PORT must be between 1 and 65535")
        
        # Validate SSL settings
        if self.ENABLE_HTTPS:
            if not self.SSL_CERT_FILE or not self.SSL_KEY_FILE:
                raise ValueError("SSL certificate and key files required when HTTPS is enabled")
        
        # Validate timeouts
        if self.REQUEST_TIMEOUT <= 0:
            raise ValueError("REQUEST_TIMEOUT must be positive")
        
        if self.RESPONSE_TIMEOUT <= 0:
            raise ValueError("RESPONSE_TIMEOUT must be positive")
        
        # Validate rate limiting
        if self.ENABLE_RATE_LIMITING:
            if self.RATE_LIMIT_REQUESTS <= 0:
                raise ValueError("RATE_LIMIT_REQUESTS must be positive")
            
            if self.RATE_LIMIT_WINDOW <= 0:
                raise ValueError("RATE_LIMIT_WINDOW must be positive")
        
        print("✅ API configuration validated")
        return True