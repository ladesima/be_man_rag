"""
Islamic RAG Configuration Package
=================================

This package contains all configuration files for the Islamic curriculum RAG system.

Modules:
- base_config: Base system configuration
- islamic_config: Islamic curriculum specific settings
- model_config: AI model configurations
- database_config: Vector database settings
- api_config: FastAPI server configuration
- performance_config: Performance optimization settings
- subjects_config: 12 mata pelajaran configuration
"""

from .base_config import BaseConfig
from .islamic_config import IslamicConfig
from .model_config import ModelConfig
from .database_config import DatabaseConfig
from .api_config import APIConfig
from .performance_config import PerformanceConfig
from .subjects_config import SubjectsConfig


class Config(
    BaseConfig,
    IslamicConfig,
    ModelConfig,
    DatabaseConfig,
    APIConfig,
    PerformanceConfig
):
    """
    Master configuration class combining all config modules
    """

    def __init__(self):
        # Initialize each config module
        BaseConfig.__init__(self)
        IslamicConfig.__init__(self)
        ModelConfig.__init__(self)
        DatabaseConfig.__init__(self)
        APIConfig.__init__(self)
        PerformanceConfig.__init__(self)
        
        # Initialize subject config as a property
        self.subjects = SubjectsConfig()

        # Validate on start
        self.validate_all_configs()

    def validate_all_configs(self):
        """Validate all configuration modules"""
        try:
            self.validate_base_config()
            self.validate_islamic_config()
            self.validate_model_config()
            self.validate_database_config()
            self.validate_api_config()
            self.validate_performance_config()

            print("✅ All configurations validated successfully")
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            raise


# Export singleton instance
config = Config()

__all__ = [
    "Config",
    "config",
    "BaseConfig",
    "IslamicConfig",
    "ModelConfig",
    "DatabaseConfig",
    "APIConfig",
    "PerformanceConfig",
    "SubjectsConfig",
]
