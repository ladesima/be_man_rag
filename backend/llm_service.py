# backend/llm_service.py - LLM service management

import os
import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline

logger = logging.getLogger(__name__)

class BaseLLMService(ABC):
    """Abstract base class untuk LLM services"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        """Tokenize input text"""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        pass

class HuggingFaceFalconLLM(BaseLLMService):
    """Hugging Face Falcon LLM service"""
    
    def __init__(self, model_name: str = "tiiuae/falcon-rw-1b"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.is_loaded = False
        
        # Generation parameters
        self.generation_config = {
            "max_new_tokens": 200,
            "temperature": 0.3,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False,
            "pad_token_id": None  # Set from tokenizer
        }
    
    def load_model(self):
        """Load Falcon model dan tokenizer"""
        if self.is_loaded:
            return
        
        try:
            token = os.getenv("HUGGINGFACE_API_TOKEN")
            if not token:
                raise ValueError("HUGGINGFACE_API_TOKEN required")
            
            logger.info(f"Loading Falcon model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=token,
                trust_remote_code=True,
                use_fast=True
            )
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=token,
                torch_dtype="float16",
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_cache=True
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                **self.generation_config,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            self.is_loaded = True
            logger.info("✅ Falcon model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Falcon model: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text menggunakan Falcon"""
        if not self.is_loaded:
            self.load_model()
        
        try:
            # Update generation config dengan kwargs
            gen_config = self.generation_config.copy()
            gen_config.update(kwargs)
            
            # Generate
            result = self.pipeline(prompt, **gen_config)
            
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
                return generated_text.strip()
            
            return ""
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text"""
        if not self.is_loaded:
            self.load_model()
        
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenize(text))
    
    def get_langchain_llm(self):
        """Get LangChain compatible LLM"""
        if not self.is_loaded:
            self.load_model()
        
        return HuggingFacePipeline(pipeline=self.pipeline)

class LLMService:
    """Main LLM service manager"""
    
    def __init__(self):
        self.llm_providers = {}
        self.active_provider = None
        
        # Initialize default provider
        self._init_default_provider()
    
    def _init_default_provider(self):
        """Initialize default Falcon provider"""
        try:
            falcon_llm = HuggingFaceFalconLLM()
            self.register_provider("falcon", falcon_llm)
            self.set_active_provider("falcon")
            
        except Exception as e:
            logger.error(f"Failed to initialize default LLM provider: {e}")
    
    def register_provider(self, name: str, provider: BaseLLMService):
        """Register LLM provider"""
        self.llm_providers[name] = provider
        logger.info(f"LLM provider '{name}' registered")
    
    def set_active_provider(self, name: str):
        """Set active LLM provider"""
        if name not in self.llm_providers:
            raise ValueError(f"LLM provider '{name}' not found")
        
        self.active_provider = name
        logger.info(f"Active LLM provider set to: {name}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate using active provider"""
        if not self.active_provider:
            raise ValueError("No active LLM provider")
        
        provider = self.llm_providers[self.active_provider]
        return provider.generate(prompt, **kwargs)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using active provider"""
        if not self.active_provider:
            raise ValueError("No active LLM provider")
        
        provider = self.llm_providers[self.active_provider]
        return provider.count_tokens(text)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return list(self.llm_providers.keys())
    
    def get_active_provider_info(self) -> Dict[str, Any]:
        """Get info about active provider"""
        if not self.active_provider:
            return {}
        
        provider = self.llm_providers[self.active_provider]
        
        return {
            "name": self.active_provider,
            "type": type(provider).__name__,
            "is_loaded": getattr(provider, 'is_loaded', False),
            "model_name": getattr(provider, 'model_name', 'unknown')
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of LLM service"""
        try:
            if not self.active_provider:
                return {"status": "unhealthy", "error": "No active provider"}
            
            # Test generation
            test_result = self.generate("Test", max_new_tokens=5)
            
            return {
                "status": "healthy",
                "active_provider": self.active_provider,
                "test_generation": len(test_result) > 0,
                "available_providers": len(self.llm_providers)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "active_provider": self.active_provider
            }

# Singleton instance
_llm_service = None

def get_llm_service() -> LLMService:
    """Get singleton LLM service instance"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service