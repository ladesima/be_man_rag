# backend/islamic_ultra_fast_rag.py - Enhanced untuk Islamic curriculum
import os
import time
import logging
import warnings
import re
from typing import Dict, Any, List, Optional
from functools import lru_cache
import threading

# Suppress warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# Islamic curriculum subject mapping
ISLAMIC_SUBJECTS = {
    'akidah': {
        'keywords': ['akidah', 'aqidah', 'tauhid', 'iman', 'rukun iman', 'asmaul husna'],
        'chunk_size': 800,
        'arabic_content': True,
        'special_processing': True
    },
    'ilmu_quran': {
        'keywords': ['quran', "qur'an", 'tajwid', 'qiraah', 'tafsir', 'makki', 'madani'],
        'chunk_size': 600,
        'arabic_content': True,
        'special_processing': True
    },
    'bahasa_arab': {
        'keywords': ['arab', 'nahwu', 'sharaf', 'balaghah', "i'rab", "fi'il"],
        'chunk_size': 500,
        'arabic_content': True,
        'special_processing': True
    },
    'fikih': {
        'keywords': ['fikih', 'fiqh', 'hukum islam', 'halal', 'haram', 'wudhu', 'shalat'],
        'chunk_size': 900,
        'arabic_content': True,
        'special_processing': True
    },
    'sejarah_kebudayaan_islam': {
        'keywords': ['sejarah islam', 'khulafaur', 'nabi', 'sahabat', "tabi'in"],
        'chunk_size': 1000,
        'arabic_content': False,
        'special_processing': True
    }
}

GENERAL_SUBJECTS = {
    'matematika': {'chunk_size': 700, 'preserve_formulas': True},
    'seni_budaya': {'chunk_size': 800, 'preserve_cultural_context': True},
    'bahasa_inggris': {'chunk_size': 600, 'preserve_grammar': True},
    'pjok': {'chunk_size': 800, 'preserve_health_concepts': True},
    'ppkn': {'chunk_size': 800, 'preserve_civic_concepts': True},
    'prakarya': {'chunk_size': 800, 'preserve_skill_steps': True},
    'sejarah_indonesia': {'chunk_size': 1000, 'preserve_chronology': True},
    'bahasa_indonesia': {'chunk_size': 600, 'preserve_linguistic_features': True}
}

class IslamicFastRAGEngine:
    """Ultra-fast RAG engine dengan Islamic curriculum optimization"""
    
    def __init__(self):
        self.vectordb = None
        self.llm = None
        self.tokenizer = None
        self.embeddings = None
        self.is_initialized = False
        
        # Islamic processing components
        self.arabic_patterns = self._init_arabic_patterns()
        self.islamic_terms = self._init_islamic_terms()
        
        logger.info(" Islamic Ultra-Fast RAG Engine initialized")
    
    def _init_arabic_patterns(self):
        """Initialize Arabic text patterns"""
        return {
            'arabic_text': re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+'),
            'quran_verse': re.compile(r'﴿.*?﴾'),
            'hadith_ref': re.compile(r'(رواه|أخرجه|صحيح|حسن|ضعيف)'),
            'transliteration': re.compile(r'[A-Za-z]+\s*\([^)]*[\u0600-\u06FF][^)]*\)'),
        }
    
    def _init_islamic_terms(self):
        """Initialize Islamic terminology dictionary"""
        return {
            'worship_terms': ['shalat', 'solat', 'puasa', 'zakat', 'haji', 'umrah', 'wudhu'],
            'belief_terms': ['allah', 'tauhid', 'iman', 'islam', 'ihsan', 'rukun'],
            'law_terms': ['halal', 'haram', 'makruh', 'sunnah', 'wajib', 'mubah'],
            'quran_terms': ['ayat', 'surah', 'juz', 'makkiyah', 'madaniyah', 'tajwid'],
            'arabic_grammar': ['nahwu', 'sharaf', 'i\'rab', 'fi\'il', 'isim', 'harf']
        }
    
    def initialize_components(self):
        """Initialize dengan Islamic optimizations"""
        if self.is_initialized:
            return
            
        start_time = time.time()
        logger.info(" Starting Islamic-optimized initialization...")
        
        try:
            # 1. Import dependencies
            self._import_dependencies()
            
            # 2. Initialize embeddings (with Arabic support)
            self._init_islamic_embeddings()
            
            # 3. Load vectorstore
            self._load_vectorstore()
            
            # 4. Initialize LLM
            self._init_fast_llm()
            
            self.is_initialized = True
            
            init_time = time.time() - start_time
            logger.info(f" Islamic RAG init completed in {init_time:.2f}s")
            
        except Exception as e:
            logger.error(f" Islamic init failed: {e}")
            raise
    
    def _import_dependencies(self):
        """Import dengan error handling"""
        try:
            from langchain_community.vectorstores import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            from langchain_community.llms import HuggingFacePipeline
            
            self.imports = {
                'Chroma': Chroma,
                'HuggingFaceEmbeddings': HuggingFaceEmbeddings,
                'AutoTokenizer': AutoTokenizer,
                'AutoModelForCausalLM': AutoModelForCausalLM,
                'pipeline': pipeline,
                'HuggingFacePipeline': HuggingFacePipeline
            }
            
        except ImportError as e:
            logger.error(f"Import failed: {e}")
            raise
    
    def _init_islamic_embeddings(self):
        """Initialize embeddings dengan Arabic support"""
        try:
            # Use multilingual embedding model for Arabic support
            self.embeddings = self.imports['HuggingFaceEmbeddings'](
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={
                    'device': 'cpu',
                    'torch_dtype': 'float32'
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32
                }
            )
            logger.info(" Islamic embeddings ready (multilingual)")
            
        except Exception as e:
            logger.error(f"Islamic embedding init failed: {e}")
            # Fallback to standard model
            self.embeddings = self.imports['HuggingFaceEmbeddings'](
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info(" Using fallback embeddings")
    
    def _load_vectorstore(self):
        """Load vectorstore dengan Islamic document structure"""
        try:
            persist_directory = "./data/chroma"
            
            if os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
                logger.info(" Loading existing Islamic vectorstore...")
                
                self.vectordb = self.imports['Chroma'](
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
                
                # Validate Islamic content
                try:
                    collection = self.vectordb._collection
                    doc_count = collection.count() if collection else 0
                    logger.info(f" Islamic vectorstore loaded: {doc_count} documents")
                except:
                    logger.warning(" Vectorstore loaded but validation failed")
                
            else:
                logger.error(" No existing vectorstore found! Please run document processing first.")
                raise FileNotFoundError("Islamic vectorstore not found. Run processing first.")
                
        except Exception as e:
            logger.error(f"Vectorstore loading failed: {e}")
            raise
    
    def _init_fast_llm(self):
        """Initialize LLM dengan Islamic content optimization"""
        try:
            model_id = "tiiuae/falcon-rw-1b"
            
            # Initialize tokenizer
            self.tokenizer = self.imports['AutoTokenizer'].from_pretrained(
                model_id,
                trust_remote_code=True,
                use_fast=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model dengan optimizations
            model = self.imports['AutoModelForCausalLM'].from_pretrained(
                model_id,
                torch_dtype="float16",
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_cache=True
            )
            
            # Create pipeline untuk Islamic content
            pipe = self.imports['pipeline'](
                "text-generation",
                model=model,
                tokenizer=self.tokenizer,
                max_new_tokens=200,  # Longer untuk educational explanations
                do_sample=True,      # Enable sampling untuk variety
                temperature=0.3,     # Lower temperature untuk factual content
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False,
                batch_size=1
            )
            
            self.llm = self.imports['HuggingFacePipeline'](pipeline=pipe)
            logger.info(" Islamic-optimized LLM ready")
            
        except Exception as e:
            logger.error(f"LLM init failed: {e}")
            raise
    
    def detect_subject_from_content(self, content: str, filepath: str = "") -> str:
        """Detect subject dari content dan filepath"""
        content_lower = content.lower()
        filepath_lower = filepath.lower()
        
        # Check Islamic subjects first
        for subject, config in ISLAMIC_SUBJECTS.items():
            if any(keyword in content_lower or keyword in filepath_lower 
                   for keyword in config['keywords']):
                return subject
        
        # Check general subjects
        if any(term in filepath_lower for term in ['math', 'matematika']):
            return 'matematika'
        elif any(term in filepath_lower for term in ['english', 'inggris']):
            return 'bahasa_inggris'
        elif any(term in filepath_lower for term in ['seni', 'budaya', 'art']):
            return 'seni_budaya'
        elif any(term in filepath_lower for term in ['pjok', 'olahraga', 'sport']):
            return 'pjok'
        elif any(term in filepath_lower for term in ['ppkn', 'pancasila', 'civic']):
            return 'ppkn'
        elif any(term in filepath_lower for term in ['prakarya', 'craft', 'kerajinan']):
            return 'prakarya'
        elif any(term in filepath_lower for term in ['sejarah indonesia', 'kemerdekaan']):
            return 'sejarah_indonesia'
        
        return 'general'
    
    def process_islamic_content(self, content: str, subject: str) -> str:
        """Enhanced processing untuk Islamic content"""
        
        if subject in ISLAMIC_SUBJECTS:
            # Preserve Arabic text
            if ISLAMIC_SUBJECTS[subject]['arabic_content']:
                content = self._preserve_arabic_formatting(content)
            
            # Subject-specific processing
            if subject == 'ilmu_quran':
                content = self._preserve_quran_verses(content)
            elif subject == 'bahasa_arab':
                content = self._enhance_arabic_grammar(content)
            elif subject == 'fikih':
                content = self._preserve_fiqh_terminology(content)
            elif subject == 'akidah':
                content = self._enhance_belief_terminology(content)
        
        return content
    
    def _preserve_arabic_formatting(self, text: str) -> str:
        """Preserve Arabic text formatting"""
        # Ensure Arabic text is properly spaced and formatted
        arabic_parts = self.arabic_patterns['arabic_text'].findall(text)
        for arabic in arabic_parts:
            # Add proper spacing around Arabic text
            text = text.replace(arabic, f" {arabic} ")
        return text
    
    def _preserve_quran_verses(self, text: str) -> str:
        """Preserve Quranic verse formatting"""
        # Find and preserve Quranic verses ﴿ ﴾
        quran_verses = self.arabic_patterns['quran_verse'].findall(text)
        for verse in quran_verses:
            # Ensure verses are properly formatted
            text = text.replace(verse, f"\n{verse}\n")
        return text
    
    @lru_cache(maxsize=512)  # Larger cache for Islamic content
    def islamic_similarity_search(self, query: str, subject: str = None, k: int = 3) -> tuple:
        """Enhanced similarity search untuk Islamic content"""
        try:
            # Enhance query dengan Islamic terminology
            enhanced_query = self._enhance_islamic_query(query, subject)
            
            docs = self.vectordb.similarity_search(enhanced_query, k=k)
            
            # Convert to tuple for caching
            return tuple(
                (
                    doc.page_content, 
                    doc.metadata.get('source', ''), 
                    doc.metadata.get('page', ''),
                    doc.metadata.get('subject', 'unknown')
                )
                for doc in docs
            )
        except Exception as e:
            logger.error(f"Islamic search failed: {e}")
            return tuple()
    
    def _enhance_islamic_query(self, query: str, subject: str = None) -> str:
        """Enhance query dengan Islamic terminology"""
        enhanced_query = query
        
        # Add Islamic synonyms
        islamic_synonyms = {
            'shalat': ['solat', 'prayer', 'صلاة'],
            'quran': ['al-quran', 'qur\'an', 'القرآن'],
            'hadith': ['hadis', 'sunnah', 'الحديث'],
            'islam': ['إسلام', 'agama islam'],
            'allah': ['الله', 'tuhan'],
        }
        
        query_lower = query.lower()
        for term, synonyms in islamic_synonyms.items():
            if term in query_lower:
                enhanced_query += f" {' '.join(synonyms)}"
        
        return enhanced_query
    
    def create_islamic_prompt(self, subject: str, question: str, context: str) -> str:
        """Create subject-specific prompt untuk Islamic curriculum"""
        
        if subject == 'akidah':
            return f"""Anda adalah ustadz yang mengajar Akidah Islamiyah. Jawab pertanyaan berikut berdasarkan ajaran Ahlus Sunnah wal Jamaah.

Konteks dari kitab/buku akidah:
{context}

Pertanyaan: {question}

Panduan jawaban:
1. Berikan penjelasan yang sesuai Al-Quran dan Sunnah
2. Sertakan dalil jika ada
3. Jelaskan dengan bahasa yang mudah dipahami
4. Hindari bid'ah dan khurafat

Jawaban:"""

        elif subject == 'ilmu_quran':
            return f"""Anda adalah ustadz yang mengajar Ilmu Al-Quran. Berikan penjelasan yang komprehensif.

Konteks dari kitab Ilmu Quran:
{context}

Pertanyaan: {question}

Panduan jawaban:
1. Jelaskan dengan detail dan akurat
2. Sertakan referensi ayat jika relevan
3. Berikan contoh praktis untuk tajwid
4. Gunakan istilah-istilah Ilmu Quran yang tepat

Jawaban:"""

        elif subject == 'bahasa_arab':
            return f"""Anda adalah ustadz Bahasa Arab. Jelaskan pertanyaan tata bahasa berikut.

Konteks dari kitab Bahasa Arab:
{context}

Pertanyaan: {question}

Panduan jawaban:
1. Berikan penjelasan grammar yang jelas
2. Sertakan contoh dalam bahasa Arab dengan harakat
3. Jelaskan kaidah-kaidah yang berlaku
4. Berikan transliterasi jika perlu

Jawaban:"""

        elif subject == 'fikih':
            return f"""Anda adalah ustadz Fikih. Jawab pertanyaan hukum Islam berikut.

Konteks dari kitab fikih:
{context}

Pertanyaan: {question}

Panduan jawaban:
1. Jelaskan hukum dengan dalil yang kuat
2. Sebutkan pendapat ulama jika ada perbedaan
3. Berikan contoh praktis dalam kehidupan
4. Gunakan istilah fikih yang tepat

Jawaban:"""

        elif subject == 'sejarah_kebudayaan_islam':
            return f"""Anda adalah guru Sejarah Kebudayaan Islam. Jelaskan materi sejarah berikut.

Konteks dari buku sejarah Islam:
{context}

Pertanyaan: {question}

Panduan jawaban:
1. Berikan penjelasan kronologis yang akurat
2. Sertakan hikmah dan pelajaran yang dapat diambil
3. Hubungkan dengan konteks masa kini
4. Gunakan sumber sejarah yang valid

Jawaban:"""

        else:
            # General subjects dengan Islamic integration
            return f"""Anda adalah guru {subject} yang mengintegrasikan nilai-nilai Islam dalam pembelajaran.

Konteks materi:
{context}

Pertanyaan: {question}

Berikan penjelasan yang jelas dan integrasikan dengan nilai-nilai Islam jika memungkinkan.

Jawaban:"""
    
    def ultra_fast_islamic_query(self, question: str) -> Dict[str, Any]:
        """Ultra-fast query dengan Islamic content intelligence"""
        start_time = time.time()
        
        try:
            if not self.is_initialized:
                self.initialize_components()
            
            # 1. Detect subject dari question
            subject = self.detect_subject_from_content(question)
            
            # 2. Islamic-enhanced similarity search
            search_start = time.time()
            search_results = self.islamic_similarity_search(question, subject, k=3)
            search_time = time.time() - search_start
            
            # 3. Prepare context dengan Islamic formatting
            context = self._prepare_islamic_context(search_results, subject)
            
            # 4. Create Islamic-specific prompt
            prompt = self.create_islamic_prompt(subject, question, context)
            
            # 5. Generate response
            gen_start = time.time()
            answer = self._generate_islamic_response(prompt)
            gen_time = time.time() - gen_start
            
            # 6. Format response dengan Islamic enhancements
            sources = self._format_islamic_sources(search_results)
            
            total_time = time.time() - start_time
            
            logger.info(f"🕌 Islamic query completed: {total_time:.2f}s "
                       f"(search: {search_time:.2f}s, gen: {gen_time:.2f}s)")
            
            return {
                "answer": answer,
                "sources": sources,
                "question": question,
                "subject": subject,
                "processing_time": round(total_time, 2),
                "search_time": round(search_time, 2),
                "generation_time": round(gen_time, 2),
                "islamic_content": subject in ISLAMIC_SUBJECTS,
                "status": "success"
            }
            
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"Islamic query failed in {error_time:.2f}s: {e}")
            
            return {
                "answer": "Maaf, terjadi kesalahan dalam memproses pertanyaan. Silakan coba lagi.",
                "sources": [],
                "question": question,
                "subject": "unknown",
                "processing_time": round(error_time, 2),
                "islamic_content": False,
                "status": "error",
                "error": str(e)
            }
    
    def _prepare_islamic_context(self, search_results: tuple, subject: str) -> str:
        """Prepare context dengan Islamic formatting"""
        if not search_results:
            return "Tidak ada konteks tersedia."
        
        context_parts = []
        total_chars = 0
        max_chars = 1000 if subject in ISLAMIC_SUBJECTS else 800
        
        for content, source, page, doc_subject in search_results:
            # Process Islamic content if needed
            if subject in ISLAMIC_SUBJECTS:
                content = self.process_islamic_content(content, subject)
            
            formatted_content = f"[{source}] {content}"
            
            if total_chars + len(formatted_content) > max_chars:
                break
            
            context_parts.append(formatted_content)
            total_chars += len(formatted_content)
        
        return "\n\n".join(context_parts)
    
    def _generate_islamic_response(self, prompt: str) -> str:
        """Generate response dengan Islamic content awareness"""
        try:
            result = self.llm.invoke(prompt)
            
            # Clean dan enhance response
            cleaned = result.strip()
            
            # Remove any repeated prompt parts
            if "Jawaban:" in cleaned:
                cleaned = cleaned.split("Jawaban:")[-1].strip()
            
            return cleaned if cleaned else "Maaf, tidak dapat memberikan jawaban yang sesuai."
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "Terjadi kesalahan dalam menghasilkan jawaban."
    
    def _format_islamic_sources(self, search_results: tuple) -> List[Dict[str, Any]]:
        """Format sources dengan Islamic context dan link ke file PDF"""
        sources = []

        for content, source, page, subject in search_results:
            source_url = f"http://localhost:8000/pdf/{source}" if source else ""

            source_info = {
                "source": source,
                "source_url": source_url,
                "page": page,
                "subject": subject,
                "content": content[:200] + "..." if len(content) > 200 else content,
                "islamic_content": subject in ISLAMIC_SUBJECTS if subject else False
            }
            sources.append(source_info)

        return sources

# Global singleton untuk Islamic RAG
_islamic_rag_instance = None
_lock = threading.Lock()

def get_islamic_rag_engine():
    """Get Islamic RAG engine singleton"""
    global _islamic_rag_instance, _lock
    
    with _lock:
        if _islamic_rag_instance is None:
            _islamic_rag_instance = IslamicFastRAGEngine()
        return _islamic_rag_instance

def islamic_ultra_fast_get_answer(question: str, use_cache: bool = True) -> Dict[str, Any]:
    """Main interface untuk Islamic ultra-fast answers"""
    
    if not question or not question.strip():
        return {
            "answer": "Pertanyaan tidak boleh kosong.",
            "sources": [],
            "question": question,
            "subject": "unknown",
            "processing_time": 0.0,
            "islamic_content": False,
            "status": "error"
        }
    
    try:
        engine = get_islamic_rag_engine()
        result = engine.ultra_fast_islamic_query(question)
        return result
        
    except Exception as e:
        logger.error(f"Islamic answer failed: {e}")
        return {
            "answer": "Sistem sedang mengalami masalah. Silakan coba lagi.",
            "sources": [],
            "question": question,
            "subject": "unknown", 
            "processing_time": 0.0,
            "islamic_content": False,
            "status": "error",
            "error": str(e)
        }

def clear_islamic_cache():
    """Clear Islamic performance cache"""
    engine = get_islamic_rag_engine()
    engine.islamic_similarity_search.cache_clear()
    logger.info("Islamic performance caches cleared")

def get_islamic_performance_stats():
    """Get Islamic performance statistics"""
    engine = get_islamic_rag_engine()
    cache_info = engine.islamic_similarity_search.cache_info()
    
    hit_rate = cache_info.hits / (cache_info.hits + cache_info.misses) if (cache_info.hits + cache_info.misses) > 0 else 0
    
    return {
        "cache_hits": cache_info.hits,
        "cache_misses": cache_info.misses,
        "cache_hit_rate": round(hit_rate, 3),
        "cache_size": cache_info.currsize,
        "is_initialized": engine.is_initialized,
        "islamic_optimizations": True,
        "supported_subjects": list(ISLAMIC_SUBJECTS.keys()) + list(GENERAL_SUBJECTS.keys())
    }

# Export
__all__ = ['islamic_ultra_fast_get_answer', 'clear_islamic_cache', 'get_islamic_performance_stats']