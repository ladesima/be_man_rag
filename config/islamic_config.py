# config/islamic_config.py - Islamic curriculum specific configuration

import os
from typing import Dict, List

class IslamicConfig:
    """Configuration for Islamic curriculum features"""
    
    def __init__(self):
        # Islamic Content Processing
        self.ARABIC_TEXT_SUPPORT = os.getenv("ARABIC_TEXT_SUPPORT", "true").lower() == "true"
        self.ISLAMIC_CONTENT_DETECTION = os.getenv("ISLAMIC_CONTENT_DETECTION", "true").lower() == "true"
        self.CROSS_SUBJECT_INTEGRATION = os.getenv("CROSS_SUBJECT_INTEGRATION", "true").lower() == "true"
        
        # Islamic Subjects Configuration
        self.ISLAMIC_SUBJECTS = {
            'akidah': {
                'keywords': ['akidah', 'aqidah', 'tauhid', 'iman', 'rukun iman', 'asmaul husna'],
                'chunk_size': 800,
                'arabic_content': True,
                'special_processing': True,
                'prompt_style': 'ustadz'
            },
            'ilmu_quran': {
                'keywords': ['quran', 'qur\'an', 'tajwid', 'qiraah', 'tafsir', 'makki', 'madani'],
                'chunk_size': 600,
                'arabic_content': True,
                'special_processing': True,
                'prompt_style': 'ustadz_quran'
            },
            'bahasa_arab': {
                'keywords': ['arab', 'nahwu', 'sharaf', 'balaghah', 'i\'rab', 'fi\'il'],
                'chunk_size': 500,
                'arabic_content': True,
                'special_processing': True,
                'prompt_style': 'ustadz_arabic'
            },
            'fikih': {
                'keywords': ['fikih', 'fiqh', 'hukum islam', 'halal', 'haram', 'wudhu', 'shalat'],
                'chunk_size': 900,
                'arabic_content': True,
                'special_processing': True,
                'prompt_style': 'ustadz_fiqh'
            },
            'sejarah_kebudayaan_islam': {
                'keywords': ['sejarah islam', 'khulafaur', 'nabi', 'sahabat', 'tabi\'in'],
                'chunk_size': 1000,
                'arabic_content': False,
                'special_processing': True,
                'prompt_style': 'ustadz_history'
            }
        }
        
        # Arabic Text Processing
        self.ARABIC_PATTERNS = {
            'arabic_text': r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+',
            'quran_verse': r'﴿.*?﴾',
            'hadith_ref': r'(رواه|أخرجه|صحيح|حسن|ضعيف)',
            'transliteration': r'[A-Za-z]+\s*\([^)]*[\u0600-\u06FF][^)]*\)',
        }
        
        # Islamic Terminology
        self.ISLAMIC_TERMS = {
            'worship_terms': ['shalat', 'solat', 'puasa', 'zakat', 'haji', 'umrah', 'wudhu'],
            'belief_terms': ['allah', 'tauhid', 'iman', 'islam', 'ihsan', 'rukun'],
            'law_terms': ['halal', 'haram', 'makruh', 'sunnah', 'wajib', 'mubah'],
            'quran_terms': ['ayat', 'surah', 'juz', 'makkiyah', 'madaniyah', 'tajwid'],
            'arabic_grammar': ['nahwu', 'sharaf', 'i\'rab', 'fi\'il', 'isim', 'harf']
        }
        
        # Prompt Templates
        self.ISLAMIC_PROMPT_TEMPLATES = {
            'ustadz': """Anda adalah ustadz yang mengajar {subject}. Jawab berdasarkan ajaran Islam yang benar.

Konteks: {context}
Pertanyaan: {question}

Panduan:
1. Berikan penjelasan sesuai Al-Quran dan Sunnah
2. Sertakan dalil jika ada
3. Gunakan bahasa yang mudah dipahami
4. Hindari bid'ah dan khurafat

Jawaban:""",
            
            'ustadz_quran': """Anda adalah ustadz Ilmu Al-Quran. Berikan penjelasan yang komprehensif.

Konteks: {context}
Pertanyaan: {question}

Panduan:
1. Jelaskan dengan detail dan akurat
2. Sertakan referensi ayat jika relevan
3. Berikan contoh praktis untuk tajwid
4. Gunakan istilah Ilmu Quran yang tepat

Jawaban:""",
            
            'general_islamic': """Anda adalah guru {subject} yang mengintegrasikan nilai-nilai Islam.

Konteks: {context}
Pertanyaan: {question}

Berikan penjelasan yang jelas dan integrasikan nilai Islam jika memungkinkan.

Jawaban:"""
        }
    
        def get_subject_config(self, subject: str) -> Dict:
            return self.ISLAMIC_SUBJECTS.get(subject, {
            'chunk_size': 800,
            'arabic_content': False,
            'special_processing': False,
            'prompt_style': 'general_islamic'
        })
    
    def is_islamic_subject(self, subject: str) -> bool:
        """Check if subject is Islamic studies"""
        return subject in self.ISLAMIC_SUBJECTS
    
    def get_prompt_template(self, style: str, subject: str) -> str:
        """Get prompt template for specific style"""
        return self.ISLAMIC_PROMPT_TEMPLATES.get(style, self.ISLAMIC_PROMPT_TEMPLATES['general_islamic'])
    
    def validate_islamic_config(self):
        """Validate Islamic configuration"""
        required_subjects = ['akidah', 'ilmu_quran', 'bahasa_arab', 'fikih', 'sejarah_kebudayaan_islam']
        
        for subject in required_subjects:
            if subject not in self.ISLAMIC_SUBJECTS:
                raise ValueError(f"Islamic subject configuration missing: {subject}")
        
        print("✅ Islamic configuration validated")
        return True

    def validate_arabic_patterns(self):
        """Validate Arabic patterns"""
        import re
        for pattern_name, pattern in self.ARABIC_PATTERNS.items():
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid Arabic pattern '{pattern_name}': {e}")

        print("✅ Arabic patterns validated")
        return True
