#!/usr/bin/env python3
"""
Indonesian Configuration - Pengaturan Bahasa Indonesia
Konfigurasi khusus untuk bahasa Indonesia dalam sistem RAG kurikulum Islam
"""

import os
from typing import Dict, List

class IndonesianConfig:
    """Konfigurasi khusus bahasa Indonesia"""
    
    # ===========================================
    # LANGUAGE SETTINGS
    # ===========================================
    PRIMARY_LANGUAGE = "indonesian"
    SUPPORTED_LANGUAGES = ["indonesian"]  # Hanya bahasa Indonesia
    DEFAULT_LANGUAGE = "indonesian"
    FORCE_INDONESIAN_RESPONSE = True
    
    # ===========================================
    # INDONESIAN TEXT PROCESSING
    # ===========================================
    # Kata-kata kunci bahasa Indonesia
    INDONESIAN_KEYWORDS = [
        "apa", "siapa", "dimana", "kapan", "mengapa", "bagaimana",
        "jelaskan", "sebutkan", "berikan", "uraikan", "gambarkan",
        "bandingkan", "contoh", "definisi", "pengertian", "makna"
    ]
    
    # Stop words bahasa Indonesia
    INDONESIAN_STOPWORDS = [
        "yang", "dan", "di", "ke", "dari", "dalam", "untuk", "pada",
        "dengan", "adalah", "akan", "atau", "juga", "dapat", "bisa",
        "sudah", "telah", "seperti", "antara", "tersebut", "ini", "itu"
    ]
    
    # ===========================================
    # INDONESIAN RESPONSE FORMATTING
    # ===========================================
    # Template respons bahasa Indonesia
    RESPONSE_TEMPLATES = {
        "greeting": "Selamat datang di sistem RAG Kurikulum Islam Indonesia!",
        "no_answer": "Maaf, saya tidak dapat menemukan jawaban yang sesuai dalam dokumen yang tersedia.",
        "error": "Terjadi kesalahan dalam memproses pertanyaan Anda. Silakan coba lagi.",
        "processing": "Sedang memproses pertanyaan Anda...",
        "success": "Berikut adalah jawaban untuk pertanyaan Anda:",
        "sources": "Sumber informasi:",
        "confidence_low": "Jawaban ini memiliki tingkat kepercayaan yang rendah.",
        "confidence_high": "Jawaban ini didukung oleh sumber yang kuat."
    }
    
    # ===========================================
    # INDONESIAN QUESTION PATTERNS
    # ===========================================
    # Pola pertanyaan bahasa Indonesia
    QUESTION_PATTERNS = {
        "definition": ["apa itu", "pengertian", "definisi", "arti"],
        "explanation": ["jelaskan", "uraikan", "gambarkan", "bagaimana"],
        "enumeration": ["sebutkan", "berikan", "daftar", "list"],
        "comparison": ["bandingkan", "perbedaan", "persamaan", "versus"],
        "procedure": ["cara", "langkah", "proses", "tahapan"],
        "reason": ["mengapa", "kenapa", "alasan", "sebab"],
        "example": ["contoh", "misal", "ilustrasi", "sampel"]
    }
    
    # ===========================================
    # INDONESIAN SUBJECT TRANSLATIONS
    # ===========================================
    # Nama mata pelajaran dalam bahasa Indonesia
    SUBJECT_NAMES_INDONESIAN = {
        "akidah": "Akidah",
        "ilmu_quran": "Ilmu Al-Quran",
        "bahasa_arab": "Bahasa Arab", 
        "fikih": "Fikih",
        "sejarah_kebudayaan_islam": "Sejarah Kebudayaan Islam",
        "matematika": "Matematika",
        "seni_budaya": "Seni Budaya",
        "bahasa_inggris": "Bahasa Inggris",
        "pjok": "Pendidikan Jasmani, Olahraga dan Kesehatan",
        "ppkn": "Pendidikan Pancasila dan Kewarganegaraan",
        "prakarya": "Prakarya dan Kewirausahaan",
        "sejarah_indonesia": "Sejarah Indonesia"
    }
    
    # ===========================================
    # INDONESIAN ISLAMIC TERMINOLOGY
    # ===========================================
    # Istilah-istilah Islam dalam bahasa Indonesia
    ISLAMIC_TERMS_INDONESIAN = {
        # Akidah
        "tauhid": "Tauhid",
        "iman": "Iman", 
        "islam": "Islam",
        "ihsan": "Ihsan",
        "rukun_iman": "Rukun Iman",
        "rukun_islam": "Rukun Islam",
        
        # Ibadah
        "shalat": "Shalat",
        "puasa": "Puasa",
        "zakat": "Zakat",
        "haji": "Haji",
        "umrah": "Umrah",
        "wudhu": "Wudhu",
        
        # Al-Quran
        "quran": "Al-Quran",
        "ayat": "Ayat",
        "surah": "Surah",
        "tajwid": "Tajwid",
        "tafsir": "Tafsir",
        
        # Fikih
        "halal": "Halal",
        "haram": "Haram",
        "makruh": "Makruh",
        "sunnah": "Sunnah",
        "wajib": "Wajib",
        "mubah": "Mubah"
    }
    
    # ===========================================
    # INDONESIAN NUMBER FORMATTING
    # ===========================================
    # Format angka bahasa Indonesia
    NUMBER_FORMATTING = {
        "decimal_separator": ",",
        "thousands_separator": ".",
        "currency_symbol": "Rp",
        "percentage_symbol": "%"
    }
    
    # ===========================================
    # INDONESIAN DATE/TIME FORMATTING
    # ===========================================
    # Format tanggal dan waktu bahasa Indonesia
    DATETIME_FORMATTING = {
        "date_format": "%d/%m/%Y",
        "time_format": "%H:%M:%S",
        "datetime_format": "%d/%m/%Y %H:%M:%S",
        "month_names": [
            "Januari", "Februari", "Maret", "April", "Mei", "Juni",
            "Juli", "Agustus", "September", "Oktober", "November", "Desember"
        ],
        "day_names": [
            "Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"
        ]
    }
    
    # ===========================================
    # INDONESIAN VALIDATION RULES
    # ===========================================
    def validate_indonesian_text(self, text: str) -> bool:
        """Validasi apakah teks menggunakan bahasa Indonesia"""
        if not text:
            return False
        
        text_lower = text.lower()
        
        # Cek kata kunci bahasa Indonesia
        indonesian_word_count = sum(
            1 for keyword in self.INDONESIAN_KEYWORDS 
            if keyword in text_lower
        )
        
        # Minimal 10% kata harus bahasa Indonesia
        words = text_lower.split()
        if len(words) > 0:
            indonesian_ratio = indonesian_word_count / len(words)
            return indonesian_ratio >= 0.1
        
        return True
    
    def format_indonesian_response(self, text: str) -> str:
        """Format respons dalam gaya bahasa Indonesia yang baik"""
        if not text:
            return self.RESPONSE_TEMPLATES["no_answer"]
        
        # Pastikan kapitalisasi yang benar
        sentences = text.split('. ')
        formatted_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Kapitalisasi awal kalimat
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                formatted_sentences.append(sentence)
        
        return '. '.join(formatted_sentences)
    
    def detect_question_type(self, question: str) -> str:
        """Deteksi jenis pertanyaan dalam bahasa Indonesia"""
        question_lower = question.lower()
        
        for question_type, patterns in self.QUESTION_PATTERNS.items():
            if any(pattern in question_lower for pattern in patterns):
                return question_type
        
        return "general"
    
    def get_subject_display_name(self, subject_code: str) -> str:
        """Get nama tampilan mata pelajaran dalam bahasa Indonesia"""
        return self.SUBJECT_NAMES_INDONESIAN.get(subject_code, subject_code.title())
    
    def validate(self) -> bool:
        """Validasi konfigurasi bahasa Indonesia"""
        # Pastikan primary language adalah Indonesian
        if self.PRIMARY_LANGUAGE != "indonesian":
            return False
        
        # Pastikan hanya mendukung bahasa Indonesia
        if len(self.SUPPORTED_LANGUAGES) != 1 or self.SUPPORTED_LANGUAGES[0] != "indonesian":
            return False
        
        # Cek template respons
        required_templates = ["greeting", "no_answer", "error", "success"]
        for template in required_templates:
            if template not in self.RESPONSE_TEMPLATES:
                return False
        
        return True
    
    def get_indonesian_summary(self) -> dict:
        """Get ringkasan konfigurasi bahasa Indonesia"""
        return {
            "bahasa_utama": self.PRIMARY_LANGUAGE,
            "bahasa_yang_didukung": self.SUPPORTED_LANGUAGES,
            "paksa_respons_indonesia": self.FORCE_INDONESIAN_RESPONSE,
            "jumlah_kata_kunci": len(self.INDONESIAN_KEYWORDS),
            "jumlah_stopwords": len(self.INDONESIAN_STOPWORDS),
            "jumlah_pola_pertanyaan": len(self.QUESTION_PATTERNS),
            "jumlah_mata_pelajaran": len(self.SUBJECT_NAMES_INDONESIAN),
            "jumlah_istilah_islam": len(self.ISLAMIC_TERMS_INDONESIAN)
        }