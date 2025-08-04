# backend/subject_detector.py - Subject detection service

import re
import logging
from typing import Dict, List, Optional, Tuple
from collections import Counter

logger = logging.getLogger(__name__)

class SubjectDetector:
    """Intelligent subject detection untuk 12 mata pelajaran"""
    
    def __init__(self):
        # Subject keywords dengan weights
        self.subject_keywords = {
            # Islamic Studies
            'akidah': {
                'primary': ['akidah', 'aqidah', 'tauhid', 'iman', 'rukun iman'],
                'secondary': ['allah', 'asmaul husna', 'malaikat', 'kitab', 'rasul'],
                'weight_multiplier': 2.0  # Higher weight untuk exact matches
            },
            'ilmu_quran': {
                'primary': ['quran', 'qur\'an', 'al-quran', 'tajwid', 'qiraah'],
                'secondary': ['tafsir', 'makki', 'madani', 'ayat', 'surah'],
                'weight_multiplier': 2.0
            },
            'bahasa_arab': {
                'primary': ['arab', 'nahwu', 'sharaf', 'balaghah', 'i\'rab'],
                'secondary': ['fi\'il', 'isim', 'harf', 'mubtada', 'khabar'],
                'weight_multiplier': 2.0
            },
            'fikih': {
                'primary': ['fikih', 'fiqh', 'hukum islam', 'halal', 'haram'],
                'secondary': ['wudhu', 'shalat', 'puasa', 'zakat', 'haji'],
                'weight_multiplier': 2.0
            },
            'sejarah_kebudayaan_islam': {
                'primary': ['sejarah islam', 'khulafaur', 'nabi muhammad', 'sahabat'],
                'secondary': ['tabi\'in', 'dinasti', 'peradaban islam', 'hijrah'],
                'weight_multiplier': 2.0
            },
            
            # General Education
            'matematika': {
                'primary': ['matematika', 'rumus', 'persamaan', 'geometri', 'aljabar'],
                'secondary': ['hitung', 'bilangan', 'grafik', 'statistik', 'trigonometri'],
                'weight_multiplier': 1.5
            },
            'seni_budaya': {
                'primary': ['seni', 'budaya', 'lukis', 'musik', 'tari'],
                'secondary': ['kebudayaan', 'tradisional', 'karya seni', 'estetika'],
                'weight_multiplier': 1.5
            },
            'bahasa_inggris': {
                'primary': ['english', 'inggris', 'grammar', 'vocabulary', 'tenses'],
                'secondary': ['speaking', 'listening', 'reading', 'writing', 'conversation'],
                'weight_multiplier': 1.5
            },
            'pjok': {
                'primary': ['pjok', 'olahraga', 'jasmani', 'kesehatan', 'sport'],
                'secondary': ['atletik', 'senam', 'permainan', 'kebugaran', 'gizi'],
                'weight_multiplier': 1.5
            },
            'ppkn': {
                'primary': ['ppkn', 'pancasila', 'kewarganegaraan', 'civics', 'negara'],
                'secondary': ['demokrasi', 'hak', 'kewajiban', 'konstitusi', 'bhinneka'],
                'weight_multiplier': 1.5
            },
            'prakarya': {
                'primary': ['prakarya', 'kerajinan', 'craft', 'keterampilan', 'wirausaha'],
                'secondary': ['kreativitas', 'inovasi', 'produk', 'bahan', 'teknik'],
                'weight_multiplier': 1.5
            },
            'sejarah_indonesia': {
                'primary': ['sejarah indonesia', 'nusantara', 'kerajaan', 'peristiwa'],
                'secondary': ['proklamasi', 'revolusi', 'pahlawan', 'kolonialisme'],
                'weight_multiplier': 1.5
            },
            'bahasa_indonesia': {
                'primary': ['bahasa indonesia', 'sastra', 'tata bahasa', 'kosa kata'],
                'secondary': ['puisi', 'cerpen', 'novel', 'drama', 'teks'],
                'weight_multiplier': 1.5
            }
        }
        # Precompile regex patterns untuk efisiensi
        self.subject_patterns = {
            subject: re.compile(r'\b(?:' + '|'.join(data['primary'] + data['secondary']) + r')\b', re.IGNORECASE)
            for subject, data in self.subject_keywords.items()
        }