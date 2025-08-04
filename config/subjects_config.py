from typing import Dict, List

class SubjectsConfig:
    """Configuration untuk 12 mata pelajaran Islamic curriculum"""

    def __init__(self):
        self.SUBJECTS_CONFIG = {
            # ========================================
            # ISLAMIC STUDIES (5 subjects)
            # ========================================
            'akidah': {
                'name': 'Akidah',
                'category': 'islamic_studies',
                'folder': '01_akidah',
                'keywords': ['akidah', 'aqidah', 'tauhid', 'iman', 'rukun iman', 'asmaul husna'],
                'chunk_size': 800,
                'chunk_overlap': 150,
                'arabic_content': True,
                'prompt_style': 'ustadz',
                'expected_pdfs': 5,
                'language': 'indonesian',
                'difficulty': 'intermediate',
                'content_types': ['theory', 'dalil', 'examples'],
                'special_processing': {
                    'preserve_dalil': True,
                    'detect_quran_references': True,
                    'detect_hadith_references': True
                }
            },
            'ilmu_quran': {
                'name': 'Ilmu Quran',
                'category': 'islamic_studies',
                'folder': '02_ilmu_quran',
                'keywords': ['quran', 'qur\'an', 'tajwid', 'qiraah', 'tafsir', 'makki', 'madani'],
                'chunk_size': 600,
                'chunk_overlap': 100,
                'arabic_content': True,
                'prompt_style': 'ustadz_quran',
                'expected_pdfs': 5,
                'language': 'mixed',
                'difficulty': 'advanced',
                'content_types': ['tajwid_rules', 'tafsir', 'qiraah', 'ulumul_quran'],
                'special_processing': {
                    'preserve_verses': True,
                    'preserve_arabic_text': True,
                    'detect_verse_numbers': True,
                    'preserve_tajwid_symbols': True
                }
            },
            'bahasa_arab': {
                'name': 'Bahasa Arab',
                'category': 'islamic_studies',
                'folder': '03_bahasa_arab',
                'keywords': ['arab', 'nahwu', 'sharaf', 'balaghah', 'i\'rab', 'fi\'il'],
                'chunk_size': 500,
                'chunk_overlap': 100,
                'arabic_content': True,
                'prompt_style': 'ustadz_arabic',
                'expected_pdfs': 5,
                'language': 'arabic',
                'difficulty': 'advanced',
                'content_types': ['grammar', 'vocabulary', 'exercises', 'examples'],
                'special_processing': {
                    'preserve_grammar_examples': True,
                    'preserve_arabic_formatting': True,
                    'detect_grammatical_terms': True,
                    'preserve_i_rab': True
                }
            },
            'fikih': {
                'name': 'Fikih',
                'category': 'islamic_studies',
                'folder': '05_fikih',
                'keywords': ['fikih', 'fiqh', 'hukum islam', 'halal', 'haram', 'wudhu', 'shalat'],
                'chunk_size': 900,
                'chunk_overlap': 200,
                'arabic_content': True,
                'prompt_style': 'ustadz_fiqh',
                'expected_pdfs': 5,
                'language': 'indonesian',
                'difficulty': 'intermediate',
                'content_types': ['hukum', 'dalil', 'praktik', 'contoh_kasus'],
                'special_processing': {
                    'preserve_hukum_context': True,
                    'detect_legal_terms': True,
                    'preserve_dalil_chains': True,
                    'categorize_by_topic': True
                }
            },
            'sejarah_kebudayaan_islam': {
                'name': 'Sejarah Kebudayaan Islam',
                'category': 'islamic_studies',
                'folder': '04_sejarah_kebudayaan_islam',
                'keywords': ['sejarah islam', 'khulafaur', 'nabi', 'sahabat', "tabi'in"],
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'arabic_content': False,
                'prompt_style': 'ustadz_history',
                'expected_pdfs': 4,
                'language': 'indonesian',
                'difficulty': 'intermediate',
                'content_types': ['chronology', 'biography', 'events', 'culture'],
                'special_processing': {
                    'preserve_chronology': True,
                    'detect_historical_figures': True,
                    'preserve_dates': True,
                    'categorize_by_period': True
                }
            },

            # ========================================
            # GENERAL EDUCATION (7 subjects)
            # ========================================
            'matematika': {
                'name': 'Matematika',
                'category': 'general_education',
                'folder': '06_matematika',
                'keywords': ['matematika', 'rumus', 'hitung', 'geometri', 'aljabar'],
                'chunk_size': 700,
                'chunk_overlap': 150,
                'arabic_content': False,
                'prompt_style': 'teacher',
                'expected_pdfs': 4,
                'language': 'indonesian',
                'difficulty': 'intermediate',
                'content_types': ['formulas', 'examples', 'exercises', 'theory'],
                'special_processing': {
                    'preserve_formulas': True,
                    'preserve_calculations': True,
                    'detect_mathematical_symbols': True,
                    'step_by_step_solutions': True
                }
            },
            'seni_budaya': {
                'name': 'Seni Budaya',
                'category': 'general_education',
                'folder': '07_seni_budaya',
                'keywords': ['seni', 'budaya', 'lukis', 'musik', 'tari', 'kebudayaan'],
                'chunk_size': 800,
                'chunk_overlap': 150,
                'arabic_content': False,
                'prompt_style': 'teacher',
                'expected_pdfs': 3,
                'language': 'indonesian',
                'difficulty': 'beginner',
                'content_types': ['theory', 'techniques', 'history', 'appreciation'],
                'special_processing': {
                    'preserve_cultural_context': True,
                    'detect_art_terminology': True,
                    'preserve_historical_periods': True,
                    'categorize_by_art_form': True
                }
            },
            'bahasa_inggris': {
                'name': 'Bahasa Inggris',
                'category': 'general_education',
                'folder': '08_bahasa_inggris',
                'keywords': ['english', 'inggris', 'grammar', 'vocabulary', 'tenses'],
                'chunk_size': 600,
                'chunk_overlap': 100,
                'arabic_content': False,
                'prompt_style': 'english_teacher',
                'expected_pdfs': 4,
                'language': 'english',
                'difficulty': 'intermediate',
                'content_types': ['grammar', 'vocabulary', 'reading', 'writing'],
                'special_processing': {
                    'preserve_grammar_rules': True,
                    'preserve_example_sentences': True,
                    'detect_language_patterns': True,
                    'categorize_by_skill': True
                }
            },
            'pjok': {
                'name': 'PJOK (Pendidikan Jasmani, Olahraga, dan Kesehatan)',
                'category': 'general_education',
                'folder': '09_pjok',
                'keywords': ['pjok', 'olahraga', 'kesehatan', 'jasmani', 'sport'],
                'chunk_size': 800,
                'chunk_overlap': 150,
                'arabic_content': False,
                'prompt_style': 'teacher',
                'expected_pdfs': 3,
                'language': 'indonesian',
                'difficulty': 'beginner',
                'content_types': ['techniques', 'health', 'rules', 'safety'],
                'special_processing': {
                    'preserve_exercise_steps': True,
                    'preserve_health_guidelines': True,
                    'detect_sports_terminology': True,
                    'categorize_by_sport_type': True
                }
            },
            'ppkn': {
                'name': 'PPKn (Pendidikan Pancasila dan Kewarganegaraan)',
                'category': 'general_education',
                'folder': '10_ppkn',
                'keywords': ['ppkn', 'pancasila', 'kewarganegaraan', 'civics', 'negara'],
                'chunk_size': 800,
                'chunk_overlap': 150,
                'arabic_content': False,
                'prompt_style': 'civics_teacher',
                'expected_pdfs': 3,
                'language': 'indonesian',
                'difficulty': 'intermediate',
                'content_types': ['theory', 'law', 'citizenship', 'values'],
                'special_processing': {
                    'preserve_legal_context': True,
                    'detect_civic_terminology': True,
                    'preserve_constitutional_references': True,
                    'categorize_by_topic': True
                }
            },
            'prakarya': {
                'name': 'Prakarya dan Kewirausahaan',
                'category': 'general_education',
                'folder': '11_prakarya',
                'keywords': ['prakarya', 'kerajinan', 'craft', 'keterampilan', 'wirausaha'],
                'chunk_size': 800,
                'chunk_overlap': 150,
                'arabic_content': False,
                'prompt_style': 'teacher',
                'expected_pdfs': 3,
                'language': 'indonesian',
                'difficulty': 'beginner',
                'content_types': ['techniques', 'materials', 'projects', 'business'],
                'special_processing': {
                    'preserve_step_by_step': True,
                    'preserve_material_lists': True,
                    'detect_craft_terminology': True,
                    'categorize_by_craft_type': True
                }
            },
            'sejarah_indonesia': {
                'name': 'Sejarah Indonesia',
                'category': 'general_education',
                'folder': '12_sejarah_indonesia',
                'keywords': ['sejarah indonesia', 'kemerdekaan', 'soekarno', 'pancasila', 'nusantara'],
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'arabic_content': False,
                'prompt_style': 'history_teacher',
                'expected_pdfs': 3,
                'language': 'indonesian',
                'difficulty': 'intermediate',
                'content_types': ['chronology', 'biography', 'events', 'analysis'],
                'special_processing': {
                    'preserve_chronology': True,
                    'detect_historical_figures': True,
                    'preserve_dates_and_periods': True,
                    'categorize_by_era': True
                }
            },
            'bahasa_indonesia': {
                'name': 'Bahasa Indonesia',
                'category': 'general_education',
                'folder': '13_bahasa_indonesia',
                'keywords': ['bahasa indonesia', 'sastra', 'tata bahasa', 'kosa kata'],
                'chunk_size': 600,
                'chunk_overlap': 100,
                'arabic_content': False,
                'prompt_style': 'indonesian_teacher',
                'expected_pdfs': 3,
                'language': 'indonesian',
                'difficulty': 'intermediate',
                'content_types': ['grammar', 'literature', 'vocabulary', 'writing'],
                'special_processing': {
                    'preserve_grammar_rules': True,
                    'preserve_literary_examples': True,
                    'detect_language_patterns': True,
                    'categorize_by_genre': True
                }
            }
        }

        self.ISLAMIC_STUDIES = ['akidah', 'ilmu_quran', 'bahasa_arab', 'fikih', 'sejarah_kebudayaan_islam']
        self.GENERAL_EDUCATION = ['matematika', 'seni_budaya', 'bahasa_inggris', 'pjok', 'ppkn', 'prakarya', 'sejarah_indonesia', 'bahasa_indonesia']
        self.ALL_SUBJECTS = self.ISLAMIC_STUDIES + self.GENERAL_EDUCATION
        self.TOTAL_EXPECTED_PDFS = sum(config['expected_pdfs'] for config in self.SUBJECTS_CONFIG.values())
        self.FOLDER_TO_SUBJECT = {config['folder']: subject for subject, config in self.SUBJECTS_CONFIG.items()}
        self.KEYWORDS_TO_SUBJECT = {}
        for subject, config in self.SUBJECTS_CONFIG.items():
            for keyword in config['keywords']:
                self.KEYWORDS_TO_SUBJECT.setdefault(keyword, []).append(subject)

    def get_subject_config(self, subject: str) -> Dict:
        return self.SUBJECTS_CONFIG.get(subject, {})

    def get_all_subjects(self) -> List[str]:
        return self.ALL_SUBJECTS

    def get_islamic_subjects(self) -> List[str]:
        return self.ISLAMIC_STUDIES

    def get_general_subjects(self) -> List[str]:
        return self.GENERAL_EDUCATION

    def is_islamic_subject(self, subject: str) -> bool:
        return subject in self.ISLAMIC_STUDIES

    def detect_subject_from_keywords(self, text: str) -> str:
        text_lower = text.lower()
        subject_scores = {}
        for keyword, subjects in self.KEYWORDS_TO_SUBJECT.items():
            if keyword in text_lower:
                for subject in subjects:
                    subject_scores[subject] = subject_scores.get(subject, 0) + 1
        return max(subject_scores, key=subject_scores.get) if subject_scores else 'unknown'

    def detect_subject_from_folder(self, folder_path: str) -> str:
        for folder, subject in self.FOLDER_TO_SUBJECT.items():
            if folder in folder_path:
                return subject
        return 'unknown'

    def get_chunk_config(self, subject: str) -> Dict:
        config = self.get_subject_config(subject)
        return {
            'chunk_size': config.get('chunk_size', 800),
            'chunk_overlap': config.get('chunk_overlap', 150),
            'special_processing': config.get('special_processing', {})
        }

    def get_expected_pdf_count(self, subject: str = None) -> int:
        if subject:
            return self.SUBJECTS_CONFIG.get(subject, {}).get('expected_pdfs', 0)
        return self.TOTAL_EXPECTED_PDFS

    def validate_subjects_config(self):
        if len(self.ALL_SUBJECTS) != 12:
            raise ValueError(f"Expected 12 subjects, got {len(self.ALL_SUBJECTS)}")
        if len(self.ISLAMIC_STUDIES) != 5:
            raise ValueError(f"Expected 5 Islamic studies subjects, got {len(self.ISLAMIC_STUDIES)}")
        if len(self.GENERAL_EDUCATION) != 7:
            raise ValueError(f"Expected 7 general education subjects, got {len(self.GENERAL_EDUCATION)}")
        required_fields = ['name', 'category', 'folder', 'keywords', 'chunk_size', 'expected_pdfs']
        for subject, config in self.SUBJECTS_CONFIG.items():
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing field '{field}' in subject '{subject}' configuration")
            if not (100 <= config['chunk_size'] <= 2000):
                raise ValueError(f"Invalid chunk_size for subject '{subject}': {config['chunk_size']}")
            if config['expected_pdfs'] <= 0:
                raise ValueError(f"Invalid expected_pdfs for subject '{subject}': {config['expected_pdfs']}")
        if self.TOTAL_EXPECTED_PDFS != 41:
            raise ValueError(f"Total expected PDFs should be 41, got {self.TOTAL_EXPECTED_PDFS}")
        print("✅ Subjects configuration validated")
        return True
