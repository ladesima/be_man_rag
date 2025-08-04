# backend/document_processor.py - Document processing service

import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator

import fitz  # PyMuPDF
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .metadata_extractor import MetadataExtractor
from .islamic_text_processor import IslamicTextProcessor
from .text_splitter import AdvancedTextSplitter

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Advanced document processing untuk Islamic curriculum"""
    
    def __init__(self):
        self.metadata_extractor = MetadataExtractor()
        self.islamic_processor = IslamicTextProcessor()
        self.text_splitter = AdvancedTextSplitter()
        
        # Processing statistics
        self.stats = {
            "total_pdfs_processed": 0,
            "total_pages_processed": 0,
            "total_chunks_created": 0,
            "total_processing_time": 0,
            "islamic_content_detected": 0,
            "arabic_text_found": 0
        }
    
    def process_pdf_file(self, pdf_path: str, subject_hint: Optional[str] = None) -> List[Document]:
        """Process single PDF file"""
        start_time = time.time()
        
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Extract basic metadata
            file_metadata = self.metadata_extractor.extract_file_metadata(pdf_path)
            
            # Open PDF
            pdf_doc = fitz.open(pdf_path)
            documents = []
            
            logger.info(f"Processing PDF: {os.path.basename(pdf_path)} ({len(pdf_doc)} pages)")
            
            for page_num in range(len(pdf_doc)):
                try:
                    page = pdf_doc[page_num]
                    text = page.get_text()
                    
                    if not text.strip():
                        continue
                    
                    # Clean and process text
                    cleaned_text = self._clean_extracted_text(text)
                    
                    if len(cleaned_text.strip()) < 50:  # Skip very short content
                        continue
                    
                    # Detect subject if not provided
                    detected_subject = self._detect_subject(cleaned_text, pdf_path, subject_hint)
                    
                    # Process Islamic content if detected
                    if self._is_islamic_content(detected_subject, cleaned_text):
                        cleaned_text = self.islamic_processor.process_islamic_text(cleaned_text)
                        self.stats["islamic_content_detected"] += 1
                    
                    # Create document with enhanced metadata
                    doc_metadata = {
                        **file_metadata,
                        "page": page_num + 1,
                        "subject": detected_subject,
                        "page_content_length": len(cleaned_text),
                        "estimated_tokens": self._estimate_tokens(cleaned_text),
                        "has_arabic": self.islamic_processor.has_arabic_text(cleaned_text),
                        "processing_timestamp": time.time()
                    }
                    
                    if doc_metadata["has_arabic"]:
                        self.stats["arabic_text_found"] += 1
                    
                    document = Document(
                        page_content=cleaned_text,
                        metadata=doc_metadata
                    )
                    
                    documents.append(document)
                    self.stats["total_pages_processed"] += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing page {page_num + 1} of {pdf_path}: {e}")
                    continue
            
            pdf_doc.close()
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats["total_pdfs_processed"] += 1
            self.stats["total_processing_time"] += processing_time
            
            logger.info(f"Processed {pdf_path}: {len(documents)} pages in {processing_time:.2f}s")
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            return []
    
    def process_pdf_folder(self, folder_path: str, batch_size: int = 10) -> Generator[List[Document], None, None]:
        """Process folder of PDFs in batches"""
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        # Get all PDF files
        pdf_files = list(folder_path.rglob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {folder_path}")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process in batches
        for i in range(0, len(pdf_files), batch_size):
            batch = pdf_files[i:i + batch_size]
            batch_documents = []
            
            logger.info(f"Processing batch {i // batch_size + 1}/{(len(pdf_files) + batch_size - 1) // batch_size}")
            
            for pdf_file in batch:
                try:
                    # Detect subject from folder structure
                    subject_hint = self._detect_subject_from_path(str(pdf_file))
                    
                    # Process PDF
                    documents = self.process_pdf_file(str(pdf_file), subject_hint)
                    batch_documents.extend(documents)
                    
                except Exception as e:
                    logger.error(f"Failed to process {pdf_file}: {e}")
                    continue
            
            if batch_documents:
                yield batch_documents
    
    def chunk_documents(self, documents: List[Document], subject: Optional[str] = None) -> List[Document]:
        """Split documents into optimized chunks"""
        try:
            chunked_documents = []
            
            for doc in documents:
                # Get subject-specific chunking config
                doc_subject = doc.metadata.get('subject', subject or 'general')
                
                # Chunk the document
                chunks = self.text_splitter.split_document(doc, doc_subject)
                
                # Add chunk-specific metadata
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        "chunk_id": i,
                        "total_chunks": len(chunks),
                        "chunk_method": "subject_optimized"
                    })
                
                chunked_documents.extend(chunks)
                self.stats["total_chunks_created"] += len(chunks)
            
            logger.info(f"Created {len(chunked_documents)} chunks from {len(documents)} documents")
            
            return chunked_documents
            
        except Exception as e:
            logger.error(f"Failed to chunk documents: {e}")
            return documents
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean extracted PDF text"""
        # Remove non-breaking spaces and excessive whitespace
        text = text.replace('\xa0', ' ')
        text = text.replace('\ufeff', '')  # Remove BOM
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            cleaned_line = line.strip()
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    
    def _detect_subject(self, text: str, file_path: str, subject_hint: Optional[str] = None) -> str:
        """Detect subject from text content and file path"""
        if subject_hint:
            return subject_hint
        
        # Try to detect from path first
        subject_from_path = self._detect_subject_from_path(file_path)
        if subject_from_path != 'unknown':
            return subject_from_path
        
        # Detect from content
        from .subject_detector import SubjectDetector
        detector = SubjectDetector()
        return detector.detect_subject_from_text(text)
    
    def _detect_subject_from_path(self, file_path: str) -> str:
        """Detect subject from file path"""
        path_lower = file_path.lower()
        
        # Subject mapping based on folder structure
        subject_indicators = {
            '01_akidah': 'akidah',
            '02_ilmu_quran': 'ilmu_quran',
            '03_bahasa_arab': 'bahasa_arab',
            '04_sejarah_kebudayaan_islam': 'sejarah_kebudayaan_islam',
            '05_fikih': 'fikih',
            '06_matematika': 'matematika',
            '07_seni_budaya': 'seni_budaya',
            '08_bahasa_inggris': 'bahasa_inggris',
            '09_pjok': 'pjok',
            '10_ppkn': 'ppkn',
            '11_prakarya': 'prakarya',
            '12_sejarah_indonesia': 'sejarah_indonesia',
            '13_bahasa_indonesia': 'bahasa_indonesia'
        }
        
        for indicator, subject in subject_indicators.items():
            if indicator in path_lower:
                return subject
        
        return 'unknown'
    
    def _is_islamic_content(self, subject: str, text: str) -> bool:
        """Check if content is Islamic"""
        islamic_subjects = ['akidah', 'ilmu_quran', 'bahasa_arab', 'fikih', 'sejarah_kebudayaan_islam']
        
        if subject in islamic_subjects:
            return True
        
        # Check content for Islamic keywords
        islamic_keywords = ['islam', 'quran', 'allah', 'muhammad', 'hadith', 'shalat', 'puasa']
        text_lower = text.lower()
        
        return any(keyword in text_lower for keyword in islamic_keywords)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count"""
        # Rough estimation: 1 token ≈ 4 characters for mixed language text
        return len(text) // 4
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            **self.stats,
            "average_processing_time_per_pdf": (
                self.stats["total_processing_time"] / self.stats["total_pdfs_processed"]
                if self.stats["total_pdfs_processed"] > 0 else 0
            ),
            "average_chunks_per_pdf": (
                self.stats["total_chunks_created"] / self.stats["total_pdfs_processed"]
                if self.stats["total_pdfs_processed"] > 0 else 0
            ),
            "islamic_content_percentage": (
                (self.stats["islamic_content_detected"] / self.stats["total_pages_processed"]) * 100
                if self.stats["total_pages_processed"] > 0 else 0
            )
        }
    
    def reset_stats(self):
        """Reset processing statistics"""
        self.stats = {
            "total_pdfs_processed": 0,
            "total_pages_processed": 0,
            "total_chunks_created": 0,
            "total_processing_time": 0,
            "islamic_content_detected": 0,
            "arabic_text_found": 0
        }
        
        logger.info("Processing statistics reset")