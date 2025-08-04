"""
Pembangun Database Vektor Studiva.AI (Versi Final dengan Format Nama File Baru)
==================================================================================
- Disesuaikan untuk format nama file 'kelas_namapelajaran.pdf' (contoh: '10_matematika.pdf').
"""
import os
import shutil
import logging
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import tiktoken

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config import config

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def semantic_chunker(elements, config_obj):
    """
    Menggabungkan elemen dari Unstructured dan menyimpan halaman sebagai string yang dipisahkan koma.
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    chunks = []
    current_chunk_elements = []
    current_chunk_tokens = 0
    current_chunk_pages = set()
    for element in elements:
        element_text = str(element.page_content)
        if not element_text.strip():
            continue
        element_tokens = len(tokenizer.encode(element_text))
        element_page = element.metadata.get("page_number")
        if current_chunk_tokens + element_tokens > config_obj.CHUNK_SIZE and current_chunk_elements:
            combined_text = "\n\n".join([str(el.page_content) for el in current_chunk_elements])
            first_element_metadata = current_chunk_elements[0].metadata
            pages_str = ",".join(map(str, sorted(list(current_chunk_pages))))
            chunk_metadata = {
                "source": first_element_metadata.get("source", ""),
                "pages": pages_str,
                "chunk_size_tokens": current_chunk_tokens
            }
            chunks.append(Document(page_content=combined_text, metadata=chunk_metadata))
            current_chunk_elements = [element]
            current_chunk_tokens = element_tokens
            current_chunk_pages = {element_page} if element_page is not None else set()
        else:
            current_chunk_elements.append(element)
            current_chunk_tokens += element_tokens
            if element_page is not None:
                current_chunk_pages.add(element_page)
    if current_chunk_elements:
        combined_text = "\n\n".join([str(el.page_content) for el in current_chunk_elements])
        first_element_metadata = current_chunk_elements[0].metadata
        pages_str = ",".join(map(str, sorted(list(current_chunk_pages))))
        chunk_metadata = {
            "source": first_element_metadata.get("source", ""),
            "pages": pages_str,
            "chunk_size_tokens": current_chunk_tokens
        }
        chunks.append(Document(page_content=combined_text, metadata=chunk_metadata))
    return chunks

def get_embedding_model(config_obj):
    logger.info("Menginisialisasi Model Embedding...")
    logger.info(f"  Model: {config_obj.EMBEDDING_MODEL_NAME}")
    logger.info(f"  Device: {config_obj.EMBEDDING_DEVICE}")
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name=config_obj.EMBEDDING_MODEL_NAME,
            cache_folder=str(config_obj.MODEL_CACHE_DIR),
            model_kwargs={"device": config_obj.EMBEDDING_DEVICE}
        )
        logger.info("  [OK] Model Embedding berhasil dimuat.")
        return embedding_model
    except Exception as e:
        logger.error(f"  [ERROR] Gagal inisialisasi model embedding: {e}", exc_info=True)
        raise

def process_and_chunk_pdf(pdf_file: Path, pdf_base_path: Path, config_obj) -> list[Document]:
    """Memproses satu file PDF dan mengembalikan chunks-nya."""
    try:
        loader = UnstructuredPDFLoader(str(pdf_file), mode="elements", strategy="fast")
        raw_elements = loader.load()
        document_chunks = semantic_chunker(raw_elements, config_obj)
        for doc in document_chunks:
            relative_path = pdf_file.relative_to(pdf_base_path)
            doc.metadata['source_path'] = str(relative_path).replace('\\', '/')
        return document_chunks
    except Exception as e:
        logger.error(f"Gagal memproses {pdf_file.name}: {e}", exc_info=True)
        return []

def build_vector_store():
    logger.info("=== MEMULAI PROSES PEMBUATAN VECTOR STORE BERDASARKAN KELAS ===")
    embedding_model = get_embedding_model(config)
    pdf_base_path = Path(config.PDF_DIR).resolve()
    
    pdf_by_grade = {"10": [], "11": [], "12": []}
    for pdf_file in pdf_base_path.glob("*.pdf"):
        # --- PERBAIKAN UTAMA ADA DI SINI ---
        # Ambil bagian PERTAMA dari nama file sebagai kelas
        grade = pdf_file.stem.split('_')[0]
        # --- AKHIR PERBAIKAN ---
        
        if grade in pdf_by_grade:
            pdf_by_grade[grade].append(pdf_file)

    for grade, pdf_files in pdf_by_grade.items():
        if not pdf_files:
            logger.info(f"Tidak ada PDF ditemukan untuk Kelas {grade}.")
            continue

        logger.info(f"--- Memproses {len(pdf_files)} PDF untuk Kelas {grade} ---")
        all_chunks = []
        for pdf_file in tqdm(pdf_files, desc=f"  Dokumen Kelas {grade}"):
            all_chunks.extend(process_and_chunk_pdf(pdf_file, pdf_base_path, config))
        
        if not all_chunks:
            logger.warning(f"Tidak ada chunk yang dihasilkan untuk Kelas {grade}.")
            continue

        db_path = f"./data/chroma_{grade}"
        collection = f"studiva_grade_{grade}"
        
        logger.info(f"Membangun Vector Store untuk Kelas {grade} di '{db_path}'")
        if os.path.exists(db_path):
            logger.warning(f"Menghapus database lama: {db_path}")
            shutil.rmtree(db_path)
        
        try:
            Chroma.from_documents(
                documents=all_chunks, 
                embedding=embedding_model,
                persist_directory=db_path,
                collection_name=collection
            )
            logger.info(f"=== Vector Store untuk Kelas {grade} berhasil dibuat! ===")
        except Exception as e:
            logger.error(f"Gagal menyimpan ke ChromaDB untuk Kelas {grade}: {e}", exc_info=True)

if __name__ == "__main__":
    build_vector_store()