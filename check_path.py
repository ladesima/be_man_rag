# check_path.py
from pathlib import Path
import os
try:
    from config import config
except ImportError:
    print("[ERROR] Gagal mengimpor 'config'. Pastikan file 'config.py' ada dan bisa diakses.")
    exit()

# --- ✍️ UBAH NAMA FILE INI ---
# Ganti dengan salah satu nama file dari hasil perintah nomor 2 di atas.
NAMA_FILE_PDF_UNTUK_TES = "11_matematika.pdf"
# ------------------------------------

print("\n" + "="*50)
print("Memulai Pengecekan Path PDF...")
print("="*50)
try:
    pdf_dir_dari_config = config.PDF_DIR
    print(f"[LANGKAH 1] Direktori PDF dari config.py: '{pdf_dir_dari_config}'")

    abs_pdf_dir = Path(pdf_dir_dari_config).resolve()
    print(f"[LANGKAH 2] Path absolut direktori PDF: '{abs_pdf_dir}'")

    path_lengkap = abs_pdf_dir / NAMA_FILE_PDF_UNTUK_TES
    print(f"[LANGKAH 3] Path lengkap yang akan dicek: '{path_lengkap}'")

    apakah_path_ada = path_lengkap.exists()
    apakah_ini_file = path_lengkap.is_file()

    print("\n" + "-"*15 + " HASIL " + "-"*15)
    print(f"Apakah path di atas ADA?  -> {apakah_path_ada}")
    print(f"Apakah path tersebut FILE? -> {apakah_ini_file}")
except Exception as e:
    print(f"\n[ERROR] Terjadi kesalahan saat menjalankan skrip: {e}")
print("="*50)