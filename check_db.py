import chromadb
from config import config

# Pengaturan untuk koneksi ke ChromaDB
DB_PATH = str(config.CHROMA_DB_PATH)
COLLECTION_NAME = config.COLLECTION_NAME

print(f"Menghubungkan ke ChromaDB di: {DB_PATH}")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

print(f"Mengambil 5 data pertama dari koleksi '{COLLECTION_NAME}'...")

# Ambil 5 item pertama dari database
results = collection.get(limit=5, include=["metadatas"])

print("\n--- Analisis Metadata ---")
if not results or not results.get("metadatas"):
    print("Tidak ada data ditemukan di database.")
else:
    for i, meta in enumerate(results["metadatas"]):
        print(f"\n[Dokumen #{i+1}]")
        print(f"  Metadata Lengkap: {meta}")

        # Cek keberadaan kunci 'pages'
        if 'pages' in meta:
            print(f"  ✅ Ditemukan kunci 'pages': {meta['pages']}")
        elif 'page' in meta:
             print(f"  ❌ KESALAHAN: Ditemukan kunci lama 'page': {meta['page']}. Database perlu dibangun ulang.")
        else:
            print("  ❌ KESALAHAN: Tidak ditemukan kunci 'page' maupun 'pages'. Database perlu dibangun ulang.")