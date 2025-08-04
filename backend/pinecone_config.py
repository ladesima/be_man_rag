import pinecone

# Inisialisasi Pinecone
pinecone.init(
    api_key="pcsk_4SGf9f_NTpLgzUwjTKe9FjBhMguHC9uATQox8LLqyv3U44KuVJPCWnMnA6jbWwBC8Q62tU",
    environment="gcp-starter"  # atau yang sesuai
)

index_name = "chat-history"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536, metric="cosine")

pinecone_index = pinecone.Index(index_name)
