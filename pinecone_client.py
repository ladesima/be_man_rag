from pinecone import Pinecone
import os

pc = Pinecone(api_key=os.getenv("pcsk_4SGf9f_NTpLgzUwjTKe9FjBhMguHC9uATQox8LLqyv3U44KuVJPCWnMnA6jbWwBC8Q62tU"))  # atau isi langsung string

# Akses index
users_index = pc.Index("users")
chat_index = pc.Index("chat-history")
