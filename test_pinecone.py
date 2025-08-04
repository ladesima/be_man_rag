from pinecone import Pinecone

pc = Pinecone(api_key="pcsk_4SGf9f_NTpLgzUwjTKe9FjBhMguHC9uATQox8LLqyv3U44KuVJPCWnMnA6jbWwBC8Q62tU")
indexes = pc.list_indexes()

print("Index yang tersedia:", indexes.names())
