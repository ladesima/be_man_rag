"""
ChromaDB Vector Store - Specialized vector database untuk Indonesian curriculum RAG
Menggunakan ChromaDB untuk penyimpanan dan retrieval vector embeddings
"""

import logging
import asyncio
import numpy as np
from typing import List, Dict, Optional, Union, Any
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import json
import uuid
from pathlib import Path

class ChromaVectorStore:
    """
    Vector store menggunakan ChromaDB untuk menyimpan dan retrieve embeddings
    Dioptimalkan untuk konten pendidikan Indonesia dengan metadata yang kaya
    """
    
    def __init__(self, 
                 db_path: str = "./data/chroma",
                 collection_name: str = "indonesian_curriculum",
                 distance_metric: str = "cosine"):
        
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        
        self.client = None
        self.collection = None
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"ChromaDB Vector Store initialized at {db_path}")
    
    async def initialize(self):
        """Initialize ChromaDB client dan collection"""
        try:
            # Create ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get atau create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": self.distance_metric}
                )
                self.logger.info(f"Loaded existing collection: {self.collection_name}")
                
            except Exception:
                # Create new collection jika belum ada
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": self.distance_metric}
                )
                self.logger.info(f"Created new collection: {self.collection_name}")
            
            # Get collection info
            count = self.collection.count()
            self.logger.info(f"Collection has {count} documents")
            
        except Exception as e:
            self.logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    async def add_documents(self, 
                           documents: List[Dict], 
                           embeddings: List[np.ndarray]) -> bool:
        """
        Add documents dengan embeddings ke collection
        
        Args:
            documents: List of document dicts dengan content dan metadata
            embeddings: List of embedding vectors
        
        Returns:
            True jika berhasil, False jika gagal
        """
        try:
            if len(documents) != len(embeddings):
                raise ValueError("Documents dan embeddings harus sama panjangnya")
            
            # Prepare data untuk ChromaDB
            ids = []
            contents = []
            metadatas = []
            embedding_list = []
            
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                # Generate unique ID
                doc_id = doc.get('id') or str(uuid.uuid4())
                ids.append(doc_id)
                
                # Extract content
                content = doc.get('content', '')
                contents.append(content)
                
                # Prepare metadata
                metadata = {
                    'subject': doc.get('subject', 'unknown'),
                    'source': doc.get('source', 'unknown'),
                    'page': doc.get('page', 0),
                    'chunk_index': doc.get('chunk_index', i),
                    'title': doc.get('title', ''),
                    'author': doc.get('author', ''),
                    'grade_level': doc.get('grade_level', ''),
                    'language': doc.get('language', 'indonesian'),
                    'content_type': doc.get('content_type', 'text'),
                    'chapter': doc.get('chapter', ''),
                    'section': doc.get('section', ''),
                    'keywords': json.dumps(doc.get('keywords', [])),
                    'created_at': doc.get('created_at', ''),
                    'updated_at': doc.get('updated_at', '')
                }
                metadatas.append(metadata)
                
                # Convert embedding ke list
                if isinstance(embedding, np.ndarray):
                    embedding_list.append(embedding.tolist())
                else:
                    embedding_list.append(list(embedding))
            
            # Add ke collection
            self.collection.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas,
                embeddings=embedding_list
            )
            
            self.logger.info(f"Added {len(documents)} documents to collection")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding documents: {e}")
            return False
    
    async def similarity_search(self, 
                               embedding: np.ndarray,
                               top_k: int = 5,
                               threshold: float = 0.0,
                               subject_filter: str = None,
                               source_filter: str = None,
                               metadata_filter: Dict = None) -> List[Dict]:
        """
        Similarity search menggunakan embedding
        
        Args:
            embedding: Query embedding vector
            top_k: Number of top results
            threshold: Minimum similarity threshold
            subject_filter: Filter by specific subject
            source_filter: Filter by specific source
            metadata_filter: Additional metadata filters
        
        Returns:
            List of similar documents dengan metadata
        """
        try:
            # Convert embedding ke list jika numpy array
            if isinstance(embedding, np.ndarray):
                query_embedding = embedding.tolist()
            else:
                query_embedding = list(embedding)
            
            # Build where clause untuk filtering
            where_clause = {}
            
            if subject_filter:
                where_clause['subject'] = subject_filter
            
            if source_filter:
                where_clause['source'] = source_filter
            
            if metadata_filter:
                where_clause.update(metadata_filter)
            
            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause if where_clause else None,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process results
            similar_docs = []
            
            if results['ids'] and len(results['ids']) > 0:
                ids = results['ids'][0]
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                
                for i, (doc_id, content, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances)
                ):
                    # Convert distance ke similarity score
                    # ChromaDB uses distance, kita convert ke similarity (1 - distance)
                    similarity_score = 1.0 - distance
                    
                    # Skip jika di bawah threshold
                    if similarity_score < threshold:
                        continue
                    
                    # Parse keywords kembali ke list
                    try:
                        keywords = json.loads(metadata.get('keywords', '[]'))
                    except:
                        keywords = []
                    
                    similar_doc = {
                        'id': doc_id,
                        'content': content,
                        'similarity_score': similarity_score,
                        'distance': distance,
                        'rank': i + 1,
                        'subject': metadata.get('subject'),
                        'source': metadata.get('source'),
                        'page': metadata.get('page'),
                        'chunk_index': metadata.get('chunk_index'),
                        'title': metadata.get('title'),
                        'author': metadata.get('author'),
                        'grade_level': metadata.get('grade_level'),
                        'chapter': metadata.get('chapter'),
                        'section': metadata.get('section'),
                        'keywords': keywords,
                        'content_type': metadata.get('content_type'),
                        'language': metadata.get('language')
                    }
                    
                    similar_docs.append(similar_doc)
            
            self.logger.debug(f"Found {len(similar_docs)} similar documents")
            return similar_docs
            
        except Exception as e:
            self.logger.error(f"Error in similarity search: {e}")
            return []
    
    async def get_by_id(self, document_id: str) -> Optional[Dict]:
        """Get document by ID"""
        try:
            results = self.collection.get(
                ids=[document_id],
                include=['documents', 'metadatas']
            )
            
            if results['ids'] and len(results['ids']) > 0:
                content = results['documents'][0]
                metadata = results['metadatas'][0]
                
                # Parse keywords
                try:
                    keywords = json.loads(metadata.get('keywords', '[]'))
                except:
                    keywords = []
                
                return {
                    'id': document_id,
                    'content': content,
                    'subject': metadata.get('subject'),
                    'source': metadata.get('source'),
                    'page': metadata.get('page'),
                    'title': metadata.get('title'),
                    'keywords': keywords,
                    **metadata
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting document by ID {document_id}: {e}")
            return None
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents by IDs"""
        try:
            self.collection.delete(ids=document_ids)
            self.logger.info(f"Deleted {len(document_ids)} documents")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting documents: {e}")
            return False
    
    async def update_document(self, document_id: str, 
                             content: str = None, 
                             metadata: Dict = None,
                             embedding: np.ndarray = None) -> bool:
        """Update document content, metadata, atau embedding"""
        try:
            update_data = {'ids': [document_id]}
            
            if content is not None:
                update_data['documents'] = [content]
            
            if metadata is not None:
                # Ensure keywords di-serialize dengan benar
                if 'keywords' in metadata and isinstance(metadata['keywords'], list):
                    metadata['keywords'] = json.dumps(metadata['keywords'])
                update_data['metadatas'] = [metadata]
            
            if embedding is not None:
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                update_data['embeddings'] = [embedding]
            
            self.collection.update(**update_data)
            self.logger.info(f"Updated document {document_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating document {document_id}: {e}")
            return False
    
    async def get_documents_by_subject(self, subject: str, limit: int = 100) -> List[Dict]:
        """Get semua documents untuk subject tertentu"""
        try:
            results = self.collection.get(
                where={'subject': subject},
                limit=limit,
                include=['documents', 'metadatas']
            )
            
            documents = []
            if results['ids']:
                for i, doc_id in enumerate(results['ids']):
                    content = results['documents'][i]
                    metadata = results['metadatas'][i]
                    
                    # Parse keywords
                    try:
                        keywords = json.loads(metadata.get('keywords', '[]'))
                    except:
                        keywords = []
                    
                    documents.append({
                        'id': doc_id,
                        'content': content,
                        'keywords': keywords,
                        **metadata
                    })
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Error getting documents by subject {subject}: {e}")
            return []
    
    async def get_all_subjects(self) -> List[str]:
        """Get list semua subjects yang ada di collection"""
        try:
            # ChromaDB doesn't have direct aggregation, jadi kita query semua metadata
            results = self.collection.get(include=['metadatas'])
            
            subjects = set()
            if results['metadatas']:
                for metadata in results['metadatas']:
                    subject = metadata.get('subject')
                    if subject and subject != 'unknown':
                        subjects.add(subject)
            
            return sorted(list(subjects))
            
        except Exception as e:
            self.logger.error(f"Error getting all subjects: {e}")
            return []
    
    async def get_stats(self) -> Dict:
        """Get statistik collection"""
        try:
            total_count = self.collection.count()
            
            # Get subjects distribution
            subjects = await self.get_all_subjects()
            subject_counts = {}
            
            for subject in subjects:
                results = self.collection.get(
                    where={'subject': subject},
                    include=['metadatas']
                )
                subject_counts[subject] = len(results['ids']) if results['ids'] else 0
            
            return {
                'total_documents': total_count,
                'total_chunks': total_count,  # Same thing dalam context ini
                'subjects': subjects,
                'subject_distribution': subject_counts,
                'collection_name': self.collection_name,
                'distance_metric': self.distance_metric
            }
            
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {}
    
    async def search_by_text(self, query_text: str, top_k: int = 5, **filters) -> List[Dict]:
        """
        Text-based search (untuk debugging atau fallback)
        Note: Ini memerlukan embedding service untuk convert text ke embedding
        """
        # This method would need embedding service to convert text to embedding
        # For now, return empty - should be implemented with embedding service dependency
        self.logger.warning("Text search not implemented - use similarity_search with embeddings")
        return []
    
    async def health_check(self) -> Dict:
        """Health check untuk vector store"""
        try:
            if not self.client or not self.collection:
                return {'status': 'unhealthy', 'reason': 'Not initialized'}
            
            # Test basic operations
            count = self.collection.count()
            
            return {
                'status': 'healthy',
                'collection_name': self.collection_name,
                'document_count': count,
                'db_path': str(self.db_path),
                'distance_metric': self.distance_metric
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def reset_collection(self):
        """Reset collection (hapus semua data)"""
        try:
            if self.client and self.collection:
                self.client.delete_collection(self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": self.distance_metric}
                )
                self.logger.info(f"Reset collection: {self.collection_name}")
            
        except Exception as e:
            self.logger.error(f"Error resetting collection: {e}")
            raise

# Utility functions
async def create_vector_store(db_path: str = None, **kwargs) -> ChromaVectorStore:
    """Factory function untuk membuat vector store"""
    store = ChromaVectorStore(db_path=db_path, **kwargs)
    await store.initialize()
    return store

# Example usage
if __name__ == "__main__":
    async def main():
        # Create vector store
        vector_store = await create_vector_store("./test_chroma")
        
        # Test add documents
        test_docs = [
            {
                'content': 'Rukun iman ada enam yaitu iman kepada Allah',
                'subject': 'akidah_akhlak',
                'source': 'buku_akidah.pdf',
                'page': 1,
                'title': 'Rukun Iman',
                'keywords': ['rukun iman', 'akidah', 'iman']
            },
            {
                'content': 'Matematika adalah ilmu tentang bilangan dan ruang',
                'subject': 'matematika',
                'source': 'buku_matematika.pdf',
                'page': 5,
                'title': 'Pengenalan Matematika',
                'keywords': ['matematika', 'bilangan', 'ruang']
            }
        ]
        
        # Dummy embeddings (dalam aplikasi nyata, gunakan embedding service)
        test_embeddings = [
            np.random.rand(384).astype(np.float32),
            np.random.rand(384).astype(np.float32)
        ]
        
        # Add documents
        success = await vector_store.add_documents(test_docs, test_embeddings)
        print(f"Add documents success: {success}")
        
        # Test similarity search
        query_embedding = np.random.rand(384).astype(np.float32)
        results = await vector_store.similarity_search(
            query_embedding, 
            top_k=2,
            subject_filter='akidah_akhlak'
        )
        
        print(f"Found {len(results)} similar documents")
        for result in results:
            print(f"- {result['title']}: {result['similarity_score']:.3f}")
        
        # Get stats
        stats = await vector_store.get_stats()
        print(f"Stats: {stats}")
        
        # Health check
        health = await vector_store.health_check()
        print(f"Health: {health}")
    
    asyncio.run(main())