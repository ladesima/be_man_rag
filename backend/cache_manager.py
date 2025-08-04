"""
Cache Manager - Advanced caching strategies untuk Indonesian RAG system
Menggunakan multi-layer caching untuk optimal performance dan user experience
"""

import logging
import asyncio
import json
import pickle
import hashlib
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
import threading
from collections import OrderedDict


# Optional Redis support untuk distributed caching
try:
    import redis
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

@dataclass
class CacheEntry:
    """Entry untuk cache dengan metadata"""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int
    ttl: Optional[float] = None
    size_bytes: int = 0
    tags: List[str] = None

@dataclass
class CacheStats:
    """Statistics cache"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    hit_rate: float = 0.0

class LRUCache:
    """
    In-memory LRU Cache dengan TTL support
    Thread-safe implementation untuk local caching
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = OrderedDict()
        self.stats = CacheStats()
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value dari cache"""
        with self.lock:
            if key not in self.cache:
                self.stats.misses += 1
                return None
            
            entry = self.cache[key]
            
            # Check TTL
            if entry.ttl and time.time() > entry.created_at + entry.ttl:
                del self.cache[key]
                self.stats.misses += 1
                self.stats.evictions += 1
                return None
            
            # Update access info
            entry.accessed_at = time.time()
            entry.access_count += 1
            
            # Move to end (most recent)
            self.cache.move_to_end(key)
            
            self.stats.hits += 1
            self._update_hit_rate()
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value ke cache"""
        with self.lock:
            # Calculate size (rough estimation)
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = len(str(value).encode('utf-8'))
            
            current_time = time.time()
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=current_time,
                accessed_at=current_time,
                access_count=1,
                ttl=ttl or self.default_ttl,
                size_bytes=size_bytes
            )
            
            # Remove existing jika ada
            if key in self.cache:
                old_entry = self.cache[key]
                self.stats.total_size_bytes -= old_entry.size_bytes
                del self.cache[key]
            
            # Check size limit
            while len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Add new entry
            self.cache[key] = entry
            self.stats.total_size_bytes += size_bytes
            self.stats.entry_count = len(self.cache)
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete entry dari cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                self.stats.total_size_bytes -= entry.size_bytes
                del self.cache[key]
                self.stats.entry_count = len(self.cache)
                return True
            return False
    
    def clear(self):
        """Clear semua cache"""
        with self.lock:
            self.cache.clear()
            self.stats = CacheStats()
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if self.cache:
            key, entry = self.cache.popitem(last=False)
            self.stats.total_size_bytes -= entry.size_bytes
            self.stats.evictions += 1
    
    def _update_hit_rate(self):
        """Update hit rate"""
        total_requests = self.stats.hits + self.stats.misses
        if total_requests > 0:
            self.stats.hit_rate = self.stats.hits / total_requests
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self.lock:
            return asdict(self.stats)

class CacheManager:
    """
    Advanced cache manager dengan multiple layers:
    1. In-memory LRU cache (fastest)
    2. File-based persistent cache
    3. Optional Redis distributed cache
    """
    
    def __init__(self, 
                 cache_dir: str = "./app/cache",
                 memory_cache_size: int = 1000,
                 default_ttl: int = 3600,
                 enable_file_cache: bool = True,
                 enable_redis: bool = False,
                 redis_url: str = "redis://localhost:6379"):
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.default_ttl = default_ttl
        self.enable_file_cache = enable_file_cache
        self.enable_redis = enable_redis and REDIS_AVAILABLE
        
        # Initialize in-memory cache
        self.memory_cache = LRUCache(memory_cache_size, default_ttl)
        
        # Initialize Redis jika enabled
        self.redis_client = None
        if self.enable_redis:
            try:
                self.redis_client = aioredis.from_url(redis_url)
                self.logger.info("Redis cache enabled")
            except Exception as e:
                self.logger.warning(f"Failed to connect to Redis: {e}")
                self.enable_redis = False
        
        # Cache categories
        self.cache_categories = {
            'query_responses': self.cache_dir / 'query_responses',
            'embeddings': self.cache_dir / 'embeddings',
            'document_metadata': self.cache_dir / 'document_metadata',
            'subject_classifications': self.cache_dir / 'subject_classifications',
            'similarity_searches': self.cache_dir / 'similarity_searches'
        }
        
        # Create category directories
        for category_dir in self.cache_categories.values():
            category_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Cache manager initialized with dir: {cache_dir}")
    
    async def get_cached_response(self, query: str) -> Optional[Any]:
        """
        Get cached response untuk query
        Check multiple cache layers secara hierarchical
        """
        cache_key = self._generate_cache_key('query', query)
        
        # 1. Check memory cache (fastest)
        result = self.memory_cache.get(cache_key)
        if result is not None:
            self.logger.debug(f"Cache hit (memory): {cache_key[:20]}...")
            return result
        
        # 2. Check Redis cache
        if self.enable_redis:
            result = await self._get_from_redis(cache_key)
            if result is not None:
                # Store ke memory cache untuk next time
                self.memory_cache.set(cache_key, result)
                self.logger.debug(f"Cache hit (redis): {cache_key[:20]}...")
                return result
        
        # 3. Check file cache
        if self.enable_file_cache:
            result = await self._get_from_file_cache('query_responses', cache_key)
            if result is not None:
                # Store ke upper layers
                self.memory_cache.set(cache_key, result)
                if self.enable_redis:
                    await self._set_to_redis(cache_key, result)
                self.logger.debug(f"Cache hit (file): {cache_key[:20]}...")
                return result
        
        return None
    
    async def cache_response(self, query: str, response: Any, ttl: Optional[int] = None):
        """Cache response untuk query dengan multi-layer storage"""
        cache_key = self._generate_cache_key('query', query)
        ttl = ttl or self.default_ttl
        
        # Store di semua cache layers
        self.memory_cache.set(cache_key, response, ttl)
        
        if self.enable_redis:
            await self._set_to_redis(cache_key, response, ttl)
        
        if self.enable_file_cache:
            await self._set_to_file_cache('query_responses', cache_key, response, ttl)
        
        self.logger.debug(f"Cached response: {cache_key[:20]}...")
    
    async def get_cached_embedding(self, text: str) -> Optional[Any]:
        """Get cached embedding untuk text"""
        cache_key = self._generate_cache_key('embedding', text)
        
        # Check memory cache
        result = self.memory_cache.get(cache_key)
        if result is not None:
            return result
        
        # Check file cache (embeddings usually stored in files)
        if self.enable_file_cache:
            result = await self._get_from_file_cache('embeddings', cache_key)
            if result is not None:
                self.memory_cache.set(cache_key, result)
                return result
        
        return None
    
    async def cache_embedding(self, text: str, embedding: Any, ttl: Optional[int] = None):
        """Cache embedding untuk text"""
        cache_key = self._generate_cache_key('embedding', text)
        ttl = ttl or self.default_ttl * 24  # Embeddings can be cached longer
        
        self.memory_cache.set(cache_key, embedding, ttl)
        
        if self.enable_file_cache:
            await self._set_to_file_cache('embeddings', cache_key, embedding, ttl)
    
    async def get_cached_classification(self, text: str) -> Optional[Any]:
        """Get cached subject classification"""
        cache_key = self._generate_cache_key('classification', text)
        
        result = self.memory_cache.get(cache_key)
        if result is not None:
            return result
        
        if self.enable_file_cache:
            result = await self._get_from_file_cache('subject_classifications', cache_key)
            if result is not None:
                self.memory_cache.set(cache_key, result)
                return result
        
        return None
    
    async def cache_classification(self, text: str, classification: Any, ttl: Optional[int] = None):
        """Cache subject classification"""
        cache_key = self._generate_cache_key('classification', text)
        ttl = ttl or self.default_ttl * 12  # Classifications can be cached longer
        
        self.memory_cache.set(cache_key, classification, ttl)
        
        if self.enable_file_cache:
            await self._set_to_file_cache('subject_classifications', cache_key, classification, ttl)
    
    async def get_cached_similarity_search(self, embedding_hash: str, filters: Dict) -> Optional[Any]:
        """Get cached similarity search results"""
        filter_key = json.dumps(filters, sort_keys=True)
        cache_key = self._generate_cache_key('similarity', f"{embedding_hash}_{filter_key}")
        
        result = self.memory_cache.get(cache_key)
        if result is not None:
            return result
        
        if self.enable_file_cache:
            result = await self._get_from_file_cache('similarity_searches', cache_key)
            if result is not None:
                self.memory_cache.set(cache_key, result)
                return result
        
        return None
    
    async def cache_similarity_search(self, embedding_hash: str, filters: Dict, 
                                    results: Any, ttl: Optional[int] = None):
        """Cache similarity search results"""
        filter_key = json.dumps(filters, sort_keys=True)
        cache_key = self._generate_cache_key('similarity', f"{embedding_hash}_{filter_key}")
        ttl = ttl or self.default_ttl * 6  # Similarity searches cached for 6 hours
        
        self.memory_cache.set(cache_key, results, ttl)
        
        if self.enable_file_cache:
            await self._set_to_file_cache('similarity_searches', cache_key, results, ttl)
    
    def _generate_cache_key(self, prefix: str, content: str) -> str:
        """Generate consistent cache key"""
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        return f"{prefix}_{content_hash}"
    
    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """Get value dari Redis"""
        try:
            if self.redis_client:
                value = await self.redis_client.get(key)
                if value:
                    return pickle.loads(value)
        except Exception as e:
            self.logger.error(f"Redis get error: {e}")
        return None
    
    async def _set_to_redis(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value ke Redis"""
        try:
            if self.redis_client:
                serialized = pickle.dumps(value)
                await self.redis_client.set(key, serialized, ex=ttl)
        except Exception as e:
            self.logger.error(f"Redis set error: {e}")
    
    async def _get_from_file_cache(self, category: str, key: str) -> Optional[Any]:
        """Get value dari file cache"""
        try:
            cache_file = self.cache_categories[category] / f"{key}.pkl"
            metadata_file = self.cache_categories[category] / f"{key}.meta"
            
            if not cache_file.exists() or not metadata_file.exists():
                return None
            
            # Check TTL dari metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            created_at = metadata.get('created_at', 0)
            ttl = metadata.get('ttl', self.default_ttl)
            
            if time.time() > created_at + ttl:
                # Expired, remove files
                cache_file.unlink(missing_ok=True)
                metadata_file.unlink(missing_ok=True)
                return None
            
            # Load value
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
                
        except Exception as e:
            self.logger.error(f"File cache get error: {e}")
            return None
    
    async def _set_to_file_cache(self, category: str, key: str, value: Any, ttl: int):
        """Set value ke file cache"""
        try:
            cache_file = self.cache_categories[category] / f"{key}.pkl"
            metadata_file = self.cache_categories[category] / f"{key}.meta"
            
            # Save value
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
            
            # Save metadata
            metadata = {
                'key': key,
                'created_at': time.time(),
                'ttl': ttl,
                'size_bytes': cache_file.stat().st_size
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
                
        except Exception as e:
            self.logger.error(f"File cache set error: {e}")
    
    async def cleanup_expired_cache(self):
        """Cleanup expired cache files"""
        try:
            current_time = time.time()
            total_cleaned = 0
            
            for category, cache_dir in self.cache_categories.items():
                if not cache_dir.exists():
                    continue
                
                meta_files = list(cache_dir.glob("*.meta"))
                
                for meta_file in meta_files:
                    try:
                        with open(meta_file, 'r') as f:
                            metadata = json.load(f)
                        
                        created_at = metadata.get('created_at', 0)
                        ttl = metadata.get('ttl', self.default_ttl)
                        
                        if current_time > created_at + ttl:
                            # Remove expired files
                            cache_key = meta_file.stem
                            cache_file = cache_dir / f"{cache_key}.pkl"
                            
                            meta_file.unlink(missing_ok=True)
                            cache_file.unlink(missing_ok=True)
                            total_cleaned += 1
                            
                    except Exception as e:
                        self.logger.error(f"Error cleaning {meta_file}: {e}")
                        continue
            
            self.logger.info(f"Cleaned {total_cleaned} expired cache entries")
            
        except Exception as e:
            self.logger.error(f"Error during cache cleanup: {e}")
    
    async def get_stats(self) -> Dict:
        """Get comprehensive cache statistics"""
        memory_stats = self.memory_cache.get_stats()
        
        # File cache stats
        file_stats = {}
        total_file_size = 0
        total_file_count = 0
        
        for category, cache_dir in self.cache_categories.items():
            if cache_dir.exists():
                cache_files = list(cache_dir.glob("*.pkl"))
                category_size = sum(f.stat().st_size for f in cache_files)
                file_stats[category] = {
                    'count': len(cache_files),
                    'size_bytes': category_size
                }
                total_file_size += category_size
                total_file_count += len(cache_files)
        
        # Redis stats (if available)
        redis_stats = {}
        if self.enable_redis and self.redis_client:
            try:
                redis_info = await self.redis_client.info('memory')
                redis_stats = {
                    'used_memory': redis_info.get('used_memory', 0),
                    'used_memory_human': redis_info.get('used_memory_human', '0B')
                }
            except:
                redis_stats = {'error': 'Could not get Redis stats'}
        
        return {
            'memory_cache': memory_stats,
            'file_cache': {
                'total_size_bytes': total_file_size,
                'total_count': total_file_count,
                'categories': file_stats
            },
            'redis_cache': redis_stats,
            'config': {
                'enable_file_cache': self.enable_file_cache,
                'enable_redis': self.enable_redis,
                'default_ttl': self.default_ttl
            }
        }
    
    async def clear_all_cache(self):
        """Clear semua cache (memory, file, Redis)"""
        # Clear memory cache
        self.memory_cache.clear()
        
        # Clear Redis cache
        if self.enable_redis and self.redis_client:
            try:
                await self.redis_client.flushdb()
            except Exception as e:
                self.logger.error(f"Error clearing Redis: {e}")
        
        # Clear file cache
        if self.enable_file_cache:
            for cache_dir in self.cache_categories.values():
                if cache_dir.exists():
                    for file_path in cache_dir.iterdir():
                        if file_path.is_file():
                            file_path.unlink()
        
        self.logger.info("All caches cleared")
    
    async def preload_common_queries(self, common_queries: List[str]):
        """Preload common queries untuk warm-up cache"""
        # This would be implemented with actual RAG engine
        # For now, just log the intent
        self.logger.info(f"Preloading {len(common_queries)} common queries")

# Background cache cleanup task
class CacheCleanupTask:
    """Background task untuk automatic cache cleanup"""
    
    def __init__(self, cache_manager: CacheManager, cleanup_interval: int = 3600):
        self.cache_manager = cache_manager
        self.cleanup_interval = cleanup_interval
        self.running = False
        self.task = None
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start background cleanup task"""
        if self.running:
            return
        
        self.running = True
        self.task = asyncio.create_task(self._cleanup_loop())
        self.logger.info("Cache cleanup task started")
    
    async def stop(self):
        """Stop background cleanup task"""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        self.logger.info("Cache cleanup task stopped")
    
    async def _cleanup_loop(self):
        """Main cleanup loop"""
        while self.running:
            try:
                await self.cache_manager.cleanup_expired_cache()
                await asyncio.sleep(self.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)  # Wait before retry

# Example usage
if __name__ == "__main__":
    async def main():
        # Create cache manager
        cache_manager = CacheManager(
            cache_dir="./test_cache",
            memory_cache_size=100,
            enable_file_cache=True,
            enable_redis=False  # Set True jika Redis tersedia
        )
        
        # Test caching
        await cache_manager.cache_response("Apa itu rukun iman?", "Rukun iman ada enam...")
        
        # Test retrieval
        cached = await cache_manager.get_cached_response("Apa itu rukun iman?")
        print(f"Cached response: {cached}")
        
        # Test embedding cache
        await cache_manager.cache_embedding("test text", [0.1, 0.2, 0.3])
        cached_embedding = await cache_manager.get_cached_embedding("test text")
        print(f"Cached embedding: {cached_embedding}")
        
        # Get stats
        stats = await cache_manager.get_stats()
        print(f"Cache stats: {json.dumps(stats, indent=2)}")
        
        # Start cleanup task
        cleanup_task = CacheCleanupTask(cache_manager)
        await cleanup_task.start()
        
        # Wait a bit then stop
        await asyncio.sleep(2)
        await cleanup_task.stop()
    
    asyncio.run(main())