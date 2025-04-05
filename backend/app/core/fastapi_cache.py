from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
from datetime import timedelta
from app.config import CACHE_TTL
import logging
from app.core.formatting import LogFormatter
from typing import Optional, Any

logger = logging.getLogger(__name__)

class CustomInMemoryBackend(InMemoryBackend):
    def __init__(self):
        """Initialize the cache backend"""
        super().__init__()
        self.cache = {}

    async def delete(self, key: str) -> bool:
        """Delete a key from the cache"""
        try:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
        except Exception as e:
            logger.error(LogFormatter.error(f"Failed to delete key {key} from cache", e))
            return False

    async def get(self, key: str) -> Any:
        """Get a value from the cache"""
        return self.cache.get(key)

    async def set(self, key: str, value: Any, expire: Optional[int] = None) -> None:
        """Set a value in the cache"""
        self.cache[key] = value

def setup_cache():
    """Initialize FastAPI Cache with in-memory backend"""
    try:
        logger.info(LogFormatter.section("CACHE INITIALIZATION"))
        FastAPICache.init(
            backend=CustomInMemoryBackend(),
            prefix="fastapi-cache"
        )
        logger.info(LogFormatter.success("Cache initialized successfully"))
    except Exception as e:
        logger.error(LogFormatter.error("Failed to initialize cache", e))
        raise

async def invalidate_cache_key(key: str):
    """Invalidate a specific cache key"""
    try:
        backend = FastAPICache.get_backend()
        if hasattr(backend, 'delete'):
            await backend.delete(key)
            logger.info(LogFormatter.success(f"Cache invalidated for key: {key}"))
        else:
            logger.warning(LogFormatter.warning("Cache backend does not support deletion"))
    except Exception as e:
        logger.error(LogFormatter.error(f"Failed to invalidate cache key: {key}", e))

def build_cache_key(*args) -> str:
    """Build a cache key from multiple arguments"""
    return ":".join(str(arg) for arg in args if arg is not None)

def cached(expire: int = CACHE_TTL, key_builder=None):
    """Decorator for caching endpoint responses
    
    Args:
        expire (int): Cache TTL in seconds
        key_builder (callable, optional): Custom key builder function
    """
    return cache(
        expire=expire,
        key_builder=key_builder
    ) 