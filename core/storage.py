"""
Base Storage Abstractions.

Generic storage interfaces and implementations to reduce duplication.
"""

import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar, Callable
from datetime import datetime


K = TypeVar('K')  # Key type
V = TypeVar('V')  # Value type


class AsyncStore(ABC, Generic[K, V]):
    """
    Generic async storage interface.
    
    Base class for version stores, schema stores, DLQ stores, etc.
    Provides a consistent interface for CRUD operations.
    """
    
    @abstractmethod
    async def save(self, key: K, value: V) -> None:
        """
        Save a value.
        
        Args:
            key: Storage key
            value: Value to store
        """
        pass
    
    @abstractmethod
    async def get(self, key: K) -> Optional[V]:
        """
        Retrieve a value by key.
        
        Args:
            key: Storage key
            
        Returns:
            Stored value or None if not found
        """
        pass
    
    @abstractmethod
    async def delete(self, key: K) -> bool:
        """
        Delete a value.
        
        Args:
            key: Storage key
            
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    async def exists(self, key: K) -> bool:
        """
        Check if a key exists.
        
        Args:
            key: Storage key
            
        Returns:
            True if exists
        """
        pass
    
    @abstractmethod
    async def list_keys(self) -> List[K]:
        """
        List all keys.
        
        Returns:
            List of all keys
        """
        pass


class InMemoryStore(AsyncStore[K, V]):
    """
    Thread-safe in-memory storage implementation.
    
    Useful for development, testing, and single-instance deployments.
    
    Example:
        store = InMemoryStore[str, dict]()
        await store.save("key1", {"data": "value"})
        value = await store.get("key1")
    """
    
    def __init__(self):
        self._data: Dict[K, V] = {}
        self._lock = threading.RLock()
    
    async def save(self, key: K, value: V) -> None:
        with self._lock:
            self._data[key] = value
    
    async def get(self, key: K) -> Optional[V]:
        with self._lock:
            return self._data.get(key)
    
    async def delete(self, key: K) -> bool:
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False
    
    async def exists(self, key: K) -> bool:
        with self._lock:
            return key in self._data
    
    async def list_keys(self) -> List[K]:
        with self._lock:
            return list(self._data.keys())
    
    async def clear(self) -> None:
        """Clear all data."""
        with self._lock:
            self._data.clear()
    
    async def size(self) -> int:
        """Get number of items."""
        with self._lock:
            return len(self._data)
    
    async def get_all(self) -> Dict[K, V]:
        """Get all items as a dictionary."""
        with self._lock:
            return dict(self._data)
    
    async def find(self, predicate: Callable[[V], bool]) -> List[V]:
        """
        Find all values matching a predicate.
        
        Args:
            predicate: Function that returns True for matching values
            
        Returns:
            List of matching values
        """
        with self._lock:
            return [v for v in self._data.values() if predicate(v)]


class NamespacedStore(AsyncStore[str, V]):
    """
    Store with namespace support for organizing data.
    
    Keys are automatically prefixed with namespace.
    
    Example:
        store = NamespacedStore(InMemoryStore(), namespace="pipelines")
        # Key "my_pipeline" becomes "pipelines:my_pipeline"
    """
    
    def __init__(self, backend: AsyncStore[str, V], namespace: str):
        self._backend = backend
        self._namespace = namespace
        self._separator = ":"
    
    def _make_key(self, key: str) -> str:
        """Create namespaced key."""
        return f"{self._namespace}{self._separator}{key}"
    
    def _strip_namespace(self, key: str) -> str:
        """Remove namespace from key."""
        prefix = f"{self._namespace}{self._separator}"
        if key.startswith(prefix):
            return key[len(prefix):]
        return key
    
    async def save(self, key: str, value: V) -> None:
        await self._backend.save(self._make_key(key), value)
    
    async def get(self, key: str) -> Optional[V]:
        return await self._backend.get(self._make_key(key))
    
    async def delete(self, key: str) -> bool:
        return await self._backend.delete(self._make_key(key))
    
    async def exists(self, key: str) -> bool:
        return await self._backend.exists(self._make_key(key))
    
    async def list_keys(self) -> List[str]:
        all_keys = await self._backend.list_keys()
        prefix = f"{self._namespace}{self._separator}"
        return [
            self._strip_namespace(k) 
            for k in all_keys 
            if k.startswith(prefix)
        ]


class TTLStore(AsyncStore[K, V]):
    """
    Store with time-to-live (TTL) support.
    
    Items automatically expire after the specified TTL.
    
    Example:
        store = TTLStore(InMemoryStore(), ttl_seconds=3600)  # 1 hour TTL
        await store.save("key", "value")
        # After 1 hour, item is automatically expired
    """
    
    def __init__(
        self,
        backend: AsyncStore[K, Dict[str, Any]],
        ttl_seconds: float,
    ):
        self._backend = backend
        self._ttl_seconds = ttl_seconds
    
    def _wrap_value(self, value: V) -> Dict[str, Any]:
        """Wrap value with expiry timestamp."""
        return {
            "value": value,
            "expires_at": datetime.now().timestamp() + self._ttl_seconds,
        }
    
    def _is_expired(self, wrapped: Dict[str, Any]) -> bool:
        """Check if wrapped value is expired."""
        return datetime.now().timestamp() > wrapped.get("expires_at", 0)
    
    async def save(self, key: K, value: V) -> None:
        await self._backend.save(key, self._wrap_value(value))
    
    async def get(self, key: K) -> Optional[V]:
        wrapped = await self._backend.get(key)
        if wrapped is None:
            return None
        
        if self._is_expired(wrapped):
            await self._backend.delete(key)
            return None
        
        return wrapped.get("value")
    
    async def delete(self, key: K) -> bool:
        return await self._backend.delete(key)
    
    async def exists(self, key: K) -> bool:
        value = await self.get(key)  # This checks expiry
        return value is not None
    
    async def list_keys(self) -> List[K]:
        # Note: This doesn't filter expired keys for efficiency
        # Use exists() to check individual keys
        return await self._backend.list_keys()
    
    async def cleanup_expired(self) -> int:
        """
        Remove all expired entries.
        
        Returns:
            Number of entries removed
        """
        removed = 0
        keys = await self._backend.list_keys()
        
        for key in keys:
            wrapped = await self._backend.get(key)
            if wrapped and self._is_expired(wrapped):
                await self._backend.delete(key)
                removed += 1
        
        return removed
