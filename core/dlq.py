"""
Dead Letter Queue (DLQ) Module.

Provides robust error handling with retry mechanisms and dead letter
queues for failed records in Analytics Engine pipelines.
"""

import asyncio
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import deque
import uuid
import hashlib


logger = logging.getLogger(__name__)


class FailureReason(str, Enum):
    """Categorized failure reasons."""
    VALIDATION_ERROR = "validation_error"
    PROCESSING_ERROR = "processing_error"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    SCHEMA_MISMATCH = "schema_mismatch"
    OPERATOR_ERROR = "operator_error"
    UNKNOWN = "unknown"


class RecordStatus(str, Enum):
    """Status of a DLQ record."""
    PENDING = "pending"
    RETRYING = "retrying"
    FAILED = "failed"
    RECOVERED = "recovered"
    DISCARDED = "discarded"


@dataclass
class RetryPolicy:
    """
    Retry policy configuration.
    
    Attributes:
        max_retries: Maximum number of retry attempts
        initial_delay_seconds: Initial delay before first retry
        max_delay_seconds: Maximum delay between retries
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delays
        retryable_errors: List of error types that should be retried
    """
    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 300.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_errors: List[str] = field(default_factory=lambda: [
        "timeout", "resource_exhausted", "processing_error"
    ])
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number."""
        delay = min(
            self.initial_delay_seconds * (self.exponential_base ** attempt),
            self.max_delay_seconds
        )
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random())
        
        return delay
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if an error should be retried."""
        if attempt >= self.max_retries:
            return False
        
        error_type = type(error).__name__.lower()
        return any(
            retryable in error_type 
            for retryable in self.retryable_errors
        )


@dataclass
class DeadLetterRecord:
    """
    A record in the Dead Letter Queue.
    
    Contains the original data, error information, and retry history.
    """
    record_id: str
    original_data: Dict[str, Any]
    pipeline_name: str
    operator_name: Optional[str]
    error_message: str
    error_type: str
    failure_reason: FailureReason
    status: RecordStatus = RecordStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    last_attempt_at: Optional[datetime] = None
    next_retry_at: Optional[datetime] = None
    attempt_count: int = 0
    retry_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.record_id:
            # Generate deterministic ID from data
            data_hash = hashlib.md5(
                json.dumps(self.original_data, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]
            self.record_id = f"dlq_{data_hash}_{int(time.time() * 1000)}"
    
    def add_retry_attempt(self, success: bool, error: Optional[str] = None) -> None:
        """Record a retry attempt."""
        self.attempt_count += 1
        self.last_attempt_at = datetime.now()
        self.retry_history.append({
            "attempt": self.attempt_count,
            "timestamp": self.last_attempt_at.isoformat(),
            "success": success,
            "error": error,
        })
        
        if success:
            self.status = RecordStatus.RECOVERED
        else:
            self.status = RecordStatus.RETRYING
    
    def mark_failed(self) -> None:
        """Mark record as permanently failed."""
        self.status = RecordStatus.FAILED
    
    def mark_discarded(self) -> None:
        """Mark record as discarded."""
        self.status = RecordStatus.DISCARDED
    
    @property
    def age_seconds(self) -> float:
        """Get age of record in seconds."""
        return (datetime.now() - self.created_at).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "original_data": self.original_data,
            "pipeline_name": self.pipeline_name,
            "operator_name": self.operator_name,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "failure_reason": self.failure_reason.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_attempt_at": self.last_attempt_at.isoformat() if self.last_attempt_at else None,
            "next_retry_at": self.next_retry_at.isoformat() if self.next_retry_at else None,
            "attempt_count": self.attempt_count,
            "retry_history": self.retry_history,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeadLetterRecord":
        """Create record from dictionary."""
        return cls(
            record_id=data["record_id"],
            original_data=data["original_data"],
            pipeline_name=data["pipeline_name"],
            operator_name=data.get("operator_name"),
            error_message=data["error_message"],
            error_type=data["error_type"],
            failure_reason=FailureReason(data["failure_reason"]),
            status=RecordStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_attempt_at=datetime.fromisoformat(data["last_attempt_at"]) if data.get("last_attempt_at") else None,
            next_retry_at=datetime.fromisoformat(data["next_retry_at"]) if data.get("next_retry_at") else None,
            attempt_count=data["attempt_count"],
            retry_history=data.get("retry_history", []),
            metadata=data.get("metadata", {}),
        )


class DLQStore(ABC):
    """Abstract base class for DLQ storage backends."""
    
    @abstractmethod
    async def push(self, record: DeadLetterRecord) -> None:
        """Add a record to the DLQ."""
        pass
    
    @abstractmethod
    async def pop(self) -> Optional[DeadLetterRecord]:
        """Remove and return the oldest record."""
        pass
    
    @abstractmethod
    async def peek(self, limit: int = 10) -> List[DeadLetterRecord]:
        """View records without removing."""
        pass
    
    @abstractmethod
    async def get(self, record_id: str) -> Optional[DeadLetterRecord]:
        """Get a specific record by ID."""
        pass
    
    @abstractmethod
    async def update(self, record: DeadLetterRecord) -> None:
        """Update a record."""
        pass
    
    @abstractmethod
    async def delete(self, record_id: str) -> bool:
        """Delete a record."""
        pass
    
    @abstractmethod
    async def get_retryable(self) -> List[DeadLetterRecord]:
        """Get records ready for retry."""
        pass
    
    @abstractmethod
    async def count(self, status: Optional[RecordStatus] = None) -> int:
        """Count records by status."""
        pass
    
    @abstractmethod
    async def clear(self) -> int:
        """Clear all records."""
        pass


class InMemoryDLQStore(DLQStore):
    """In-memory DLQ store for development/testing."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._records: Dict[str, DeadLetterRecord] = {}
        self._queue: deque = deque()
        self._lock = threading.Lock()
    
    async def push(self, record: DeadLetterRecord) -> None:
        with self._lock:
            if len(self._records) >= self.max_size:
                # Remove oldest
                if self._queue:
                    oldest_id = self._queue.popleft()
                    self._records.pop(oldest_id, None)
            
            self._records[record.record_id] = record
            self._queue.append(record.record_id)
    
    async def pop(self) -> Optional[DeadLetterRecord]:
        with self._lock:
            if not self._queue:
                return None
            record_id = self._queue.popleft()
            return self._records.pop(record_id, None)
    
    async def peek(self, limit: int = 10) -> List[DeadLetterRecord]:
        with self._lock:
            records = []
            for record_id in list(self._queue)[:limit]:
                if record_id in self._records:
                    records.append(self._records[record_id])
            return records
    
    async def get(self, record_id: str) -> Optional[DeadLetterRecord]:
        return self._records.get(record_id)
    
    async def update(self, record: DeadLetterRecord) -> None:
        with self._lock:
            if record.record_id in self._records:
                self._records[record.record_id] = record
    
    async def delete(self, record_id: str) -> bool:
        with self._lock:
            if record_id in self._records:
                del self._records[record_id]
                try:
                    self._queue.remove(record_id)
                except ValueError:
                    pass
                return True
            return False
    
    async def get_retryable(self) -> List[DeadLetterRecord]:
        now = datetime.now()
        retryable = []
        with self._lock:
            for record in self._records.values():
                if record.status == RecordStatus.RETRYING:
                    if record.next_retry_at and record.next_retry_at <= now:
                        retryable.append(record)
                elif record.status == RecordStatus.PENDING:
                    retryable.append(record)
        return retryable
    
    async def count(self, status: Optional[RecordStatus] = None) -> int:
        with self._lock:
            if status is None:
                return len(self._records)
            return sum(1 for r in self._records.values() if r.status == status)
    
    async def clear(self) -> int:
        with self._lock:
            count = len(self._records)
            self._records.clear()
            self._queue.clear()
            return count


class RedisDLQStore(DLQStore):
    """Redis-backed DLQ store for production."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        key_prefix: str = "ae:dlq:",
        ttl_days: int = 30,
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_days * 86400
        self._client = None
    
    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            import redis.asyncio as redis
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
            )
        except ImportError:
            raise ImportError("redis package required: pip install redis")
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._client:
            await self._client.close()
    
    def _key(self, record_id: str) -> str:
        return f"{self.key_prefix}record:{record_id}"
    
    def _queue_key(self) -> str:
        return f"{self.key_prefix}queue"
    
    async def push(self, record: DeadLetterRecord) -> None:
        if not self._client:
            await self.connect()
        
        key = self._key(record.record_id)
        await self._client.set(
            key,
            json.dumps(record.to_dict(), default=str),
            ex=self.ttl_seconds,
        )
        await self._client.lpush(self._queue_key(), record.record_id)
    
    async def pop(self) -> Optional[DeadLetterRecord]:
        if not self._client:
            await self.connect()
        
        record_id = await self._client.rpop(self._queue_key())
        if not record_id:
            return None
        
        return await self.get(record_id)
    
    async def peek(self, limit: int = 10) -> List[DeadLetterRecord]:
        if not self._client:
            await self.connect()
        
        record_ids = await self._client.lrange(self._queue_key(), -limit, -1)
        records = []
        for record_id in record_ids:
            record = await self.get(record_id)
            if record:
                records.append(record)
        return records
    
    async def get(self, record_id: str) -> Optional[DeadLetterRecord]:
        if not self._client:
            await self.connect()
        
        key = self._key(record_id)
        data = await self._client.get(key)
        if data:
            return DeadLetterRecord.from_dict(json.loads(data))
        return None
    
    async def update(self, record: DeadLetterRecord) -> None:
        if not self._client:
            await self.connect()
        
        key = self._key(record.record_id)
        await self._client.set(
            key,
            json.dumps(record.to_dict(), default=str),
            ex=self.ttl_seconds,
        )
    
    async def delete(self, record_id: str) -> bool:
        if not self._client:
            await self.connect()
        
        key = self._key(record_id)
        deleted = await self._client.delete(key)
        await self._client.lrem(self._queue_key(), 0, record_id)
        return deleted > 0
    
    async def get_retryable(self) -> List[DeadLetterRecord]:
        """Get records ready for retry."""
        # Scan all records - in production, use a sorted set with retry times
        records = await self.peek(limit=1000)
        now = datetime.now()
        
        retryable = []
        for record in records:
            if record.status == RecordStatus.PENDING:
                retryable.append(record)
            elif record.status == RecordStatus.RETRYING:
                if record.next_retry_at and record.next_retry_at <= now:
                    retryable.append(record)
        
        return retryable
    
    async def count(self, status: Optional[RecordStatus] = None) -> int:
        if not self._client:
            await self.connect()
        
        if status is None:
            return await self._client.llen(self._queue_key())
        
        # Count by status requires scanning
        records = await self.peek(limit=10000)
        return sum(1 for r in records if r.status == status)
    
    async def clear(self) -> int:
        if not self._client:
            await self.connect()
        
        count = await self._client.llen(self._queue_key())
        await self._client.delete(self._queue_key())
        # Note: Individual record keys will expire via TTL
        return count


class DeadLetterQueue:
    """
    Dead Letter Queue manager for Analytics Engine.
    
    Handles failed records with configurable retry policies,
    persistence, and recovery mechanisms.
    
    Example:
        dlq = DeadLetterQueue(
            store=RedisDLQStore(host="localhost"),
            retry_policy=RetryPolicy(max_retries=5),
        )
        
        # When processing fails
        await dlq.push_failed(
            data={"sensor_id": "s1", "value": 42},
            error=processing_error,
            pipeline_name="temperature_alerts",
            operator_name="ThresholdEvaluator",
        )
        
        # Process retries
        await dlq.process_retries(retry_handler)
        
        # Inspect DLQ
        failed = await dlq.get_failed_records(limit=50)
    """
    
    def __init__(
        self,
        store: Optional[DLQStore] = None,
        retry_policy: Optional[RetryPolicy] = None,
        on_recovered: Optional[Callable[[DeadLetterRecord], None]] = None,
        on_failed: Optional[Callable[[DeadLetterRecord], None]] = None,
        max_age_hours: int = 168,  # 7 days
    ):
        """
        Initialize Dead Letter Queue.
        
        Args:
            store: Storage backend (defaults to in-memory)
            retry_policy: Retry policy configuration
            on_recovered: Callback when record is recovered
            on_failed: Callback when record permanently fails
            max_age_hours: Maximum age before auto-discard
        """
        self.store = store or InMemoryDLQStore()
        self.retry_policy = retry_policy or RetryPolicy()
        self.on_recovered = on_recovered
        self.on_failed = on_failed
        self.max_age_seconds = max_age_hours * 3600
        self._running = False
        self._retry_task: Optional[asyncio.Task] = None
    
    async def push_failed(
        self,
        data: Dict[str, Any],
        error: Exception,
        pipeline_name: str,
        operator_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DeadLetterRecord:
        """
        Push a failed record to the DLQ.
        
        Args:
            data: Original data that failed
            error: The exception that occurred
            pipeline_name: Name of the pipeline
            operator_name: Name of the operator that failed
            metadata: Additional metadata
            
        Returns:
            Created DLQ record
        """
        failure_reason = self._categorize_error(error)
        
        record = DeadLetterRecord(
            record_id="",
            original_data=data,
            pipeline_name=pipeline_name,
            operator_name=operator_name,
            error_message=str(error),
            error_type=type(error).__name__,
            failure_reason=failure_reason,
            metadata=metadata or {},
        )
        
        # Schedule first retry
        if self.retry_policy.should_retry(error, 0):
            record.status = RecordStatus.RETRYING
            record.next_retry_at = datetime.now() + timedelta(
                seconds=self.retry_policy.get_delay(0)
            )
        else:
            record.status = RecordStatus.FAILED
        
        await self.store.push(record)
        
        logger.warning(
            f"Record pushed to DLQ: {record.record_id} "
            f"pipeline={pipeline_name} operator={operator_name} "
            f"error={type(error).__name__}"
        )
        
        return record
    
    def _categorize_error(self, error: Exception) -> FailureReason:
        """Categorize an error into a failure reason."""
        error_type = type(error).__name__.lower()
        error_msg = str(error).lower()
        
        if "validation" in error_type or "validation" in error_msg:
            return FailureReason.VALIDATION_ERROR
        elif "timeout" in error_type or "timeout" in error_msg:
            return FailureReason.TIMEOUT
        elif "schema" in error_type or "schema" in error_msg:
            return FailureReason.SCHEMA_MISMATCH
        elif "resource" in error_type or "memory" in error_msg:
            return FailureReason.RESOURCE_EXHAUSTED
        else:
            return FailureReason.PROCESSING_ERROR
    
    async def process_retries(
        self,
        retry_handler: Callable[[Dict[str, Any]], bool],
        batch_size: int = 10,
    ) -> Dict[str, int]:
        """
        Process pending retries.
        
        Args:
            retry_handler: Function to retry processing (returns True on success)
            batch_size: Number of records to process at once
            
        Returns:
            Dict with counts of recovered, failed, and skipped records
        """
        stats = {"recovered": 0, "failed": 0, "skipped": 0}
        
        retryable = await self.store.get_retryable()
        
        for record in retryable[:batch_size]:
            # Check age
            if record.age_seconds > self.max_age_seconds:
                record.mark_discarded()
                await self.store.update(record)
                stats["skipped"] += 1
                continue
            
            # Check retry limit
            if record.attempt_count >= self.retry_policy.max_retries:
                record.mark_failed()
                await self.store.update(record)
                if self.on_failed:
                    self.on_failed(record)
                stats["failed"] += 1
                continue
            
            # Attempt retry
            try:
                success = retry_handler(record.original_data)
                record.add_retry_attempt(success)
                
                if success:
                    await self.store.delete(record.record_id)
                    if self.on_recovered:
                        self.on_recovered(record)
                    stats["recovered"] += 1
                else:
                    record.next_retry_at = datetime.now() + timedelta(
                        seconds=self.retry_policy.get_delay(record.attempt_count)
                    )
                    await self.store.update(record)
                    
            except Exception as e:
                record.add_retry_attempt(False, str(e))
                record.next_retry_at = datetime.now() + timedelta(
                    seconds=self.retry_policy.get_delay(record.attempt_count)
                )
                await self.store.update(record)
        
        return stats
    
    async def start_retry_worker(
        self,
        retry_handler: Callable[[Dict[str, Any]], bool],
        interval_seconds: int = 30,
    ) -> None:
        """Start background retry worker."""
        self._running = True
        
        async def worker():
            while self._running:
                try:
                    stats = await self.process_retries(retry_handler)
                    if any(stats.values()):
                        logger.info(f"DLQ retry stats: {stats}")
                except Exception as e:
                    logger.error(f"DLQ retry worker error: {e}")
                
                await asyncio.sleep(interval_seconds)
        
        self._retry_task = asyncio.create_task(worker())
    
    async def stop_retry_worker(self) -> None:
        """Stop background retry worker."""
        self._running = False
        if self._retry_task:
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass
    
    async def get_failed_records(
        self,
        limit: int = 100,
        pipeline_name: Optional[str] = None,
    ) -> List[DeadLetterRecord]:
        """Get failed records for inspection."""
        records = await self.store.peek(limit=limit * 2)
        
        failed = [
            r for r in records 
            if r.status == RecordStatus.FAILED
        ]
        
        if pipeline_name:
            failed = [r for r in failed if r.pipeline_name == pipeline_name]
        
        return failed[:limit]
    
    async def replay_record(
        self,
        record_id: str,
        handler: Callable[[Dict[str, Any]], bool],
    ) -> bool:
        """Manually replay a failed record."""
        record = await self.store.get(record_id)
        if not record:
            return False
        
        try:
            success = handler(record.original_data)
            if success:
                await self.store.delete(record_id)
                logger.info(f"Successfully replayed record {record_id}")
                return True
            else:
                logger.warning(f"Replay failed for record {record_id}")
                return False
        except Exception as e:
            logger.error(f"Error replaying record {record_id}: {e}")
            return False
    
    async def discard_record(self, record_id: str) -> bool:
        """Manually discard a record."""
        record = await self.store.get(record_id)
        if record:
            record.mark_discarded()
            await self.store.delete(record_id)
            return True
        return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get DLQ statistics."""
        return {
            "total": await self.store.count(),
            "pending": await self.store.count(RecordStatus.PENDING),
            "retrying": await self.store.count(RecordStatus.RETRYING),
            "failed": await self.store.count(RecordStatus.FAILED),
            "recovered": await self.store.count(RecordStatus.RECOVERED),
        }


# -------------------------------------------------------------------------
# Global Instance
# -------------------------------------------------------------------------

_dlq_instance: Optional[DeadLetterQueue] = None


def get_dlq() -> DeadLetterQueue:
    """Get the global DLQ instance."""
    global _dlq_instance
    if _dlq_instance is None:
        _dlq_instance = DeadLetterQueue()
    return _dlq_instance


def set_dlq(dlq: DeadLetterQueue) -> None:
    """Set the global DLQ instance."""
    global _dlq_instance
    _dlq_instance = dlq
