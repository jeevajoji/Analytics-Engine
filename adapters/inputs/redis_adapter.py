"""
Redis Input/Output Adapters.

Provides Redis Streams consumer for real-time telemetry
and Redis writer for state/caching.
"""

import asyncio
from typing import Any, Dict, List, Optional, AsyncIterator
from datetime import datetime
import logging
import json

from ..lanes.ruleset_lane import InputAdapter, OutputAdapter


logger = logging.getLogger(__name__)


class RedisStreamAdapter(InputAdapter):
    """
    Redis Streams input adapter for real-time telemetry.
    
    Consumes messages from Redis Streams for the Ruleset Lane.
    
    Example:
        adapter = RedisStreamAdapter(
            host="localhost",
            port=6379,
            streams=["telemetry:sensors"],
            consumer_group="analytics_engine",
            consumer_name="worker_1"
        )
        
        lane.set_input_adapter(adapter)
        await lane.start_async()
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        streams: Optional[List[str]] = None,
        consumer_group: str = "analytics_engine",
        consumer_name: str = "worker_1",
        batch_size: int = 100,
        block_ms: int = 1000,
        auto_ack: bool = True,
    ):
        """
        Initialize Redis Stream adapter.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            streams: List of stream names to consume
            consumer_group: Consumer group name
            consumer_name: Consumer name within group
            batch_size: Number of messages to fetch per call
            block_ms: Blocking timeout in milliseconds
            auto_ack: Automatically acknowledge messages
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.streams = streams or ["telemetry"]
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name
        self.batch_size = batch_size
        self.block_ms = block_ms
        self.auto_ack = auto_ack
        
        self._client = None
        self._running = False
    
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
            
            # Create consumer groups if they don't exist
            for stream in self.streams:
                try:
                    await self._client.xgroup_create(
                        stream, 
                        self.consumer_group, 
                        id="0",
                        mkstream=True
                    )
                except redis.ResponseError as e:
                    if "BUSYGROUP" not in str(e):
                        raise
            
            self._running = True
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            
        except ImportError:
            raise ImportError("redis package required: pip install redis")
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        self._running = False
        if self._client:
            await self._client.close()
            self._client = None
        logger.info("Disconnected from Redis")
    
    async def consume(self) -> AsyncIterator[Dict[str, Any]]:
        """Consume messages from Redis Streams."""
        if not self._client:
            raise RuntimeError("Not connected to Redis")
        
        streams_dict = {stream: ">" for stream in self.streams}
        
        while self._running:
            try:
                messages = await self._client.xreadgroup(
                    self.consumer_group,
                    self.consumer_name,
                    streams_dict,
                    count=self.batch_size,
                    block=self.block_ms,
                )
                
                for stream_name, stream_messages in messages:
                    for message_id, data in stream_messages:
                        # Parse message data
                        record = self._parse_message(data)
                        record["_stream"] = stream_name
                        record["_message_id"] = message_id
                        record["_received_at"] = datetime.now().isoformat()
                        
                        yield record
                        
                        # Auto-acknowledge
                        if self.auto_ack:
                            await self._client.xack(
                                stream_name, 
                                self.consumer_group, 
                                message_id
                            )
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error consuming from Redis: {e}")
                await asyncio.sleep(1)
    
    def _parse_message(self, data: Dict[str, str]) -> Dict[str, Any]:
        """Parse Redis message data."""
        result = {}
        for key, value in data.items():
            try:
                # Try to parse JSON values
                result[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                result[key] = value
        return result


class RedisOutputAdapter(OutputAdapter):
    """
    Redis output adapter for state and caching.
    
    Writes events/incidents to Redis for real-time access.
    
    Example:
        adapter = RedisOutputAdapter(
            host="localhost",
            key_prefix="events:",
            ttl_seconds=3600
        )
        
        lane.add_output_adapter(adapter)
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        key_prefix: str = "events:",
        ttl_seconds: Optional[int] = 3600,
        use_stream: bool = False,
        stream_name: str = "events",
    ):
        """
        Initialize Redis output adapter.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            key_prefix: Prefix for keys
            ttl_seconds: TTL for keys (None = no expiry)
            use_stream: Write to Redis Stream instead of keys
            stream_name: Stream name if use_stream=True
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds
        self.use_stream = use_stream
        self.stream_name = stream_name
        
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
            
            logger.info(f"Redis output adapter connected to {self.host}:{self.port}")
            
        except ImportError:
            raise ImportError("redis package required: pip install redis")
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._client:
            await self._client.close()
            self._client = None
    
    async def write(self, record: Dict[str, Any]) -> None:
        """Write a single record to Redis."""
        if not self._client:
            raise RuntimeError("Not connected to Redis")
        
        if self.use_stream:
            # Write to stream
            data = {k: json.dumps(v) if not isinstance(v, str) else v 
                    for k, v in record.items()}
            await self._client.xadd(self.stream_name, data)
        else:
            # Write to key
            key = self._get_key(record)
            value = json.dumps(record)
            
            if self.ttl_seconds:
                await self._client.setex(key, self.ttl_seconds, value)
            else:
                await self._client.set(key, value)
    
    async def write_batch(self, records: List[Dict[str, Any]]) -> None:
        """Write multiple records to Redis."""
        if not self._client:
            raise RuntimeError("Not connected to Redis")
        
        pipe = self._client.pipeline()
        
        for record in records:
            if self.use_stream:
                data = {k: json.dumps(v) if not isinstance(v, str) else v 
                        for k, v in record.items()}
                pipe.xadd(self.stream_name, data)
            else:
                key = self._get_key(record)
                value = json.dumps(record)
                if self.ttl_seconds:
                    pipe.setex(key, self.ttl_seconds, value)
                else:
                    pipe.set(key, value)
        
        await pipe.execute()
    
    def _get_key(self, record: Dict[str, Any]) -> str:
        """Generate Redis key for record."""
        # Use event_id or generate timestamp-based key
        event_id = record.get("event_id") or record.get("id")
        if event_id:
            return f"{self.key_prefix}{event_id}"
        return f"{self.key_prefix}{datetime.now().timestamp()}"
