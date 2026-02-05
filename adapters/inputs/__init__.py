"""
Input adapters for the Analytics Engine.

Redis Streams and InfluxDB adapters for consuming data.
"""

from .redis_adapter import (
    RedisConfig,
    RedisStreamConsumer,
)
from .influxdb_adapter import (
    InfluxDBConfig,
    InfluxDBAdapter,
)

__all__ = [
    # Redis
    "RedisConfig",
    "RedisStreamConsumer",
    # InfluxDB
    "InfluxDBConfig",
    "InfluxDBAdapter",
]
