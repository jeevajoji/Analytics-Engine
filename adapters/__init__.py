"""
Analytics Engine Adapters.

Input and output adapters for data sources and sinks.
"""

# Input Adapters
from .inputs import (
    RedisConfig,
    RedisStreamConsumer,
    InfluxDBConfig,
    InfluxDBAdapter,
)

# Output Adapters
from .outputs import (
    PostgresConfig,
    PostgresAdapter,
    NotificationConfig,
    NotificationType,
    NotificationAdapter,
)

__all__ = [
    # Input Adapters
    "RedisConfig",
    "RedisStreamConsumer",
    "InfluxDBConfig",
    "InfluxDBAdapter",
    # Output Adapters
    "PostgresConfig",
    "PostgresAdapter",
    "NotificationConfig",
    "NotificationType",
    "NotificationAdapter",
]
