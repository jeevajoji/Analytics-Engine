"""
Output adapters for the Analytics Engine.

PostgreSQL and Notification adapters for writing events/alerts.
"""

from .postgres_adapter import (
    PostgresConfig,
    PostgresAdapter,
)
from .notification_adapter import (
    NotificationConfig,
    NotificationType,
    NotificationAdapter,
)

__all__ = [
    # PostgreSQL
    "PostgresConfig",
    "PostgresAdapter",
    # Notifications
    "NotificationConfig",
    "NotificationType",
    "NotificationAdapter",
]
