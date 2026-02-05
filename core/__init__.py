"""
Core components of the Analytics Engine.
"""

from .operator import Operator
from .registry import OperatorRegistry
from .pipeline import Pipeline, PipelineExecutor
from .schema import DataSchema, SchemaField

# New enhancement modules
from .metrics import (
    MetricsCollector,
    get_metrics,
    timed,
    traced,
    StructuredLogger,
    get_logger,
)
from .dlq import (
    DeadLetterQueue,
    DeadLetterRecord,
    RetryPolicy,
    get_dlq,
    set_dlq,
)
from .versioning import (
    PipelineVersionManager,
    SemanticVersion,
    PipelineVersion,
    get_version_manager,
)
from .schema_registry import (
    SchemaRegistry,
    get_schema_registry,
    get_pipeline_input_subject,
    get_pipeline_output_subject,
)

# Exceptions
from .exceptions import (
    AnalyticsEngineError,
    OperatorError,
    OperatorNotFoundError,
    OperatorConfigError,
    OperatorExecutionError,
    PipelineError,
    PipelineExecutionError,
    SchemaError,
    SchemaValidationError,
    PackError,
    PackLoadError,
)

# Enums
from .enums import (
    ComparisonOperator,
    AggregationType,
    WindowType,
    JoinType,
    SeverityLevel,
    FilterMode,
    LaneType,
)

# Storage
from .storage import AsyncStore, InMemoryStore

__all__ = [
    # Core
    "Operator",
    "OperatorRegistry",
    "Pipeline",
    "PipelineExecutor",
    "DataSchema",
    "SchemaField",
    # Metrics
    "MetricsCollector",
    "get_metrics",
    "timed",
    "traced",
    "StructuredLogger",
    "get_logger",
    # DLQ
    "DeadLetterQueue",
    "DeadLetterRecord",
    "RetryPolicy",
    "get_dlq",
    "set_dlq",
    # Versioning
    "PipelineVersionManager",
    "SemanticVersion",
    "PipelineVersion",
    "get_version_manager",
    # Schema Registry
    "SchemaRegistry",
    "get_schema_registry",
    "get_pipeline_input_subject",
    "get_pipeline_output_subject",
    # Exceptions
    "AnalyticsEngineError",
    "OperatorError",
    "OperatorNotFoundError",
    "OperatorConfigError",
    "OperatorExecutionError",
    "PipelineError",
    "PipelineExecutionError",
    "SchemaError",
    "SchemaValidationError",
    "PackError",
    "PackLoadError",
    # Enums
    "ComparisonOperator",
    "AggregationType",
    "WindowType",
    "JoinType",
    "SeverityLevel",
    "FilterMode",
    "LaneType",
    # Storage
    "AsyncStore",
    "InMemoryStore",
]
