"""
Analytics Engine v1.0

A generic, reusable, parameter-driven analytics framework for the IOP platform.
"""

__version__ = "1.0.0"

from .core.operator import Operator
from .core.registry import OperatorRegistry
from .core.pipeline import Pipeline, PipelineExecutor

# Enhancement modules
from .core.metrics import MetricsCollector, get_metrics
from .core.dlq import DeadLetterQueue, get_dlq
from .core.versioning import PipelineVersionManager, get_version_manager
from .core.schema_registry import SchemaRegistry, get_schema_registry

__all__ = [
    # Core
    "Operator",
    "OperatorRegistry", 
    "Pipeline",
    "PipelineExecutor",
    # Enhancements
    "MetricsCollector",
    "get_metrics",
    "DeadLetterQueue",
    "get_dlq",
    "PipelineVersionManager",
    "get_version_manager",
    "SchemaRegistry",
    "get_schema_registry",
]
