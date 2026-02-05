"""
Analytics Engine Exception Hierarchy.

Centralized exception definitions for consistent error handling.
"""


class AnalyticsEngineError(Exception):
    """Base exception for all Analytics Engine errors."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - {self.details}"
        return self.message


# -----------------------------------------------------------------------------
# Operator Errors
# -----------------------------------------------------------------------------

class OperatorError(AnalyticsEngineError):
    """Base exception for operator-related errors."""
    pass


class OperatorNotFoundError(OperatorError):
    """Raised when an operator is not found in the registry."""
    
    def __init__(self, operator_name: str):
        super().__init__(
            f"Operator not found: '{operator_name}'",
            {"operator_name": operator_name}
        )
        self.operator_name = operator_name


class OperatorConfigError(OperatorError):
    """Raised when operator configuration is invalid."""
    
    def __init__(self, operator_name: str, message: str):
        super().__init__(
            f"Invalid configuration for {operator_name}: {message}",
            {"operator_name": operator_name}
        )
        self.operator_name = operator_name


class OperatorExecutionError(OperatorError):
    """Raised when an operator fails during execution."""
    
    def __init__(self, operator_name: str, message: str, original_error: Exception = None):
        super().__init__(
            f"Operator '{operator_name}' execution failed: {message}",
            {"operator_name": operator_name, "original_error": str(original_error) if original_error else None}
        )
        self.operator_name = operator_name
        self.original_error = original_error


# -----------------------------------------------------------------------------
# Pipeline Errors
# -----------------------------------------------------------------------------

class PipelineError(AnalyticsEngineError):
    """Base exception for pipeline-related errors."""
    pass


class PipelineNotFoundError(PipelineError):
    """Raised when a pipeline is not found."""
    
    def __init__(self, pipeline_name: str):
        super().__init__(
            f"Pipeline not found: '{pipeline_name}'",
            {"pipeline_name": pipeline_name}
        )
        self.pipeline_name = pipeline_name


class PipelineConfigError(PipelineError):
    """Raised when pipeline configuration is invalid."""
    pass


class PipelineExecutionError(PipelineError):
    """Raised when pipeline execution fails."""
    
    def __init__(self, pipeline_name: str, message: str, failed_operator: str = None):
        super().__init__(
            f"Pipeline '{pipeline_name}' execution failed: {message}",
            {"pipeline_name": pipeline_name, "failed_operator": failed_operator}
        )
        self.pipeline_name = pipeline_name
        self.failed_operator = failed_operator


# -----------------------------------------------------------------------------
# Schema Errors
# -----------------------------------------------------------------------------

class SchemaError(AnalyticsEngineError):
    """Base exception for schema-related errors."""
    pass


class SchemaValidationError(SchemaError):
    """Raised when data fails schema validation."""
    
    def __init__(self, schema_name: str, errors: list):
        super().__init__(
            f"Schema validation failed for '{schema_name}'",
            {"schema_name": schema_name, "errors": errors}
        )
        self.schema_name = schema_name
        self.errors = errors


class SchemaCompatibilityError(SchemaError):
    """Raised when schemas are not compatible."""
    
    def __init__(self, message: str, issues: list = None):
        super().__init__(message, {"issues": issues or []})
        self.issues = issues or []


# -----------------------------------------------------------------------------
# Pack Errors
# -----------------------------------------------------------------------------

class PackError(AnalyticsEngineError):
    """Base exception for pack-related errors."""
    pass


class PackLoadError(PackError):
    """Raised when a pack cannot be loaded."""
    
    def __init__(self, pack_id: str, message: str):
        super().__init__(
            f"Failed to load pack '{pack_id}': {message}",
            {"pack_id": pack_id}
        )
        self.pack_id = pack_id


class PackValidationError(PackError):
    """Raised when pack validation fails."""
    
    def __init__(self, pack_id: str, errors: list):
        super().__init__(
            f"Pack validation failed for '{pack_id}'",
            {"pack_id": pack_id, "errors": errors}
        )
        self.pack_id = pack_id
        self.errors = errors


class PackDependencyError(PackError):
    """Raised when pack dependencies cannot be resolved."""
    
    def __init__(self, pack_id: str, missing_deps: list):
        super().__init__(
            f"Missing dependencies for pack '{pack_id}'",
            {"pack_id": pack_id, "missing_deps": missing_deps}
        )
        self.pack_id = pack_id
        self.missing_deps = missing_deps


# -----------------------------------------------------------------------------
# Adapter Errors
# -----------------------------------------------------------------------------

class AdapterError(AnalyticsEngineError):
    """Base exception for adapter-related errors."""
    pass


class ConnectionError(AdapterError):
    """Raised when connection to external system fails."""
    
    def __init__(self, adapter_name: str, message: str):
        super().__init__(
            f"Connection failed for {adapter_name}: {message}",
            {"adapter_name": adapter_name}
        )
        self.adapter_name = adapter_name


class DataFetchError(AdapterError):
    """Raised when data fetching fails."""
    pass


class DataWriteError(AdapterError):
    """Raised when data writing fails."""
    pass


# -----------------------------------------------------------------------------
# Version Errors
# -----------------------------------------------------------------------------

class VersionError(AnalyticsEngineError):
    """Base exception for versioning errors."""
    pass


class VersionNotFoundError(VersionError):
    """Raised when a version is not found."""
    
    def __init__(self, pipeline_name: str, version: str):
        super().__init__(
            f"Version not found: {pipeline_name}@{version}",
            {"pipeline_name": pipeline_name, "version": version}
        )
        self.pipeline_name = pipeline_name
        self.version = version


class VersionConflictError(VersionError):
    """Raised when there's a version conflict."""
    pass


# -----------------------------------------------------------------------------
# DLQ Errors
# -----------------------------------------------------------------------------

class DLQError(AnalyticsEngineError):
    """Base exception for Dead Letter Queue errors."""
    pass


class DLQFullError(DLQError):
    """Raised when DLQ is at capacity."""
    pass


class RetryExhaustedError(DLQError):
    """Raised when all retry attempts are exhausted."""
    
    def __init__(self, record_id: str, attempts: int):
        super().__init__(
            f"Retry exhausted for record '{record_id}' after {attempts} attempts",
            {"record_id": record_id, "attempts": attempts}
        )
        self.record_id = record_id
        self.attempts = attempts
