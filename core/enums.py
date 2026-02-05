"""
Centralized Enum Definitions.

Shared enumerations used across multiple operators and modules.
"""

from enum import Enum


# -----------------------------------------------------------------------------
# Comparison Operators
# -----------------------------------------------------------------------------

class ComparisonOperator(str, Enum):
    """
    Comparison operators for filtering and threshold evaluation.
    
    Used by: FilterOperator, ThresholdEvaluator, expression parser
    """
    EQ = "eq"           # Equal to
    NE = "ne"           # Not equal to
    GT = "gt"           # Greater than
    GE = "ge"           # Greater than or equal to
    LT = "lt"           # Less than
    LE = "le"           # Less than or equal to
    IN = "in"           # Value in list
    NOT_IN = "not_in"   # Value not in list
    BETWEEN = "between" # Value between two bounds
    OUTSIDE = "outside" # Value outside bounds
    CONTAINS = "contains"       # String contains
    STARTS_WITH = "starts_with" # String starts with
    ENDS_WITH = "ends_with"     # String ends with
    IS_NULL = "is_null"         # Value is null
    IS_NOT_NULL = "is_not_null" # Value is not null
    REGEX = "regex"             # Regex match


# Aliases for common operator representations
COMPARISON_ALIASES = {
    "==": ComparisonOperator.EQ,
    "=": ComparisonOperator.EQ,
    "!=": ComparisonOperator.NE,
    "<>": ComparisonOperator.NE,
    ">": ComparisonOperator.GT,
    ">=": ComparisonOperator.GE,
    "<": ComparisonOperator.LT,
    "<=": ComparisonOperator.LE,
}


# -----------------------------------------------------------------------------
# Aggregation Types
# -----------------------------------------------------------------------------

class AggregationType(str, Enum):
    """
    Aggregation functions for data summarization.
    
    Used by: Aggregator, WindowSelector
    """
    SUM = "sum"
    MEAN = "mean"
    AVG = "avg"         # Alias for mean
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    COUNT_DISTINCT = "count_distinct"
    FIRST = "first"
    LAST = "last"
    MEDIAN = "median"
    STD = "std"         # Standard deviation
    VAR = "var"         # Variance
    RANGE = "range"     # max - min
    PERCENTILE = "percentile"


# -----------------------------------------------------------------------------
# Window Types
# -----------------------------------------------------------------------------

class WindowType(str, Enum):
    """
    Types of time windows for temporal analysis.
    
    Used by: WindowSelector
    """
    TUMBLING = "tumbling"   # Non-overlapping fixed windows
    SLIDING = "sliding"     # Overlapping windows
    SESSION = "session"     # Gap-based sessions
    EXPANDING = "expanding" # Cumulative from start
    ROLLING = "rolling"     # Rolling count-based window


# -----------------------------------------------------------------------------
# Join Types
# -----------------------------------------------------------------------------

class JoinType(str, Enum):
    """
    Types of joins for combining data.
    
    Used by: JoinOperator
    """
    INNER = "inner"
    LEFT = "left"
    RIGHT = "right"
    OUTER = "outer"
    CROSS = "cross"
    ASOF = "asof"           # Time-based join
    ASOF_BACKWARD = "asof_backward"
    ASOF_FORWARD = "asof_forward"


# -----------------------------------------------------------------------------
# Severity Levels
# -----------------------------------------------------------------------------

class SeverityLevel(str, Enum):
    """
    Severity levels for events and alerts.
    
    Used by: EventBuilder, ThresholdEvaluator
    """
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# -----------------------------------------------------------------------------
# Filter Modes
# -----------------------------------------------------------------------------

class FilterMode(str, Enum):
    """
    Modes for combining multiple filter conditions.
    
    Used by: FilterOperator
    """
    AND = "and"     # All conditions must match
    OR = "or"       # Any condition must match
    ALL = "all"     # Alias for AND
    ANY = "any"     # Alias for OR


# -----------------------------------------------------------------------------
# ML/Analytics Related
# -----------------------------------------------------------------------------

class AnomalyMethod(str, Enum):
    """
    Methods for anomaly detection.
    
    Used by: AnomalyDetector
    """
    ZSCORE = "zscore"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"
    MAD = "mad"         # Median Absolute Deviation
    DBSCAN = "dbscan"
    LOF = "lof"         # Local Outlier Factor
    ROLLING_ZSCORE = "rolling_zscore"


class ForecastMethod(str, Enum):
    """
    Methods for time series forecasting.
    
    Used by: ForecastOperator
    """
    NAIVE = "naive"
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    HOLT = "holt"
    HOLT_WINTERS = "holt_winters"
    ARIMA = "arima"


class ClusterMethod(str, Enum):
    """
    Methods for clustering.
    
    Used by: ClusterOperator
    """
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"
    SPECTRAL = "spectral"


class RegressionMethod(str, Enum):
    """
    Methods for regression analysis.
    
    Used by: RegressionOperator
    """
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"


class ClassificationMethod(str, Enum):
    """
    Methods for classification.
    
    Used by: ClassificationOperator
    """
    LOGISTIC = "logistic"
    RANDOM_FOREST = "random_forest"
    DECISION_TREE = "decision_tree"
    SVM = "svm"
    KNN = "knn"
    NAIVE_BAYES = "naive_bayes"


# -----------------------------------------------------------------------------
# Data Types
# -----------------------------------------------------------------------------

class DataType(str, Enum):
    """
    Data types for schema definitions.
    
    Used by: DataSchema, SchemaRegistry
    """
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    DATE = "date"
    TIME = "time"
    DURATION = "duration"
    LIST = "list"
    OBJECT = "object"
    BINARY = "binary"
    ANY = "any"


# -----------------------------------------------------------------------------
# Status Types
# -----------------------------------------------------------------------------

class PipelineStatus(str, Enum):
    """
    Status of a pipeline execution.
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OperatorStatus(str, Enum):
    """
    Status of an operator within a pipeline.
    """
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


# -----------------------------------------------------------------------------
# Lane Types
# -----------------------------------------------------------------------------

class LaneType(str, Enum):
    """
    Types of processing lanes.
    
    Used by: Pipeline configuration
    """
    RULESET = "ruleset"     # Real-time, <100ms
    ANALYTICS = "analytics" # Batch processing
