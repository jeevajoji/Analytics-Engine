"""
Analytics Engine Operators.

All core operators for the Analytics Engine.
"""

# Tier 1: Foundation
from .window_selector import (
    WindowSelector,
    WindowSelectorConfig,
    WindowType,
    parse_duration,
)
from .filter_operator import (
    FilterOperator,
    FilterOperatorConfig,
    FilterCondition,
    FilterMode,
    ComparisonOperator,
    filter_data,
)
from .aggregator import (
    Aggregator,
    AggregatorConfig,
    MetricConfig,
    AggregateFunction,
    aggregate,
    summarize,
)

# Tier 2: Data Combination
from .join_operator import (
    JoinOperator,
    JoinOperatorConfig,
    JoinType,
    AsofStrategy,
    MultiJoin,
    join_dataframes,
)

# Tier 3: Rule Engine Core
from .threshold_evaluator import (
    ThresholdEvaluator,
    ThresholdEvaluatorConfig,
    ThresholdRule,
    ThresholdType,
    evaluate_threshold,
)
from .event_builder import (
    EventBuilder,
    EventBuilderConfig,
    Event,
    EventSeverity,
    EventStatus,
    build_events,
)

# Tier 4: ML & Analytics
from .anomaly_detector import (
    AnomalyDetector,
    AnomalyDetectorConfig,
    AnomalyMethod,
)
from .forecast_operator import (
    ForecastOperator,
    ForecastOperatorConfig,
    ForecastMethod,
    SeasonalityType,
)
from .cluster_operator import (
    ClusterOperator,
    ClusterOperatorConfig,
    ClusterMethod,
    LinkageType,
)

# Tier 5: ML Wrappers
from .classification_operator import (
    ClassificationOperator,
    ClassificationOperatorConfig,
    ModelType,
    classify,
)
from .regression_operator import (
    RegressionOperator,
    RegressionOperatorConfig,
    RegressionModelType,
    regress,
)

# Tier 7: Output
from .reporter import (
    Reporter,
    ReporterConfig,
    OutputDestination,
    FileFormat,
    LogLevel,
    report,
)

__all__ = [
    # Tier 1
    "WindowSelector",
    "WindowSelectorConfig",
    "WindowType",
    "parse_duration",
    "FilterOperator",
    "FilterOperatorConfig",
    "FilterCondition",
    "FilterMode",
    "ComparisonOperator",
    "filter_data",
    "Aggregator",
    "AggregatorConfig",
    "MetricConfig",
    "AggregateFunction",
    "aggregate",
    "summarize",
    # Tier 2
    "JoinOperator",
    "JoinOperatorConfig",
    "JoinType",
    "AsofStrategy",
    "MultiJoin",
    "join_dataframes",
    # Tier 3
    "ThresholdEvaluator",
    "ThresholdEvaluatorConfig",
    "ThresholdRule",
    "ThresholdType",
    "evaluate_threshold",
    "EventBuilder",
    "EventBuilderConfig",
    "Event",
    "EventSeverity",
    "EventStatus",
    "build_events",
    # Tier 4
    "AnomalyDetector",
    "AnomalyDetectorConfig",
    "AnomalyMethod",
    "ForecastOperator",
    "ForecastOperatorConfig",
    "ForecastMethod",
    "SeasonalityType",
    "ClusterOperator",
    "ClusterOperatorConfig",
    "ClusterMethod",
    "LinkageType",
    # Tier 5
    "ClassificationOperator",
    "ClassificationOperatorConfig",
    "ModelType",
    "classify",
    "RegressionOperator",
    "RegressionOperatorConfig",
    "RegressionModelType",
    "regress",
    # Tier 7
    "Reporter",
    "ReporterConfig",
    "OutputDestination",
    "FileFormat",
    "LogLevel",
    "report",
]


