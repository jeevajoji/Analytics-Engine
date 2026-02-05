"""
Metrics & Observability Module.

Provides comprehensive metrics collection, tracing, and structured logging
for monitoring Analytics Engine in production environments.
"""

import time
import logging
import threading
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
from functools import wraps
import json
import uuid


logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class MetricUnit(str, Enum):
    """Units for metrics."""
    MILLISECONDS = "ms"
    SECONDS = "s"
    BYTES = "bytes"
    ROWS = "rows"
    COUNT = "count"
    PERCENT = "percent"


@dataclass
class MetricValue:
    """A single metric value."""
    name: str
    value: float
    metric_type: MetricType
    unit: MetricUnit = MetricUnit.COUNT
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "unit": self.unit.value,
            "labels": self.labels,
            "timestamp": self.timestamp.isoformat(),
        }
    
    def to_prometheus(self) -> str:
        """Convert to Prometheus text format."""
        labels_str = ""
        if self.labels:
            label_pairs = [f'{k}="{v}"' for k, v in self.labels.items()]
            labels_str = "{" + ",".join(label_pairs) + "}"
        return f"{self.name}{labels_str} {self.value}"


@dataclass
class HistogramBucket:
    """Histogram bucket for latency distributions."""
    le: float  # less than or equal
    count: int = 0


class Histogram:
    """Histogram for tracking distributions (e.g., latencies)."""
    
    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    
    def __init__(
        self,
        name: str,
        buckets: Optional[List[float]] = None,
        labels: Optional[Dict[str, str]] = None,
    ):
        self.name = name
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self.labels = labels or {}
        self._counts = {b: 0 for b in self.buckets}
        self._counts[float('inf')] = 0
        self._sum = 0.0
        self._count = 0
        self._lock = threading.Lock()
    
    def observe(self, value: float) -> None:
        """Record an observation."""
        with self._lock:
            self._sum += value
            self._count += 1
            for bucket in self.buckets:
                if value <= bucket:
                    self._counts[bucket] += 1
            self._counts[float('inf')] += 1
    
    def get_percentile(self, percentile: float) -> float:
        """Get approximate percentile value."""
        if self._count == 0:
            return 0.0
        
        target = percentile * self._count
        cumulative = 0
        
        for bucket in self.buckets:
            cumulative = self._counts[bucket]
            if cumulative >= target:
                return bucket
        
        return self.buckets[-1] if self.buckets else 0.0
    
    @property
    def mean(self) -> float:
        return self._sum / self._count if self._count > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "buckets": {str(k): v for k, v in self._counts.items()},
            "sum": self._sum,
            "count": self._count,
            "mean": self.mean,
            "p50": self.get_percentile(0.5),
            "p95": self.get_percentile(0.95),
            "p99": self.get_percentile(0.99),
        }


@dataclass
class Span:
    """A trace span for distributed tracing."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "ok"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "attributes": attributes or {},
        })
    
    def set_error(self, error: Exception) -> None:
        """Mark span as error."""
        self.status = "error"
        self.attributes["error.type"] = type(error).__name__
        self.attributes["error.message"] = str(error)
    
    def finish(self) -> None:
        """Finish the span."""
        self.end_time = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
        }


class Tracer:
    """Distributed tracing support."""
    
    _current_span: Optional[Span] = None
    _spans: List[Span] = []
    _lock = threading.Lock()
    
    def __init__(self, service_name: str = "analytics_engine"):
        self.service_name = service_name
        self._spans = []
    
    @contextmanager
    def start_span(
        self,
        operation_name: str,
        parent: Optional[Span] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """Start a new span."""
        trace_id = parent.trace_id if parent else str(uuid.uuid4())
        span = Span(
            trace_id=trace_id,
            span_id=str(uuid.uuid4())[:16],
            parent_span_id=parent.span_id if parent else None,
            operation_name=operation_name,
            start_time=datetime.now(),
            attributes=attributes or {},
        )
        span.attributes["service.name"] = self.service_name
        
        previous_span = self._current_span
        self._current_span = span
        
        try:
            yield span
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            span.finish()
            self._current_span = previous_span
            with self._lock:
                self._spans.append(span)
    
    def get_current_span(self) -> Optional[Span]:
        return self._current_span
    
    def get_spans(self, limit: int = 100) -> List[Span]:
        """Get recent spans."""
        with self._lock:
            return self._spans[-limit:]
    
    def clear_spans(self) -> None:
        """Clear stored spans."""
        with self._lock:
            self._spans.clear()


class MetricsCollector:
    """
    Central metrics collection for Analytics Engine.
    
    Collects and exposes metrics for:
    - Pipeline execution (duration, success/failure, rows processed)
    - Operator performance (per-operator latency, throughput)
    - Lane metrics (Ruleset latency, Analytics job duration)
    - System metrics (memory, queue depth)
    
    Example:
        metrics = MetricsCollector()
        
        with metrics.measure_pipeline("my_pipeline"):
            # Execute pipeline
            pass
        
        metrics.increment_counter("records_processed", 1000)
        metrics.set_gauge("queue_depth", 42)
        
        # Export for Prometheus
        print(metrics.to_prometheus())
    """
    
    _instance: Optional["MetricsCollector"] = None
    
    def __new__(cls) -> "MetricsCollector":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._labels: Dict[str, Dict[str, str]] = {}
        self._lock = threading.Lock()
        self._tracer = Tracer()
        self._initialized = True
        
        # Pre-define standard metrics
        self._define_standard_metrics()
    
    def _define_standard_metrics(self) -> None:
        """Define standard Analytics Engine metrics."""
        # Pipeline metrics
        self._histograms["ae_pipeline_duration_seconds"] = Histogram(
            "ae_pipeline_duration_seconds",
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
        )
        
        # Operator metrics
        self._histograms["ae_operator_duration_seconds"] = Histogram(
            "ae_operator_duration_seconds",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        )
        
        # Ruleset Lane metrics
        self._histograms["ae_ruleset_latency_ms"] = Histogram(
            "ae_ruleset_latency_ms",
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
        )
    
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance."""
        cls._instance = None
    
    # -------------------------------------------------------------------------
    # Counter Methods
    # -------------------------------------------------------------------------
    
    def increment_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter metric."""
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + value
            if labels:
                self._labels[key] = labels
    
    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get counter value."""
        key = self._make_key(name, labels)
        return self._counters.get(key, 0.0)
    
    # -------------------------------------------------------------------------
    # Gauge Methods
    # -------------------------------------------------------------------------
    
    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge metric."""
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] = value
            if labels:
                self._labels[key] = labels
    
    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get gauge value."""
        key = self._make_key(name, labels)
        return self._gauges.get(key, 0.0)
    
    # -------------------------------------------------------------------------
    # Histogram Methods
    # -------------------------------------------------------------------------
    
    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a histogram observation."""
        key = self._make_key(name, labels)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = Histogram(name)
            self._histograms[key].observe(value)
            if labels:
                self._labels[key] = labels
    
    def get_histogram(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[Histogram]:
        """Get histogram."""
        key = self._make_key(name, labels)
        return self._histograms.get(key)
    
    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------
    
    @contextmanager
    def measure_time(
        self,
        histogram_name: str,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Context manager to measure execution time."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.observe_histogram(histogram_name, duration, labels)
    
    @contextmanager
    def measure_pipeline(self, pipeline_name: str):
        """Measure pipeline execution."""
        labels = {"pipeline": pipeline_name}
        self.increment_counter("ae_pipeline_executions_total", labels=labels)
        
        with self._tracer.start_span(f"pipeline:{pipeline_name}") as span:
            start = time.perf_counter()
            try:
                yield span
                self.increment_counter("ae_pipeline_success_total", labels=labels)
            except Exception as e:
                self.increment_counter("ae_pipeline_errors_total", labels=labels)
                raise
            finally:
                duration = time.perf_counter() - start
                self.observe_histogram("ae_pipeline_duration_seconds", duration, labels)
    
    @contextmanager
    def measure_operator(self, operator_name: str, pipeline_name: Optional[str] = None):
        """Measure operator execution."""
        labels = {"operator": operator_name}
        if pipeline_name:
            labels["pipeline"] = pipeline_name
        
        parent_span = self._tracer.get_current_span()
        with self._tracer.start_span(f"operator:{operator_name}", parent=parent_span) as span:
            start = time.perf_counter()
            try:
                yield span
            finally:
                duration = time.perf_counter() - start
                self.observe_histogram("ae_operator_duration_seconds", duration, labels)
    
    def record_rows_processed(
        self,
        count: int,
        pipeline_name: Optional[str] = None,
        operator_name: Optional[str] = None,
    ) -> None:
        """Record number of rows processed."""
        labels = {}
        if pipeline_name:
            labels["pipeline"] = pipeline_name
        if operator_name:
            labels["operator"] = operator_name
        
        self.increment_counter("ae_rows_processed_total", count, labels)
    
    def record_event_generated(
        self,
        event_type: str,
        severity: str,
        pipeline_name: Optional[str] = None,
    ) -> None:
        """Record event generation."""
        labels = {"event_type": event_type, "severity": severity}
        if pipeline_name:
            labels["pipeline"] = pipeline_name
        self.increment_counter("ae_events_generated_total", labels=labels)
    
    # -------------------------------------------------------------------------
    # Tracing
    # -------------------------------------------------------------------------
    
    @property
    def tracer(self) -> Tracer:
        """Get the tracer instance."""
        return self._tracer
    
    # -------------------------------------------------------------------------
    # Export Methods
    # -------------------------------------------------------------------------
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Create a unique key for a metric with labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Export all metrics as dictionary."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {k: v.to_dict() for k, v in self._histograms.items()},
                "collected_at": datetime.now().isoformat(),
            }
    
    def to_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        
        with self._lock:
            # Counters
            for key, value in self._counters.items():
                lines.append(f"{key} {value}")
            
            # Gauges
            for key, value in self._gauges.items():
                lines.append(f"{key} {value}")
            
            # Histograms
            for name, histogram in self._histograms.items():
                for bucket, count in histogram._counts.items():
                    le = "+Inf" if bucket == float('inf') else bucket
                    lines.append(f'{name}_bucket{{le="{le}"}} {count}')
                lines.append(f"{name}_sum {histogram._sum}")
                lines.append(f"{name}_count {histogram._count}")
        
        return "\n".join(lines)
    
    def to_json(self) -> str:
        """Export metrics as JSON."""
        return json.dumps(self.to_dict(), indent=2)


# -------------------------------------------------------------------------
# Structured Logging
# -------------------------------------------------------------------------

class StructuredLogger:
    """
    Structured logging with correlation IDs for Analytics Engine.
    
    Example:
        log = StructuredLogger("pipeline.executor")
        
        with log.context(pipeline="my_pipeline", trace_id="abc123"):
            log.info("Starting pipeline execution")
            log.metric("rows_processed", 1000)
    """
    
    _context: Dict[str, Any] = {}
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._local_context: Dict[str, Any] = {}
    
    @contextmanager
    def context(self, **kwargs):
        """Add context to all log messages in this scope."""
        previous = self._local_context.copy()
        self._local_context.update(kwargs)
        try:
            yield
        finally:
            self._local_context = previous
    
    def _format_message(self, message: str, **kwargs) -> str:
        """Format message with context."""
        ctx = {**self._local_context, **kwargs}
        if ctx:
            ctx_str = " ".join(f"{k}={json.dumps(v) if isinstance(v, (dict, list)) else v}" 
                              for k, v in ctx.items())
            return f"{message} | {ctx_str}"
        return message
    
    def _log(self, level: int, message: str, **kwargs) -> None:
        """Log with structured context."""
        formatted = self._format_message(message, **kwargs)
        self.logger.log(level, formatted)
    
    def debug(self, message: str, **kwargs) -> None:
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        self._log(logging.ERROR, message, **kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        self._log(logging.ERROR, message, **kwargs)
        self.logger.exception("")
    
    def metric(self, name: str, value: float, **kwargs) -> None:
        """Log a metric value."""
        self.info(f"metric:{name}={value}", **kwargs)


# -------------------------------------------------------------------------
# Decorators
# -------------------------------------------------------------------------

def timed(histogram_name: str = "ae_function_duration_seconds"):
    """Decorator to measure function execution time."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            metrics = MetricsCollector()
            with metrics.measure_time(histogram_name, {"function": func.__name__}):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def traced(operation_name: Optional[str] = None):
    """Decorator to add tracing to a function."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            metrics = MetricsCollector()
            op_name = operation_name or func.__name__
            with metrics.tracer.start_span(op_name) as span:
                return func(*args, **kwargs)
        return wrapper
    return decorator


# -------------------------------------------------------------------------
# Global Instance
# -------------------------------------------------------------------------

def get_metrics() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return MetricsCollector()


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger."""
    return StructuredLogger(name)
