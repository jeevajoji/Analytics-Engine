"""
EventBuilder Operator

Creates Event/Incident records from threshold violations.
Essential for the Ruleset Lane output.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Any, Dict, Callable
from uuid import uuid4
import polars as pl

from analytics_engine.core.operator import Operator, OperatorConfig, OperatorResult
from analytics_engine.core.registry import register_operator
from analytics_engine.core.schema import DataSchema, SchemaField, DataType


class EventSeverity(str, Enum):
    """Severity levels for events."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EventStatus(str, Enum):
    """Status of an event/incident."""
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


class EventBuilderConfig(OperatorConfig):
    """Configuration for EventBuilder operator.
    
    Attributes:
        event_type: Type identifier for the event (e.g., "TEMPERATURE_ALERT")
        severity: Default severity level
        severity_column: Column containing per-row severity (overrides default)
        title_template: Template for event title (supports {column} placeholders)
        description_template: Template for event description
        violation_column: Column indicating violations (default: "is_violation")
        include_columns: Columns to include in event payload
        exclude_columns: Columns to exclude from event payload
        asset_id_column: Column containing asset identifier
        timestamp_column: Column containing event timestamp
        group_by: Group violations into single events by these columns
        dedup_window: Deduplicate events within this time window (e.g., "5m")
        tags: Static tags to add to all events
        metadata: Static metadata to add to all events
    """
    event_type: str = "GENERIC_EVENT"
    severity: EventSeverity = EventSeverity.WARNING
    severity_column: Optional[str] = None
    title_template: Optional[str] = None
    description_template: Optional[str] = None
    violation_column: str = "is_violation"
    include_columns: Optional[List[str]] = None
    exclude_columns: Optional[List[str]] = None
    asset_id_column: Optional[str] = "asset_id"
    timestamp_column: str = "timestamp"
    group_by: Optional[List[str]] = None
    dedup_window: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class Event:
    """Represents a single event/incident.
    
    This is the output format that can be serialized to database or API.
    """
    
    def __init__(
        self,
        event_id: str,
        event_type: str,
        severity: EventSeverity,
        title: str,
        description: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        asset_id: Optional[str] = None,
        status: EventStatus = EventStatus.OPEN,
        payload: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.event_id = event_id
        self.event_type = event_type
        self.severity = severity
        self.title = title
        self.description = description
        self.timestamp = timestamp or datetime.utcnow()
        self.asset_id = asset_id
        self.status = status
        self.payload = payload or {}
        self.tags = tags or []
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "asset_id": self.asset_id,
            "status": self.status.value,
            "payload": self.payload,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=data["event_type"],
            severity=EventSeverity(data["severity"]),
            title=data["title"],
            description=data.get("description"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None,
            asset_id=data.get("asset_id"),
            status=EventStatus(data.get("status", "open")),
            payload=data.get("payload", {}),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


@register_operator
class EventBuilder(Operator[EventBuilderConfig]):
    """Event builder operator.
    
    Converts threshold violations into structured Event/Incident records.
    
    Example:
        config = EventBuilderConfig(
            event_type="TEMPERATURE_ALERT",
            severity=EventSeverity.WARNING,
            title_template="High temperature on {asset_id}",
            description_template="Temperature {temperature}°C exceeds threshold",
            asset_id_column="sensor_id",
            group_by=["sensor_id"],
        )
        
        builder = EventBuilder(config=config)
        result = builder.process(violation_data)
        
        # Access events
        events = result.metadata["events"]
    
    Output:
        - DataFrame with event columns added
        - List of Event objects in metadata["events"]
    """
    
    name = "EventBuilder"
    description = "Create Event/Incident records from violations"
    version = "1.0.0"
    config_class = EventBuilderConfig
    
    input_schema = DataSchema(
        fields=[
            SchemaField(
                name="is_violation",
                dtype=DataType.BOOLEAN,
                required=False,
                description="Violation indicator from ThresholdEvaluator"
            ),
        ]
    )
    
    output_schema = DataSchema(
        fields=[
            SchemaField(name="event_id", dtype=DataType.STRING, required=True),
            SchemaField(name="event_type", dtype=DataType.STRING, required=True),
            SchemaField(name="severity", dtype=DataType.STRING, required=True),
            SchemaField(name="event_title", dtype=DataType.STRING, required=True),
        ]
    )
    
    def process(self, data: pl.DataFrame) -> OperatorResult:
        """Build events from violation data.
        
        Args:
            data: Input DataFrame with violations
            
        Returns:
            OperatorResult with events
        """
        try:
            config = self._config or EventBuilderConfig()
            
            # Filter to only violations
            violation_col = config.violation_column
            if violation_col in data.columns:
                violations = data.filter(pl.col(violation_col) == True)
            else:
                # No violation column - treat all rows as violations
                violations = data
            
            if len(violations) == 0:
                return OperatorResult(
                    success=True,
                    data=data,
                    metadata={"events": [], "event_count": 0}
                )
            
            # Group violations if configured
            if config.group_by:
                events_df, events = self._build_grouped_events(violations, config)
            else:
                events_df, events = self._build_individual_events(violations, config)
            
            # Apply deduplication if configured
            if config.dedup_window:
                events = self._deduplicate_events(events, config.dedup_window)
            
            return OperatorResult(
                success=True,
                data=events_df,
                metadata={
                    "events": [e.to_dict() for e in events],
                    "event_count": len(events),
                    "violation_count": len(violations),
                }
            )
            
        except Exception as e:
            return OperatorResult(success=False, error=str(e))
    
    def _build_individual_events(
        self,
        data: pl.DataFrame,
        config: EventBuilderConfig,
    ) -> tuple[pl.DataFrame, List[Event]]:
        """Build one event per violation row."""
        events = []
        event_ids = []
        
        # Convert to rows for processing
        rows = data.to_dicts()
        
        for row in rows:
            event = self._create_event(row, config)
            events.append(event)
            event_ids.append(event.event_id)
        
        # Add event columns to DataFrame
        result = data.with_columns([
            pl.Series("event_id", event_ids),
            pl.lit(config.event_type).alias("event_type"),
            pl.lit(config.severity.value).alias("severity"),
        ])
        
        # Add title column
        if config.title_template:
            titles = [self._format_template(config.title_template, row) for row in rows]
            result = result.with_columns(pl.Series("event_title", titles))
        else:
            result = result.with_columns(
                pl.lit(f"{config.event_type} Event").alias("event_title")
            )
        
        return result, events
    
    def _build_grouped_events(
        self,
        data: pl.DataFrame,
        config: EventBuilderConfig,
    ) -> tuple[pl.DataFrame, List[Event]]:
        """Build one event per group of violations."""
        events = []
        
        # Group by specified columns
        groups = data.group_by(config.group_by).agg([
            pl.len().alias("_violation_count"),
            pl.col(config.timestamp_column).min().alias("_first_timestamp"),
            pl.col(config.timestamp_column).max().alias("_last_timestamp"),
        ])
        
        # Add columns from first row of each group for templates
        first_rows = data.group_by(config.group_by).first()
        groups = groups.join(first_rows, on=config.group_by, how="left", suffix="_first")
        
        # Create events for each group
        group_rows = groups.to_dicts()
        event_ids = []
        
        for row in group_rows:
            event = self._create_event(row, config, is_grouped=True)
            event.payload["violation_count"] = row.get("_violation_count", 1)
            event.payload["first_timestamp"] = str(row.get("_first_timestamp"))
            event.payload["last_timestamp"] = str(row.get("_last_timestamp"))
            events.append(event)
            event_ids.append(event.event_id)
        
        # Add event info to groups DataFrame
        result = groups.with_columns([
            pl.Series("event_id", event_ids),
            pl.lit(config.event_type).alias("event_type"),
            pl.lit(config.severity.value).alias("severity"),
        ])
        
        return result, events
    
    def _create_event(
        self,
        row: Dict[str, Any],
        config: EventBuilderConfig,
        is_grouped: bool = False,
    ) -> Event:
        """Create a single Event from a row."""
        # Generate event ID
        event_id = str(uuid4())
        
        # Determine severity
        if config.severity_column and config.severity_column in row:
            severity_val = row[config.severity_column]
            try:
                severity = EventSeverity(severity_val)
            except ValueError:
                severity = config.severity
        else:
            severity = config.severity
        
        # Build title
        if config.title_template:
            title = self._format_template(config.title_template, row)
        else:
            title = f"{config.event_type} Event"
        
        # Build description
        if config.description_template:
            description = self._format_template(config.description_template, row)
        else:
            description = None
        
        # Get timestamp
        timestamp = None
        ts_col = config.timestamp_column
        if ts_col in row and row[ts_col] is not None:
            ts_val = row[ts_col]
            if isinstance(ts_val, datetime):
                timestamp = ts_val
            elif isinstance(ts_val, str):
                timestamp = datetime.fromisoformat(ts_val)
        
        # Get asset ID
        asset_id = None
        if config.asset_id_column and config.asset_id_column in row:
            asset_id = str(row[config.asset_id_column])
        
        # Build payload
        payload = self._build_payload(row, config)
        
        # Combine tags
        tags = list(config.tags) if config.tags else []
        if "violation_labels" in row and row["violation_labels"]:
            labels = row["violation_labels"]
            if isinstance(labels, list):
                tags.extend(labels)
        
        # Combine metadata
        metadata = dict(config.metadata) if config.metadata else {}
        metadata["is_grouped"] = is_grouped
        
        return Event(
            event_id=event_id,
            event_type=config.event_type,
            severity=severity,
            title=title,
            description=description,
            timestamp=timestamp,
            asset_id=asset_id,
            payload=payload,
            tags=tags,
            metadata=metadata,
        )
    
    def _format_template(self, template: str, row: Dict[str, Any]) -> str:
        """Format a template string with row values."""
        try:
            # Replace {column} with row values
            result = template
            for key, value in row.items():
                placeholder = "{" + key + "}"
                if placeholder in result:
                    result = result.replace(placeholder, str(value) if value is not None else "N/A")
            return result
        except Exception:
            return template
    
    def _build_payload(
        self,
        row: Dict[str, Any],
        config: EventBuilderConfig,
    ) -> Dict[str, Any]:
        """Build event payload from row data."""
        payload = {}
        
        # Determine which columns to include
        if config.include_columns:
            columns = config.include_columns
        else:
            columns = list(row.keys())
        
        # Apply exclusions
        exclude = set(config.exclude_columns or [])
        exclude.update(["_violation_count", "_first_timestamp", "_last_timestamp"])
        
        for col in columns:
            if col not in exclude and col in row:
                value = row[col]
                # Convert non-serializable types
                if isinstance(value, datetime):
                    value = value.isoformat()
                elif hasattr(value, "tolist"):  # numpy arrays
                    value = value.tolist()
                payload[col] = value
        
        return payload
    
    def _deduplicate_events(
        self,
        events: List[Event],
        window: str,
    ) -> List[Event]:
        """Deduplicate events within a time window."""
        from analytics_engine.operators.window_selector import parse_duration
        
        window_td = parse_duration(window)
        
        # Group by (event_type, asset_id)
        groups: Dict[tuple, List[Event]] = {}
        for event in events:
            key = (event.event_type, event.asset_id)
            if key not in groups:
                groups[key] = []
            groups[key].append(event)
        
        # Keep only first event within each window
        deduped = []
        for key, group_events in groups.items():
            # Sort by timestamp
            group_events.sort(key=lambda e: e.timestamp or datetime.min)
            
            last_timestamp = None
            for event in group_events:
                if last_timestamp is None:
                    deduped.append(event)
                    last_timestamp = event.timestamp
                elif event.timestamp and (event.timestamp - last_timestamp) > window_td:
                    deduped.append(event)
                    last_timestamp = event.timestamp
        
        return deduped
    
    def get_events(self, result: OperatorResult) -> List[Event]:
        """Extract Event objects from operator result.
        
        Args:
            result: OperatorResult from process()
            
        Returns:
            List of Event objects
        """
        if not result.success or not result.metadata:
            return []
        
        event_dicts = result.metadata.get("events", [])
        return [Event.from_dict(e) for e in event_dicts]


# Convenience function
def build_events(
    data: pl.DataFrame,
    event_type: str,
    severity: str = "warning",
    violation_column: str = "is_violation",
) -> List[Event]:
    """Quick event building from violation data.
    
    Args:
        data: DataFrame with violations
        event_type: Type of event
        severity: Severity level
        violation_column: Column indicating violations
        
    Returns:
        List of Event objects
    """
    config = EventBuilderConfig(
        event_type=event_type,
        severity=EventSeverity(severity),
        violation_column=violation_column,
    )
    
    op = EventBuilder(config=config)
    result = op.process(data)
    
    if result.success:
        return op.get_events(result)
    else:
        raise ValueError(f"Event building failed: {result.error}")
