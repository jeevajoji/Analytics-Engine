"""
Reporter Operator.

Emits final output to various destinations:
- PostgreSQL: Events, incidents, insights
- Notifications: Email, SMS, Push, Webhook
- Logs: Structured logging
- Files: CSV, JSON, Parquet
- Redis: Real-time state updates
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import polars as pl
from pydantic import BaseModel, Field, field_validator

from ..core.operator import Operator, OperatorConfig, OperatorResult
from ..core.registry import register_operator
from ..core.schema import DataSchema, DataType, SchemaField


class OutputDestination(str, Enum):
    """Supported output destinations."""
    
    POSTGRES = "postgres"
    NOTIFICATION = "notification"
    LOG = "log"
    FILE = "file"
    REDIS = "redis"
    WEBHOOK = "webhook"
    CONSOLE = "console"


class FileFormat(str, Enum):
    """Supported file output formats."""
    
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    NDJSON = "ndjson"  # Newline-delimited JSON


class LogLevel(str, Enum):
    """Log levels for logging output."""
    
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ReporterConfig(BaseModel):
    """Configuration for Reporter operator."""
    
    # Destination settings
    destination: OutputDestination = Field(
        default=OutputDestination.CONSOLE,
        description="Output destination type"
    )
    
    # PostgreSQL settings
    postgres_connection: Optional[str] = Field(
        default=None,
        description="PostgreSQL connection string"
    )
    postgres_table: str = Field(
        default="analytics_output",
        description="Target table name"
    )
    postgres_schema: str = Field(
        default="public",
        description="Database schema"
    )
    postgres_if_exists: str = Field(
        default="append",
        description="What to do if table exists: 'append', 'replace', 'fail'"
    )
    
    # Notification settings
    notification_type: str = Field(
        default="webhook",
        description="Notification type: 'email', 'sms', 'push', 'webhook'"
    )
    notification_url: Optional[str] = Field(
        default=None,
        description="Webhook URL or notification service endpoint"
    )
    notification_template: Optional[str] = Field(
        default=None,
        description="Message template with {column} placeholders"
    )
    notification_severity_column: Optional[str] = Field(
        default=None,
        description="Column containing severity for filtering"
    )
    notification_min_severity: Optional[str] = Field(
        default=None,
        description="Minimum severity to trigger notification"
    )
    
    # File settings
    file_path: Optional[str] = Field(
        default=None,
        description="Output file path"
    )
    file_format: FileFormat = Field(
        default=FileFormat.JSON,
        description="File output format"
    )
    file_append: bool = Field(
        default=False,
        description="Append to existing file"
    )
    file_partition_by: Optional[str] = Field(
        default=None,
        description="Column to partition output by (creates directories)"
    )
    
    # Log settings
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Log level for output"
    )
    log_message_template: str = Field(
        default="Pipeline output: {record_count} records",
        description="Log message template"
    )
    log_include_data: bool = Field(
        default=False,
        description="Include data records in log"
    )
    log_max_records: int = Field(
        default=10,
        description="Maximum records to include in log"
    )
    
    # Redis settings
    redis_url: Optional[str] = Field(
        default=None,
        description="Redis connection URL"
    )
    redis_key: str = Field(
        default="analytics:output",
        description="Redis key or key pattern"
    )
    redis_key_column: Optional[str] = Field(
        default=None,
        description="Column to use for Redis key suffix"
    )
    redis_ttl: Optional[int] = Field(
        default=None,
        description="TTL for Redis keys in seconds"
    )
    
    # Webhook settings
    webhook_url: Optional[str] = Field(
        default=None,
        description="Webhook URL to POST data"
    )
    webhook_headers: Optional[dict[str, str]] = Field(
        default=None,
        description="Custom headers for webhook"
    )
    webhook_batch_size: int = Field(
        default=100,
        ge=1,
        description="Number of records per webhook call"
    )
    
    # Column selection
    output_columns: Optional[list[str]] = Field(
        default=None,
        description="Columns to include in output (None = all)"
    )
    exclude_columns: Optional[list[str]] = Field(
        default=None,
        description="Columns to exclude from output"
    )
    
    # Metadata
    include_metadata: bool = Field(
        default=True,
        description="Include pipeline metadata in output"
    )
    timestamp_column: str = Field(
        default="reported_at",
        description="Name for timestamp column"
    )
    add_timestamp: bool = Field(
        default=True,
        description="Add timestamp to each record"
    )


@register_operator("Reporter")
class Reporter(Operator):
    """
    Emits pipeline output to various destinations.
    
    Supported Destinations:
    - postgres: Write to PostgreSQL table
    - notification: Send alerts via email/SMS/push/webhook
    - log: Structured logging output
    - file: CSV, JSON, Parquet files
    - redis: Redis key-value or stream
    - webhook: HTTP POST to endpoint
    - console: Print to stdout
    
    Example Config:
        {
            "destination": "postgres",
            "postgres_table": "insights",
            "output_columns": ["entity_id", "metric", "value", "timestamp"],
            "add_timestamp": true
        }
    """
    
    name = "Reporter"
    
    input_schema = DataSchema(
        fields=[
            SchemaField(name="*", dtype=DataType.STRING, required=False),
        ],
        description="Any DataFrame to report"
    )
    
    output_schema = DataSchema(
        fields=[
            SchemaField(name="reported", dtype=DataType.BOOLEAN, required=True),
        ],
        description="Report status"
    )
    
    def __init__(self, config: Optional[OperatorConfig] = None):
        super().__init__(config)
        self._parsed_config: Optional[ReporterConfig] = None
    
    def validate_config(self) -> bool:
        """Validate the operator configuration."""
        if not self.config or not self.config.params:
            return False
        
        try:
            self._parsed_config = ReporterConfig(**self.config.params)
            return True
        except Exception as e:
            raise ValueError(f"Invalid Reporter config: {e}")
    
    def process(self, data: pl.DataFrame) -> OperatorResult:
        """
        Report data to configured destination.
        
        Args:
            data: Input DataFrame to report
            
        Returns:
            OperatorResult with report status
        """
        if self._parsed_config is None:
            self.validate_config()
        
        config = self._parsed_config
        
        try:
            # Prepare data
            output_data = self._prepare_data(data, config)
            
            # Route to destination handler
            handlers = {
                OutputDestination.POSTGRES: self._report_postgres,
                OutputDestination.NOTIFICATION: self._report_notification,
                OutputDestination.LOG: self._report_log,
                OutputDestination.FILE: self._report_file,
                OutputDestination.REDIS: self._report_redis,
                OutputDestination.WEBHOOK: self._report_webhook,
                OutputDestination.CONSOLE: self._report_console,
            }
            
            handler = handlers.get(config.destination)
            if not handler:
                raise ValueError(f"Unknown destination: {config.destination}")
            
            report_result = handler(output_data, config)
            
            return OperatorResult(
                success=True,
                data=data,  # Pass through original data
                metadata={
                    "destination": config.destination.value,
                    "record_count": len(output_data),
                    "reported_at": datetime.utcnow().isoformat(),
                    **report_result,
                }
            )
            
        except Exception as e:
            return OperatorResult(
                success=False,
                data=data,
                error=f"Reporting failed: {e}"
            )
    
    def _prepare_data(
        self,
        data: pl.DataFrame,
        config: ReporterConfig
    ) -> pl.DataFrame:
        """Prepare data for output."""
        
        result = data
        
        # Add timestamp
        if config.add_timestamp:
            result = result.with_columns([
                pl.lit(datetime.utcnow()).alias(config.timestamp_column),
            ])
        
        # Select columns
        if config.output_columns:
            cols_to_select = [c for c in config.output_columns if c in result.columns]
            if config.add_timestamp and config.timestamp_column not in cols_to_select:
                cols_to_select.append(config.timestamp_column)
            result = result.select(cols_to_select)
        
        # Exclude columns
        if config.exclude_columns:
            cols_to_drop = [c for c in config.exclude_columns if c in result.columns]
            result = result.drop(cols_to_drop)
        
        return result
    
    def _report_postgres(
        self,
        data: pl.DataFrame,
        config: ReporterConfig
    ) -> dict:
        """Write data to PostgreSQL."""
        
        if not config.postgres_connection:
            raise ValueError("PostgreSQL connection string required")
        
        try:
            import sqlalchemy
            from sqlalchemy import create_engine
        except ImportError:
            raise ImportError(
                "sqlalchemy is required for PostgreSQL output. "
                "Install with: pip install sqlalchemy psycopg2-binary"
            )
        
        engine = create_engine(config.postgres_connection)
        
        # Convert to pandas for SQLAlchemy compatibility
        df_pandas = data.to_pandas()
        
        df_pandas.to_sql(
            config.postgres_table,
            engine,
            schema=config.postgres_schema,
            if_exists=config.postgres_if_exists,
            index=False,
        )
        
        return {
            "table": f"{config.postgres_schema}.{config.postgres_table}",
            "rows_written": len(data),
        }
    
    def _report_notification(
        self,
        data: pl.DataFrame,
        config: ReporterConfig
    ) -> dict:
        """Send notifications."""
        
        import json
        
        # Filter by severity if configured
        if config.notification_severity_column and config.notification_min_severity:
            severity_order = ["debug", "info", "warning", "error", "critical"]
            min_idx = severity_order.index(config.notification_min_severity.lower())
            
            if config.notification_severity_column in data.columns:
                data = data.filter(
                    pl.col(config.notification_severity_column).str.to_lowercase().is_in(
                        severity_order[min_idx:]
                    )
                )
        
        if len(data) == 0:
            return {"notifications_sent": 0, "reason": "no_matching_records"}
        
        # Send to webhook
        if config.notification_url:
            try:
                import httpx
                
                payload = {
                    "records": data.to_dicts(),
                    "count": len(data),
                    "timestamp": datetime.utcnow().isoformat(),
                }
                
                response = httpx.post(
                    config.notification_url,
                    json=payload,
                    timeout=30.0,
                )
                response.raise_for_status()
                
                return {
                    "notifications_sent": len(data),
                    "status_code": response.status_code,
                }
                
            except ImportError:
                # Fallback without httpx
                import urllib.request
                
                payload = json.dumps({
                    "records": data.to_dicts(),
                    "count": len(data),
                }).encode()
                
                req = urllib.request.Request(
                    config.notification_url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return {
                        "notifications_sent": len(data),
                        "status_code": resp.status,
                    }
        
        return {"notifications_sent": 0, "reason": "no_notification_url"}
    
    def _report_log(
        self,
        data: pl.DataFrame,
        config: ReporterConfig
    ) -> dict:
        """Log data using Python logging."""
        
        import logging
        
        logger = logging.getLogger("analytics_engine.reporter")
        
        log_func = {
            LogLevel.DEBUG: logger.debug,
            LogLevel.INFO: logger.info,
            LogLevel.WARNING: logger.warning,
            LogLevel.ERROR: logger.error,
            LogLevel.CRITICAL: logger.critical,
        }[config.log_level]
        
        # Format message
        message = config.log_message_template.format(
            record_count=len(data),
            columns=list(data.columns),
        )
        
        log_func(message)
        
        # Optionally log records
        if config.log_include_data:
            records = data.head(config.log_max_records).to_dicts()
            for i, record in enumerate(records):
                log_func(f"  Record {i+1}: {record}")
        
        return {
            "log_level": config.log_level.value,
            "records_logged": min(len(data), config.log_max_records) if config.log_include_data else 0,
        }
    
    def _report_file(
        self,
        data: pl.DataFrame,
        config: ReporterConfig
    ) -> dict:
        """Write data to file."""
        
        if not config.file_path:
            raise ValueError("File path required for file output")
        
        file_path = Path(config.file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Handle partitioning
        if config.file_partition_by and config.file_partition_by in data.columns:
            written_files = []
            for partition_val, partition_data in data.group_by(config.file_partition_by):
                partition_dir = file_path.parent / str(partition_val)
                partition_dir.mkdir(parents=True, exist_ok=True)
                partition_file = partition_dir / file_path.name
                
                self._write_file(partition_data, partition_file, config)
                written_files.append(str(partition_file))
            
            return {"files_written": written_files, "partitions": len(written_files)}
        
        self._write_file(data, file_path, config)
        
        return {"file": str(file_path), "rows_written": len(data)}
    
    def _write_file(
        self,
        data: pl.DataFrame,
        path: Path,
        config: ReporterConfig
    ) -> None:
        """Write DataFrame to file in specified format."""
        
        if config.file_format == FileFormat.CSV:
            if config.file_append and path.exists():
                existing = pl.read_csv(path)
                data = pl.concat([existing, data])
            data.write_csv(path)
            
        elif config.file_format == FileFormat.JSON:
            data.write_json(path)
            
        elif config.file_format == FileFormat.NDJSON:
            data.write_ndjson(path)
            
        elif config.file_format == FileFormat.PARQUET:
            data.write_parquet(path)
    
    def _report_redis(
        self,
        data: pl.DataFrame,
        config: ReporterConfig
    ) -> dict:
        """Write data to Redis."""
        
        if not config.redis_url:
            raise ValueError("Redis URL required for Redis output")
        
        try:
            import redis
        except ImportError:
            raise ImportError(
                "redis is required for Redis output. "
                "Install with: pip install redis"
            )
        
        import json
        
        client = redis.from_url(config.redis_url)
        
        records_written = 0
        
        if config.redis_key_column and config.redis_key_column in data.columns:
            # Write each record with individual key
            for record in data.to_dicts():
                key_suffix = record.get(config.redis_key_column, records_written)
                key = f"{config.redis_key}:{key_suffix}"
                
                client.set(key, json.dumps(record))
                if config.redis_ttl:
                    client.expire(key, config.redis_ttl)
                
                records_written += 1
        else:
            # Write all data under single key
            client.set(config.redis_key, json.dumps(data.to_dicts()))
            if config.redis_ttl:
                client.expire(config.redis_key, config.redis_ttl)
            records_written = len(data)
        
        return {"redis_key": config.redis_key, "records_written": records_written}
    
    def _report_webhook(
        self,
        data: pl.DataFrame,
        config: ReporterConfig
    ) -> dict:
        """POST data to webhook."""
        
        if not config.webhook_url:
            raise ValueError("Webhook URL required")
        
        import json
        
        try:
            import httpx
            client = httpx
        except ImportError:
            client = None
        
        records = data.to_dicts()
        batches_sent = 0
        
        # Send in batches
        for i in range(0, len(records), config.webhook_batch_size):
            batch = records[i:i + config.webhook_batch_size]
            
            payload = {
                "records": batch,
                "batch_index": i // config.webhook_batch_size,
                "total_records": len(records),
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            headers = {"Content-Type": "application/json"}
            if config.webhook_headers:
                headers.update(config.webhook_headers)
            
            if client:
                response = client.post(
                    config.webhook_url,
                    json=payload,
                    headers=headers,
                    timeout=30.0,
                )
                response.raise_for_status()
            else:
                import urllib.request
                
                req = urllib.request.Request(
                    config.webhook_url,
                    data=json.dumps(payload).encode(),
                    headers=headers,
                )
                urllib.request.urlopen(req, timeout=30)
            
            batches_sent += 1
        
        return {
            "webhook_url": config.webhook_url,
            "batches_sent": batches_sent,
            "records_sent": len(records),
        }
    
    def _report_console(
        self,
        data: pl.DataFrame,
        config: ReporterConfig
    ) -> dict:
        """Print data to console."""
        
        print(f"\n=== Analytics Engine Report ===")
        print(f"Timestamp: {datetime.utcnow().isoformat()}")
        print(f"Records: {len(data)}")
        print(f"Columns: {data.columns}")
        print(f"\n{data}\n")
        
        return {"console_output": True, "records_printed": len(data)}


def report(
    data: pl.DataFrame,
    destination: str = "console",
    **kwargs
) -> dict:
    """
    Convenience function to report data.
    
    Args:
        data: DataFrame to report
        destination: Output destination
        **kwargs: Additional config options
        
    Returns:
        Report metadata
    """
    config = OperatorConfig(
        params={
            "destination": destination,
            **kwargs,
        }
    )
    
    operator = Reporter(config)
    result = operator.process(data)
    
    if not result.success:
        raise ValueError(result.error)
    
    return result.metadata
