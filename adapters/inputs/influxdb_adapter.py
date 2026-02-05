"""
InfluxDB Input Adapter.

Provides time-series data fetching for the Analytics Lane.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import logging
import polars as pl

from ..lanes.analytics_lane import DataFetcher


logger = logging.getLogger(__name__)


class InfluxDBAdapter(DataFetcher):
    """
    InfluxDB input adapter for time-series data.
    
    Fetches historical data for batch processing in the Analytics Lane.
    
    Example:
        adapter = InfluxDBAdapter(
            url="http://localhost:8086",
            token="my-token",
            org="my-org",
            bucket="telemetry"
        )
        
        lane.set_data_fetcher(adapter)
    """
    
    def __init__(
        self,
        url: str = "http://localhost:8086",
        token: Optional[str] = None,
        org: str = "default",
        bucket: str = "telemetry",
        timeout: int = 30000,
    ):
        """
        Initialize InfluxDB adapter.
        
        Args:
            url: InfluxDB URL
            token: Authentication token
            org: Organization name
            bucket: Default bucket name
            timeout: Query timeout in milliseconds
        """
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.timeout = timeout
        
        self._client = None
        self._query_api = None
    
    async def connect(self) -> None:
        """Connect to InfluxDB."""
        try:
            from influxdb_client import InfluxDBClient
            from influxdb_client.client.query_api import QueryApi
            
            self._client = InfluxDBClient(
                url=self.url,
                token=self.token,
                org=self.org,
                timeout=self.timeout,
            )
            self._query_api = self._client.query_api()
            
            logger.info(f"Connected to InfluxDB at {self.url}")
            
        except ImportError:
            raise ImportError("influxdb-client package required: pip install influxdb-client")
    
    async def disconnect(self) -> None:
        """Disconnect from InfluxDB."""
        if self._client:
            self._client.close()
            self._client = None
            self._query_api = None
    
    async def fetch(
        self,
        pipeline_name: str,
        window: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        measurement: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> pl.DataFrame:
        """
        Fetch data from InfluxDB.
        
        Args:
            pipeline_name: Name of the pipeline (used for measurement if not specified)
            window: Time window (e.g., "1h", "1d")
            start_time: Start time (overrides window)
            end_time: End time (default: now)
            measurement: Measurement name (default: pipeline_name)
            filters: Additional tag filters
            
        Returns:
            Polars DataFrame with fetched data
        """
        if not self._query_api:
            raise RuntimeError("Not connected to InfluxDB")
        
        # Determine time range
        end = end_time or datetime.utcnow()
        if start_time:
            start = start_time
        elif window:
            start = end - self._parse_window(window)
        else:
            start = end - timedelta(hours=1)
        
        # Build Flux query
        measurement_name = measurement or pipeline_name
        query = self._build_query(
            bucket=self.bucket,
            measurement=measurement_name,
            start=start,
            end=end,
            filters=filters,
        )
        
        logger.debug(f"Executing InfluxDB query: {query}")
        
        # Execute query
        tables = self._query_api.query(query, org=self.org)
        
        # Convert to Polars DataFrame
        return self._tables_to_polars(tables)
    
    def _parse_window(self, window: str) -> timedelta:
        """Parse window string to timedelta."""
        from analytics_engine.operators.window_selector import parse_duration
        return parse_duration(window)
    
    def _build_query(
        self,
        bucket: str,
        measurement: str,
        start: datetime,
        end: datetime,
        filters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build Flux query string."""
        # Format times for Flux
        start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str = end.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        query_parts = [
            f'from(bucket: "{bucket}")',
            f'  |> range(start: {start_str}, stop: {end_str})',
            f'  |> filter(fn: (r) => r._measurement == "{measurement}")',
        ]
        
        # Add tag filters
        if filters:
            for key, value in filters.items():
                if isinstance(value, list):
                    values_str = " or ".join([f'r.{key} == "{v}"' for v in value])
                    query_parts.append(f'  |> filter(fn: (r) => {values_str})')
                else:
                    query_parts.append(f'  |> filter(fn: (r) => r.{key} == "{value}")')
        
        # Pivot to get field values as columns
        query_parts.append('  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")')
        
        return "\n".join(query_parts)
    
    def _tables_to_polars(self, tables) -> pl.DataFrame:
        """Convert InfluxDB tables to Polars DataFrame."""
        records = []
        
        for table in tables:
            for record in table.records:
                row = {
                    "timestamp": record.get_time(),
                    "measurement": record.get_measurement(),
                }
                
                # Add all values
                values = record.values
                for key, value in values.items():
                    if key not in ["_start", "_stop", "_time", "_measurement", "result", "table"]:
                        row[key] = value
                
                records.append(row)
        
        if not records:
            return pl.DataFrame()
        
        return pl.DataFrame(records)
    
    def query(self, flux_query: str) -> pl.DataFrame:
        """Execute a raw Flux query.
        
        Args:
            flux_query: Flux query string
            
        Returns:
            Polars DataFrame with results
        """
        if not self._query_api:
            raise RuntimeError("Not connected to InfluxDB")
        
        tables = self._query_api.query(flux_query, org=self.org)
        return self._tables_to_polars(tables)
