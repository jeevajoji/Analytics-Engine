"""
PostgreSQL Output Adapter.

Writes events, incidents, and insights to PostgreSQL.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import logging
import json

from ..lanes.ruleset_lane import OutputAdapter
from ..lanes.analytics_lane import AnalyticsJob, InsightWriter


logger = logging.getLogger(__name__)


class PostgresOutputAdapter(OutputAdapter, InsightWriter):
    """
    PostgreSQL output adapter for events and insights.
    
    Writes events, incidents, and insights to PostgreSQL tables.
    
    Example:
        adapter = PostgresOutputAdapter(
            host="localhost",
            database="iop",
            user="analytics",
            password="secret",
            events_table="events",
            insights_table="insights"
        )
        
        lane.add_output_adapter(adapter)
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "iop",
        user: str = "postgres",
        password: Optional[str] = None,
        events_table: str = "events",
        incidents_table: str = "incidents",
        insights_table: str = "insights",
        schema: str = "public",
        pool_size: int = 5,
    ):
        """
        Initialize PostgreSQL adapter.
        
        Args:
            host: PostgreSQL host
            port: PostgreSQL port
            database: Database name
            user: Username
            password: Password
            events_table: Table for events
            incidents_table: Table for incidents
            insights_table: Table for insights
            schema: Schema name
            pool_size: Connection pool size
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.events_table = events_table
        self.incidents_table = incidents_table
        self.insights_table = insights_table
        self.schema = schema
        self.pool_size = pool_size
        
        self._pool = None
    
    async def connect(self) -> None:
        """Connect to PostgreSQL."""
        try:
            import asyncpg
            
            self._pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=1,
                max_size=self.pool_size,
            )
            
            # Ensure tables exist
            await self._ensure_tables()
            
            logger.info(f"Connected to PostgreSQL at {self.host}:{self.port}/{self.database}")
            
        except ImportError:
            raise ImportError("asyncpg package required: pip install asyncpg")
    
    async def disconnect(self) -> None:
        """Disconnect from PostgreSQL."""
        if self._pool:
            await self._pool.close()
            self._pool = None
    
    async def _ensure_tables(self) -> None:
        """Create tables if they don't exist."""
        async with self._pool.acquire() as conn:
            # Events table
            await conn.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.schema}.{self.events_table} (
                    id SERIAL PRIMARY KEY,
                    event_id VARCHAR(64) UNIQUE NOT NULL,
                    event_type VARCHAR(100) NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    title TEXT,
                    description TEXT,
                    timestamp TIMESTAMPTZ,
                    asset_id VARCHAR(100),
                    status VARCHAR(20) DEFAULT 'open',
                    payload JSONB,
                    tags TEXT[],
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            ''')
            
            # Create index on event_type and timestamp
            await conn.execute(f'''
                CREATE INDEX IF NOT EXISTS idx_{self.events_table}_type_time 
                ON {self.schema}.{self.events_table} (event_type, timestamp DESC)
            ''')
            
            # Insights table
            await conn.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.schema}.{self.insights_table} (
                    id SERIAL PRIMARY KEY,
                    insight_id VARCHAR(64),
                    pipeline_name VARCHAR(100),
                    job_id VARCHAR(100),
                    insight_type VARCHAR(100),
                    data JSONB,
                    window_start TIMESTAMPTZ,
                    window_end TIMESTAMPTZ,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            ''')
    
    async def write(self, record: Dict[str, Any]) -> None:
        """Write a single record."""
        if not self._pool:
            raise RuntimeError("Not connected to PostgreSQL")
        
        # Determine which table to write to
        if "event_type" in record:
            await self._write_event(record)
        else:
            await self._write_insight(record)
    
    async def write_batch(self, records: List[Dict[str, Any]]) -> None:
        """Write multiple records."""
        if not self._pool:
            raise RuntimeError("Not connected to PostgreSQL")
        
        events = [r for r in records if "event_type" in r]
        insights = [r for r in records if "event_type" not in r]
        
        if events:
            await self._write_events_batch(events)
        if insights:
            await self._write_insights_batch(insights)
    
    async def _write_event(self, record: Dict[str, Any]) -> None:
        """Write a single event."""
        async with self._pool.acquire() as conn:
            await conn.execute(f'''
                INSERT INTO {self.schema}.{self.events_table}
                (event_id, event_type, severity, title, description, 
                 timestamp, asset_id, status, payload, tags, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (event_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    metadata = EXCLUDED.metadata
            ''',
                record.get("event_id"),
                record.get("event_type"),
                record.get("severity", "warning"),
                record.get("title"),
                record.get("description"),
                self._parse_timestamp(record.get("timestamp")),
                record.get("asset_id"),
                record.get("status", "open"),
                json.dumps(record.get("payload", {})),
                record.get("tags", []),
                json.dumps(record.get("metadata", {})),
            )
    
    async def _write_events_batch(self, records: List[Dict[str, Any]]) -> None:
        """Write multiple events in batch."""
        async with self._pool.acquire() as conn:
            await conn.executemany(f'''
                INSERT INTO {self.schema}.{self.events_table}
                (event_id, event_type, severity, title, description,
                 timestamp, asset_id, status, payload, tags, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (event_id) DO NOTHING
            ''', [
                (
                    r.get("event_id"),
                    r.get("event_type"),
                    r.get("severity", "warning"),
                    r.get("title"),
                    r.get("description"),
                    self._parse_timestamp(r.get("timestamp")),
                    r.get("asset_id"),
                    r.get("status", "open"),
                    json.dumps(r.get("payload", {})),
                    r.get("tags", []),
                    json.dumps(r.get("metadata", {})),
                )
                for r in records
            ])
    
    async def _write_insight(self, record: Dict[str, Any]) -> None:
        """Write a single insight."""
        async with self._pool.acquire() as conn:
            await conn.execute(f'''
                INSERT INTO {self.schema}.{self.insights_table}
                (insight_id, pipeline_name, job_id, insight_type, data,
                 window_start, window_end)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            ''',
                record.get("insight_id"),
                record.get("pipeline_name"),
                record.get("job_id"),
                record.get("insight_type"),
                json.dumps(record),
                self._parse_timestamp(record.get("window_start")),
                self._parse_timestamp(record.get("window_end")),
            )
    
    async def _write_insights_batch(self, records: List[Dict[str, Any]]) -> None:
        """Write multiple insights in batch."""
        async with self._pool.acquire() as conn:
            await conn.executemany(f'''
                INSERT INTO {self.schema}.{self.insights_table}
                (insight_id, pipeline_name, job_id, insight_type, data,
                 window_start, window_end)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            ''', [
                (
                    r.get("insight_id"),
                    r.get("pipeline_name"),
                    r.get("job_id"),
                    r.get("insight_type"),
                    json.dumps(r),
                    self._parse_timestamp(r.get("window_start")),
                    self._parse_timestamp(r.get("window_end")),
                )
                for r in records
            ])
    
    async def write(self, job: AnalyticsJob) -> None:
        """Write job results to insights table (InsightWriter interface)."""
        if job.result is None or job.result.output_data is None:
            return
        
        records = job.result.output_data.to_dicts()
        for record in records:
            record["pipeline_name"] = job.pipeline.name
            record["job_id"] = job.job_id
        
        await self._write_insights_batch(records)
    
    def _parse_timestamp(self, ts: Any) -> Optional[datetime]:
        """Parse timestamp from various formats."""
        if ts is None:
            return None
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                return None
        return None
