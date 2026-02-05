"""
Analytics Engine Configuration.

Settings and configuration management.
"""

from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class EngineSettings(BaseSettings):
    """Global settings for the Analytics Engine."""
    
    # General
    engine_name: str = Field(default="analytics_engine", description="Engine name")
    version: str = Field(default="1.0.0", description="Engine version")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Processing
    default_chunk_size: int = Field(default=10000, description="Default chunk size")
    max_memory_mb: int = Field(default=1024, description="Max memory usage in MB")
    
    # Real-time lane
    realtime_timeout_ms: int = Field(default=100, description="Real-time processing timeout")
    
    # Batch lane
    batch_max_workers: int = Field(default=4, description="Max parallel workers for batch")
    
    # Redis (for real-time)
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database")
    
    # InfluxDB (for time-series)
    influxdb_url: Optional[str] = Field(default=None, description="InfluxDB URL")
    influxdb_token: Optional[str] = Field(default=None, description="InfluxDB token")
    influxdb_org: Optional[str] = Field(default=None, description="InfluxDB organization")
    influxdb_bucket: Optional[str] = Field(default=None, description="InfluxDB bucket")
    
    # PostgreSQL (for events/incidents)
    postgres_host: str = Field(default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_db: str = Field(default="iop", description="PostgreSQL database")
    postgres_user: str = Field(default="iop", description="PostgreSQL user")
    postgres_password: str = Field(default="", description="PostgreSQL password")
    
    class Config:
        env_prefix = "AE_"
        env_file = ".env"


# Global settings instance
_settings: Optional[EngineSettings] = None


def get_settings() -> EngineSettings:
    """Get global settings instance."""
    global _settings
    if _settings is None:
        _settings = EngineSettings()
    return _settings


def configure(settings: EngineSettings) -> None:
    """Set global settings."""
    global _settings
    _settings = settings
