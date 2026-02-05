"""
WindowSelector Operator

Selects data within time windows (sliding, tumbling, session).
This is the foundation operator for time-series analytics.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Union
import polars as pl

from analytics_engine.core.operator import Operator, OperatorConfig, OperatorResult
from analytics_engine.core.registry import register_operator
from analytics_engine.core.schema import DataSchema, SchemaField, DataType
from analytics_engine.utils.time import parse_duration, duration_to_polars_interval


class WindowType(str, Enum):
    """Types of time windows."""
    TUMBLING = "tumbling"    # Non-overlapping fixed windows
    SLIDING = "sliding"      # Overlapping windows
    SESSION = "session"      # Gap-based windows


class WindowSelectorConfig(OperatorConfig):
    """Configuration for WindowSelector operator.
    
    Attributes:
        window: Window size as duration string (e.g., "1h", "30m", "1d")
        window_type: Type of window - tumbling, sliding, or session
        timestamp_column: Name of the timestamp column
        slide: Slide interval for sliding windows (e.g., "15m")
        session_gap: Maximum gap for session windows (e.g., "5m")
        start_time: Optional start time for the window range
        end_time: Optional end time for the window range
        include_window_bounds: Whether to include window start/end columns
    """
    window: str = "1h"
    window_type: WindowType = WindowType.TUMBLING
    timestamp_column: str = "timestamp"
    slide: Optional[str] = None  # For sliding windows
    session_gap: Optional[str] = None  # For session windows
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    include_window_bounds: bool = True


# Use centralized time utilities - parse_duration imported from utils.time


def duration_to_polars(duration_str: str) -> str:
    """Convert our duration format to Polars duration format.
    
    Polars uses: ns, us, ms, s, m, h, d, w
    """
    return duration_to_polars_interval(duration_str)


@register_operator
class WindowSelector(Operator[WindowSelectorConfig]):
    """Selects data within time windows.
    
    Supports three window types:
    1. Tumbling: Non-overlapping fixed-size windows
    2. Sliding: Overlapping windows with configurable slide
    3. Session: Windows based on activity gaps
    
    Example:
        # Select hourly tumbling windows
        config = WindowSelectorConfig(window="1h", window_type=WindowType.TUMBLING)
        selector = WindowSelector(config=config)
        result = selector.process(data)
        
        # Select 1-hour sliding windows with 15-minute slide
        config = WindowSelectorConfig(
            window="1h", 
            window_type=WindowType.SLIDING,
            slide="15m"
        )
    """
    
    name = "WindowSelector"
    description = "Select data within time windows (sliding, tumbling, session)"
    version = "1.0.0"
    config_class = WindowSelectorConfig
    
    input_schema = DataSchema(
        fields=[
            SchemaField(
                name="timestamp",
                dtype=DataType.DATETIME,
                required=True,
                description="Timestamp column for windowing"
            ),
        ]
    )
    
    output_schema = DataSchema(
        fields=[
            SchemaField(
                name="window_start",
                dtype=DataType.DATETIME,
                required=False,
                description="Start of the window"
            ),
            SchemaField(
                name="window_end",
                dtype=DataType.DATETIME,
                required=False,
                description="End of the window"
            ),
        ]
    )
    
    def process(self, data: pl.DataFrame) -> OperatorResult:
        """Apply window selection to the data.
        
        Args:
            data: Input DataFrame with timestamp column
            
        Returns:
            OperatorResult with windowed data
        """
        try:
            config = self._config or WindowSelectorConfig()
            ts_col = config.timestamp_column
            
            # Validate timestamp column exists
            if ts_col not in data.columns:
                return OperatorResult(
                    success=False,
                    error=f"Timestamp column '{ts_col}' not found in data"
                )
            
            # Ensure timestamp is datetime type
            if data[ts_col].dtype != pl.Datetime:
                data = data.with_columns(
                    pl.col(ts_col).cast(pl.Datetime).alias(ts_col)
                )
            
            # Apply time range filter if specified
            if config.start_time:
                data = data.filter(pl.col(ts_col) >= config.start_time)
            if config.end_time:
                data = data.filter(pl.col(ts_col) <= config.end_time)
            
            # Sort by timestamp
            data = data.sort(ts_col)
            
            # Apply windowing based on type
            if config.window_type == WindowType.TUMBLING:
                result = self._apply_tumbling_window(data, config)
            elif config.window_type == WindowType.SLIDING:
                result = self._apply_sliding_window(data, config)
            elif config.window_type == WindowType.SESSION:
                result = self._apply_session_window(data, config)
            else:
                return OperatorResult(
                    success=False,
                    error=f"Unknown window type: {config.window_type}"
                )
            
            return OperatorResult(
                success=True,
                data=result,
                metadata={
                    "window_type": config.window_type.value,
                    "window_size": config.window,
                    "row_count": len(result),
                }
            )
            
        except Exception as e:
            return OperatorResult(success=False, error=str(e))
    
    def _apply_tumbling_window(
        self, 
        data: pl.DataFrame, 
        config: WindowSelectorConfig
    ) -> pl.DataFrame:
        """Apply tumbling (non-overlapping) windows.
        
        Tumbling windows partition data into fixed-size, non-overlapping intervals.
        """
        ts_col = config.timestamp_column
        window_duration = parse_duration(config.window)
        window_td = duration_to_polars(config.window)
        
        # Calculate window boundaries using truncation
        result = data.with_columns([
            pl.col(ts_col).dt.truncate(window_td).alias("window_start"),
        ])
        
        # Add window end
        if config.include_window_bounds:
            result = result.with_columns([
                (pl.col("window_start") + window_duration).alias("window_end"),
            ])
        
        return result
    
    def _apply_sliding_window(
        self, 
        data: pl.DataFrame, 
        config: WindowSelectorConfig
    ) -> pl.DataFrame:
        """Apply sliding (overlapping) windows.
        
        Sliding windows overlap based on the slide interval.
        Each row may appear in multiple windows.
        """
        ts_col = config.timestamp_column
        window_duration = parse_duration(config.window)
        slide_duration = parse_duration(config.slide or config.window)
        
        # Get time range
        min_ts = data[ts_col].min()
        max_ts = data[ts_col].max()
        
        # Generate window starts
        window_starts = []
        current = min_ts
        while current <= max_ts:
            window_starts.append(current)
            current = current + slide_duration
        
        # For each window, filter data and add window bounds
        result_frames = []
        for window_start in window_starts:
            window_end = window_start + window_duration
            
            # Filter data within this window
            window_data = data.filter(
                (pl.col(ts_col) >= window_start) & 
                (pl.col(ts_col) < window_end)
            )
            
            if len(window_data) > 0:
                if config.include_window_bounds:
                    window_data = window_data.with_columns([
                        pl.lit(window_start).alias("window_start"),
                        pl.lit(window_end).alias("window_end"),
                    ])
                result_frames.append(window_data)
        
        if result_frames:
            return pl.concat(result_frames)
        return data.head(0)  # Empty frame with schema
    
    def _apply_session_window(
        self, 
        data: pl.DataFrame, 
        config: WindowSelectorConfig
    ) -> pl.DataFrame:
        """Apply session windows based on activity gaps.
        
        Session windows group records that are within a gap threshold of each other.
        A new session starts when the gap exceeds the threshold.
        """
        ts_col = config.timestamp_column
        gap_duration = parse_duration(config.session_gap or "5m")
        
        if len(data) == 0:
            if config.include_window_bounds:
                return data.with_columns([
                    pl.lit(None).cast(pl.Datetime).alias("window_start"),
                    pl.lit(None).cast(pl.Datetime).alias("window_end"),
                    pl.lit(None).cast(pl.Int64).alias("session_id"),
                ])
            return data
        
        # Calculate time differences
        data = data.with_columns([
            (pl.col(ts_col) - pl.col(ts_col).shift(1)).alias("_time_diff"),
        ])
        
        # Mark new sessions where gap exceeds threshold
        data = data.with_columns([
            (
                (pl.col("_time_diff").is_null()) |
                (pl.col("_time_diff") > gap_duration)
            ).alias("_is_new_session"),
        ])
        
        # Assign session IDs
        data = data.with_columns([
            pl.col("_is_new_session").cum_sum().alias("session_id"),
        ])
        
        # Calculate window bounds per session
        if config.include_window_bounds:
            session_bounds = data.group_by("session_id").agg([
                pl.col(ts_col).min().alias("window_start"),
                pl.col(ts_col).max().alias("window_end"),
            ])
            
            data = data.join(session_bounds, on="session_id", how="left")
        
        # Clean up temporary columns
        data = data.drop(["_time_diff", "_is_new_session"])
        
        return data
    
    def get_windows(
        self, 
        data: pl.DataFrame
    ) -> List[pl.DataFrame]:
        """Split data into individual window DataFrames.
        
        Useful for processing each window separately.
        
        Args:
            data: DataFrame with window_start column (after process())
            
        Returns:
            List of DataFrames, one per window
        """
        if "window_start" not in data.columns:
            result = self.process(data)
            if not result.success:
                return []
            data = result.data
        
        windows = []
        for window_start in data["window_start"].unique().sort():
            window_data = data.filter(pl.col("window_start") == window_start)
            windows.append(window_data)
        
        return windows
