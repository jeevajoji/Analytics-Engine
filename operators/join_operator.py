"""
JoinOperator

Merge multiple data streams on key columns or time-based alignment.
Supports inner, left, right, outer, and asof joins.
"""

from enum import Enum
from typing import Optional, List, Union
import polars as pl

from analytics_engine.core.operator import Operator, OperatorConfig, OperatorResult
from analytics_engine.core.registry import register_operator
from analytics_engine.core.schema import DataSchema, SchemaField, DataType


class JoinType(str, Enum):
    """Types of joins supported."""
    INNER = "inner"       # Only matching rows
    LEFT = "left"         # All from left, matching from right
    RIGHT = "right"       # All from right, matching from left
    OUTER = "outer"       # All rows from both
    CROSS = "cross"       # Cartesian product
    SEMI = "semi"         # Left rows that have a match
    ANTI = "anti"         # Left rows that have no match
    ASOF = "asof"         # Time-based inexact join


class AsofStrategy(str, Enum):
    """Strategy for asof join matching."""
    BACKWARD = "backward"  # Match previous value (≤)
    FORWARD = "forward"    # Match next value (≥)
    NEAREST = "nearest"    # Match closest value


class JoinOperatorConfig(OperatorConfig):
    """Configuration for JoinOperator.
    
    Attributes:
        join_type: Type of join (inner, left, right, outer, asof, etc.)
        on: Column(s) to join on (when same name in both)
        left_on: Column(s) from left DataFrame
        right_on: Column(s) from right DataFrame
        suffix: Suffix for duplicate column names
        asof_on: Time column for asof join
        asof_by: Grouping columns for asof join
        asof_tolerance: Max time difference for asof match
        asof_strategy: Direction for asof matching
        validate: Validation mode (one_to_one, one_to_many, many_to_one, many_to_many)
    """
    join_type: JoinType = JoinType.INNER
    on: Optional[Union[str, List[str]]] = None
    left_on: Optional[Union[str, List[str]]] = None
    right_on: Optional[Union[str, List[str]]] = None
    suffix: str = "_right"
    # Asof join settings
    asof_on: Optional[str] = None
    asof_by: Optional[Union[str, List[str]]] = None
    asof_tolerance: Optional[str] = None  # e.g., "5m", "1h"
    asof_strategy: AsofStrategy = AsofStrategy.BACKWARD
    # Validation
    validate: Optional[str] = None


@register_operator
class JoinOperator(Operator[JoinOperatorConfig]):
    """Join operator for merging multiple data streams.
    
    Supports standard SQL-style joins and time-series specific asof joins.
    
    Standard Join Example:
        config = JoinOperatorConfig(
            join_type=JoinType.LEFT,
            on="sensor_id"
        )
        joiner = JoinOperator(config=config)
        result = joiner.process(left_data, right_data)
    
    Asof Join Example (time-based):
        config = JoinOperatorConfig(
            join_type=JoinType.ASOF,
            asof_on="timestamp",
            asof_by="sensor_id",
            asof_strategy=AsofStrategy.BACKWARD,
            asof_tolerance="5m"
        )
        joiner = JoinOperator(config=config)
        result = joiner.process(left_data, right_data)
    
    Note: JoinOperator expects two DataFrames. Use `process_join()` method
    or pass the right DataFrame in the config.
    """
    
    name = "JoinOperator"
    description = "Merge multiple streams on key/time"
    version = "1.0.0"
    config_class = JoinOperatorConfig
    
    # Generic operator
    input_schema = None
    output_schema = None
    
    def __init__(self, config: Optional[JoinOperatorConfig] = None):
        super().__init__(config)
        self._right_data: Optional[pl.DataFrame] = None
    
    def set_right_data(self, data: pl.DataFrame) -> "JoinOperator":
        """Set the right DataFrame for joining.
        
        Args:
            data: Right DataFrame
            
        Returns:
            Self for chaining
        """
        self._right_data = data
        return self
    
    def process(self, data: pl.DataFrame) -> OperatorResult:
        """Join left data with stored right data.
        
        Args:
            data: Left DataFrame
            
        Returns:
            OperatorResult with joined data
        """
        if self._right_data is None:
            return OperatorResult(
                success=False,
                error="Right DataFrame not set. Use set_right_data() or process_join()"
            )
        
        return self.process_join(data, self._right_data)
    
    def process_join(
        self, 
        left: pl.DataFrame, 
        right: pl.DataFrame
    ) -> OperatorResult:
        """Join two DataFrames.
        
        Args:
            left: Left DataFrame
            right: Right DataFrame
            
        Returns:
            OperatorResult with joined data
        """
        try:
            config = self._config or JoinOperatorConfig()
            
            # Store for potential re-use
            self._right_data = right
            
            # Dispatch to appropriate join method
            if config.join_type == JoinType.ASOF:
                result = self._asof_join(left, right, config)
            elif config.join_type == JoinType.CROSS:
                result = self._cross_join(left, right, config)
            else:
                result = self._standard_join(left, right, config)
            
            return OperatorResult(
                success=True,
                data=result,
                metadata={
                    "join_type": config.join_type.value,
                    "left_rows": len(left),
                    "right_rows": len(right),
                    "result_rows": len(result),
                }
            )
            
        except Exception as e:
            return OperatorResult(success=False, error=str(e))
    
    def _standard_join(
        self,
        left: pl.DataFrame,
        right: pl.DataFrame,
        config: JoinOperatorConfig,
    ) -> pl.DataFrame:
        """Perform standard join (inner, left, right, outer, semi, anti)."""
        # Determine join columns
        if config.on:
            on = config.on if isinstance(config.on, list) else [config.on]
            return left.join(
                right,
                on=on,
                how=config.join_type.value,
                suffix=config.suffix,
            )
        elif config.left_on and config.right_on:
            left_on = config.left_on if isinstance(config.left_on, list) else [config.left_on]
            right_on = config.right_on if isinstance(config.right_on, list) else [config.right_on]
            return left.join(
                right,
                left_on=left_on,
                right_on=right_on,
                how=config.join_type.value,
                suffix=config.suffix,
            )
        else:
            raise ValueError("Join requires 'on' or 'left_on'/'right_on' columns")
    
    def _asof_join(
        self,
        left: pl.DataFrame,
        right: pl.DataFrame,
        config: JoinOperatorConfig,
    ) -> pl.DataFrame:
        """Perform asof (time-based inexact) join.
        
        Asof joins match rows based on the nearest key value,
        useful for joining time-series data with different timestamps.
        """
        if not config.asof_on:
            raise ValueError("Asof join requires 'asof_on' (time column)")
        
        # Ensure sorted by asof column
        left = left.sort(config.asof_on)
        right = right.sort(config.asof_on)
        
        # Build asof join parameters
        join_kwargs = {
            "other": right,
            "on": config.asof_on,
            "strategy": config.asof_strategy.value,
            "suffix": config.suffix,
        }
        
        # Add grouping if specified
        if config.asof_by:
            by = config.asof_by if isinstance(config.asof_by, list) else [config.asof_by]
            join_kwargs["by"] = by
        
        # Add tolerance if specified
        if config.asof_tolerance:
            # Parse tolerance string to timedelta
            from analytics_engine.operators.window_selector import parse_duration
            tolerance = parse_duration(config.asof_tolerance)
            join_kwargs["tolerance"] = tolerance
        
        return left.join_asof(**join_kwargs)
    
    def _cross_join(
        self,
        left: pl.DataFrame,
        right: pl.DataFrame,
        config: JoinOperatorConfig,
    ) -> pl.DataFrame:
        """Perform cross join (Cartesian product)."""
        return left.join(
            right,
            how="cross",
            suffix=config.suffix,
        )


# Multi-stream join helper
class MultiJoin:
    """Helper for joining multiple DataFrames in sequence.
    
    Example:
        result = (
            MultiJoin(base_df)
            .join(sensors_df, on="sensor_id", how="left")
            .join(assets_df, on="asset_id", how="left")
            .join(telemetry_df, asof_on="timestamp", asof_by="sensor_id")
            .result()
        )
    """
    
    def __init__(self, data: pl.DataFrame):
        """Initialize with base DataFrame.
        
        Args:
            data: Base DataFrame to join onto
        """
        self._data = data
    
    def join(
        self,
        other: pl.DataFrame,
        on: Optional[Union[str, List[str]]] = None,
        left_on: Optional[Union[str, List[str]]] = None,
        right_on: Optional[Union[str, List[str]]] = None,
        how: Union[JoinType, str] = JoinType.INNER,
        suffix: str = "_right",
    ) -> "MultiJoin":
        """Add a standard join.
        
        Args:
            other: DataFrame to join
            on: Column(s) to join on
            left_on: Left columns (if different names)
            right_on: Right columns (if different names)
            how: Join type
            suffix: Suffix for duplicates
            
        Returns:
            Self for chaining
        """
        how_value = how.value if isinstance(how, JoinType) else how
        
        config = JoinOperatorConfig(
            join_type=JoinType(how_value),
            on=on,
            left_on=left_on,
            right_on=right_on,
            suffix=suffix,
        )
        
        op = JoinOperator(config=config)
        result = op.process_join(self._data, other)
        
        if result.success:
            self._data = result.data
        else:
            raise ValueError(f"Join failed: {result.error}")
        
        return self
    
    def asof_join(
        self,
        other: pl.DataFrame,
        on: str,
        by: Optional[Union[str, List[str]]] = None,
        tolerance: Optional[str] = None,
        strategy: AsofStrategy = AsofStrategy.BACKWARD,
        suffix: str = "_right",
    ) -> "MultiJoin":
        """Add an asof (time-based) join.
        
        Args:
            other: DataFrame to join
            on: Time column for matching
            by: Grouping column(s)
            tolerance: Max time difference
            strategy: Matching direction
            suffix: Suffix for duplicates
            
        Returns:
            Self for chaining
        """
        config = JoinOperatorConfig(
            join_type=JoinType.ASOF,
            asof_on=on,
            asof_by=by,
            asof_tolerance=tolerance,
            asof_strategy=strategy,
            suffix=suffix,
        )
        
        op = JoinOperator(config=config)
        result = op.process_join(self._data, other)
        
        if result.success:
            self._data = result.data
        else:
            raise ValueError(f"Asof join failed: {result.error}")
        
        return self
    
    def result(self) -> pl.DataFrame:
        """Get the final joined DataFrame.
        
        Returns:
            Joined DataFrame
        """
        return self._data


# Convenience function
def join_dataframes(
    left: pl.DataFrame,
    right: pl.DataFrame,
    on: Optional[Union[str, List[str]]] = None,
    how: Union[JoinType, str] = "inner",
) -> pl.DataFrame:
    """Convenience function for simple joins.
    
    Args:
        left: Left DataFrame
        right: Right DataFrame
        on: Column(s) to join on
        how: Join type
        
    Returns:
        Joined DataFrame
    """
    how_value = how.value if isinstance(how, JoinType) else how
    
    config = JoinOperatorConfig(
        join_type=JoinType(how_value),
        on=on,
    )
    
    op = JoinOperator(config=config)
    result = op.process_join(left, right)
    
    if result.success:
        return result.data
    else:
        raise ValueError(f"Join failed: {result.error}")
