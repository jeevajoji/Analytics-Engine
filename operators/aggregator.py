"""
Aggregator Operator

Performs aggregation functions: sum, mean, count, min, max, stddev, percentiles.
Works on grouped data (by window, by key, or entire dataset).
"""

from enum import Enum
from typing import Optional, List, Union, Any
import polars as pl

from analytics_engine.core.operator import Operator, OperatorConfig, OperatorResult
from analytics_engine.core.registry import register_operator
from analytics_engine.core.schema import DataSchema, SchemaField, DataType


class AggregateFunction(str, Enum):
    """Supported aggregation functions."""
    SUM = "sum"
    MEAN = "mean"
    AVG = "avg"  # Alias for mean
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    STDDEV = "stddev"
    VARIANCE = "variance"
    MEDIAN = "median"
    FIRST = "first"
    LAST = "last"
    # Percentiles handled separately
    P25 = "p25"
    P50 = "p50"
    P75 = "p75"
    P90 = "p90"
    P95 = "p95"
    P99 = "p99"


class MetricConfig(OperatorConfig):
    """Configuration for a single metric aggregation.
    
    Attributes:
        column: Column to aggregate
        functions: List of aggregation functions to apply
        alias_prefix: Optional prefix for output column names
    """
    column: str
    functions: List[Union[AggregateFunction, str]]
    alias_prefix: Optional[str] = None


class AggregatorConfig(OperatorConfig):
    """Configuration for Aggregator operator.
    
    Attributes:
        metrics: List of metric configurations
        group_by: Columns to group by (e.g., ["window_start", "asset_id"])
        include_count: Whether to include row count in output
        drop_nulls: Whether to drop null values before aggregating
    """
    metrics: List[MetricConfig]
    group_by: Optional[List[str]] = None
    include_count: bool = True
    drop_nulls: bool = False


@register_operator
class Aggregator(Operator[AggregatorConfig]):
    """Aggregation operator for computing statistics.
    
    Supports multiple aggregation functions applied to multiple columns,
    with optional grouping.
    
    Example:
        config = AggregatorConfig(
            metrics=[
                MetricConfig(column="temperature", functions=["mean", "max", "stddev"]),
                MetricConfig(column="pressure", functions=["mean", "min"]),
            ],
            group_by=["window_start", "sensor_id"]
        )
        
        aggregator = Aggregator(config=config)
        result = aggregator.process(data)
        
        # Output columns: window_start, sensor_id, temperature_mean, 
        #                 temperature_max, temperature_stddev, 
        #                 pressure_mean, pressure_min, _count
    """
    
    name = "Aggregator"
    description = "Aggregate data using sum, mean, count, min, max, stddev, percentiles"
    version = "1.0.0"
    config_class = AggregatorConfig
    
    # Generic operator - accepts any schema
    input_schema = None
    output_schema = None
    
    def process(self, data: pl.DataFrame) -> OperatorResult:
        """Apply aggregations to data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            OperatorResult with aggregated data
        """
        try:
            config = self._config
            if not config or not config.metrics:
                return OperatorResult(
                    success=False,
                    error="Aggregator requires metrics configuration"
                )
            
            # Drop nulls if requested
            if config.drop_nulls:
                cols_to_check = [m.column for m in config.metrics]
                data = data.drop_nulls(subset=cols_to_check)
            
            # Build aggregation expressions
            agg_exprs = []
            
            for metric in config.metrics:
                col = metric.column
                prefix = metric.alias_prefix or col
                
                for func in metric.functions:
                    func_str = func.value if isinstance(func, AggregateFunction) else func.lower()
                    expr = self._build_agg_expr(col, func_str, prefix)
                    if expr is not None:
                        agg_exprs.append(expr)
            
            # Add count if requested
            if config.include_count:
                agg_exprs.append(pl.len().alias("_count"))
            
            # Apply grouping or aggregate entire dataset
            if config.group_by:
                result = data.group_by(config.group_by).agg(agg_exprs)
                # Sort by group columns
                result = result.sort(config.group_by)
            else:
                result = data.select(agg_exprs)
            
            return OperatorResult(
                success=True,
                data=result,
                metadata={
                    "metrics_count": len(config.metrics),
                    "group_by": config.group_by,
                    "output_rows": len(result),
                }
            )
            
        except Exception as e:
            return OperatorResult(success=False, error=str(e))
    
    def _build_agg_expr(
        self, 
        column: str, 
        function: str, 
        prefix: str
    ) -> Optional[pl.Expr]:
        """Build Polars aggregation expression.
        
        Args:
            column: Column name
            function: Aggregation function name
            prefix: Prefix for output column name
            
        Returns:
            Polars expression or None if unknown function
        """
        col = pl.col(column)
        alias = f"{prefix}_{function}"
        
        func_map = {
            "sum": col.sum(),
            "mean": col.mean(),
            "avg": col.mean(),
            "count": col.count(),
            "min": col.min(),
            "max": col.max(),
            "stddev": col.std(),
            "std": col.std(),
            "variance": col.var(),
            "var": col.var(),
            "median": col.median(),
            "first": col.first(),
            "last": col.last(),
            # Percentiles
            "p25": col.quantile(0.25),
            "p50": col.quantile(0.50),
            "p75": col.quantile(0.75),
            "p90": col.quantile(0.90),
            "p95": col.quantile(0.95),
            "p99": col.quantile(0.99),
        }
        
        # Check for custom percentile (e.g., "p80")
        if function.startswith("p") and function[1:].isdigit():
            percentile = int(function[1:]) / 100
            return col.quantile(percentile).alias(alias)
        
        if function in func_map:
            return func_map[function].alias(alias)
        
        # Unknown function - skip
        return None


# Convenience functions for quick aggregation
def aggregate(
    data: pl.DataFrame,
    columns: Union[str, List[str]],
    functions: Union[str, List[str]],
    group_by: Optional[List[str]] = None,
) -> pl.DataFrame:
    """Convenience function for quick aggregation.
    
    Args:
        data: Input DataFrame
        columns: Column(s) to aggregate
        functions: Function(s) to apply
        group_by: Optional grouping columns
        
    Returns:
        Aggregated DataFrame
    """
    if isinstance(columns, str):
        columns = [columns]
    if isinstance(functions, str):
        functions = [functions]
    
    metrics = [
        MetricConfig(column=col, functions=functions)
        for col in columns
    ]
    
    config = AggregatorConfig(
        metrics=metrics,
        group_by=group_by,
        include_count=False,
    )
    
    op = Aggregator(config=config)
    result = op.process(data)
    
    if result.success:
        return result.data
    else:
        raise ValueError(f"Aggregation failed: {result.error}")


def summarize(data: pl.DataFrame, columns: Optional[List[str]] = None) -> pl.DataFrame:
    """Quick summary statistics for numeric columns.
    
    Args:
        data: Input DataFrame
        columns: Columns to summarize (default: all numeric)
        
    Returns:
        DataFrame with summary statistics
    """
    if columns is None:
        # Get all numeric columns
        columns = [
            name for name, dtype in zip(data.columns, data.dtypes)
            if dtype in [pl.Float32, pl.Float64, pl.Int8, pl.Int16, 
                        pl.Int32, pl.Int64, pl.UInt8, pl.UInt16,
                        pl.UInt32, pl.UInt64]
        ]
    
    metrics = [
        MetricConfig(
            column=col, 
            functions=["count", "mean", "stddev", "min", "p25", "median", "p75", "max"]
        )
        for col in columns
    ]
    
    config = AggregatorConfig(metrics=metrics, include_count=False)
    op = Aggregator(config=config)
    result = op.process(data)
    
    if result.success:
        return result.data
    else:
        raise ValueError(f"Summary failed: {result.error}")
