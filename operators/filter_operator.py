"""
FilterOperator

Conditional row filtering using expressions.
Generic filter that can handle any boolean condition.
"""

from enum import Enum
from typing import Optional, List, Any, Union
import polars as pl

from analytics_engine.core.operator import Operator, OperatorConfig, OperatorResult
from analytics_engine.core.registry import register_operator
from analytics_engine.core.schema import DataSchema, SchemaField, DataType


class FilterMode(str, Enum):
    """How to combine multiple conditions."""
    AND = "and"  # All conditions must be true
    OR = "or"    # Any condition must be true


class ComparisonOperator(str, Enum):
    """Comparison operators for filter conditions."""
    EQ = "eq"           # Equal ==
    NE = "ne"           # Not equal !=
    GT = "gt"           # Greater than >
    GE = "ge"           # Greater than or equal >=
    LT = "lt"           # Less than <
    LE = "le"           # Less than or equal <=
    IN = "in"           # In list
    NOT_IN = "not_in"   # Not in list
    BETWEEN = "between" # Between two values
    IS_NULL = "is_null"         # Is null
    IS_NOT_NULL = "is_not_null" # Is not null
    CONTAINS = "contains"       # String contains
    STARTS_WITH = "starts_with" # String starts with
    ENDS_WITH = "ends_with"     # String ends with
    REGEX = "regex"             # Regex match


class FilterCondition(OperatorConfig):
    """A single filter condition.
    
    Attributes:
        column: Column name to filter on
        operator: Comparison operator
        value: Value to compare against (not needed for is_null/is_not_null)
        value2: Second value for between operator
    """
    column: str
    operator: ComparisonOperator
    value: Optional[Any] = None
    value2: Optional[Any] = None  # For BETWEEN


class FilterOperatorConfig(OperatorConfig):
    """Configuration for FilterOperator.
    
    Supports two modes:
    1. Simple: Single expression string (e.g., "temperature > 100")
    2. Structured: List of FilterCondition objects
    
    Attributes:
        expression: Simple filter expression string
        conditions: List of structured filter conditions
        mode: How to combine multiple conditions (and/or)
        negate: Whether to negate the final result
    """
    expression: Optional[str] = None
    conditions: Optional[List[FilterCondition]] = None
    mode: FilterMode = FilterMode.AND
    negate: bool = False


@register_operator
class FilterOperator(Operator[FilterOperatorConfig]):
    """Conditional row filtering operator.
    
    Supports both simple expression-based filtering and structured conditions.
    
    Expression Mode Examples:
        - "temperature > 100"
        - "status == 'active'"
        - "speed >= 60 and speed <= 120"
    
    Condition Mode Examples:
        conditions=[
            FilterCondition(column="temperature", operator="gt", value=100),
            FilterCondition(column="status", operator="eq", value="active"),
        ]
    
    Usage:
        # Simple expression
        config = FilterOperatorConfig(expression="temperature > 100")
        
        # Structured conditions
        config = FilterOperatorConfig(
            conditions=[
                FilterCondition(column="temp", operator=ComparisonOperator.GT, value=100),
                FilterCondition(column="humidity", operator=ComparisonOperator.LT, value=80),
            ],
            mode=FilterMode.AND
        )
        
        filter_op = FilterOperator(config=config)
        result = filter_op.process(data)
    """
    
    name = "FilterOperator"
    description = "Conditional row filtering using expressions or structured conditions"
    version = "1.0.0"
    config_class = FilterOperatorConfig
    
    # FilterOperator is generic - accepts any schema
    input_schema = None
    output_schema = None
    
    def process(self, data: pl.DataFrame) -> OperatorResult:
        """Apply filter to data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            OperatorResult with filtered data
        """
        try:
            config = self._config or FilterOperatorConfig()
            original_count = len(data)
            
            # Determine filter mode
            if config.expression:
                result = self._filter_by_expression(data, config.expression)
            elif config.conditions:
                result = self._filter_by_conditions(data, config.conditions, config.mode)
            else:
                # No filter specified - return as-is
                return OperatorResult(
                    success=True,
                    data=data,
                    metadata={"rows_filtered": 0, "rows_remaining": len(data)}
                )
            
            # Apply negation if requested
            if config.negate:
                # Get the complement
                result = data.join(
                    result.select(data.columns), 
                    on=data.columns, 
                    how="anti"
                )
            
            filtered_count = original_count - len(result)
            
            return OperatorResult(
                success=True,
                data=result,
                metadata={
                    "rows_filtered": filtered_count,
                    "rows_remaining": len(result),
                    "filter_rate": filtered_count / original_count if original_count > 0 else 0,
                }
            )
            
        except Exception as e:
            return OperatorResult(success=False, error=str(e))
    
    def _filter_by_expression(self, data: pl.DataFrame, expression: str) -> pl.DataFrame:
        """Filter using a simple expression string.
        
        Parses expressions like:
        - "column > 100"
        - "column == 'value'"
        - "column >= 10 and column <= 20"
        """
        # Parse and convert expression to Polars expression
        expr = self._parse_expression(expression)
        return data.filter(expr)
    
    def _parse_expression(self, expression: str) -> pl.Expr:
        """Parse expression string to Polars expression.
        
        Supports basic comparisons and boolean operators.
        """
        expression = expression.strip()
        
        # Handle boolean operators (and, or)
        if " and " in expression.lower():
            parts = expression.lower().split(" and ")
            exprs = [self._parse_simple_expression(p.strip()) for p in parts]
            result = exprs[0]
            for expr in exprs[1:]:
                result = result & expr
            return result
        
        if " or " in expression.lower():
            parts = expression.lower().split(" or ")
            exprs = [self._parse_simple_expression(p.strip()) for p in parts]
            result = exprs[0]
            for expr in exprs[1:]:
                result = result | expr
            return result
        
        return self._parse_simple_expression(expression)
    
    def _parse_simple_expression(self, expression: str) -> pl.Expr:
        """Parse a simple comparison expression."""
        # Operator patterns (order matters - check longer patterns first)
        operators = [
            (">=", lambda col, val: pl.col(col) >= val),
            ("<=", lambda col, val: pl.col(col) <= val),
            ("!=", lambda col, val: pl.col(col) != val),
            ("==", lambda col, val: pl.col(col) == val),
            (">", lambda col, val: pl.col(col) > val),
            ("<", lambda col, val: pl.col(col) < val),
            ("=", lambda col, val: pl.col(col) == val),
        ]
        
        for op_str, op_func in operators:
            if op_str in expression:
                parts = expression.split(op_str, 1)
                column = parts[0].strip()
                value_str = parts[1].strip()
                
                # Parse value
                value = self._parse_value(value_str)
                
                return op_func(column, value)
        
        raise ValueError(f"Could not parse expression: {expression}")
    
    def _parse_value(self, value_str: str) -> Any:
        """Parse a value string to appropriate type."""
        value_str = value_str.strip()
        
        # String (quoted)
        if (value_str.startswith('"') and value_str.endswith('"')) or \
           (value_str.startswith("'") and value_str.endswith("'")):
            return value_str[1:-1]
        
        # Boolean
        if value_str.lower() == "true":
            return True
        if value_str.lower() == "false":
            return False
        
        # None/null
        if value_str.lower() in ("none", "null"):
            return None
        
        # Try numeric
        try:
            if "." in value_str:
                return float(value_str)
            return int(value_str)
        except ValueError:
            # Return as string
            return value_str
    
    def _filter_by_conditions(
        self, 
        data: pl.DataFrame, 
        conditions: List[FilterCondition],
        mode: FilterMode
    ) -> pl.DataFrame:
        """Filter using structured conditions."""
        if not conditions:
            return data
        
        # Build expressions for each condition
        exprs = [self._build_condition_expr(cond) for cond in conditions]
        
        # Combine expressions
        if mode == FilterMode.AND:
            combined = exprs[0]
            for expr in exprs[1:]:
                combined = combined & expr
        else:  # OR
            combined = exprs[0]
            for expr in exprs[1:]:
                combined = combined | expr
        
        return data.filter(combined)
    
    def _build_condition_expr(self, condition: FilterCondition) -> pl.Expr:
        """Build Polars expression from FilterCondition."""
        col = pl.col(condition.column)
        op = condition.operator
        val = condition.value
        
        if op == ComparisonOperator.EQ:
            return col == val
        elif op == ComparisonOperator.NE:
            return col != val
        elif op == ComparisonOperator.GT:
            return col > val
        elif op == ComparisonOperator.GE:
            return col >= val
        elif op == ComparisonOperator.LT:
            return col < val
        elif op == ComparisonOperator.LE:
            return col <= val
        elif op == ComparisonOperator.IN:
            return col.is_in(val)
        elif op == ComparisonOperator.NOT_IN:
            return ~col.is_in(val)
        elif op == ComparisonOperator.BETWEEN:
            return (col >= val) & (col <= condition.value2)
        elif op == ComparisonOperator.IS_NULL:
            return col.is_null()
        elif op == ComparisonOperator.IS_NOT_NULL:
            return col.is_not_null()
        elif op == ComparisonOperator.CONTAINS:
            return col.str.contains(val)
        elif op == ComparisonOperator.STARTS_WITH:
            return col.str.starts_with(val)
        elif op == ComparisonOperator.ENDS_WITH:
            return col.str.ends_with(val)
        elif op == ComparisonOperator.REGEX:
            return col.str.contains(val)
        else:
            raise ValueError(f"Unknown operator: {op}")


# Convenience function for quick filtering
def filter_data(
    data: pl.DataFrame,
    expression: Optional[str] = None,
    conditions: Optional[List[FilterCondition]] = None,
    mode: FilterMode = FilterMode.AND,
) -> pl.DataFrame:
    """Convenience function to filter data without creating operator instance.
    
    Args:
        data: Input DataFrame
        expression: Simple filter expression
        conditions: List of FilterCondition objects
        mode: How to combine conditions
        
    Returns:
        Filtered DataFrame
    """
    config = FilterOperatorConfig(
        expression=expression,
        conditions=conditions,
        mode=mode,
    )
    op = FilterOperator(config=config)
    result = op.process(data)
    
    if result.success:
        return result.data
    else:
        raise ValueError(f"Filter failed: {result.error}")
