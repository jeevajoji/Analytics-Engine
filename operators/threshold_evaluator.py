"""
ThresholdEvaluator Operator

Evaluates conditions and thresholds on data.
Core operator for the Ruleset Lane (safety violations, compliance).
"""

from enum import Enum
from typing import Optional, List, Any, Union, Dict
import polars as pl

from analytics_engine.core.operator import Operator, OperatorConfig, OperatorResult
from analytics_engine.core.registry import register_operator
from analytics_engine.core.schema import DataSchema, SchemaField, DataType


class ThresholdType(str, Enum):
    """Types of threshold comparisons."""
    GT = "gt"               # Greater than >
    GE = "ge"               # Greater than or equal >=
    LT = "lt"               # Less than <
    LE = "le"               # Less than or equal <=
    EQ = "eq"               # Equal ==
    NE = "ne"               # Not equal !=
    BETWEEN = "between"     # Between two values (inclusive)
    OUTSIDE = "outside"     # Outside two values
    IN = "in"               # In set of values
    NOT_IN = "not_in"       # Not in set of values
    RATE_OF_CHANGE = "rate_of_change"  # Change rate exceeds threshold
    DEVIATION = "deviation"  # Deviation from baseline


class ThresholdRule(OperatorConfig):
    """A single threshold rule.
    
    Attributes:
        column: Column to evaluate
        threshold_type: Type of comparison
        value: Primary threshold value
        value2: Secondary value (for between/outside)
        baseline_column: Column for baseline comparison (deviation)
        output_column: Name for the result column (default: {column}_violation)
        label: Human-readable label for this rule
    """
    column: str
    threshold_type: ThresholdType
    value: Optional[Any] = None
    value2: Optional[Any] = None  # For BETWEEN, OUTSIDE
    baseline_column: Optional[str] = None  # For DEVIATION
    output_column: Optional[str] = None
    label: Optional[str] = None


class ThresholdEvaluatorConfig(OperatorConfig):
    """Configuration for ThresholdEvaluator operator.
    
    Supports two modes:
    1. Simple: Single expression string
    2. Structured: List of ThresholdRule objects
    
    Attributes:
        expression: Simple expression (e.g., "temperature > 100")
        rules: List of structured threshold rules
        mode: How to combine multiple rules ("and" or "or")
        output_column: Name for combined violation column
        include_details: Whether to include per-rule violation columns
    """
    expression: Optional[str] = None
    rules: Optional[List[ThresholdRule]] = None
    mode: str = "or"  # "and" = all rules must trigger, "or" = any rule triggers
    output_column: str = "is_violation"
    include_details: bool = True


@register_operator
class ThresholdEvaluator(Operator[ThresholdEvaluatorConfig]):
    """Threshold evaluation operator.
    
    Evaluates data against thresholds and marks violations.
    Essential for the Ruleset Lane (real-time safety/compliance).
    
    Simple Expression Mode:
        config = ThresholdEvaluatorConfig(
            expression="temperature > 100"
        )
    
    Structured Rules Mode:
        config = ThresholdEvaluatorConfig(
            rules=[
                ThresholdRule(
                    column="temperature",
                    threshold_type=ThresholdType.GT,
                    value=100,
                    label="High Temperature"
                ),
                ThresholdRule(
                    column="speed",
                    threshold_type=ThresholdType.BETWEEN,
                    value=0,
                    value2=120,
                    label="Speed in Range"
                ),
            ],
            mode="or"
        )
    
    Output:
        Adds violation columns to the data:
        - is_violation: Boolean, True if any rule triggered
        - {column}_violation: Boolean per rule (if include_details=True)
        - violation_labels: List of triggered rule labels
    """
    
    name = "ThresholdEvaluator"
    description = "Evaluate thresholds and conditions"
    version = "1.0.0"
    config_class = ThresholdEvaluatorConfig
    
    input_schema = None  # Generic
    output_schema = DataSchema(
        fields=[
            SchemaField(
                name="is_violation",
                dtype=DataType.BOOLEAN,
                required=True,
                description="Whether any threshold was violated"
            ),
        ]
    )
    
    def process(self, data: pl.DataFrame) -> OperatorResult:
        """Evaluate thresholds on data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            OperatorResult with violation columns added
        """
        try:
            config = self._config or ThresholdEvaluatorConfig()
            
            # Determine evaluation mode
            if config.expression:
                result = self._evaluate_expression(data, config.expression)
            elif config.rules:
                result = self._evaluate_rules(data, config.rules, config)
            else:
                # No rules - no violations
                result = data.with_columns(
                    pl.lit(False).alias(config.output_column)
                )
            
            # Count violations
            violation_count = result[config.output_column].sum()
            
            return OperatorResult(
                success=True,
                data=result,
                metadata={
                    "total_rows": len(result),
                    "violation_count": violation_count,
                    "violation_rate": violation_count / len(result) if len(result) > 0 else 0,
                }
            )
            
        except Exception as e:
            return OperatorResult(success=False, error=str(e))
    
    def _evaluate_expression(
        self, 
        data: pl.DataFrame, 
        expression: str
    ) -> pl.DataFrame:
        """Evaluate a simple expression."""
        # Reuse FilterOperator's expression parsing logic
        from analytics_engine.operators.filter_operator import FilterOperator, FilterOperatorConfig
        
        filter_config = FilterOperatorConfig(expression=expression)
        filter_op = FilterOperator(config=filter_config)
        
        # Parse expression to get the condition
        expr = filter_op._parse_expression(expression)
        
        config = self._config or ThresholdEvaluatorConfig()
        return data.with_columns(
            expr.alias(config.output_column)
        )
    
    def _evaluate_rules(
        self,
        data: pl.DataFrame,
        rules: List[ThresholdRule],
        config: ThresholdEvaluatorConfig,
    ) -> pl.DataFrame:
        """Evaluate structured rules."""
        violation_columns = []
        labels_per_row: List[pl.Expr] = []
        
        for i, rule in enumerate(rules):
            # Build violation expression for this rule
            violation_expr = self._build_rule_expr(rule)
            
            # Column name for this rule's result
            col_name = rule.output_column or f"{rule.column}_violation"
            
            if config.include_details:
                data = data.with_columns(
                    violation_expr.alias(col_name)
                )
            
            violation_columns.append(col_name if config.include_details else violation_expr)
            
            # Track label for this rule
            if rule.label:
                labels_per_row.append(
                    pl.when(violation_expr)
                    .then(pl.lit(rule.label))
                    .otherwise(pl.lit(None))
                )
        
        # Combine all rule violations
        if config.include_details:
            # Use column references
            if config.mode == "and":
                combined = pl.all_horizontal([pl.col(c) for c in violation_columns])
            else:  # "or"
                combined = pl.any_horizontal([pl.col(c) for c in violation_columns])
        else:
            # Use expressions directly
            if config.mode == "and":
                combined = pl.all_horizontal(violation_columns)
            else:  # "or"
                combined = pl.any_horizontal(violation_columns)
        
        data = data.with_columns(
            combined.alias(config.output_column)
        )
        
        # Add violation labels column
        if labels_per_row and config.include_details:
            # Concatenate non-null labels
            data = data.with_columns(
                pl.concat_list(labels_per_row)
                .list.drop_nulls()
                .alias("violation_labels")
            )
        
        return data
    
    def _build_rule_expr(self, rule: ThresholdRule) -> pl.Expr:
        """Build Polars expression for a single rule."""
        col = pl.col(rule.column)
        val = rule.value
        
        if rule.threshold_type == ThresholdType.GT:
            return col > val
        elif rule.threshold_type == ThresholdType.GE:
            return col >= val
        elif rule.threshold_type == ThresholdType.LT:
            return col < val
        elif rule.threshold_type == ThresholdType.LE:
            return col <= val
        elif rule.threshold_type == ThresholdType.EQ:
            return col == val
        elif rule.threshold_type == ThresholdType.NE:
            return col != val
        elif rule.threshold_type == ThresholdType.BETWEEN:
            return (col >= val) & (col <= rule.value2)
        elif rule.threshold_type == ThresholdType.OUTSIDE:
            return (col < val) | (col > rule.value2)
        elif rule.threshold_type == ThresholdType.IN:
            return col.is_in(val)
        elif rule.threshold_type == ThresholdType.NOT_IN:
            return ~col.is_in(val)
        elif rule.threshold_type == ThresholdType.RATE_OF_CHANGE:
            # Rate of change: |current - previous| / previous > threshold
            return ((col - col.shift(1)).abs() / col.shift(1).abs()) > val
        elif rule.threshold_type == ThresholdType.DEVIATION:
            # Deviation from baseline column
            if rule.baseline_column:
                baseline = pl.col(rule.baseline_column)
                return (col - baseline).abs() > val
            else:
                # Deviation from mean
                return (col - col.mean()).abs() > val
        else:
            raise ValueError(f"Unknown threshold type: {rule.threshold_type}")


# Convenience function
def evaluate_threshold(
    data: pl.DataFrame,
    column: str,
    operator: str,
    value: Any,
    value2: Optional[Any] = None,
) -> pl.DataFrame:
    """Quick threshold evaluation.
    
    Args:
        data: Input DataFrame
        column: Column to check
        operator: Comparison operator (gt, lt, between, etc.)
        value: Threshold value
        value2: Second value for between/outside
        
    Returns:
        DataFrame with is_violation column
    """
    rule = ThresholdRule(
        column=column,
        threshold_type=ThresholdType(operator),
        value=value,
        value2=value2,
    )
    
    config = ThresholdEvaluatorConfig(rules=[rule])
    op = ThresholdEvaluator(config=config)
    result = op.process(data)
    
    if result.success:
        return result.data
    else:
        raise ValueError(f"Threshold evaluation failed: {result.error}")
