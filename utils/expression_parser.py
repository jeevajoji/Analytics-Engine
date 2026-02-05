"""
Expression Parser Utility.

Centralized expression parsing for filter and threshold operations.
"""

import re
from typing import Any, List, Optional, Tuple, Union
from enum import Enum
from datetime import datetime

import polars as pl


class ComparisonOperator(str, Enum):
    """Comparison operators for expressions."""
    EQ = "eq"       # Equal
    NE = "ne"       # Not equal
    GT = "gt"       # Greater than
    GE = "ge"       # Greater than or equal
    LT = "lt"       # Less than
    LE = "le"       # Less than or equal
    IN = "in"       # In list
    NOT_IN = "not_in"  # Not in list
    BETWEEN = "between"  # Between two values
    CONTAINS = "contains"  # String contains
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"
    REGEX = "regex"  # Regular expression match


# Operator string mappings
OPERATOR_ALIASES = {
    "==": ComparisonOperator.EQ,
    "=": ComparisonOperator.EQ,
    "eq": ComparisonOperator.EQ,
    "!=": ComparisonOperator.NE,
    "<>": ComparisonOperator.NE,
    "ne": ComparisonOperator.NE,
    ">": ComparisonOperator.GT,
    "gt": ComparisonOperator.GT,
    ">=": ComparisonOperator.GE,
    "ge": ComparisonOperator.GE,
    "<": ComparisonOperator.LT,
    "lt": ComparisonOperator.LT,
    "<=": ComparisonOperator.LE,
    "le": ComparisonOperator.LE,
    "in": ComparisonOperator.IN,
    "not_in": ComparisonOperator.NOT_IN,
    "not in": ComparisonOperator.NOT_IN,
    "between": ComparisonOperator.BETWEEN,
    "contains": ComparisonOperator.CONTAINS,
    "like": ComparisonOperator.CONTAINS,
    "starts_with": ComparisonOperator.STARTS_WITH,
    "ends_with": ComparisonOperator.ENDS_WITH,
    "is_null": ComparisonOperator.IS_NULL,
    "is null": ComparisonOperator.IS_NULL,
    "is_not_null": ComparisonOperator.IS_NOT_NULL,
    "is not null": ComparisonOperator.IS_NOT_NULL,
    "regex": ComparisonOperator.REGEX,
    "matches": ComparisonOperator.REGEX,
}


class ExpressionParser:
    """
    Parse simple expressions into Polars expressions.
    
    Supports formats like:
    - "column > 100"
    - "status == 'active'"
    - "temperature between 20 and 30"
    - "category in ['A', 'B', 'C']"
    
    Example:
        parser = ExpressionParser()
        
        # Parse a simple comparison
        expr = parser.parse("temperature > 100")
        filtered_df = df.filter(expr)
        
        # Parse with explicit operator
        expr = parser.parse_condition("temperature", "gt", 100)
    """
    
    # Pattern for simple expressions: column operator value
    SIMPLE_PATTERN = re.compile(
        r'^(\w+)\s*(==|!=|<>|>=|<=|>|<|=)\s*(.+)$'
    )
    
    # Pattern for keyword expressions: column KEYWORD value
    KEYWORD_PATTERN = re.compile(
        r'^(\w+)\s+(eq|ne|gt|ge|lt|le|in|not_in|between|contains|like|starts_with|ends_with|is_null|is_not_null|regex|matches)\s*(.*)$',
        re.IGNORECASE
    )
    
    # Pattern for BETWEEN: column between X and Y
    BETWEEN_PATTERN = re.compile(
        r'^(\w+)\s+between\s+(.+?)\s+and\s+(.+)$',
        re.IGNORECASE
    )
    
    def parse(self, expression: str) -> pl.Expr:
        """
        Parse an expression string into a Polars expression.
        
        Args:
            expression: Expression string (e.g., "column > 100")
            
        Returns:
            Polars expression
            
        Raises:
            ValueError: If expression cannot be parsed
        """
        expression = expression.strip()
        
        # Try BETWEEN pattern first
        between_match = self.BETWEEN_PATTERN.match(expression)
        if between_match:
            column = between_match.group(1)
            low = self._parse_value(between_match.group(2).strip())
            high = self._parse_value(between_match.group(3).strip())
            return pl.col(column).is_between(low, high)
        
        # Try simple pattern (with symbols)
        simple_match = self.SIMPLE_PATTERN.match(expression)
        if simple_match:
            column = simple_match.group(1)
            op_str = simple_match.group(2)
            value_str = simple_match.group(3).strip()
            
            operator = OPERATOR_ALIASES.get(op_str)
            if operator is None:
                raise ValueError(f"Unknown operator: {op_str}")
            
            value = self._parse_value(value_str)
            return self._build_expression(column, operator, value)
        
        # Try keyword pattern
        keyword_match = self.KEYWORD_PATTERN.match(expression)
        if keyword_match:
            column = keyword_match.group(1)
            op_str = keyword_match.group(2).lower()
            value_str = keyword_match.group(3).strip() if keyword_match.group(3) else None
            
            operator = OPERATOR_ALIASES.get(op_str)
            if operator is None:
                raise ValueError(f"Unknown operator: {op_str}")
            
            value = self._parse_value(value_str) if value_str else None
            return self._build_expression(column, operator, value)
        
        raise ValueError(f"Cannot parse expression: '{expression}'")
    
    def parse_condition(
        self,
        column: str,
        operator: Union[str, ComparisonOperator],
        value: Any = None,
    ) -> pl.Expr:
        """
        Build a Polars expression from explicit components.
        
        Args:
            column: Column name
            operator: Comparison operator (string or enum)
            value: Value to compare against
            
        Returns:
            Polars expression
        """
        if isinstance(operator, str):
            operator = OPERATOR_ALIASES.get(operator.lower())
            if operator is None:
                raise ValueError(f"Unknown operator: {operator}")
        
        return self._build_expression(column, operator, value)
    
    def _build_expression(
        self,
        column: str,
        operator: ComparisonOperator,
        value: Any,
    ) -> pl.Expr:
        """Build Polars expression from components."""
        col = pl.col(column)
        
        if operator == ComparisonOperator.EQ:
            return col == value
        elif operator == ComparisonOperator.NE:
            return col != value
        elif operator == ComparisonOperator.GT:
            return col > value
        elif operator == ComparisonOperator.GE:
            return col >= value
        elif operator == ComparisonOperator.LT:
            return col < value
        elif operator == ComparisonOperator.LE:
            return col <= value
        elif operator == ComparisonOperator.IN:
            if isinstance(value, (list, tuple)):
                return col.is_in(list(value))
            return col.is_in([value])
        elif operator == ComparisonOperator.NOT_IN:
            if isinstance(value, (list, tuple)):
                return ~col.is_in(list(value))
            return ~col.is_in([value])
        elif operator == ComparisonOperator.BETWEEN:
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                return col.is_between(value[0], value[1])
            raise ValueError("BETWEEN requires two values")
        elif operator == ComparisonOperator.CONTAINS:
            return col.str.contains(str(value))
        elif operator == ComparisonOperator.STARTS_WITH:
            return col.str.starts_with(str(value))
        elif operator == ComparisonOperator.ENDS_WITH:
            return col.str.ends_with(str(value))
        elif operator == ComparisonOperator.IS_NULL:
            return col.is_null()
        elif operator == ComparisonOperator.IS_NOT_NULL:
            return col.is_not_null()
        elif operator == ComparisonOperator.REGEX:
            return col.str.contains(str(value))
        else:
            raise ValueError(f"Unsupported operator: {operator}")
    
    def _parse_value(self, value_str: str) -> Any:
        """Parse a value string to appropriate Python type."""
        if value_str is None:
            return None
        
        value_str = value_str.strip()
        
        # Empty string
        if not value_str:
            return None
        
        # Boolean
        if value_str.lower() == 'true':
            return True
        if value_str.lower() == 'false':
            return False
        
        # None/null
        if value_str.lower() in ('none', 'null'):
            return None
        
        # Quoted string
        if (value_str.startswith('"') and value_str.endswith('"')) or \
           (value_str.startswith("'") and value_str.endswith("'")):
            return value_str[1:-1]
        
        # List
        if value_str.startswith('[') and value_str.endswith(']'):
            return self._parse_list(value_str)
        
        # Number
        try:
            if '.' in value_str:
                return float(value_str)
            return int(value_str)
        except ValueError:
            pass
        
        # ISO datetime
        if 'T' in value_str or re.match(r'^\d{4}-\d{2}-\d{2}', value_str):
            try:
                return datetime.fromisoformat(value_str.replace('Z', '+00:00'))
            except ValueError:
                pass
        
        # Default: return as string
        return value_str
    
    def _parse_list(self, list_str: str) -> List[Any]:
        """Parse a list string like "[1, 2, 3]" or "['a', 'b']"."""
        inner = list_str[1:-1].strip()
        if not inner:
            return []
        
        # Simple split on comma (doesn't handle nested structures)
        items = []
        for item in inner.split(','):
            items.append(self._parse_value(item.strip()))
        
        return items


# Convenience functions
def parse_expression(expression: str) -> pl.Expr:
    """Parse an expression string into a Polars expression."""
    return ExpressionParser().parse(expression)


def build_expression(column: str, operator: str, value: Any) -> pl.Expr:
    """Build a Polars expression from components."""
    return ExpressionParser().parse_condition(column, operator, value)
