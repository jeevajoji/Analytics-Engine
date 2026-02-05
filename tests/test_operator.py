"""
Tests for core operator functionality.
"""

import pytest
import polars as pl
from typing import Optional

from analytics_engine.core.operator import Operator, OperatorConfig, OperatorResult
from analytics_engine.core.schema import DataSchema, SchemaField, DataType


# Test Operator Implementation
class TestOperatorConfig(OperatorConfig):
    """Config for test operator."""
    multiplier: float = 1.0


class TestOperator(Operator[TestOperatorConfig]):
    """Simple test operator that multiplies a column."""
    
    name = "test_operator"
    description = "Test operator for unit tests"
    version = "1.0.0"
    
    input_schema = DataSchema(
        fields=[
            SchemaField(name="value", dtype=DataType.FLOAT, required=True),
        ]
    )
    
    def process(self, data: pl.DataFrame) -> OperatorResult:
        """Multiply the 'value' column by the configured multiplier."""
        try:
            multiplier = self._config.multiplier if self._config else 1.0
            result = data.with_columns(
                (pl.col("value") * multiplier).alias("value")
            )
            return OperatorResult(success=True, data=result)
        except Exception as e:
            return OperatorResult(success=False, error=str(e))


class TestOperatorBase:
    """Tests for the base Operator class."""
    
    def test_operator_instantiation(self):
        """Test basic operator creation."""
        op = TestOperator()
        assert op.name == "test_operator"
        assert op.version == "1.0.0"
    
    def test_operator_with_config(self):
        """Test operator with configuration."""
        config = TestOperatorConfig(multiplier=2.0)
        op = TestOperator(config=config)
        assert op.config.multiplier == 2.0
    
    def test_operator_process(self):
        """Test operator processing."""
        config = TestOperatorConfig(multiplier=2.0)
        op = TestOperator(config=config)
        
        data = pl.DataFrame({"value": [1.0, 2.0, 3.0]})
        result = op.process(data)
        
        assert result.success
        assert result.data is not None
        assert result.data["value"].to_list() == [2.0, 4.0, 6.0]
    
    def test_operator_validate_input_valid(self):
        """Test input validation with valid data."""
        op = TestOperator()
        data = pl.DataFrame({"value": [1.0, 2.0, 3.0]})
        
        is_valid, errors = op.validate_input(data)
        assert is_valid
        assert len(errors) == 0
    
    def test_operator_validate_input_missing_column(self):
        """Test input validation with missing required column."""
        op = TestOperator()
        data = pl.DataFrame({"other": [1.0, 2.0, 3.0]})
        
        is_valid, errors = op.validate_input(data)
        assert not is_valid
        assert len(errors) > 0
    
    def test_operator_to_dict(self):
        """Test operator serialization."""
        config = TestOperatorConfig(multiplier=2.0)
        op = TestOperator(config=config)
        
        d = op.to_dict()
        assert d["name"] == "test_operator"
        assert d["config"]["multiplier"] == 2.0
    
    def test_operator_repr(self):
        """Test operator string representation."""
        op = TestOperator()
        repr_str = repr(op)
        assert "TestOperator" in repr_str
        assert "test_operator" in repr_str


class TestOperatorResult:
    """Tests for OperatorResult."""
    
    def test_success_result(self):
        """Test successful result creation."""
        data = pl.DataFrame({"x": [1, 2, 3]})
        result = OperatorResult(success=True, data=data)
        
        assert result.success
        assert result.data is not None
        assert result.error is None
    
    def test_failure_result(self):
        """Test failure result creation."""
        result = OperatorResult(success=False, error="Something went wrong")
        
        assert not result.success
        assert result.error == "Something went wrong"
        assert result.data is None
    
    def test_result_with_metadata(self):
        """Test result with metadata."""
        result = OperatorResult(
            success=True,
            metadata={"rows_processed": 100, "time_ms": 50}
        )
        
        assert result.metadata["rows_processed"] == 100
        assert result.metadata["time_ms"] == 50


class TestChunkedProcessing:
    """Tests for chunked processing."""
    
    def test_process_chunked(self):
        """Test processing data in chunks."""
        op = TestOperator(config=TestOperatorConfig(multiplier=2.0))
        data = pl.DataFrame({"value": list(range(100))})
        
        chunks = list(op.process_chunked(data, chunk_size=25))
        
        assert len(chunks) == 4
        for chunk_result in chunks:
            assert chunk_result.success
    
    def test_process_chunked_uneven(self):
        """Test chunked processing with uneven division."""
        op = TestOperator(config=TestOperatorConfig(multiplier=1.0))
        data = pl.DataFrame({"value": list(range(30))})
        
        chunks = list(op.process_chunked(data, chunk_size=10))
        
        assert len(chunks) == 3
        
        # Verify all data was processed
        total_rows = sum(len(c.data) for c in chunks)
        assert total_rows == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
