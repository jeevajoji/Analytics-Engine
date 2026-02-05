"""
Tests for the Pipeline Executor.
"""

import pytest
import polars as pl

from analytics_engine.core.operator import Operator, OperatorConfig, OperatorResult
from analytics_engine.core.registry import OperatorRegistry, register_operator
from analytics_engine.core.pipeline import (
    Pipeline,
    PipelineConfig,
    PipelineExecutor,
    OperatorNode,
    LaneType,
    create_pipeline,
)


# Test operators for pipeline tests
class MultiplyConfig(OperatorConfig):
    multiplier: float = 2.0


class AddConfig(OperatorConfig):
    addend: float = 10.0


@register_operator
class MultiplyOperator(Operator[MultiplyConfig]):
    name = "multiply"
    description = "Multiply values"
    version = "1.0.0"
    config_class = MultiplyConfig
    
    def process(self, data: pl.DataFrame) -> OperatorResult:
        multiplier = self._config.multiplier if self._config else 2.0
        result = data.with_columns(
            (pl.col("value") * multiplier).alias("value")
        )
        return OperatorResult(success=True, data=result)


@register_operator
class AddOperator(Operator[AddConfig]):
    name = "add"
    description = "Add to values"
    version = "1.0.0"
    config_class = AddConfig
    
    def process(self, data: pl.DataFrame) -> OperatorResult:
        addend = self._config.addend if self._config else 10.0
        result = data.with_columns(
            (pl.col("value") + addend).alias("value")
        )
        return OperatorResult(success=True, data=result)


@register_operator  
class FailingOperator(Operator):
    name = "failing"
    description = "Always fails"
    version = "1.0.0"
    
    def process(self, data: pl.DataFrame) -> OperatorResult:
        return OperatorResult(success=False, error="Intentional failure")


class TestPipelineConfig:
    """Tests for PipelineConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = PipelineConfig(name="test")
        
        assert config.name == "test"
        assert config.lane == LaneType.BATCH
        assert config.chunk_size == 10000
    
    def test_realtime_config(self):
        """Test real-time lane configuration."""
        config = PipelineConfig(
            name="realtime_test",
            lane=LaneType.REALTIME,
        )
        
        assert config.lane == LaneType.REALTIME


class TestOperatorNode:
    """Tests for OperatorNode."""
    
    def test_node_creation(self):
        """Test creating an operator node."""
        node = OperatorNode(
            operator_name="multiply",
            params={"multiplier": 3.0}
        )
        
        assert node.operator_name == "multiply"
        assert node.params["multiplier"] == 3.0
        assert len(node.id) == 8  # UUID prefix
    
    def test_node_with_custom_id(self):
        """Test creating node with custom ID."""
        node = OperatorNode(
            id="custom_id",
            operator_name="add",
            params={}
        )
        
        assert node.id == "custom_id"


class TestPipeline:
    """Tests for Pipeline."""
    
    def test_create_pipeline(self):
        """Test creating a pipeline."""
        pipeline = Pipeline(
            config=PipelineConfig(name="test_pipeline")
        )
        
        assert pipeline.name == "test_pipeline"
        assert pipeline.lane == LaneType.BATCH
        assert len(pipeline.nodes) == 0
    
    def test_add_operator(self):
        """Test adding operators to pipeline."""
        pipeline = Pipeline(
            config=PipelineConfig(name="test")
        )
        
        node_id = pipeline.add_operator("multiply", {"multiplier": 2.0})
        
        assert len(pipeline.nodes) == 1
        assert pipeline.nodes[0].operator_name == "multiply"
    
    def test_operator_chaining(self):
        """Test that operators are automatically chained."""
        pipeline = Pipeline(
            config=PipelineConfig(name="test")
        )
        
        id1 = pipeline.add_operator("multiply")
        id2 = pipeline.add_operator("add")
        
        assert id2 in pipeline.nodes[0].next_nodes
    
    def test_get_node(self):
        """Test getting a node by ID."""
        pipeline = Pipeline(
            config=PipelineConfig(name="test")
        )
        
        node_id = pipeline.add_operator("multiply", node_id="my_node")
        node = pipeline.get_node("my_node")
        
        assert node is not None
        assert node.operator_name == "multiply"
    
    def test_to_dict(self):
        """Test pipeline serialization."""
        pipeline = Pipeline(
            config=PipelineConfig(name="test")
        )
        pipeline.add_operator("multiply", {"multiplier": 2.0})
        
        d = pipeline.to_dict()
        
        assert d["config"]["name"] == "test"
        assert len(d["nodes"]) == 1
    
    def test_from_dict(self):
        """Test pipeline deserialization."""
        data = {
            "config": {"name": "test", "lane": "batch"},
            "nodes": [
                {"id": "n1", "operator_name": "multiply", "params": {}}
            ]
        }
        
        pipeline = Pipeline.from_dict(data)
        
        assert pipeline.name == "test"
        assert len(pipeline.nodes) == 1


class TestPipelineExecutor:
    """Tests for PipelineExecutor."""
    
    def setup_method(self):
        """Reset registry before each test."""
        OperatorRegistry.reset()
        # Re-register test operators
        OperatorRegistry().register(MultiplyOperator)
        OperatorRegistry().register(AddOperator)
        OperatorRegistry().register(FailingOperator)
    
    def test_execute_single_operator(self):
        """Test executing pipeline with single operator."""
        pipeline = create_pipeline(
            name="single",
            operators=[
                {"name": "multiply", "params": {"multiplier": 2.0}}
            ]
        )
        
        executor = PipelineExecutor()
        data = pl.DataFrame({"value": [1.0, 2.0, 3.0]})
        
        result = executor.execute(pipeline, data)
        
        assert result.success
        assert result.output_data["value"].to_list() == [2.0, 4.0, 6.0]
    
    def test_execute_chained_operators(self):
        """Test executing pipeline with chained operators."""
        pipeline = create_pipeline(
            name="chained",
            operators=[
                {"name": "multiply", "params": {"multiplier": 2.0}},
                {"name": "add", "params": {"addend": 10.0}},
            ]
        )
        
        executor = PipelineExecutor()
        data = pl.DataFrame({"value": [1.0, 2.0, 3.0]})
        
        result = executor.execute(pipeline, data)
        
        # (value * 2) + 10
        assert result.success
        assert result.output_data["value"].to_list() == [12.0, 14.0, 16.0]
    
    def test_execute_failing_operator(self):
        """Test pipeline with failing operator."""
        pipeline = create_pipeline(
            name="failing",
            operators=[
                {"name": "failing", "params": {}}
            ]
        )
        
        executor = PipelineExecutor()
        data = pl.DataFrame({"value": [1.0]})
        
        result = executor.execute(pipeline, data)
        
        assert not result.success
        assert result.error is not None
    
    def test_execute_chunked(self):
        """Test chunked execution."""
        pipeline = create_pipeline(
            name="chunked",
            operators=[
                {"name": "multiply", "params": {"multiplier": 2.0}}
            ]
        )
        
        executor = PipelineExecutor()
        data = pl.DataFrame({"value": list(range(100))})
        
        chunks = list(executor.execute_chunked(pipeline, data, chunk_size=25))
        
        assert len(chunks) == 4
        for chunk in chunks:
            assert chunk.success
    
    def test_validate_pipeline_valid(self):
        """Test validating a valid pipeline."""
        pipeline = create_pipeline(
            name="valid",
            operators=[
                {"name": "multiply", "params": {}}
            ]
        )
        
        executor = PipelineExecutor()
        is_valid, errors = executor.validate_pipeline(pipeline)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_pipeline_unknown_operator(self):
        """Test validating pipeline with unknown operator."""
        pipeline = Pipeline(
            config=PipelineConfig(name="invalid")
        )
        pipeline.nodes.append(
            OperatorNode(operator_name="unknown_operator", params={})
        )
        
        executor = PipelineExecutor()
        is_valid, errors = executor.validate_pipeline(pipeline)
        
        assert not is_valid
        assert len(errors) > 0


class TestCreatePipeline:
    """Tests for create_pipeline helper."""
    
    def test_create_simple_pipeline(self):
        """Test creating a simple pipeline."""
        pipeline = create_pipeline(
            name="simple",
            operators=[
                {"name": "multiply", "params": {"multiplier": 2.0}},
                {"name": "add", "params": {"addend": 5.0}},
            ]
        )
        
        assert pipeline.name == "simple"
        assert len(pipeline.nodes) == 2
    
    def test_create_realtime_pipeline(self):
        """Test creating a real-time pipeline."""
        pipeline = create_pipeline(
            name="realtime",
            operators=[{"name": "multiply"}],
            lane=LaneType.REALTIME,
        )
        
        assert pipeline.lane == LaneType.REALTIME
    
    def test_create_scheduled_pipeline(self):
        """Test creating a scheduled batch pipeline."""
        pipeline = create_pipeline(
            name="scheduled",
            operators=[{"name": "multiply"}],
            lane=LaneType.BATCH,
            schedule="0 * * * *",
        )
        
        assert pipeline.config.schedule == "0 * * * *"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
