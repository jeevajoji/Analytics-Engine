"""
Pipeline Executor for the Analytics Engine.

Provides DAG-based pipeline execution with support for:
- Sequential operator chaining
- Branching (one output → multiple operators)
- Merging (multiple inputs → one operator via JoinOperator)
- Chunked processing for memory efficiency
"""

from typing import Any, Dict, List, Optional, Iterator, Literal
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
import polars as pl
from datetime import datetime
import uuid

from .operator import Operator, OperatorResult, OperatorConfig
from .registry import OperatorRegistry, get_registry


class LaneType(str, Enum):
    """Pipeline execution lane type."""
    REALTIME = "realtime"  # Per-event, no buffering
    BATCH = "batch"        # Window-based, scheduled


class OperatorNode(BaseModel):
    """
    A node in the pipeline DAG representing an operator.
    """
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    operator_name: str = Field(..., description="Name of the operator from registry")
    params: Dict[str, Any] = Field(default_factory=dict, description="Operator parameters")
    next_nodes: List[str] = Field(default_factory=list, description="IDs of downstream nodes")
    
    model_config = ConfigDict(extra="allow")


class PipelineConfig(BaseModel):
    """Configuration for a pipeline."""
    
    name: str = Field(..., description="Pipeline name")
    description: Optional[str] = Field(default=None)
    lane: LaneType = Field(default=LaneType.BATCH, description="Execution lane")
    schedule: Optional[str] = Field(default=None, description="Cron expression for batch")
    chunk_size: int = Field(default=10000, description="Chunk size for batch processing")
    
    model_config = ConfigDict(use_enum_values=True)


class Pipeline(BaseModel):
    """
    A pipeline definition containing operator nodes.
    """
    
    config: PipelineConfig
    nodes: List[OperatorNode] = Field(default_factory=list)
    
    model_config = ConfigDict(extra="allow")
    
    @property
    def name(self) -> str:
        return self.config.name
    
    @property
    def lane(self) -> LaneType:
        return LaneType(self.config.lane)
    
    def add_operator(
        self,
        operator_name: str,
        params: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ) -> str:
        """
        Add an operator to the pipeline.
        
        Args:
            operator_name: Name of the operator from registry
            params: Operator parameters
            node_id: Optional custom node ID
            
        Returns:
            Node ID
        """
        node = OperatorNode(
            id=node_id or str(uuid.uuid4())[:8],
            operator_name=operator_name,
            params=params or {},
        )
        
        # Link to previous node if exists
        if self.nodes:
            self.nodes[-1].next_nodes.append(node.id)
        
        self.nodes.append(node)
        return node.id
    
    def get_node(self, node_id: str) -> Optional[OperatorNode]:
        """Get a node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_root_nodes(self) -> List[OperatorNode]:
        """Get nodes with no incoming edges (entry points)."""
        # For simple linear pipelines, just return first node
        if self.nodes:
            return [self.nodes[0]]
        return []
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize pipeline to dictionary."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pipeline":
        """Create pipeline from dictionary."""
        return cls.model_validate(data)
    
    @classmethod
    def from_yaml(cls, yaml_content: str) -> "Pipeline":
        """Create pipeline from YAML string."""
        import yaml
        data = yaml.safe_load(yaml_content)
        return cls.from_dict(data)


class PipelineExecutionResult(BaseModel):
    """Result of pipeline execution."""
    
    pipeline_name: str
    success: bool
    started_at: datetime
    completed_at: Optional[datetime] = None
    output_data: Optional[Any] = None
    node_results: Dict[str, OperatorResult] = Field(default_factory=dict)
    error: Optional[str] = None
    rows_processed: int = 0
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class PipelineExecutor:
    """
    Executes pipelines by running operators in sequence.
    
    Supports:
    - Batch execution (full DataFrame)
    - Chunked execution (for memory efficiency)
    - Streaming execution (for real-time lane)
    """
    
    def __init__(self, registry: Optional[OperatorRegistry] = None):
        """
        Initialize executor.
        
        Args:
            registry: Operator registry to use (defaults to global)
        """
        self.registry = registry or get_registry()
        self._instantiated_operators: Dict[str, Operator] = {}
    
    def _get_operator(self, node: OperatorNode) -> Operator:
        """
        Get or create an operator instance for a node.
        
        Args:
            node: The operator node
            
        Returns:
            Instantiated operator
        """
        if node.id in self._instantiated_operators:
            return self._instantiated_operators[node.id]
        
        # Get operator class and create config from params
        operator_class = self.registry.get_class(node.operator_name)
        
        # Try to create config from params
        config = None
        if node.params:
            # Check if operator has a specific config class
            config_class = getattr(operator_class, 'config_class', OperatorConfig)
            try:
                config = config_class(**node.params)
            except Exception:
                # Fall back to generic config
                config = OperatorConfig(**node.params)
        
        operator = operator_class(config=config)
        self._instantiated_operators[node.id] = operator
        return operator
    
    def execute(
        self,
        pipeline: Pipeline,
        input_data: pl.DataFrame,
    ) -> PipelineExecutionResult:
        """
        Execute a pipeline on input data.
        
        Args:
            pipeline: The pipeline to execute
            input_data: Input DataFrame
            
        Returns:
            PipelineExecutionResult with output data or error
        """
        result = PipelineExecutionResult(
            pipeline_name=pipeline.name,
            success=False,
            started_at=datetime.now(),
            rows_processed=len(input_data),
        )
        
        try:
            current_data = input_data
            
            # Execute operators in sequence
            for node in pipeline.nodes:
                operator = self._get_operator(node)
                
                # Validate input
                is_valid, errors = operator.validate_input(current_data)
                if not is_valid:
                    raise ValueError(f"Input validation failed for {node.operator_name}: {errors}")
                
                # Process data
                op_result = operator.process(current_data)
                result.node_results[node.id] = op_result
                
                if not op_result.success:
                    raise RuntimeError(
                        f"Operator {node.operator_name} failed: {op_result.error}"
                    )
                
                # Pass output to next operator
                current_data = op_result.data
                
                # Handle None output (terminal operators)
                if current_data is None:
                    break
            
            result.success = True
            result.output_data = current_data
            result.completed_at = datetime.now()
            
        except Exception as e:
            result.error = str(e)
            result.completed_at = datetime.now()
        
        # Cleanup instantiated operators
        self._instantiated_operators.clear()
        
        return result
    
    def execute_chunked(
        self,
        pipeline: Pipeline,
        input_data: pl.DataFrame,
        chunk_size: Optional[int] = None,
    ) -> Iterator[PipelineExecutionResult]:
        """
        Execute pipeline in chunks for memory efficiency.
        
        Args:
            pipeline: The pipeline to execute
            input_data: Input DataFrame
            chunk_size: Rows per chunk (defaults to pipeline config)
            
        Yields:
            PipelineExecutionResult for each chunk
        """
        chunk_size = chunk_size or pipeline.config.chunk_size
        total_rows = len(input_data)
        
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            chunk = input_data.slice(start, end - start)
            
            result = self.execute(pipeline, chunk)
            result.rows_processed = len(chunk)
            yield result
    
    def execute_stream(
        self,
        pipeline: Pipeline,
        stream: Iterator[Dict[str, Any]],
    ) -> Iterator[PipelineExecutionResult]:
        """
        Execute pipeline on streaming data (for real-time lane).
        
        Args:
            pipeline: The pipeline to execute
            stream: Iterator of data records
            
        Yields:
            PipelineExecutionResult for each record
        """
        for record in stream:
            # Convert single record to DataFrame
            df = pl.DataFrame([record])
            result = self.execute(pipeline, df)
            yield result
    
    def validate_pipeline(self, pipeline: Pipeline) -> tuple[bool, List[str]]:
        """
        Validate a pipeline definition.
        
        Checks:
        - All operators exist in registry
        - Node references are valid
        
        Args:
            pipeline: Pipeline to validate
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        node_ids = {node.id for node in pipeline.nodes}
        
        for node in pipeline.nodes:
            # Check operator exists
            if not self.registry.exists(node.operator_name):
                errors.append(f"Operator '{node.operator_name}' not found in registry")
            
            # Check next_nodes references are valid
            for next_id in node.next_nodes:
                if next_id not in node_ids:
                    errors.append(f"Node '{node.id}' references unknown node '{next_id}'")
        
        return len(errors) == 0, errors


def create_pipeline(
    name: str,
    operators: List[Dict[str, Any]],
    lane: LaneType = LaneType.BATCH,
    schedule: Optional[str] = None,
) -> Pipeline:
    """
    Helper function to create a pipeline from operator definitions.
    
    Args:
        name: Pipeline name
        operators: List of operator configs with 'name' and 'params'
        lane: Execution lane type
        schedule: Cron schedule for batch pipelines
        
    Returns:
        Configured Pipeline
    """
    pipeline = Pipeline(
        config=PipelineConfig(
            name=name,
            lane=lane,
            schedule=schedule,
        )
    )
    
    for op_def in operators:
        pipeline.add_operator(
            operator_name=op_def["name"],
            params=op_def.get("params", {}),
        )
    
    return pipeline
