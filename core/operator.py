"""
Base Operator class for the Analytics Engine.

All operators must inherit from this base class and implement the process() method.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional, TypeVar, Generic
from pydantic import BaseModel, Field, ConfigDict
import polars as pl

from .schema import DataSchema


class OperatorConfig(BaseModel):
    """Base configuration for all operators."""
    
    model_config = ConfigDict(extra="allow")
    
    # Subclasses should define their specific parameters here


T = TypeVar("T", bound=OperatorConfig)


class OperatorResult(BaseModel):
    """Result returned by an operator after processing."""
    
    success: bool = Field(default=True, description="Whether processing succeeded")
    data: Optional[Any] = Field(default=None, description="Output data (DataFrame or records)")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Operator(ABC, Generic[T]):
    """
    Abstract base class for all Analytics Engine operators.
    
    Operators are stateless, reusable processing units that:
    - Accept data (Polars DataFrame or iterator of records)
    - Apply transformations based on parameters
    - Produce output data
    
    All operators must be GENERIC - no domain-specific logic.
    Business logic comes from parameters, not code.
    """
    
    # Class-level attributes (override in subclasses)
    name: str = "base_operator"
    description: str = "Base operator class"
    version: str = "1.0.0"
    
    # Schema definitions (override in subclasses)
    input_schema: Optional[DataSchema] = None
    output_schema: Optional[DataSchema] = None
    
    def __init__(self, config: Optional[T] = None):
        """
        Initialize operator with configuration.
        
        Args:
            config: Operator-specific configuration parameters
        """
        self._config = config
        self._validate_config()
    
    @property
    def config(self) -> Optional[T]:
        """Get operator configuration."""
        return self._config
    
    def _validate_config(self) -> None:
        """
        Validate operator configuration.
        Override in subclasses for custom validation.
        """
        pass
    
    @abstractmethod
    def process(self, data: pl.DataFrame) -> OperatorResult:
        """
        Process input data and produce output.
        
        This is the main processing method that subclasses must implement.
        
        Args:
            data: Input Polars DataFrame
            
        Returns:
            OperatorResult containing processed data or error
        """
        pass
    
    def process_stream(self, stream: Iterator[Dict[str, Any]]) -> Iterator[OperatorResult]:
        """
        Process streaming data (for real-time lane).
        
        Default implementation converts each record to single-row DataFrame.
        Override for more efficient streaming implementations.
        
        Args:
            stream: Iterator of data records
            
        Yields:
            OperatorResult for each processed record
        """
        for record in stream:
            df = pl.DataFrame([record])
            yield self.process(df)
    
    def process_chunked(
        self, 
        data: pl.DataFrame, 
        chunk_size: int = 10000
    ) -> Iterator[OperatorResult]:
        """
        Process data in chunks to limit memory usage.
        
        Args:
            data: Input Polars DataFrame
            chunk_size: Number of rows per chunk
            
        Yields:
            OperatorResult for each chunk
        """
        total_rows = len(data)
        
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            chunk = data.slice(start, end - start)
            yield self.process(chunk)
    
    def validate_input(self, data: pl.DataFrame) -> tuple[bool, list[str]]:
        """
        Validate input data against input schema.
        
        Args:
            data: Input DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        if self.input_schema is None:
            return True, []
        
        errors = []
        data_columns = set(data.columns)
        
        # Check required fields exist
        missing = self.input_schema.required_fields - data_columns
        if missing:
            errors.append(f"Missing required columns: {missing}")
        
        return len(errors) == 0, errors
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, version={self.version})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize operator to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "config": self._config.model_dump() if self._config else None,
        }
