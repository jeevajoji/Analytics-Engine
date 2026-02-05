"""
Schema definitions for data validation in the Analytics Engine.

Provides lightweight schema validation for operator inputs/outputs.
"""

from typing import Any, Dict, List, Optional, Set, Literal
from enum import Enum
from pydantic import BaseModel, Field


class DataType(str, Enum):
    """Supported data types for schema fields."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    ARRAY = "array"
    OBJECT = "object"
    ANY = "any"


class SchemaField(BaseModel):
    """Definition of a single field in a data schema."""
    
    name: str = Field(..., description="Field name")
    dtype: DataType = Field(..., description="Data type of the field")
    required: bool = Field(default=True, description="Whether field is required")
    nullable: bool = Field(default=False, description="Whether field can be null")
    description: Optional[str] = Field(default=None, description="Field description")
    
    class Config:
        use_enum_values = True


class DataSchema(BaseModel):
    """
    Schema definition for operator input/output data.
    
    Used to validate data flowing between operators in a pipeline.
    """
    
    fields: List[SchemaField] = Field(default_factory=list, description="List of fields")
    allow_extra: bool = Field(default=True, description="Allow fields not in schema")
    
    @property
    def required_fields(self) -> Set[str]:
        """Get set of required field names."""
        return {f.name for f in self.fields if f.required}
    
    @property
    def field_names(self) -> Set[str]:
        """Get set of all field names."""
        return {f.name for f in self.fields}
    
    def validate_data(self, data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate data against schema.
        
        Args:
            data: Dictionary of field names to values
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Check required fields
        missing = self.required_fields - set(data.keys())
        if missing:
            errors.append(f"Missing required fields: {missing}")
        
        # Check extra fields if not allowed
        if not self.allow_extra:
            extra = set(data.keys()) - self.field_names
            if extra:
                errors.append(f"Unexpected fields: {extra}")
        
        return len(errors) == 0, errors
    
    def merge_with(self, other: "DataSchema") -> "DataSchema":
        """
        Merge this schema with another, combining fields.
        
        Args:
            other: Another DataSchema to merge with
            
        Returns:
            New DataSchema with combined fields
        """
        existing_names = self.field_names
        new_fields = self.fields.copy()
        
        for field in other.fields:
            if field.name not in existing_names:
                new_fields.append(field)
        
        return DataSchema(
            fields=new_fields,
            allow_extra=self.allow_extra and other.allow_extra
        )


# Common schema templates
TELEMETRY_SCHEMA = DataSchema(
    fields=[
        SchemaField(name="timestamp", dtype=DataType.DATETIME, required=True),
        SchemaField(name="asset_id", dtype=DataType.STRING, required=True),
        SchemaField(name="value", dtype=DataType.FLOAT, required=True),
    ]
)

EVENT_SCHEMA = DataSchema(
    fields=[
        SchemaField(name="event_id", dtype=DataType.STRING, required=True),
        SchemaField(name="timestamp", dtype=DataType.DATETIME, required=True),
        SchemaField(name="event_type", dtype=DataType.STRING, required=True),
        SchemaField(name="severity", dtype=DataType.STRING, required=True),
        SchemaField(name="asset_id", dtype=DataType.STRING, required=False),
        SchemaField(name="payload", dtype=DataType.OBJECT, required=False),
    ]
)

INSIGHT_SCHEMA = DataSchema(
    fields=[
        SchemaField(name="insight_id", dtype=DataType.STRING, required=True),
        SchemaField(name="timestamp", dtype=DataType.DATETIME, required=True),
        SchemaField(name="insight_type", dtype=DataType.STRING, required=True),
        SchemaField(name="message", dtype=DataType.STRING, required=True),
        SchemaField(name="confidence", dtype=DataType.FLOAT, required=False),
        SchemaField(name="metadata", dtype=DataType.OBJECT, required=False),
    ]
)
