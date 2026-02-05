"""
Central Schema Registry Module.

Provides centralized schema management for the Analytics Engine:
- Schema storage and retrieval
- Schema validation
- Schema evolution and compatibility
- Auto-generation from data samples
"""

import json
import hashlib
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
import copy

import polars as pl
from pydantic import BaseModel, Field

from analytics_engine.core.schema import DataSchema, SchemaField, DataType


logger = logging.getLogger(__name__)


class SchemaCompatibility(str, Enum):
    """Schema compatibility modes."""
    NONE = "none"  # No compatibility checks
    BACKWARD = "backward"  # New schema can read old data
    FORWARD = "forward"  # Old schema can read new data
    FULL = "full"  # Both backward and forward compatible
    TRANSITIVE_BACKWARD = "transitive_backward"  # Backward compatible with all versions
    TRANSITIVE_FORWARD = "transitive_forward"  # Forward compatible with all versions
    TRANSITIVE_FULL = "transitive_full"  # Full compatible with all versions


class SchemaType(str, Enum):
    """Types of schemas in the registry."""
    INPUT = "input"  # Pipeline input schema
    OUTPUT = "output"  # Pipeline output schema
    INTERMEDIATE = "intermediate"  # Between operators
    EVENT = "event"  # Event/notification schema
    OPERATOR_CONFIG = "operator_config"  # Operator configuration


@dataclass
class SchemaVersion:
    """
    A versioned schema in the registry.
    
    Attributes:
        schema_id: Unique schema identifier
        subject: Schema subject (e.g., "pipeline.my_pipeline.input")
        version: Version number
        schema_def: The actual schema definition
        schema_hash: Hash of the schema for quick comparison
        created_at: Creation timestamp
        compatibility: Compatibility mode
        deprecated: Whether this version is deprecated
        metadata: Additional metadata
    """
    schema_id: str
    subject: str
    version: int
    schema_def: DataSchema
    schema_hash: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    compatibility: SchemaCompatibility = SchemaCompatibility.BACKWARD
    deprecated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.schema_hash:
            self.schema_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute hash of the schema."""
        content = json.dumps(self.schema_def.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "subject": self.subject,
            "version": self.version,
            "schema_def": self.schema_def.to_dict(),
            "schema_hash": self.schema_hash,
            "created_at": self.created_at.isoformat(),
            "compatibility": self.compatibility.value,
            "deprecated": self.deprecated,
            "metadata": self.metadata,
        }


class SchemaStore(ABC):
    """Abstract base class for schema storage."""
    
    @abstractmethod
    async def save(self, schema_version: SchemaVersion) -> None:
        """Save a schema version."""
        pass
    
    @abstractmethod
    async def get(self, subject: str, version: int) -> Optional[SchemaVersion]:
        """Get a specific schema version."""
        pass
    
    @abstractmethod
    async def get_latest(self, subject: str) -> Optional[SchemaVersion]:
        """Get the latest version of a schema."""
        pass
    
    @abstractmethod
    async def get_by_id(self, schema_id: str) -> Optional[SchemaVersion]:
        """Get schema by ID."""
        pass
    
    @abstractmethod
    async def list_versions(self, subject: str) -> List[SchemaVersion]:
        """List all versions of a subject."""
        pass
    
    @abstractmethod
    async def list_subjects(self) -> List[str]:
        """List all subjects."""
        pass
    
    @abstractmethod
    async def delete(self, subject: str, version: Optional[int] = None) -> bool:
        """Delete a schema version or all versions of a subject."""
        pass


class InMemorySchemaStore(SchemaStore):
    """In-memory schema store for development/testing."""
    
    def __init__(self):
        self._schemas: Dict[str, Dict[int, SchemaVersion]] = {}
        self._by_id: Dict[str, SchemaVersion] = {}
        self._lock = threading.Lock()
    
    async def save(self, schema_version: SchemaVersion) -> None:
        with self._lock:
            if schema_version.subject not in self._schemas:
                self._schemas[schema_version.subject] = {}
            
            self._schemas[schema_version.subject][schema_version.version] = schema_version
            self._by_id[schema_version.schema_id] = schema_version
    
    async def get(self, subject: str, version: int) -> Optional[SchemaVersion]:
        with self._lock:
            if subject not in self._schemas:
                return None
            return self._schemas[subject].get(version)
    
    async def get_latest(self, subject: str) -> Optional[SchemaVersion]:
        with self._lock:
            if subject not in self._schemas:
                return None
            
            versions = self._schemas[subject]
            if not versions:
                return None
            
            max_version = max(versions.keys())
            return versions[max_version]
    
    async def get_by_id(self, schema_id: str) -> Optional[SchemaVersion]:
        with self._lock:
            return self._by_id.get(schema_id)
    
    async def list_versions(self, subject: str) -> List[SchemaVersion]:
        with self._lock:
            if subject not in self._schemas:
                return []
            return sorted(
                self._schemas[subject].values(),
                key=lambda s: s.version,
                reverse=True,
            )
    
    async def list_subjects(self) -> List[str]:
        with self._lock:
            return list(self._schemas.keys())
    
    async def delete(self, subject: str, version: Optional[int] = None) -> bool:
        with self._lock:
            if subject not in self._schemas:
                return False
            
            if version is not None:
                if version in self._schemas[subject]:
                    schema = self._schemas[subject].pop(version)
                    self._by_id.pop(schema.schema_id, None)
                    return True
                return False
            else:
                for schema in self._schemas[subject].values():
                    self._by_id.pop(schema.schema_id, None)
                del self._schemas[subject]
                return True


class SchemaEvolutionChecker:
    """Checks schema compatibility for evolution."""
    
    def is_backward_compatible(
        self,
        new_schema: DataSchema,
        old_schema: DataSchema,
    ) -> Tuple[bool, List[str]]:
        """
        Check if new schema can read data written with old schema.
        
        Rules:
        - Can add optional fields
        - Cannot remove required fields
        - Cannot change field types to incompatible types
        
        Returns:
            Tuple of (is_compatible, list of issues)
        """
        issues = []
        
        old_fields = {f.name: f for f in old_schema.fields}
        new_fields = {f.name: f for f in new_schema.fields}
        
        # Check removed fields
        for name, old_field in old_fields.items():
            if name not in new_fields:
                if old_field.required:
                    issues.append(f"Required field '{name}' was removed")
                else:
                    issues.append(f"Warning: Optional field '{name}' was removed")
            else:
                # Check type compatibility
                new_field = new_fields[name]
                if not self._types_compatible(old_field.data_type, new_field.data_type):
                    issues.append(
                        f"Field '{name}' type changed from "
                        f"{old_field.data_type} to {new_field.data_type}"
                    )
        
        # New required fields without defaults break backward compatibility
        for name, new_field in new_fields.items():
            if name not in old_fields:
                if new_field.required and new_field.default is None:
                    issues.append(
                        f"New required field '{name}' added without default"
                    )
        
        # Filter out warnings for compatibility check
        breaking_issues = [i for i in issues if not i.startswith("Warning:")]
        return len(breaking_issues) == 0, issues
    
    def is_forward_compatible(
        self,
        new_schema: DataSchema,
        old_schema: DataSchema,
    ) -> Tuple[bool, List[str]]:
        """
        Check if old schema can read data written with new schema.
        
        Rules:
        - Can remove optional fields
        - Cannot add required fields
        - Cannot change field types to incompatible types
        
        Returns:
            Tuple of (is_compatible, list of issues)
        """
        issues = []
        
        old_fields = {f.name: f for f in old_schema.fields}
        new_fields = {f.name: f for f in new_schema.fields}
        
        # New required fields break forward compatibility
        for name, new_field in new_fields.items():
            if name not in old_fields:
                if new_field.required:
                    issues.append(f"New required field '{name}' breaks forward compatibility")
        
        # Check type compatibility for existing fields
        for name in old_fields.keys() & new_fields.keys():
            old_field = old_fields[name]
            new_field = new_fields[name]
            
            if not self._types_compatible(new_field.data_type, old_field.data_type):
                issues.append(
                    f"Field '{name}' type changed from "
                    f"{old_field.data_type} to {new_field.data_type}"
                )
        
        return len(issues) == 0, issues
    
    def is_full_compatible(
        self,
        new_schema: DataSchema,
        old_schema: DataSchema,
    ) -> Tuple[bool, List[str]]:
        """Check both backward and forward compatibility."""
        backward_ok, backward_issues = self.is_backward_compatible(new_schema, old_schema)
        forward_ok, forward_issues = self.is_forward_compatible(new_schema, old_schema)
        
        all_issues = list(set(backward_issues + forward_issues))
        return backward_ok and forward_ok, all_issues
    
    def _types_compatible(
        self,
        source_type: DataType,
        target_type: DataType,
    ) -> bool:
        """Check if source type can be safely converted to target type."""
        if source_type == target_type:
            return True
        
        # Define type compatibility matrix
        compatible_promotions = {
            (DataType.INTEGER, DataType.FLOAT): True,
            (DataType.INTEGER, DataType.STRING): True,
            (DataType.FLOAT, DataType.STRING): True,
            (DataType.BOOLEAN, DataType.STRING): True,
            (DataType.BOOLEAN, DataType.INTEGER): True,
        }
        
        return compatible_promotions.get((source_type, target_type), False)


class SchemaGenerator:
    """Generates schemas from data samples."""
    
    def from_dataframe(
        self,
        df: pl.DataFrame,
        name: str = "auto_generated",
        sample_size: int = 100,
    ) -> DataSchema:
        """
        Generate a DataSchema from a Polars DataFrame.
        
        Args:
            df: Source DataFrame
            name: Schema name
            sample_size: Number of rows to analyze for nullable detection
            
        Returns:
            Generated DataSchema
        """
        fields = []
        
        for col_name in df.columns:
            col = df[col_name]
            
            # Map Polars dtype to DataType
            data_type = self._polars_to_datatype(col.dtype)
            
            # Check if column has nulls (use sample for efficiency)
            sample = df.head(min(sample_size, len(df)))
            has_nulls = sample[col_name].null_count() > 0
            
            fields.append(SchemaField(
                name=col_name,
                data_type=data_type,
                required=not has_nulls,
                description=f"Auto-generated from column {col_name}",
            ))
        
        return DataSchema(name=name, fields=fields)
    
    def from_dict_list(
        self,
        data: List[Dict[str, Any]],
        name: str = "auto_generated",
    ) -> DataSchema:
        """
        Generate a DataSchema from a list of dictionaries.
        
        Args:
            data: List of dictionaries
            name: Schema name
            
        Returns:
            Generated DataSchema
        """
        if not data:
            return DataSchema(name=name, fields=[])
        
        # Collect all unique keys and their types
        field_types: Dict[str, Set[type]] = {}
        field_nullability: Dict[str, bool] = {}
        
        for record in data:
            for key, value in record.items():
                if key not in field_types:
                    field_types[key] = set()
                    field_nullability[key] = False
                
                if value is None:
                    field_nullability[key] = True
                else:
                    field_types[key].add(type(value))
        
        fields = []
        for key, types in field_types.items():
            data_type = self._python_types_to_datatype(types)
            
            fields.append(SchemaField(
                name=key,
                data_type=data_type,
                required=not field_nullability[key],
                description=f"Auto-generated from field {key}",
            ))
        
        return DataSchema(name=name, fields=fields)
    
    def from_json_sample(
        self,
        json_str: str,
        name: str = "auto_generated",
    ) -> DataSchema:
        """Generate schema from JSON sample."""
        data = json.loads(json_str)
        
        if isinstance(data, list):
            return self.from_dict_list(data, name)
        elif isinstance(data, dict):
            return self.from_dict_list([data], name)
        else:
            raise ValueError("JSON must be an object or array of objects")
    
    def _polars_to_datatype(self, dtype) -> DataType:
        """Map Polars dtype to DataType."""
        dtype_str = str(dtype).lower()
        
        if "int" in dtype_str:
            return DataType.INTEGER
        elif "float" in dtype_str or "decimal" in dtype_str:
            return DataType.FLOAT
        elif "bool" in dtype_str:
            return DataType.BOOLEAN
        elif "datetime" in dtype_str or "date" in dtype_str:
            return DataType.DATETIME
        elif "duration" in dtype_str:
            return DataType.STRING  # Store durations as ISO strings
        elif "list" in dtype_str:
            return DataType.LIST
        elif "struct" in dtype_str:
            return DataType.OBJECT
        else:
            return DataType.STRING
    
    def _python_types_to_datatype(self, types: Set[type]) -> DataType:
        """Infer DataType from a set of Python types."""
        # Remove NoneType
        types = {t for t in types if t is not type(None)}
        
        if not types:
            return DataType.STRING
        
        if len(types) == 1:
            t = next(iter(types))
            return self._python_type_to_datatype(t)
        
        # Multiple types - try to find common supertype
        if int in types and float in types:
            return DataType.FLOAT
        
        # Default to string for mixed types
        return DataType.STRING
    
    def _python_type_to_datatype(self, t: type) -> DataType:
        """Map Python type to DataType."""
        mapping = {
            int: DataType.INTEGER,
            float: DataType.FLOAT,
            bool: DataType.BOOLEAN,
            str: DataType.STRING,
            list: DataType.LIST,
            dict: DataType.OBJECT,
            datetime: DataType.DATETIME,
        }
        return mapping.get(t, DataType.STRING)


class SchemaRegistry:
    """
    Central schema registry for the Analytics Engine.
    
    Features:
    - Register and retrieve schemas
    - Version management
    - Compatibility checking
    - Auto-generation from samples
    
    Example:
        registry = SchemaRegistry()
        
        # Register a schema
        schema = DataSchema(
            name="temperature_reading",
            fields=[
                SchemaField("device_id", DataType.STRING, required=True),
                SchemaField("temperature", DataType.FLOAT, required=True),
                SchemaField("timestamp", DataType.DATETIME, required=True),
            ],
        )
        
        schema_id = await registry.register(
            subject="pipeline.temperature_alerts.input",
            schema_def=schema,
        )
        
        # Get latest schema
        latest = await registry.get_latest("pipeline.temperature_alerts.input")
        
        # Validate data against schema
        is_valid, errors = await registry.validate(df, subject)
    """
    
    def __init__(
        self,
        store: Optional[SchemaStore] = None,
        default_compatibility: SchemaCompatibility = SchemaCompatibility.BACKWARD,
    ):
        """
        Initialize schema registry.
        
        Args:
            store: Schema storage backend
            default_compatibility: Default compatibility mode
        """
        self.store = store or InMemorySchemaStore()
        self.default_compatibility = default_compatibility
        self.evolution_checker = SchemaEvolutionChecker()
        self.generator = SchemaGenerator()
        
        # Subject compatibility overrides
        self._subject_compatibility: Dict[str, SchemaCompatibility] = {}
        self._lock = threading.Lock()
    
    async def register(
        self,
        subject: str,
        schema_def: DataSchema,
        compatibility: Optional[SchemaCompatibility] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register a new schema version.
        
        Args:
            subject: Schema subject (e.g., "pipeline.my_pipeline.input")
            schema_def: The schema definition
            compatibility: Compatibility mode (uses default if not specified)
            metadata: Additional metadata
            
        Returns:
            Schema ID
            
        Raises:
            ValueError: If schema is not compatible with previous versions
        """
        compat = compatibility or self._get_subject_compatibility(subject)
        
        # Get latest version for comparison
        latest = await self.store.get_latest(subject)
        
        if latest and compat != SchemaCompatibility.NONE:
            # Check compatibility
            is_compatible, issues = self._check_compatibility(
                schema_def, latest.schema_def, compat
            )
            
            if not is_compatible:
                raise ValueError(
                    f"Schema not compatible ({compat.value}): {'; '.join(issues)}"
                )
        
        # Determine version number
        version = (latest.version + 1) if latest else 1
        
        # Generate unique ID
        hash_content = f"{subject}:{version}:{datetime.now().isoformat()}"
        schema_id = hashlib.sha256(hash_content.encode()).hexdigest()[:12]
        
        # Create schema version
        schema_version = SchemaVersion(
            schema_id=schema_id,
            subject=subject,
            version=version,
            schema_def=schema_def,
            compatibility=compat,
            metadata=metadata or {},
        )
        
        await self.store.save(schema_version)
        
        logger.info(f"Registered schema {subject} v{version} (ID: {schema_id})")
        
        return schema_id
    
    async def get(
        self,
        subject: str,
        version: Optional[int] = None,
    ) -> Optional[SchemaVersion]:
        """
        Get a schema version.
        
        Args:
            subject: Schema subject
            version: Specific version (latest if not specified)
            
        Returns:
            SchemaVersion or None
        """
        if version is not None:
            return await self.store.get(subject, version)
        return await self.store.get_latest(subject)
    
    async def get_by_id(self, schema_id: str) -> Optional[SchemaVersion]:
        """Get schema by ID."""
        return await self.store.get_by_id(schema_id)
    
    async def get_latest(self, subject: str) -> Optional[SchemaVersion]:
        """Get the latest version of a schema."""
        return await self.store.get_latest(subject)
    
    async def list_versions(self, subject: str) -> List[SchemaVersion]:
        """List all versions of a subject."""
        return await self.store.list_versions(subject)
    
    async def list_subjects(self) -> List[str]:
        """List all subjects in the registry."""
        return await self.store.list_subjects()
    
    async def delete(
        self,
        subject: str,
        version: Optional[int] = None,
    ) -> bool:
        """Delete a schema or subject."""
        return await self.store.delete(subject, version)
    
    async def deprecate(
        self,
        subject: str,
        version: int,
    ) -> bool:
        """Mark a schema version as deprecated."""
        schema = await self.store.get(subject, version)
        if not schema:
            return False
        
        schema.deprecated = True
        await self.store.save(schema)
        return True
    
    # -------------------------------------------------------------------------
    # Compatibility
    # -------------------------------------------------------------------------
    
    def set_subject_compatibility(
        self,
        subject: str,
        compatibility: SchemaCompatibility,
    ) -> None:
        """Set compatibility mode for a subject."""
        with self._lock:
            self._subject_compatibility[subject] = compatibility
    
    def _get_subject_compatibility(self, subject: str) -> SchemaCompatibility:
        """Get compatibility mode for a subject."""
        with self._lock:
            return self._subject_compatibility.get(subject, self.default_compatibility)
    
    def _check_compatibility(
        self,
        new_schema: DataSchema,
        old_schema: DataSchema,
        mode: SchemaCompatibility,
    ) -> Tuple[bool, List[str]]:
        """Check schema compatibility."""
        if mode == SchemaCompatibility.NONE:
            return True, []
        elif mode == SchemaCompatibility.BACKWARD:
            return self.evolution_checker.is_backward_compatible(new_schema, old_schema)
        elif mode == SchemaCompatibility.FORWARD:
            return self.evolution_checker.is_forward_compatible(new_schema, old_schema)
        elif mode == SchemaCompatibility.FULL:
            return self.evolution_checker.is_full_compatible(new_schema, old_schema)
        else:
            # Transitive checks require checking against all versions
            # For simplicity, just check against the latest
            if "backward" in mode.value:
                return self.evolution_checker.is_backward_compatible(new_schema, old_schema)
            elif "forward" in mode.value:
                return self.evolution_checker.is_forward_compatible(new_schema, old_schema)
            else:
                return self.evolution_checker.is_full_compatible(new_schema, old_schema)
    
    async def check_compatibility(
        self,
        subject: str,
        schema_def: DataSchema,
    ) -> Tuple[bool, List[str]]:
        """
        Check if a schema is compatible with the current version.
        
        Args:
            subject: Schema subject
            schema_def: Schema to check
            
        Returns:
            Tuple of (is_compatible, list of issues)
        """
        latest = await self.store.get_latest(subject)
        if not latest:
            return True, []
        
        compat = self._get_subject_compatibility(subject)
        return self._check_compatibility(schema_def, latest.schema_def, compat)
    
    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    
    async def validate(
        self,
        data: pl.DataFrame,
        subject: str,
        version: Optional[int] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Validate data against a schema.
        
        Args:
            data: DataFrame to validate
            subject: Schema subject
            version: Schema version (latest if not specified)
            
        Returns:
            Tuple of (is_valid, list of errors)
        """
        schema = await self.get(subject, version)
        if not schema:
            return False, [f"Schema not found: {subject}"]
        
        return self._validate_dataframe(data, schema.schema_def)
    
    def _validate_dataframe(
        self,
        df: pl.DataFrame,
        schema: DataSchema,
    ) -> Tuple[bool, List[str]]:
        """Validate a DataFrame against a schema."""
        errors = []
        
        # Check required fields
        for field in schema.fields:
            if field.required and field.name not in df.columns:
                errors.append(f"Missing required field: {field.name}")
            elif field.name in df.columns:
                # Check type compatibility
                expected_type = self._datatype_to_polars_class(field.data_type)
                actual_dtype = str(df[field.name].dtype).lower()
                
                if not self._dtype_matches(actual_dtype, field.data_type):
                    errors.append(
                        f"Field '{field.name}' has type {actual_dtype}, "
                        f"expected {field.data_type.value}"
                    )
                
                # Check null constraint
                if field.required and df[field.name].null_count() > 0:
                    errors.append(f"Required field '{field.name}' contains null values")
        
        return len(errors) == 0, errors
    
    def _datatype_to_polars_class(self, data_type: DataType) -> str:
        """Get Polars dtype string for a DataType."""
        mapping = {
            DataType.INTEGER: "int",
            DataType.FLOAT: "float",
            DataType.BOOLEAN: "bool",
            DataType.STRING: "str",
            DataType.DATETIME: "datetime",
            DataType.DATE: "date",
            DataType.LIST: "list",
            DataType.OBJECT: "struct",
        }
        return mapping.get(data_type, "str")
    
    def _dtype_matches(self, actual: str, expected: DataType) -> bool:
        """Check if actual dtype matches expected DataType."""
        expected_str = self._datatype_to_polars_class(expected)
        return expected_str.lower() in actual.lower()
    
    # -------------------------------------------------------------------------
    # Auto-generation
    # -------------------------------------------------------------------------
    
    async def register_from_dataframe(
        self,
        df: pl.DataFrame,
        subject: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate and register a schema from a DataFrame.
        
        Args:
            df: Source DataFrame
            subject: Schema subject
            name: Schema name
            metadata: Additional metadata
            
        Returns:
            Schema ID
        """
        schema = self.generator.from_dataframe(df, name or subject)
        return await self.register(subject, schema, metadata=metadata)
    
    async def register_from_json(
        self,
        json_str: str,
        subject: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate and register a schema from JSON sample.
        
        Args:
            json_str: JSON sample
            subject: Schema subject
            name: Schema name
            metadata: Additional metadata
            
        Returns:
            Schema ID
        """
        schema = self.generator.from_json_sample(json_str, name or subject)
        return await self.register(subject, schema, metadata=metadata)


# -------------------------------------------------------------------------
# Subject Naming Helpers
# -------------------------------------------------------------------------

def get_pipeline_input_subject(pipeline_name: str) -> str:
    """Get the subject name for a pipeline's input schema."""
    return f"pipeline.{pipeline_name}.input"


def get_pipeline_output_subject(pipeline_name: str) -> str:
    """Get the subject name for a pipeline's output schema."""
    return f"pipeline.{pipeline_name}.output"


def get_operator_schema_subject(
    pipeline_name: str,
    operator_name: str,
    schema_type: SchemaType = SchemaType.OUTPUT,
) -> str:
    """Get the subject name for an operator's schema."""
    return f"pipeline.{pipeline_name}.{operator_name}.{schema_type.value}"


def get_event_schema_subject(event_type: str) -> str:
    """Get the subject name for an event schema."""
    return f"event.{event_type}"


# -------------------------------------------------------------------------
# Global Instance
# -------------------------------------------------------------------------

_schema_registry: Optional[SchemaRegistry] = None


def get_schema_registry() -> SchemaRegistry:
    """Get the global schema registry instance."""
    global _schema_registry
    if _schema_registry is None:
        _schema_registry = SchemaRegistry()
    return _schema_registry


def set_schema_registry(registry: SchemaRegistry) -> None:
    """Set the global schema registry instance."""
    global _schema_registry
    _schema_registry = registry
