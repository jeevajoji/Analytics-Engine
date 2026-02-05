"""
Serialization Utilities.

Provides consistent serialization/deserialization patterns.
"""

from typing import Any, Dict, Type, TypeVar, get_type_hints
from datetime import datetime, date, timedelta
from dataclasses import dataclass, fields, is_dataclass, asdict
from enum import Enum
import json


T = TypeVar('T')


def serialize_value(value: Any) -> Any:
    """
    Serialize a value for JSON/dict conversion.
    
    Handles common types:
    - datetime -> ISO string
    - date -> ISO string  
    - timedelta -> total seconds
    - Enum -> value
    - dataclass -> dict
    - list/tuple -> recursively serialized list
    - dict -> recursively serialized dict
    
    Args:
        value: Value to serialize
        
    Returns:
        JSON-serializable value
    """
    if value is None:
        return None
    
    if isinstance(value, datetime):
        return value.isoformat()
    
    if isinstance(value, date):
        return value.isoformat()
    
    if isinstance(value, timedelta):
        return value.total_seconds()
    
    if isinstance(value, Enum):
        return value.value
    
    if is_dataclass(value) and not isinstance(value, type):
        return {k: serialize_value(v) for k, v in asdict(value).items()}
    
    if isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    
    if isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    
    # Primitives pass through
    if isinstance(value, (str, int, float, bool)):
        return value
    
    # Fallback: convert to string
    return str(value)


def deserialize_value(value: Any, target_type: Type = None) -> Any:
    """
    Deserialize a value from JSON/dict.
    
    Args:
        value: Value to deserialize
        target_type: Optional type hint for conversion
        
    Returns:
        Deserialized value
    """
    if value is None:
        return None
    
    if target_type is None:
        return value
    
    # Handle Optional types
    origin = getattr(target_type, '__origin__', None)
    if origin is type(None):
        return None
    
    # datetime
    if target_type == datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return datetime.fromisoformat(value)
    
    # date
    if target_type == date:
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            return date.fromisoformat(value)
    
    # timedelta
    if target_type == timedelta:
        if isinstance(value, timedelta):
            return value
        if isinstance(value, (int, float)):
            return timedelta(seconds=value)
    
    # Enum
    if isinstance(target_type, type) and issubclass(target_type, Enum):
        if isinstance(value, target_type):
            return value
        return target_type(value)
    
    return value


class SerializableMixin:
    """
    Mixin providing consistent serialization for dataclasses.
    
    Add to any dataclass to get to_dict() and from_dict() methods.
    
    Example:
        @dataclass
        class MyConfig(SerializableMixin):
            name: str
            created_at: datetime
            
        config = MyConfig(name="test", created_at=datetime.now())
        data = config.to_dict()  # {"name": "test", "created_at": "2024-..."}
        restored = MyConfig.from_dict(data)
    """
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with serialized values."""
        if not is_dataclass(self):
            raise TypeError(f"{type(self).__name__} is not a dataclass")
        
        result = {}
        for field in fields(self):
            value = getattr(self, field.name)
            result[field.name] = serialize_value(value)
        
        return result
    
    def to_json(self, indent: int = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create instance from dictionary."""
        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} is not a dataclass")
        
        # Get type hints for proper deserialization
        hints = get_type_hints(cls) if hasattr(cls, '__annotations__') else {}
        
        kwargs = {}
        for field in fields(cls):
            if field.name in data:
                value = data[field.name]
                target_type = hints.get(field.name)
                kwargs[field.name] = deserialize_value(value, target_type)
        
        return cls(**kwargs)
    
    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """Create instance from JSON string."""
        return cls.from_dict(json.loads(json_str))


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """
    Convert a dataclass instance to a dictionary.
    
    Convenience function for non-mixin dataclasses.
    """
    if not is_dataclass(obj):
        raise TypeError(f"{type(obj).__name__} is not a dataclass")
    
    return {k: serialize_value(v) for k, v in asdict(obj).items()}
