"""
Operator Registry for the Analytics Engine.

Provides a central registry for all available operators.
Supports lazy loading and dynamic registration.
"""

from typing import Dict, List, Optional, Type, Any
from .operator import Operator, OperatorConfig
from .exceptions import OperatorNotFoundError, OperatorConfigError


class OperatorRegistrationError(Exception):
    """Raised when operator registration fails."""
    pass


class OperatorRegistry:
    """
    Central registry for all Analytics Engine operators.
    
    Supports:
    - Registration of operator classes
    - Lazy instantiation of operators
    - Operator discovery and listing
    """
    
    _instance: Optional["OperatorRegistry"] = None
    _operators: Dict[str, Type[Operator]] = {}
    
    def __new__(cls) -> "OperatorRegistry":
        """Singleton pattern - only one registry instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._operators = {}
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """Reset the registry (useful for testing)."""
        cls._instance = None
        cls._operators = {}
    
    def register(self, operator_class: Type[Operator]) -> None:
        """
        Register an operator class.
        
        Args:
            operator_class: The operator class to register
            
        Raises:
            OperatorRegistrationError: If registration fails
        """
        if not issubclass(operator_class, Operator):
            raise OperatorRegistrationError(
                f"{operator_class} must be a subclass of Operator"
            )
        
        name = operator_class.name
        if not name or name == "base_operator":
            raise OperatorRegistrationError(
                f"Operator class must define a unique 'name' attribute"
            )
        
        if name in self._operators:
            # Allow re-registration (for updates)
            pass
        
        self._operators[name] = operator_class
    
    def unregister(self, name: str) -> None:
        """
        Remove an operator from the registry.
        
        Args:
            name: Name of the operator to remove
        """
        if name in self._operators:
            del self._operators[name]
    
    def get(self, name: str, config: Optional[OperatorConfig] = None) -> Operator:
        """
        Get an instantiated operator by name.
        
        Args:
            name: Name of the operator
            config: Optional configuration for the operator
            
        Returns:
            Instantiated operator
            
        Raises:
            OperatorNotFoundError: If operator not found
        """
        if name not in self._operators:
            raise OperatorNotFoundError(f"Operator '{name}' not found in registry")
        
        operator_class = self._operators[name]
        return operator_class(config=config)
    
    def get_class(self, name: str) -> Type[Operator]:
        """
        Get an operator class by name (without instantiating).
        
        Args:
            name: Name of the operator
            
        Returns:
            Operator class
            
        Raises:
            OperatorNotFoundError: If operator not found
        """
        if name not in self._operators:
            raise OperatorNotFoundError(f"Operator '{name}' not found in registry")
        
        return self._operators[name]
    
    def exists(self, name: str) -> bool:
        """Check if an operator is registered."""
        return name in self._operators
    
    def list_operators(self) -> List[str]:
        """Get list of all registered operator names."""
        return list(self._operators.keys())
    
    def get_all_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all registered operators.
        
        Returns:
            List of operator info dictionaries
        """
        return [
            {
                "name": op_class.name,
                "description": op_class.description,
                "version": op_class.version,
            }
            for op_class in self._operators.values()
        ]
    
    def __contains__(self, name: str) -> bool:
        return self.exists(name)
    
    def __len__(self) -> int:
        return len(self._operators)
    
    def __repr__(self) -> str:
        return f"OperatorRegistry(operators={self.list_operators()})"


# Decorator for easy operator registration
def register_operator(cls_or_name: Type[Operator] | str | None = None) -> Type[Operator]:
    """
    Decorator to register an operator class.
    
    Usage:
        @register_operator
        class MyOperator(Operator):
            name = "my_operator"
            ...
            
        # Or with explicit name:
        @register_operator("CustomName")
        class MyOperator(Operator):
            ...
    """
    def decorator(cls: Type[Operator]) -> Type[Operator]:
        # If a custom name was provided, set it on the class
        if isinstance(cls_or_name, str):
            cls.name = cls_or_name
        registry = OperatorRegistry()
        registry.register(cls)
        return cls
    
    # Called without arguments: @register_operator
    if isinstance(cls_or_name, type):
        # cls_or_name is actually the class
        registry = OperatorRegistry()
        registry.register(cls_or_name)
        return cls_or_name
    
    # Called with arguments: @register_operator("name") or @register_operator()
    return decorator


# Global registry instance
_registry = OperatorRegistry()


def get_registry() -> OperatorRegistry:
    """Get the global operator registry instance."""
    return _registry
