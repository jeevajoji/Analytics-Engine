"""
Tests for the Operator Registry.
"""

import pytest
import polars as pl

from analytics_engine.core.operator import Operator, OperatorConfig, OperatorResult
from analytics_engine.core.registry import (
    OperatorRegistry,
    OperatorNotFoundError,
    OperatorRegistrationError,
    register_operator,
    get_registry,
)


# Test operators for registry tests
class DummyConfig(OperatorConfig):
    value: int = 0


class DummyOperator(Operator[DummyConfig]):
    name = "dummy_operator"
    description = "Dummy operator for testing"
    version = "1.0.0"
    
    def process(self, data: pl.DataFrame) -> OperatorResult:
        return OperatorResult(success=True, data=data)


class AnotherDummyOperator(Operator[DummyConfig]):
    name = "another_dummy"
    description = "Another dummy operator"
    version = "1.0.0"
    
    def process(self, data: pl.DataFrame) -> OperatorResult:
        return OperatorResult(success=True, data=data)


class TestOperatorRegistry:
    """Tests for OperatorRegistry."""
    
    def setup_method(self):
        """Reset registry before each test."""
        OperatorRegistry.reset()
    
    def test_singleton_pattern(self):
        """Test that registry is a singleton."""
        reg1 = OperatorRegistry()
        reg2 = OperatorRegistry()
        assert reg1 is reg2
    
    def test_register_operator(self):
        """Test registering an operator."""
        registry = OperatorRegistry()
        registry.register(DummyOperator)
        
        assert registry.exists("dummy_operator")
        assert "dummy_operator" in registry
    
    def test_register_non_operator_fails(self):
        """Test that registering non-operator class fails."""
        registry = OperatorRegistry()
        
        class NotAnOperator:
            pass
        
        with pytest.raises(OperatorRegistrationError):
            registry.register(NotAnOperator)
    
    def test_get_operator(self):
        """Test getting an operator instance."""
        registry = OperatorRegistry()
        registry.register(DummyOperator)
        
        op = registry.get("dummy_operator")
        
        assert isinstance(op, DummyOperator)
        assert op.name == "dummy_operator"
    
    def test_get_operator_with_config(self):
        """Test getting operator with configuration."""
        registry = OperatorRegistry()
        registry.register(DummyOperator)
        
        config = DummyConfig(value=42)
        op = registry.get("dummy_operator", config=config)
        
        assert op.config.value == 42
    
    def test_get_nonexistent_operator_fails(self):
        """Test that getting non-existent operator fails."""
        registry = OperatorRegistry()
        
        with pytest.raises(OperatorNotFoundError):
            registry.get("nonexistent")
    
    def test_get_class(self):
        """Test getting operator class without instantiating."""
        registry = OperatorRegistry()
        registry.register(DummyOperator)
        
        op_class = registry.get_class("dummy_operator")
        
        assert op_class is DummyOperator
    
    def test_unregister_operator(self):
        """Test unregistering an operator."""
        registry = OperatorRegistry()
        registry.register(DummyOperator)
        
        assert registry.exists("dummy_operator")
        
        registry.unregister("dummy_operator")
        
        assert not registry.exists("dummy_operator")
    
    def test_list_operators(self):
        """Test listing all operators."""
        registry = OperatorRegistry()
        registry.register(DummyOperator)
        registry.register(AnotherDummyOperator)
        
        operators = registry.list_operators()
        
        assert "dummy_operator" in operators
        assert "another_dummy" in operators
        assert len(operators) == 2
    
    def test_get_all_info(self):
        """Test getting info about all operators."""
        registry = OperatorRegistry()
        registry.register(DummyOperator)
        
        info = registry.get_all_info()
        
        assert len(info) == 1
        assert info[0]["name"] == "dummy_operator"
        assert info[0]["version"] == "1.0.0"
    
    def test_len(self):
        """Test registry length."""
        registry = OperatorRegistry()
        assert len(registry) == 0
        
        registry.register(DummyOperator)
        assert len(registry) == 1
    
    def test_repr(self):
        """Test registry string representation."""
        registry = OperatorRegistry()
        registry.register(DummyOperator)
        
        repr_str = repr(registry)
        
        assert "OperatorRegistry" in repr_str
        assert "dummy_operator" in repr_str


class TestRegisterDecorator:
    """Tests for the register_operator decorator."""
    
    def setup_method(self):
        """Reset registry before each test."""
        OperatorRegistry.reset()
    
    def test_decorator_registers_operator(self):
        """Test that decorator registers the operator."""
        
        @register_operator
        class DecoratedOperator(Operator):
            name = "decorated_operator"
            description = "Decorated operator"
            version = "1.0.0"
            
            def process(self, data: pl.DataFrame) -> OperatorResult:
                return OperatorResult(success=True, data=data)
        
        registry = get_registry()
        assert registry.exists("decorated_operator")
    
    def test_decorator_returns_class(self):
        """Test that decorator returns the original class."""
        
        @register_operator
        class DecoratedOperator(Operator):
            name = "decorated_operator_2"
            description = "Decorated operator"
            version = "1.0.0"
            
            def process(self, data: pl.DataFrame) -> OperatorResult:
                return OperatorResult(success=True, data=data)
        
        assert DecoratedOperator.name == "decorated_operator_2"


class TestGetRegistry:
    """Tests for get_registry function."""
    
    def setup_method(self):
        """Reset registry before each test."""
        OperatorRegistry.reset()
    
    def test_get_registry_returns_singleton(self):
        """Test that get_registry returns the singleton."""
        reg1 = get_registry()
        reg2 = get_registry()
        
        assert reg1 is reg2
    
    def test_get_registry_same_as_constructor(self):
        """Test that get_registry returns same instance as constructor."""
        reg1 = get_registry()
        reg2 = OperatorRegistry()
        
        assert reg1 is reg2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
