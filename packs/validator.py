"""
Pack Validator Module.

Validates pack structure, manifest, and contents.
"""

import os
import re
import logging
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path


from analytics_engine.packs.models import (
    PackManifest,
    PackType,
    PackVersion,
    PackDependency,
    OperatorDefinition,
    ConnectorDefinition,
    PipelineTemplateDefinition,
)


logger = logging.getLogger(__name__)


class ValidationError:
    """Represents a validation error."""
    
    def __init__(
        self,
        code: str,
        message: str,
        path: Optional[str] = None,
        severity: str = "error",
    ):
        self.code = code
        self.message = message
        self.path = path
        self.severity = severity
    
    def __str__(self) -> str:
        if self.path:
            return f"[{self.code}] {self.path}: {self.message}"
        return f"[{self.code}] {self.message}"


class PackValidator:
    """
    Validates pack structure and content.
    
    Performs validation at multiple levels:
    - Manifest structure
    - Pack contents
    - Dependencies
    - Code quality (optional)
    
    Example:
        validator = PackValidator()
        
        # Validate manifest
        is_valid, errors = validator.validate_manifest(manifest)
        
        # Validate full pack
        is_valid, errors = validator.validate_pack(pack_directory)
    """
    
    # Regex patterns for validation
    PACK_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")
    VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
    MODULE_PATH_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.]*$")
    
    # Required manifest fields
    REQUIRED_MANIFEST_FIELDS = ["pack_id", "name", "version", "pack_type"]
    
    def __init__(
        self,
        strict: bool = False,
        check_code: bool = False,
    ):
        """
        Initialize validator.
        
        Args:
            strict: Enable strict validation
            check_code: Validate Python code syntax
        """
        self.strict = strict
        self.check_code = check_code
    
    def validate_manifest(
        self,
        manifest: PackManifest,
    ) -> Tuple[bool, List[str]]:
        """
        Validate a pack manifest.
        
        Args:
            manifest: PackManifest to validate
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors: List[ValidationError] = []
        
        # Validate pack_id
        if not self.PACK_ID_PATTERN.match(manifest.pack_id):
            errors.append(ValidationError(
                "INVALID_PACK_ID",
                f"Pack ID must be lowercase alphanumeric with underscores/hyphens, "
                f"starting with a letter. Got: {manifest.pack_id}",
                "pack_id",
            ))
        
        # Validate name
        if not manifest.name or len(manifest.name) < 2:
            errors.append(ValidationError(
                "INVALID_NAME",
                "Pack name must be at least 2 characters",
                "name",
            ))
        
        if len(manifest.name) > 100:
            errors.append(ValidationError(
                "NAME_TOO_LONG",
                "Pack name must be 100 characters or less",
                "name",
            ))
        
        # Validate version
        if not self.VERSION_PATTERN.match(str(manifest.version)):
            errors.append(ValidationError(
                "INVALID_VERSION",
                f"Version must be in format X.Y.Z. Got: {manifest.version}",
                "version",
            ))
        
        # Validate AE compatibility
        if manifest.ae_version_min:
            if not self.VERSION_PATTERN.match(manifest.ae_version_min):
                errors.append(ValidationError(
                    "INVALID_AE_VERSION",
                    f"ae_version_min must be in format X.Y.Z",
                    "ae_version_min",
                ))
        
        # Validate operators
        operator_names = set()
        for i, op in enumerate(manifest.operators):
            op_errors = self._validate_operator(op, f"operators[{i}]")
            errors.extend(op_errors)
            
            if op.name in operator_names:
                errors.append(ValidationError(
                    "DUPLICATE_OPERATOR",
                    f"Duplicate operator name: {op.name}",
                    f"operators[{i}].name",
                ))
            operator_names.add(op.name)
        
        # Validate connectors
        connector_names = set()
        for i, conn in enumerate(manifest.connectors):
            conn_errors = self._validate_connector(conn, f"connectors[{i}]")
            errors.extend(conn_errors)
            
            if conn.name in connector_names:
                errors.append(ValidationError(
                    "DUPLICATE_CONNECTOR",
                    f"Duplicate connector name: {conn.name}",
                    f"connectors[{i}].name",
                ))
            connector_names.add(conn.name)
        
        # Validate templates
        template_names = set()
        for i, tmpl in enumerate(manifest.pipeline_templates):
            tmpl_errors = self._validate_template(tmpl, f"pipeline_templates[{i}]")
            errors.extend(tmpl_errors)
            
            if tmpl.name in template_names:
                errors.append(ValidationError(
                    "DUPLICATE_TEMPLATE",
                    f"Duplicate template name: {tmpl.name}",
                    f"pipeline_templates[{i}].name",
                ))
            template_names.add(tmpl.name)
        
        # Validate dependencies
        for i, dep in enumerate(manifest.dependencies):
            dep_errors = self._validate_dependency(dep, f"dependencies[{i}]")
            errors.extend(dep_errors)
        
        # Strict mode checks
        if self.strict:
            if not manifest.description:
                errors.append(ValidationError(
                    "MISSING_DESCRIPTION",
                    "Description is required in strict mode",
                    "description",
                    "warning",
                ))
            
            if not manifest.authors:
                errors.append(ValidationError(
                    "MISSING_AUTHORS",
                    "At least one author is required in strict mode",
                    "authors",
                    "warning",
                ))
        
        # Filter by severity for final result
        error_messages = [
            str(e) for e in errors if e.severity == "error"
        ]
        
        return len(error_messages) == 0, error_messages
    
    def validate_pack(
        self,
        pack_directory: str,
    ) -> Tuple[bool, List[str]]:
        """
        Validate a complete pack (directory + contents).
        
        Args:
            pack_directory: Path to pack directory
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors: List[str] = []
        pack_path = Path(pack_directory)
        
        # Check directory exists
        if not pack_path.exists():
            return False, [f"Pack directory not found: {pack_directory}"]
        
        # Find and load manifest
        manifest_path = None
        for name in ["pack.json", "manifest.json", "package.json"]:
            if (pack_path / name).exists():
                manifest_path = pack_path / name
                break
        
        if not manifest_path:
            return False, ["No manifest file found (pack.json)"]
        
        try:
            import json
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            manifest = PackManifest.from_dict(manifest_data)
        except Exception as e:
            return False, [f"Failed to parse manifest: {e}"]
        
        # Validate manifest
        is_valid, manifest_errors = self.validate_manifest(manifest)
        errors.extend(manifest_errors)
        
        # Validate file existence
        for op in manifest.operators:
            module_file = self._module_to_file(pack_path, op.module_path)
            if not module_file:
                errors.append(
                    f"Operator module not found: {op.module_path}"
                )
        
        for conn in manifest.connectors:
            module_file = self._module_to_file(pack_path, conn.module_path)
            if not module_file:
                errors.append(
                    f"Connector module not found: {conn.module_path}"
                )
        
        for tmpl in manifest.pipeline_templates:
            tmpl_file = pack_path / tmpl.template_file
            if not tmpl_file.exists():
                errors.append(
                    f"Template file not found: {tmpl.template_file}"
                )
        
        for schema_path in manifest.schemas:
            schema_file = pack_path / schema_path
            if not schema_file.exists():
                errors.append(
                    f"Schema file not found: {schema_path}"
                )
        
        # Validate Python code (optional)
        if self.check_code:
            code_errors = self._validate_python_files(pack_path)
            errors.extend(code_errors)
        
        return len(errors) == 0, errors
    
    def _validate_operator(
        self,
        op: OperatorDefinition,
        path: str,
    ) -> List[ValidationError]:
        """Validate an operator definition."""
        errors = []
        
        if not op.name:
            errors.append(ValidationError(
                "MISSING_NAME",
                "Operator name is required",
                f"{path}.name",
            ))
        
        if not op.class_name:
            errors.append(ValidationError(
                "MISSING_CLASS_NAME",
                "Operator class_name is required",
                f"{path}.class_name",
            ))
        
        if not op.module_path:
            errors.append(ValidationError(
                "MISSING_MODULE_PATH",
                "Operator module_path is required",
                f"{path}.module_path",
            ))
        elif not self.MODULE_PATH_PATTERN.match(op.module_path):
            errors.append(ValidationError(
                "INVALID_MODULE_PATH",
                f"Invalid module path format: {op.module_path}",
                f"{path}.module_path",
            ))
        
        return errors
    
    def _validate_connector(
        self,
        conn: ConnectorDefinition,
        path: str,
    ) -> List[ValidationError]:
        """Validate a connector definition."""
        errors = []
        
        if not conn.name:
            errors.append(ValidationError(
                "MISSING_NAME",
                "Connector name is required",
                f"{path}.name",
            ))
        
        if conn.connector_type not in ["input", "output"]:
            errors.append(ValidationError(
                "INVALID_CONNECTOR_TYPE",
                "Connector type must be 'input' or 'output'",
                f"{path}.connector_type",
            ))
        
        if not conn.class_name:
            errors.append(ValidationError(
                "MISSING_CLASS_NAME",
                "Connector class_name is required",
                f"{path}.class_name",
            ))
        
        if not conn.module_path:
            errors.append(ValidationError(
                "MISSING_MODULE_PATH",
                "Connector module_path is required",
                f"{path}.module_path",
            ))
        
        return errors
    
    def _validate_template(
        self,
        tmpl: PipelineTemplateDefinition,
        path: str,
    ) -> List[ValidationError]:
        """Validate a pipeline template definition."""
        errors = []
        
        if not tmpl.name:
            errors.append(ValidationError(
                "MISSING_NAME",
                "Template name is required",
                f"{path}.name",
            ))
        
        if not tmpl.template_file:
            errors.append(ValidationError(
                "MISSING_TEMPLATE_FILE",
                "Template file path is required",
                f"{path}.template_file",
            ))
        
        return errors
    
    def _validate_dependency(
        self,
        dep: PackDependency,
        path: str,
    ) -> List[ValidationError]:
        """Validate a dependency definition."""
        errors = []
        
        if not dep.pack_id:
            errors.append(ValidationError(
                "MISSING_PACK_ID",
                "Dependency pack_id is required",
                f"{path}.pack_id",
            ))
        
        if not dep.version_constraint:
            errors.append(ValidationError(
                "MISSING_VERSION_CONSTRAINT",
                "Dependency version_constraint is required",
                f"{path}.version_constraint",
            ))
        
        # Validate constraint format
        valid_prefixes = [">=", "<=", "^", "~", "==", ""]
        has_valid_prefix = any(
            dep.version_constraint.startswith(p) for p in valid_prefixes
        )
        
        if not has_valid_prefix:
            errors.append(ValidationError(
                "INVALID_VERSION_CONSTRAINT",
                f"Invalid version constraint format: {dep.version_constraint}",
                f"{path}.version_constraint",
            ))
        
        return errors
    
    def _module_to_file(
        self,
        base_path: Path,
        module_path: str,
    ) -> Optional[Path]:
        """Convert module path to file path."""
        parts = module_path.split(".")
        
        # Try as a module file
        module_file = base_path / "/".join(parts[:-1]) / f"{parts[-1]}.py"
        if module_file.exists():
            return module_file
        
        module_file = base_path / f"{'/'.join(parts)}.py"
        if module_file.exists():
            return module_file
        
        # Try as a package
        package_path = base_path / "/".join(parts)
        if package_path.exists() and (package_path / "__init__.py").exists():
            return package_path / "__init__.py"
        
        return None
    
    def _validate_python_files(self, pack_path: Path) -> List[str]:
        """Validate Python files for syntax errors."""
        errors = []
        
        import ast
        
        for py_file in pack_path.rglob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    source = f.read()
                ast.parse(source)
            except SyntaxError as e:
                errors.append(
                    f"Syntax error in {py_file.relative_to(pack_path)}: "
                    f"line {e.lineno}: {e.msg}"
                )
        
        return errors


def validate_pack_manifest(manifest_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Convenience function to validate a manifest dictionary.
    
    Args:
        manifest_dict: Manifest as dictionary
        
    Returns:
        Tuple of (is_valid, list of errors)
    """
    try:
        manifest = PackManifest.from_dict(manifest_dict)
    except Exception as e:
        return False, [f"Failed to parse manifest: {e}"]
    
    validator = PackValidator()
    return validator.validate_manifest(manifest)
