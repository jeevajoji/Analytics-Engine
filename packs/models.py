"""
Pack Models and Data Structures.

Defines the core models for the pack system including:
- Pack manifest schema
- Pack types
- Dependency management
"""

import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class PackType(str, Enum):
    """Types of packs available in the marketplace."""
    OPERATOR_PACK = "operator_pack"  # Collection of operators
    PIPELINE_TEMPLATE = "pipeline_template"  # Pre-built pipeline
    CONNECTOR_PACK = "connector_pack"  # Input/output adapters
    SCHEMA_PACK = "schema_pack"  # Schema definitions
    DOMAIN_PACK = "domain_pack"  # Domain-specific operators + templates
    UTILITY_PACK = "utility_pack"  # Helper utilities


class PackStatus(str, Enum):
    """Status of a pack in the registry."""
    DRAFT = "draft"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class PackVersion:
    """Version information for a pack."""
    major: int = 1
    minor: int = 0
    patch: int = 0
    
    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def __lt__(self, other: "PackVersion") -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PackVersion):
            return False
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)
    
    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch))
    
    @classmethod
    def parse(cls, version_str: str) -> "PackVersion":
        """Parse version string."""
        parts = version_str.split(".")
        return cls(
            major=int(parts[0]) if len(parts) > 0 else 1,
            minor=int(parts[1]) if len(parts) > 1 else 0,
            patch=int(parts[2]) if len(parts) > 2 else 0,
        )


@dataclass
class PackDependency:
    """Dependency on another pack."""
    pack_id: str
    version_constraint: str  # e.g., ">=1.0.0", "^1.2.0", "~1.2.3"
    optional: bool = False
    
    def is_satisfied_by(self, version: PackVersion) -> bool:
        """Check if a version satisfies this dependency constraint."""
        # Parse the constraint
        if self.version_constraint.startswith(">="):
            min_version = PackVersion.parse(self.version_constraint[2:])
            return version >= min_version
        elif self.version_constraint.startswith("<="):
            max_version = PackVersion.parse(self.version_constraint[2:])
            return version <= max_version
        elif self.version_constraint.startswith("^"):
            # Caret: allow changes that don't modify left-most non-zero
            min_version = PackVersion.parse(self.version_constraint[1:])
            return (
                version.major == min_version.major and
                version >= min_version
            )
        elif self.version_constraint.startswith("~"):
            # Tilde: allow patch-level changes
            min_version = PackVersion.parse(self.version_constraint[1:])
            return (
                version.major == min_version.major and
                version.minor == min_version.minor and
                version >= min_version
            )
        elif self.version_constraint.startswith("=="):
            exact_version = PackVersion.parse(self.version_constraint[2:])
            return version == exact_version
        else:
            # Assume exact match
            exact_version = PackVersion.parse(self.version_constraint)
            return version == exact_version
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "version_constraint": self.version_constraint,
            "optional": self.optional,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PackDependency":
        return cls(
            pack_id=data["pack_id"],
            version_constraint=data["version_constraint"],
            optional=data.get("optional", False),
        )


@dataclass
class PackAuthor:
    """Author information for a pack."""
    name: str
    email: Optional[str] = None
    url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "email": self.email,
            "url": self.url,
        }
    
    @classmethod
    def from_dict(cls, data: Union[str, Dict[str, Any]]) -> "PackAuthor":
        if isinstance(data, str):
            return cls(name=data)
        return cls(
            name=data["name"],
            email=data.get("email"),
            url=data.get("url"),
        )


@dataclass
class OperatorDefinition:
    """Definition of an operator included in a pack."""
    name: str
    class_name: str
    module_path: str
    description: str = ""
    config_schema: Optional[Dict[str, Any]] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "class_name": self.class_name,
            "module_path": self.module_path,
            "description": self.description,
            "config_schema": self.config_schema,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OperatorDefinition":
        return cls(
            name=data["name"],
            class_name=data["class_name"],
            module_path=data["module_path"],
            description=data.get("description", ""),
            config_schema=data.get("config_schema"),
            input_schema=data.get("input_schema"),
            output_schema=data.get("output_schema"),
            tags=data.get("tags", []),
        )


@dataclass
class PipelineTemplateDefinition:
    """Definition of a pipeline template in a pack."""
    name: str
    description: str
    template_file: str  # Relative path to YAML/JSON template
    variables: Dict[str, Any] = field(default_factory=dict)  # Template variables
    required_operators: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "template_file": self.template_file,
            "variables": self.variables,
            "required_operators": self.required_operators,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineTemplateDefinition":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            template_file=data["template_file"],
            variables=data.get("variables", {}),
            required_operators=data.get("required_operators", []),
            tags=data.get("tags", []),
        )


@dataclass
class ConnectorDefinition:
    """Definition of a connector in a pack."""
    name: str
    connector_type: str  # "input" or "output"
    class_name: str
    module_path: str
    description: str = ""
    config_schema: Optional[Dict[str, Any]] = None
    supports_streaming: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "connector_type": self.connector_type,
            "class_name": self.class_name,
            "module_path": self.module_path,
            "description": self.description,
            "config_schema": self.config_schema,
            "supports_streaming": self.supports_streaming,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConnectorDefinition":
        return cls(
            name=data["name"],
            connector_type=data["connector_type"],
            class_name=data["class_name"],
            module_path=data["module_path"],
            description=data.get("description", ""),
            config_schema=data.get("config_schema"),
            supports_streaming=data.get("supports_streaming", False),
        )


@dataclass
class PackManifest:
    """
    Manifest file for a pack (pack.json).
    
    Contains all metadata and content definitions for a pack.
    """
    pack_id: str
    name: str
    version: PackVersion
    pack_type: PackType
    description: str = ""
    authors: List[PackAuthor] = field(default_factory=list)
    license: str = "MIT"
    homepage: Optional[str] = None
    repository: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    
    # Analytics Engine compatibility
    ae_version_min: str = "1.0.0"
    ae_version_max: Optional[str] = None
    
    # Dependencies
    dependencies: List[PackDependency] = field(default_factory=list)
    python_dependencies: List[str] = field(default_factory=list)  # pip packages
    
    # Content
    operators: List[OperatorDefinition] = field(default_factory=list)
    pipeline_templates: List[PipelineTemplateDefinition] = field(default_factory=list)
    connectors: List[ConnectorDefinition] = field(default_factory=list)
    schemas: List[str] = field(default_factory=list)  # Schema file paths
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: PackStatus = PackStatus.DRAFT
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "name": self.name,
            "version": str(self.version),
            "pack_type": self.pack_type.value,
            "description": self.description,
            "authors": [a.to_dict() for a in self.authors],
            "license": self.license,
            "homepage": self.homepage,
            "repository": self.repository,
            "keywords": self.keywords,
            "ae_version_min": self.ae_version_min,
            "ae_version_max": self.ae_version_max,
            "dependencies": [d.to_dict() for d in self.dependencies],
            "python_dependencies": self.python_dependencies,
            "operators": [o.to_dict() for o in self.operators],
            "pipeline_templates": [t.to_dict() for t in self.pipeline_templates],
            "connectors": [c.to_dict() for c in self.connectors],
            "schemas": self.schemas,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status.value,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PackManifest":
        return cls(
            pack_id=data["pack_id"],
            name=data["name"],
            version=PackVersion.parse(data["version"]),
            pack_type=PackType(data["pack_type"]),
            description=data.get("description", ""),
            authors=[PackAuthor.from_dict(a) for a in data.get("authors", [])],
            license=data.get("license", "MIT"),
            homepage=data.get("homepage"),
            repository=data.get("repository"),
            keywords=data.get("keywords", []),
            ae_version_min=data.get("ae_version_min", "1.0.0"),
            ae_version_max=data.get("ae_version_max"),
            dependencies=[
                PackDependency.from_dict(d) for d in data.get("dependencies", [])
            ],
            python_dependencies=data.get("python_dependencies", []),
            operators=[
                OperatorDefinition.from_dict(o) for o in data.get("operators", [])
            ],
            pipeline_templates=[
                PipelineTemplateDefinition.from_dict(t)
                for t in data.get("pipeline_templates", [])
            ],
            connectors=[
                ConnectorDefinition.from_dict(c) for c in data.get("connectors", [])
            ],
            schemas=data.get("schemas", []),
            created_at=datetime.fromisoformat(data["created_at"])
                if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"])
                if "updated_at" in data else datetime.now(),
            status=PackStatus(data.get("status", "draft")),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "PackManifest":
        return cls.from_dict(json.loads(json_str))


@dataclass
class Pack:
    """
    A loaded pack with all its contents.
    
    This represents a fully loaded pack that can be used by the Analytics Engine.
    """
    manifest: PackManifest
    base_path: str  # Path to pack directory
    loaded_operators: Dict[str, type] = field(default_factory=dict)
    loaded_connectors: Dict[str, type] = field(default_factory=dict)
    loaded_schemas: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def pack_id(self) -> str:
        return self.manifest.pack_id
    
    @property
    def name(self) -> str:
        return self.manifest.name
    
    @property
    def version(self) -> PackVersion:
        return self.manifest.version
    
    @property
    def pack_type(self) -> PackType:
        return self.manifest.pack_type
    
    def get_operator(self, name: str) -> Optional[type]:
        """Get a loaded operator class by name."""
        return self.loaded_operators.get(name)
    
    def get_connector(self, name: str) -> Optional[type]:
        """Get a loaded connector class by name."""
        return self.loaded_connectors.get(name)
    
    def list_operators(self) -> List[str]:
        """List all available operator names."""
        return [op.name for op in self.manifest.operators]
    
    def list_templates(self) -> List[str]:
        """List all available pipeline template names."""
        return [t.name for t in self.manifest.pipeline_templates]
    
    def list_connectors(self) -> List[str]:
        """List all available connector names."""
        return [c.name for c in self.manifest.connectors]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest": self.manifest.to_dict(),
            "base_path": self.base_path,
            "loaded_operators": list(self.loaded_operators.keys()),
            "loaded_connectors": list(self.loaded_connectors.keys()),
            "loaded_schemas": list(self.loaded_schemas.keys()),
        }
